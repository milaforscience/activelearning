import torch
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, cast
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.module import Module

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation
from activelearning.dataset.dataset import Dataset


class BoTorchSurrogate(Surrogate):
    """A highly flexible Gaussian Process surrogate using BoTorch. 
    
    Automatically handles single-fidelity and multi-fidelity configurations, 
    data scaling, and supports both full retraining and fast low-rank updates.
    """
   
    def __init__(
        self,
        normalize_inputs: bool = True,
        standardize_outputs: bool = True,
        optimize_hyperparameters: bool = True,
        fit_kwargs: Optional[dict[str, Any]] = None,
        custom_fit_function: Optional[Callable] = None,
        covar_module: Optional[Module] = None,
        use_partial_updates: bool = False,
        custom_fidelity_kernel: bool = False
    ):
        """
        Parameters
        ----------
        normalize_inputs : bool, default=True
            If True, scales X values to [0, 1] internally.
        standardize_outputs : bool, default=True
            If True, scales Y values to mean=0, var=1 internally.
        optimize_hyperparameters : bool, default=True
            If True, fits kernel hyperparameters during fit(). Set to False if you want 
            to use a custom kernel's default initialization or if you are injecting a 
            pre-trained state_dict.
        fit_kwargs : dict, optional
            Keyword arguments passed directly to the optimizer (e.g., BoTorch's 
            fit_gpytorch_mll or the custom_fit_function).
        custom_fit_function : callable, optional
            A custom function to handle hyperparameter optimization (e.g., a custom 
            Adam training loop for Deep Kernel Learning). If provided, this overrides 
            BoTorch's default L-BFGS-B optimizer.
        covar_module : Module, optional
            Optional custom GPyTorch kernel for non-fidelity dimensions. If None, 
            defaults to BoTorch's standard Matérn/RBF kernels.
        use_partial_updates : bool, default=False
            If True, update() uses fast incremental Cholesky updates when the model is 
            already fitted. If False, update() always performs full retraining for 
            maximum reliability. Beginners should use False.
        custom_fidelity_kernel : bool, default=False
            If True, bypasses BoTorch's default multi-fidelity model and allows the 
            custom `covar_module` to handle fidelity dimensions internally.
        """
        self.model = None
        self.mll = None
        
        # Toggles and configurations
        self.normalize_inputs = normalize_inputs
        self.standardize_outputs = standardize_outputs
        self.optimize_hyperparameters = optimize_hyperparameters
        self.fit_kwargs = fit_kwargs or {}
        self.custom_fit_function = custom_fit_function
        self.covar_module = covar_module
        self.use_partial_updates = use_partial_updates
        self.custom_fidelity_kernel = custom_fidelity_kernel
        
        # Internal state tracking
        self._is_multi_fidelity = False
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None
        self._fidelity_confidences: dict[int, float] = {}
        self._pending_state_dict: Optional[dict[str, torch.Tensor]] = None

    def _parse_observations(self, observations: Iterable[Observation]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts generic Observations into BoTorch-ready tensors.

        Also updates ``_is_multi_fidelity`` based on whether the supplied
        observations contain fidelity values. This flag is reset on every
        call, so repeated calls with different observation types always
        leave the surrogate in a consistent state.

        Parameters
        ----------
        observations : Iterable[Observation]
            An iterable of Observation objects containing features, labels, and
            optional fidelity metadata.

        Returns
        -------
        train_X : torch.Tensor
            A tensor of input features, with mapped fidelity confidences appended
            as the final column if multi-fidelity.
        train_Y : torch.Tensor
            A tensor of observed labels reshaped for BoTorch.

        Raises
        ------
        ValueError
            If multi-fidelity observations are provided but not all observations
            contain a fidelity value.
        """
        # Reset on every call so that switching from MF → SF (or vice versa)
        # never leaves stale state from a previous invocation.
        self._is_multi_fidelity = False

        x_data, y_data, fidelities = [], [], []
        obs_list = list(observations)

        for obs in obs_list:
            # Safely convert x to a 1D tensor
            x_tensor = torch.atleast_1d(torch.as_tensor(obs.x, dtype=torch.float64))
            x_data.append(x_tensor)
            y_data.append(torch.as_tensor(obs.y, dtype=torch.float64))
           
            if obs.fidelity is not None:
                # Map the integer ID to the continuous confidence value.
                mapped_fid = self._fidelity_confidences.get(obs.fidelity, obs.fidelity)
                fidelities.append(mapped_fid)

        train_X = torch.stack(x_data)
        train_Y = torch.stack(y_data).view(-1, 1)  # BoTorch expects shape (n, 1)

        # Handle multi-fidelity concatenation
        if fidelities:
            if len(fidelities) != len(obs_list):
                raise ValueError("If using multi-fidelity, all observations must have a fidelity.")
           
            self._is_multi_fidelity = True
            fid_tensor = torch.tensor(fidelities, dtype=torch.float64).view(-1, 1)
            # Append fidelity as the final column of the training data
            train_X = torch.cat([train_X, fid_tensor], dim=-1)

        return train_X, train_Y

    def _parse_candidates(self, candidates: Sequence[Candidate]) -> torch.Tensor:
        """Converts Candidates into BoTorch-ready tensors.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            A sequence of Candidate objects to be evaluated.

        Returns
        -------
        test_X : torch.Tensor
            A tensor of candidate features, with mapped fidelity confidences appended 
            as the final column if multi-fidelity.

        Raises
        ------
        ValueError
            If the model was trained on multi-fidelity data, but candidates are 
            missing fidelity values.
        """
        x_data, fidelities = [], []

        for cand in candidates:
            x_tensor = torch.atleast_1d(torch.as_tensor(cand.x, dtype=torch.float64))
            x_data.append(x_tensor)
           
            if cand.fidelity is not None:
                # Map the integer ID to the continuous confidence value.
                mapped_fid = self._fidelity_confidences.get(cand.fidelity, cand.fidelity)
                fidelities.append(mapped_fid)

        test_X = torch.stack(x_data)

        if self._is_multi_fidelity:
            if len(fidelities) != len(candidates):
                raise ValueError("Surrogate was fitted on multi-fidelity data. Candidates require fidelities.")
           
            fid_tensor = torch.tensor(fidelities, dtype=torch.float64).view(-1, 1)
            test_X = torch.cat([test_X, fid_tensor], dim=-1)

        return test_X

    def _build_and_fit_model(self) -> None:
        """Constructs and optionally optimizes the Gaussian Process model."""
        # _build_and_fit_model is only called after _parse_observations has populated
        # these tensors; assert here so the type checker sees them as non-None.
        assert self._train_X is not None and self._train_Y is not None

        # 1. Setup Transforms
        input_transform = Normalize(d=self._train_X.shape[-1]) if self.normalize_inputs else None
        outcome_transform = Standardize(m=self._train_Y.shape[-1]) if self.standardize_outputs else None

        # 2. Build Model Configuration
        # If MF is true AND they aren't using a custom fidelity kernel, use BoTorch's default MF model
        if self._is_multi_fidelity and not self.custom_fidelity_kernel:
            self.model = SingleTaskMultiFidelityGP(
                self._train_X, 
                self._train_Y, 
                data_fidelities=[-1],
                outcome_transform=outcome_transform,
                input_transform=input_transform
            )
        else:
            # Fallback for Single Fidelity OR Custom Multi-Fidelity Kernels (like DKL)
            self.model = SingleTaskGP(
                self._train_X, 
                self._train_Y,
                covar_module=self.covar_module, # The user's DKL kernel is passed here
                outcome_transform=outcome_transform,
                input_transform=input_transform
            )
           
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        # 3. Inject pre-trained hyperparameters if the user provided them before fitting
        if self._pending_state_dict is not None:
            self.model.load_state_dict(self._pending_state_dict)
            self._pending_state_dict = None
        
        # 4. Optimize Hyperparameters (if toggled on)
        if self.optimize_hyperparameters:
            if self.custom_fit_function is not None:
                # Use the user's custom fit function if provided
                self.custom_fit_function(self.mll, **self.fit_kwargs)
            else:
                # Default BoTorch fitting procedure
                fit_gpytorch_mll(self.mll, **self.fit_kwargs)
            
        self.model.eval()
    
    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        """Stores fidelity confidences and passes them to the custom kernel if supported.

        Parameters
        ----------
        confidences : dict[int, float]
            Mapping of fidelity levels (integer indices) to confidence
            values in the range [0, 1].
        """
        self._fidelity_confidences = confidences
        
        # Pass to custom kernel if it exists.  cast to Any because update_confidences
        # is an optional protocol not declared on gpytorch.Module.
        if self.covar_module is not None and hasattr(self.covar_module, "update_confidences"):
            cast(Any, self.covar_module).update_confidences(confidences)

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fits the GP model from scratch, overwriting any previous data.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of observations to train on.
        """
        self._train_X, self._train_Y = self._parse_observations(observations)
        self._build_and_fit_model()

    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, Any]:
        """Returns predictions including the mean, std, and the full BoTorch posterior.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to predict.

        Returns
        -------
        result : Mapping[str, Any]
            Dictionary containing prediction keys:
            - "mean": List[float] representing the predicted means.
            - "std": List[float] representing the predicted standard deviations.
            - "posterior": The raw BoTorch posterior object.

        Raises
        ------
        RuntimeError
            If called before the surrogate model has been fitted.
        """
        if self.model is None:
            raise RuntimeError("Surrogate model must be fitted before calling predict().")
           
        test_X = self._parse_candidates(candidates)
       
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_X)
           
        return {
            "mean": posterior.mean.squeeze(-1).tolist(),
            "std": posterior.variance.sqrt().squeeze(-1).tolist(),
            "posterior": posterior
        }
    
    def update(self, dataset: "Dataset") -> None:
        """Updates the surrogate with observations from the dataset.
        
        This method decides internally whether to do a full refit or partial update
        based on the `use_partial_updates` configuration.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset containing all observations and latest observations.
            
        Notes
        -----
        - If `use_partial_updates=False` (default): Always performs full retraining
          from scratch using all observations in the dataset. This is the most
          reliable option and recommended for beginners and small-to-medium datasets.
        - If `use_partial_updates=True` and model is already fitted: Uses only the
          latest observations for a fast Cholesky update without retraining
          hyperparameters. This is faster but may accumulate numerical errors.
        - First call always does a full fit (no existing model to update).
        """
        if self.use_partial_updates and self.model is not None:
            # Fast incremental update using only latest observations
            new_observations = dataset.get_latest_observations_iterable()
            self._partial_update(new_observations)
        else:
            # Full refit from scratch using all observations
            all_observations = dataset.get_observations_iterable()
            self.fit(all_observations)
    
    def _partial_update(self, new_observations: Iterable[Observation]) -> None:
        """Fast, low-rank update of the GP without retraining hyperparameters.
        
        This is an internal method called by `update()` when `use_partial_updates=True`.

        Parameters
        ----------
        new_observations : Iterable[Observation]
            Iterable of strictly new observations to condition the GP on.
        """
        if self.model is None or self._train_X is None:
            # Fallback to a full fit if the model hasn't been initialized
            self.fit(new_observations)
            return

        obs_list = list(new_observations)
        if not obs_list:
            return

        new_X, new_Y = self._parse_observations(obs_list)

        # _train_X/_train_Y are non-None here: the guard at the top of this
        # method returns early when self._train_X is None.
        assert self._train_X is not None and self._train_Y is not None

        # Update historical state for future full refits
        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        # Fast Cholesky update (internal data scaling transforms apply automatically)
        # Note: condition_on_observations requires the model to have made at least one prediction
        # to populate internal caches. We need to trigger a prediction first.
        self.model.eval()
        with torch.no_grad():
            # Make a dummy prediction to initialize caches
            _ = self.model.posterior(new_X[:1])
        
        self.model = self.model.condition_on_observations(X=new_X, Y=new_Y)

    def state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        """Extracts the model's hyperparameters (lengthscales, noise, etc.).

        Returns
        -------
        state_dict : dict or None
            A dictionary containing the PyTorch model state, or None if the model 
            has not been fitted yet.
        """
        if self.model is not None:
            return self.model.state_dict()
        return None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Injects a pre-trained hyperparameter dictionary into the model.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing the previously saved PyTorch model state.
        """
        if self.model is not None:
            self.model.load_state_dict(state_dict)
        else:
            # Store it for when _build_and_fit_model is called
            self._pending_state_dict = state_dict
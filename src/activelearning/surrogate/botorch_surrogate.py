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
from activelearning.utils.types import Candidate, Observation, observations_to_tensors, candidates_to_tensor


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
            Optional custom GPyTorch kernel passed to ``SingleTaskGP``. In the
            single-fidelity case it covers only the input dimensions. In the
            multi-fidelity case, providing a ``covar_module`` bypasses
            ``SingleTaskMultiFidelityGP`` in favour of ``SingleTaskGP``, and
            the kernel receives all dimensions including the fidelity column
            (appended as the last column), so it is responsible for both input
            and fidelity dimensions (e.g. when using a composite or product
            kernel that encodes fidelity correlations directly). If None,
            defaults to BoTorch's standard Matérn/RBF kernels; for
            multi-fidelity data this means ``SingleTaskMultiFidelityGP`` is used.
        use_partial_updates : bool, default=False
            If True, update() uses fast incremental Cholesky updates when the model is
            already fitted. If False, update() always performs full retraining for
            maximum reliability. Beginners should use False.
        """
        # Toggles and configurations
        self.normalize_inputs = normalize_inputs
        self.standardize_outputs = standardize_outputs
        self.optimize_hyperparameters = optimize_hyperparameters
        self.fit_kwargs = fit_kwargs or {}
        self.custom_fit_function = custom_fit_function
        self.covar_module = covar_module
        self.use_partial_updates = use_partial_updates

        if custom_fit_function is not None and not optimize_hyperparameters:
            raise ValueError(
                "custom_fit_function is provided but optimize_hyperparameters=False. "
                "The custom function will never be called. Either set "
                "optimize_hyperparameters=True or remove custom_fit_function."
            )

        # Internal state tracking
        self.model = None
        self.mll = None
        self._is_multi_fidelity = False
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None
        self._fidelity_confidences: dict[int, float] = {}
        self._pending_state_dict: Optional[dict[str, torch.Tensor]] = None

    def _parse_observations(
        self, observations: Iterable[Observation]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        X, y, fidelities = observations_to_tensors(
            observations, self._fidelity_confidences
        )
        # BoTorch requires 2D inputs: ensure (n, d) for X and (n, 1) for Y
        train_X = torch.atleast_2d(X)
        train_Y = y.view(-1, 1)
        obs_count = train_X.shape[0]

        # Handle multi-fidelity concatenation
        if fidelities:
            if len(fidelities) != obs_count:
                raise ValueError(
                    "If using multi-fidelity, all observations must have a fidelity."
                )

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
        test_X, fidelities = candidates_to_tensor(candidates, self._fidelity_confidences)
        # BoTorch requires 2D inputs: ensure (n, d)
        test_X = torch.atleast_2d(test_X)

        if self._is_multi_fidelity:
            if len(fidelities) != len(candidates):
                raise ValueError(
                    "Surrogate was fitted on multi-fidelity data. Candidates require fidelities."
                )

            fid_tensor = torch.tensor(fidelities, dtype=torch.float64).view(-1, 1)
            test_X = torch.cat([test_X, fid_tensor], dim=-1)

        return test_X

    def _build_and_fit_model(self) -> None:
        """Constructs and optionally optimizes the Gaussian Process model."""
        assert self._train_X is not None and self._train_Y is not None, (
            "_parse_observations() must be called before _build_and_fit_model()."
        )

        # 1. Setup Transforms
        # When multi-fidelity, normalize only the feature columns (all except the last),
        # leaving the fidelity confidence column in its original [0, 1] range.
        n_dims = self._train_X.shape[-1]
        if self.normalize_inputs:
            if self._is_multi_fidelity:
                feature_indices = list(range(n_dims - 1))
                input_transform = Normalize(d=n_dims, indices=feature_indices)
            else:
                input_transform = Normalize(d=n_dims)
        else:
            input_transform = None
        outcome_transform = (
            Standardize(m=self._train_Y.shape[-1]) if self.standardize_outputs else None
        )

        # 2. Build Model Configuration
        # Use BoTorch's default MF model only when MF data is present and no custom
        # kernel is provided. A custom covar_module implies the user wants SingleTaskGP
        # with full control over the kernel (including fidelity dimensions).
        if self._is_multi_fidelity and self.covar_module is None:
            self.model = SingleTaskMultiFidelityGP(
                self._train_X,
                self._train_Y,
                data_fidelities=[-1],
                outcome_transform=outcome_transform,
                input_transform=input_transform,
            )
        else:
            # Single-fidelity OR multi-fidelity with a custom kernel
            self.model = SingleTaskGP(
                self._train_X,
                self._train_Y,
                covar_module=self.covar_module,  # The user's kernel is passed here
                outcome_transform=outcome_transform,
                input_transform=input_transform,
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
        if self.covar_module is not None and hasattr(
            self.covar_module, "update_confidences"
        ):
            cast(Any, self.covar_module).update_confidences(confidences)

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fits the GP model from scratch, overwriting any previous data.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of observations to train on.

        Raises
        ------
        ValueError
            If observations is empty.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("Cannot fit on an empty observation list.")
        self._train_X, self._train_Y = self._parse_observations(obs_list)
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
            raise RuntimeError(
                "Surrogate model must be fitted before calling predict()."
            )

        test_X = self._parse_candidates(candidates)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_X)

        return {
            "mean": posterior.mean.squeeze(-1).tolist(),
            "std": posterior.variance.sqrt().squeeze(-1).tolist(),
            "posterior": posterior,
        }

    def updates_from_latest(self) -> bool:
        """Return True when the model is fitted and partial updates are enabled.

        The active learning loop uses this to decide whether to call
        ``update(latest_observations)`` (incremental Cholesky) or
        ``fit(all_observations)`` (full retraining).

        Returns
        -------
        bool
            ``True`` if ``use_partial_updates=True`` and the model has already
            been fitted at least once; ``False`` otherwise.
        """
        return self.use_partial_updates and self.model is not None

    def update(self, observations: Iterable[Observation]) -> None:
        """Incrementally update the GP with the latest (new) observations.

        Called by the active learning loop only when ``updates_from_latest()``
        returns ``True``. Uses a fast low-rank Cholesky conditioning step without
        retraining hyperparameters.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of the most recent observations to condition the GP on.
        """
        self._partial_update(observations)

    def _partial_update(self, new_observations: Iterable[Observation]) -> None:
        """Fast, low-rank update of the GP without retraining hyperparameters.

        This is an internal method called by `update()` when `use_partial_updates=True`.

        Parameters
        ----------
        new_observations : Iterable[Observation]
            The latest batch of new observations to condition the GP on. Must be
            a finite iterable; it will be fully materialized into a list internally.
        """
        if self.model is None or self._train_X is None:
            # Fallback to a full fit if the model hasn't been initialized
            self.fit(new_observations)
            return

        obs_list = list(new_observations)
        if not obs_list:
            return

        # Validate fidelity consistency before parsing: _parse_observations resets
        # _is_multi_fidelity based only on the new batch, so a batch missing fidelities
        # would silently produce wrong-shaped tensors and fail during torch.cat.
        if self._is_multi_fidelity:
            missing = [i for i, obs in enumerate(obs_list) if obs.fidelity is None]
            if missing:
                raise ValueError(
                    f"Surrogate was fitted on multi-fidelity data but new observations "
                    f"at indices {missing} are missing fidelity values."
                )

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

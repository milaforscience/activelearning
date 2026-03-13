import torch
from typing import Any, Callable, Iterable, Mapping, Optional, cast

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.module import Module

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import (
    Candidate,
    Observation,
    candidates_to_tensor,
    observations_to_tensors,
)


class BoTorchGPSurrogate(Surrogate):
    """A highly flexible Gaussian Process surrogate using BoTorch.

    Automatically handles single-fidelity and multi-fidelity configurations,
    data scaling, and supports both full retraining and fast low-rank updates.
    """

    def __init__(
        self,
        scale_inputs: bool = True,
        standardize_outputs: bool = True,
        optimize_hyperparameters: bool = True,
        fit_kwargs: Optional[dict[str, Any]] = None,
        custom_fit_function: Optional[Callable[..., Any]] = None,
        covar_module: Optional[Module] = None,
        use_partial_updates: bool = False,
    ) -> None:
        """Initialize the BoTorch GP surrogate.

        Parameters
        ----------
        scale_inputs : bool, default=True
            If True, scales input values to [0, 1] internally.
        standardize_outputs : bool, default=True
            If True, normalizes output values to mean=0, var=1 internally.
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
        if custom_fit_function is not None and not optimize_hyperparameters:
            raise ValueError(
                "custom_fit_function is provided but optimize_hyperparameters=False. "
                "The custom function will never be called. Either set "
                "optimize_hyperparameters=True or remove custom_fit_function."
            )
        # Toggles and configurations
        self.scale_inputs = scale_inputs
        self.standardize_outputs = standardize_outputs
        self.optimize_hyperparameters = optimize_hyperparameters
        self.fit_kwargs = fit_kwargs or {}
        self.custom_fit_function = custom_fit_function
        self.covar_module = covar_module
        self.use_partial_updates = use_partial_updates

        # Internal state tracking
        self.model: Optional[SingleTaskGP | SingleTaskMultiFidelityGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None
        self._is_multi_fidelity: bool = False
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None
        self._fidelity_confidences: dict[int, float] = {}
        self._pending_state_dict: Optional[dict[str, torch.Tensor]] = None

    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        """Stores fidelity confidences and passes them to the custom kernel if supported.

        Parameters
        ----------
        confidences : dict[int, float]
            Mapping of fidelity levels (integer indices) to confidence
            values in the range [0, 1].
        """
        copied_confidences = dict(confidences)
        if (
            self.model is not None
            or self._train_X is not None
            or self._train_Y is not None
        ) and copied_confidences != self._fidelity_confidences:
            raise RuntimeError(
                "Cannot change fidelity confidences after the surrogate has been "
                "fitted. Set them before calling fit() or update()."
            )
        self._fidelity_confidences = copied_confidences

        # Optional protocol for custom kernels that need access to the mapping.
        if self.covar_module is not None and hasattr(
            self.covar_module, "update_confidences"
        ):
            cast(Any, self.covar_module).update_confidences(copied_confidences)

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the surrogate from scratch on all provided observations.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of observations to train on. Materialized to a list internally.

        Raises
        ------
        ValueError
            If observations is empty or structurally incompatible.
        """
        obs_list = list(observations)
        if not obs_list:
            # No-op on empty data: surrogate stays unfitted. The AL loop checks
            # is_fitted() and will use random candidate selection for this round.
            return

        train_X, train_Y, is_multi_fidelity = self._parse_observations(obs_list)
        self._is_multi_fidelity = is_multi_fidelity
        self._train_X = train_X
        self._train_Y = train_Y

        self._build_model(train_X, train_Y)

        assert self.model is not None
        assert self.mll is not None

        if self.optimize_hyperparameters:
            if self.custom_fit_function is not None:
                self.custom_fit_function(self.mll, **self.fit_kwargs)
            else:
                fit_gpytorch_mll(self.mll, **self.fit_kwargs)

        # Switch the model to evaluation mode so it behaves in inference mode
        # (disables training-specific behavior in PyTorch modules).
        self.model.eval()

    def update(self, observations: Iterable[Observation]) -> None:
        """Update the fitted surrogate using newly acquired observations.

        When ``use_partial_updates=True`` and a model is already fitted, this uses
        fast incremental conditioning. Otherwise, it rebuilds the model on the
        accumulated training data so direct ``update()`` calls still respect the
        constructor contract.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of newly acquired observations.

        Raises
        ------
        ValueError
            If the incoming observations are incompatible with the fitted model
            mode (single-fidelity vs multi-fidelity).
        """
        if self.model is None or self._train_X is None or self._train_Y is None:
            # Fallback to a full fit if the model hasn't been initialized
            self.fit(observations)
            return

        obs_list = list(observations)
        if not obs_list:
            return

        incoming_is_multi_fidelity = self._infer_is_multi_fidelity(obs_list)

        if self._is_multi_fidelity and not incoming_is_multi_fidelity:
            missing = [i for i, obs in enumerate(obs_list) if obs.fidelity is None]
            raise ValueError(
                "Surrogate was fitted in multi-fidelity mode, but new observations "
                f"at indices {missing} are missing fidelity values."
            )
        if not self._is_multi_fidelity and incoming_is_multi_fidelity:
            raise ValueError(
                "Surrogate was fitted in single-fidelity mode, but new observations "
                "include fidelity values."
            )

        new_X, new_Y, parsed_is_multi_fidelity = self._parse_observations(obs_list)
        if parsed_is_multi_fidelity != self._is_multi_fidelity:
            raise ValueError(
                "Incoming observation structure is incompatible with the fitted surrogate."
            )

        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        if not self.use_partial_updates:
            self._build_model(self._train_X, self._train_Y)
            assert self.model is not None
            assert self.mll is not None

            if self.optimize_hyperparameters:
                if self.custom_fit_function is not None:
                    self.custom_fit_function(self.mll, **self.fit_kwargs)
                else:
                    fit_gpytorch_mll(self.mll, **self.fit_kwargs)

            self.model.eval()
            return

        # Fast Cholesky update (internal data scaling transforms apply automatically)
        # Note: condition_on_observations requires the model to have made at least one prediction
        # to populate internal caches. We need to trigger a prediction first.
        self.model.eval()
        with torch.no_grad():
            # Make a dummy prediction to initialize caches
            _ = self.model.posterior(new_X[:1])

        self.model = cast(
            SingleTaskGP | SingleTaskMultiFidelityGP,
            self.model.condition_on_observations(X=new_X, Y=new_Y),
        )
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.model.eval()

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

    def is_fitted(self) -> bool:
        """Return whether the GP model has been built on at least one observation.

        Returns
        -------
        bool
            ``True`` once a GP model has been built; ``False`` before the first
            non-empty ``fit()`` call.
        """
        return self.model is not None

    def predict(self, candidates: Iterable[Candidate]) -> Mapping[str, Any]:
        """Predict posterior quantities for candidates.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Iterable of candidates to evaluate.

        Returns
        -------
        result : Mapping[str, Any]
            Dictionary containing:
            - ``"mean"``: posterior means as a list for single-output models,
              or a nested list of shape ``(N, m)`` for multi-output models
            - ``"std"``: posterior standard deviations with the same shape
              convention as ``"mean"``
            - ``"posterior"``: raw BoTorch posterior object

        Raises
        ------
        RuntimeError
            If called before the surrogate has been fitted.
        """
        model = self.get_model()
        test_X = self.encode_candidates(candidates)

        # Re-assert evaluation mode in case external code switched the model
        # back to training mode between fit() and predict().
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(test_X)

        mean = posterior.mean
        std = posterior.variance.sqrt()

        if mean.ndim >= 2 and mean.shape[-1] == 1:
            mean_out = mean.squeeze(-1).tolist()
            std_out = std.squeeze(-1).tolist()
        else:
            mean_out = mean.tolist()
            std_out = std.tolist()

        return {
            "mean": mean_out,
            "std": std_out,
            "posterior": posterior,
        }

    def state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        """Return the fitted model state dictionary, if available.

        Returns
        -------
        result : Optional[dict[str, torch.Tensor]]
            Fitted model state dictionary, or ``None`` if the model has not yet
            been built.
        """
        if self.model is None:
            return None
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a model state dictionary immediately or defer until model build.

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            Model parameters to load.
        """
        if self.model is not None:
            self.model.load_state_dict(state_dict)
        else:
            # Store it for when _build_model is called
            self._pending_state_dict = state_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def get_model(self) -> SingleTaskGP | SingleTaskMultiFidelityGP:
        """Return the fitted BoTorch model.

        Returns
        -------
        result : SingleTaskGP | SingleTaskMultiFidelityGP
            The fitted BoTorch model.

        Raises
        ------
        RuntimeError
            If called before the surrogate has been fitted.
        """
        if self.model is None:
            raise RuntimeError("BoTorch surrogate model has not been fitted yet.")
        return self.model

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Encode singleton candidates into model-space input tensors.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Iterable of candidates to encode. Materialized to a list internally.

        Returns
        -------
        test_X : torch.Tensor
            Encoded candidate tensor of shape ``(N, d)``.

        Raises
        ------
        ValueError
            If candidates are incompatible with the fitted model mode.
        """
        cand_list = list(candidates)
        if not cand_list:
            raise ValueError("Cannot encode an empty candidate iterable.")

        has_fidelity = [cand.fidelity is not None for cand in cand_list]
        if any(has_fidelity) and not all(has_fidelity):
            missing = [i for i, present in enumerate(has_fidelity) if not present]
            raise ValueError(
                "Mixed fidelity specification detected: either all candidates must "
                f"provide a fidelity or none should. Missing indices: {missing}."
            )

        incoming_is_multi_fidelity = all(has_fidelity)

        if not self._is_multi_fidelity and incoming_is_multi_fidelity:
            raise ValueError(
                "Surrogate was fitted in single-fidelity mode. "
                "Candidates must not provide fidelity values."
            )

        if self._is_multi_fidelity and not incoming_is_multi_fidelity:
            raise ValueError(
                "Surrogate was fitted in multi-fidelity mode. "
                "All candidates must provide a fidelity."
            )

        if self._is_multi_fidelity:
            unknown = [
                i
                for i, cand in enumerate(cand_list)
                if cand.fidelity not in self._fidelity_confidences
            ]
            if unknown:
                raise ValueError(
                    "Some candidate fidelities are not present in the fidelity-confidence "
                    f"map. Invalid candidate indices: {unknown}."
                )

        fidelity_confidences = self._fidelity_confidences or None
        test_X, fidelities = candidates_to_tensor(cand_list, fidelity_confidences)
        test_X = self._ensure_feature_matrix(test_X)

        if self._is_multi_fidelity:
            fid_tensor = torch.tensor(fidelities, dtype=torch.float64).view(-1, 1)
            test_X = torch.cat([test_X, fid_tensor], dim=-1)

        return test_X

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_feature_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize scalar and vector features to BoTorch's ``(N, d)`` layout.

        Parameters
        ----------
        X : torch.Tensor
            Raw feature tensor returned by the generic conversion helpers.

        Returns
        -------
        result : torch.Tensor
            Feature tensor with scalar inputs reshaped to ``(N, 1)`` while
            leaving already batched feature tensors unchanged.
        """
        if X.ndim == 0:
            return X.view(1, 1)
        if X.ndim == 1:
            return X.unsqueeze(-1)
        return X

    def _parse_observations(
        self,
        observations: Iterable[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Converts generic Observations into BoTorch-ready tensors.

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
        is_multi_fidelity : bool
            Whether the observations are structurally multi-fidelity.

        Raises
        ------
        ValueError
            If some but not all observations provide fidelity values.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("Cannot parse an empty observation iterable.")

        is_multi_fidelity = self._infer_is_multi_fidelity(obs_list)

        fidelity_confidences = self._fidelity_confidences or None
        X, y, fidelities = observations_to_tensors(obs_list, fidelity_confidences)
        train_X = self._ensure_feature_matrix(X)
        if y.ndim == 0:
            train_Y = y.view(1, 1)
        elif y.ndim == 1:
            train_Y = y.unsqueeze(-1)
        else:
            train_Y = y
        obs_count = len(obs_list)

        if is_multi_fidelity:
            if len(fidelities) != obs_count:
                raise ValueError(
                    "If using multi-fidelity observations, all observations must "
                    "provide a fidelity value present in the fidelity-confidence map."
                )
            fid_tensor = torch.tensor(fidelities, dtype=torch.float64).view(-1, 1)
            train_X = torch.cat([train_X, fid_tensor], dim=-1)

        return train_X, train_Y, is_multi_fidelity

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Construct the BoTorch Gaussian Process model and marginal log likelihood.

        Parameters
        ----------
        train_X : torch.Tensor
            Training input tensor.
        train_Y : torch.Tensor
            Training target tensor.
        """

        # 1. Setup Transforms
        # When multi-fidelity, scale only the feature columns (all except the last),
        # leaving the fidelity confidence column in its original [0, 1] range.
        n_dims = train_X.shape[-1]

        if self.scale_inputs:
            if self._is_multi_fidelity:
                feature_indices = list(range(n_dims - 1))
                input_transform = Normalize(d=n_dims, indices=feature_indices)
            else:
                input_transform = Normalize(d=n_dims)
        else:
            input_transform = None

        outcome_transform = (
            Standardize(m=train_Y.shape[-1]) if self.standardize_outputs else None
        )

        # 2. Build Model Configuration
        # Use BoTorch's default MF model only when MF data is present and no custom
        # kernel is provided. A custom covar_module implies the user wants SingleTaskGP
        # with full control over the kernel (including fidelity dimensions).
        if self._is_multi_fidelity and self.covar_module is None:
            self.model = SingleTaskMultiFidelityGP(
                train_X,
                train_Y,
                data_fidelities=[-1],
                outcome_transform=outcome_transform,
                input_transform=input_transform,
            )
        else:
            # Single-fidelity OR multi-fidelity with a custom kernel
            # When covar_module is None, BoTorch currently defaults to an RBFKernel
            # inside SingleTaskGP (this default may change across BoTorch versions).
            self.model = SingleTaskGP(
                train_X,
                train_Y,
                covar_module=self.covar_module,  # The user's kernel is passed here
                outcome_transform=outcome_transform,
                input_transform=input_transform,
            )

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # 3. Inject pre-trained hyperparameters if the user provided them before fitting
        if self._pending_state_dict is not None:
            self.model.load_state_dict(self._pending_state_dict)
            self._pending_state_dict = None

    def _infer_is_multi_fidelity(
        self,
        observations: Iterable[Observation],
    ) -> bool:
        """Infer whether a batch of observations is structurally multi-fidelity.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of observations to inspect.

        Returns
        -------
        result : bool
            ``True`` if all observations provide a fidelity, ``False`` if none do.

        Raises
        ------
        ValueError
            If only some observations provide fidelity values.
        """
        obs_list = list(observations)
        has_fidelity = [obs.fidelity is not None for obs in obs_list]

        if any(has_fidelity) and not all(has_fidelity):
            missing = [i for i, present in enumerate(has_fidelity) if not present]
            raise ValueError(
                "Mixed fidelity specification detected: either all observations must "
                f"provide a fidelity or none should. Missing indices: {missing}."
            )

        return all(has_fidelity)

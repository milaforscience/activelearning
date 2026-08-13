"""Deep Kernel Learning surrogates for encoded active-learning inputs.

Two variants are provided, both inheriting from :class:`BoTorchGPSurrogate`:

``ExactDKLSurrogate``
    Uses BoTorch's ``SingleTaskGP`` with an encoder kernel as the
    covariance module.  Compatible with **all BoTorch acquisition functions**.
    Training minimises ``ExactMarginalLogLikelihood + MLM loss`` via Adam.

``VariationalDKLSurrogate``
    Uses a sparse variational GP head, following the reference
    ``DeepKernelRegressor``.  Training minimises
    ``VariationalELBO + MLM loss`` via Adam.

**Multi-fidelity** is controlled via the ``is_multi_fidelity`` constructor
argument (and the ``is_multi_fidelity`` key in the YAML config). When enabled,
the surrogate appends the BoTorch-facing **fidelity confidence** from
``set_fidelity_confidences()`` as the last column of the feature tensor before
the GP. This keeps the fidelity coordinate in the continuous space used by
BoTorch's multi-fidelity helpers such as ``project_to_target_fidelity``.

The input representation is supplied through an ``input_adapter`` callable.
This keeps conversion from domain objects to model-space tensors outside the
generic DKL implementation. Floating-point tensors follow
``RuntimeContext.dtype`` (``torch.float64`` by default).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Iterable, Optional, Protocol

import gpytorch
import torch
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from torch.optim import Adam

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.dkl.kernel import EncoderKernel
from activelearning.utils.types import Candidate, Observation


class InputAdapter(Protocol):
    """Convert domain-specific inputs into batched DKL model inputs."""

    def __call__(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode domain values into a model-space tensor.

        Parameters
        ----------
        values : Sequence[Any]
            Domain values extracted from candidates or observations.
        device : torch.device
            Device on which the returned tensor should be allocated.

        Returns
        -------
        torch.Tensor
            Batched model-space inputs, with one row per value.
        """
        ...


class DeepKernelSurrogate(BoTorchGPSurrogate):
    """Base class for exact and variational deep-kernel surrogates.

    The class owns the representation-independent training and multi-fidelity
    logic. ``input_adapter`` is the only domain-specific boundary: it converts
    the ``x`` values carried by candidates and observations into a batched
    tensor accepted by the encoder and GP.

    Not intended to be instantiated directly. Use :class:`ExactDKLSurrogate`
    or :class:`VariationalDKLSurrogate`.

    Parameters
    ----------
    encoder : torch.nn.Module
        Feature encoder jointly optimised with the GP where its parameters
        are trainable.
    input_adapter : InputAdapter
        Callable converting a sequence of domain inputs into a batched tensor.
    training_params : object
        Training hyper-parameters (epochs, lr, mask_ratio, pretrain_epochs).
        The masking parameters are used only when the encoder exposes an
        optional ``mlm_loss`` method.
    is_multi_fidelity : bool
        Whether to append the encoded fidelity confidence to each feature
        vector. Should match the ``is_multi_fidelity`` key in the YAML run config.
    target_fidelity : int, optional
        The target (highest) fidelity level.  **Required when
        ``is_multi_fidelity=True``**; tells BoTorch's ``project_to_target_fidelity``
        which encoded fidelity value to project to. Typically this is the
        highest fidelity level, and it is mapped to its configured confidence
        internally.
    **botorch_kwargs
        Forwarded to :class:`BoTorchGPSurrogate`.

    Subclass contract
    -----------------
    Concrete subclasses **must** implement:

    * :meth:`_build_model` -- construct the GP model (sets self.model).
    * :meth:`_make_mll` -- return the MLL / ELBO objective.
    * :meth:`_gp_forward` -- run the GP forward pass on model-space tensors.
    * :meth:`_make_optimizer` -- return an Adam optimiser over all parameters.

    Subclasses **may** override:

    * :meth:`_prepare_targets` -- standardise targets before training.
    * :meth:`_set_train_mode` / :meth:`_set_eval_mode` -- train/eval toggling.
    * :meth:`predict` -- if the GP posterior is not a BoTorch posterior object.
    """

    def __init__(
        self,
        encoder: Any,
        input_adapter: InputAdapter,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        **botorch_kwargs: Any,
    ) -> None:
        """Initialize a representation-independent DKL surrogate.

        Parameters
        ----------
        encoder : torch.nn.Module
            Feature encoder jointly optimized with the GP when its parameters
            are trainable.
        input_adapter : InputAdapter
            Callable that converts candidate and observation values into
            batched model-space tensors.
        training_params : object
            Training settings containing ``epochs`` and ``lr``. Encoders with
            an ``mlm_loss`` method may also use ``pretrain_epochs`` and
            ``mask_ratio``.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidences to model inputs.
        target_fidelity : int, optional
            Fidelity level to use for target-fidelity projections. Required
            when ``is_multi_fidelity`` is true.
        **botorch_kwargs : Any
            Additional keyword arguments passed to
            :class:`BoTorchGPSurrogate`.

        Raises
        ------
        ValueError
            If multi-fidelity mode is enabled without a target fidelity.
        TypeError
            If ``input_adapter`` is not callable.
        """
        if is_multi_fidelity and target_fidelity is None:
            raise ValueError(
                "target_fidelity must be set when is_multi_fidelity=True. "
                "Set it to the maximum fidelity level (e.g. max(fidelity_costs))."
            )
        if not callable(input_adapter):
            raise TypeError("input_adapter must be callable.")
        self._encoder = encoder
        self._input_adapter = input_adapter
        self._training = training_params

        self._target_fidelity_level = target_fidelity if is_multi_fidelity else None
        botorch_kwargs.setdefault("optimize_hyperparameters", False)
        botorch_kwargs.setdefault("is_multi_fidelity", is_multi_fidelity)
        super().__init__(**botorch_kwargs)

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Move the DKL stack onto the shared runtime device and dtype.

        Parameters
        ----------
        runtime_context : RuntimeContext
            Runtime device and floating-point dtype used by the active-learning
            loop.
        """
        super().bind_runtime_context(runtime_context)
        self._apply_runtime_context()

    def _apply_runtime_context(self) -> None:
        """Apply the currently bound runtime device/dtype to all learnable modules."""
        # Runtime precision applies to trainable DKL components; frozen
        # pretrained encoders may restore their checkpoint dtype below.
        self._encoder = self._encoder.to(device=self.device, dtype=self.dtype)
        if self.model is not None:
            self.model = self.model.to(device=self.device, dtype=self.dtype)
        restore_backbone_dtype = getattr(self._encoder, "restore_backbone_dtype", None)
        if restore_backbone_dtype is not None:
            restore_backbone_dtype()
        if hasattr(self, "_gp_model") and self._gp_model is not None:
            self._gp_model = self._gp_model.to(device=self.device, dtype=self.dtype)
        if hasattr(self, "_likelihood") and self._likelihood is not None:
            self._likelihood = self._likelihood.to(device=self.device, dtype=self.dtype)
        if hasattr(self, "_botorch_adapter") and self._botorch_adapter is not None:
            self._botorch_adapter = self._botorch_adapter.to(
                device=self.device, dtype=self.dtype
            )

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the encoder and GP jointly to a collection of observations.

        Parameters
        ----------
        observations : Iterable[Observation]
            Observations whose ``x`` values are converted by the input adapter
            and whose ``y`` values are used as regression targets. An empty
            iterable leaves the surrogate unchanged.
        """
        obs_list = list(observations)
        if not obs_list:
            return
        self._train_X, self._train_Y, self._is_multi_fidelity = (
            self._parse_observations(obs_list)
        )
        self._train_Y = self._prepare_targets(self._train_Y)
        self._build_model(self._train_X, self._train_Y)
        self._apply_runtime_context()
        self._remove_noise_prior()
        self._joint_train(self._make_mll(len(obs_list)))

    def updates_from_latest(self) -> bool:
        """Report whether fitting can reuse the latest observations.

        Returns
        -------
        bool
            Always ``False`` because DKL fitting rebuilds and retrains the GP
            stack from the complete observation set.
        """
        return False

    def get_target_fidelity_value(self) -> float | None:
        """Return the encoded target fidelity value used by BoTorch.

        Overrides :meth:`BoTorchGPSurrogate.get_target_fidelity_value` so the
        configured target fidelity level is converted through the active
        confidence mapping before BoTorch uses it.

        Returns
        -------
        float or None
            Encoded target-fidelity confidence in multi-fidelity mode, or
            ``None`` for single-fidelity surrogates.
        """
        if not self._is_multi_fidelity or self._target_fidelity_level is None:
            return None
        return self._encode_fidelity_level(self._target_fidelity_level)

    # Template methods -- must be implemented by subclasses

    @abstractmethod
    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Construct the GP model and store it in self.model."""

    @abstractmethod
    def _make_mll(self, num_data: int) -> Any:
        """Return the MLL / ELBO objective for this GP variant."""

    @abstractmethod
    def _gp_forward(self, model_X: torch.Tensor) -> Any:
        """Run the GP forward pass and return the output distribution."""

    @abstractmethod
    def _make_optimizer(self) -> Adam:
        """Return an Adam optimiser over all trainable parameters."""

    def _prepare_targets(self, train_Y: torch.Tensor) -> torch.Tensor:
        """Optional hook for target standardisation. No-op by default."""
        return train_Y

    @property
    def _has_mlm_loss(self) -> bool:
        """Return whether the encoder provides an MLM auxiliary objective."""
        return callable(getattr(self._encoder, "mlm_loss", None))

    # Shared training loop

    def _joint_train(self, mll: Any) -> None:
        """Run the joint auxiliary-loss + GP Adam training loop."""
        optimizer = self._make_optimizer()
        all_params = [p for group in optimizer.param_groups for p in group["params"]]
        train_X = self._train_X.to(device=self.device, dtype=self.dtype)
        targets = self._train_Y.squeeze(-1).to(device=self.device, dtype=self.dtype)

        if self._has_mlm_loss:
            # Auxiliary masked-token objectives consume integer token IDs;
            # ordinary DKL encoders never enter this branch.
            mlm_tokens = (
                train_X[:, :-1].long() if self._is_multi_fidelity else train_X.long()
            )
            for _ in range(self._training.pretrain_epochs):
                self._encoder.train()
                optimizer.zero_grad()
                self._encoder.mlm_loss(mlm_tokens, self._training.mask_ratio).backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

        for _ in range(self._training.epochs):
            self._set_train_mode()
            optimizer.zero_grad()
            mlm_loss = None
            if self._has_mlm_loss:
                mlm_loss = self._encoder.mlm_loss(mlm_tokens, self._training.mask_ratio)
            # Extra Cholesky jitter stabilises early training when embeddings are similar
            with gpytorch.settings.cholesky_jitter(1e-1):
                gp_loss = -mll(self._gp_forward(train_X), targets)
            (gp_loss if mlm_loss is None else mlm_loss + gp_loss).backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

        self._set_eval_mode()

    # Domain input conversion

    def _parse_observations(
        self, observations: Iterable[Observation]
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Convert observations to model-space inputs and runtime-dtype targets."""
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("Cannot parse an empty observation iterable.")
        inputs = self._encode_inputs_with_fidelity(obs_list)
        train_Y = torch.as_tensor([o.y for o in obs_list], dtype=self.dtype).unsqueeze(
            -1
        )
        return inputs, train_Y, self._is_multi_fidelity

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Convert candidates to model-space inputs.

        Appends the encoded fidelity confidence as the last column when
        ``is_multi_fidelity=True``.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates to encode.

        Returns
        -------
        torch.Tensor
            Batched model-space inputs on the active runtime device and dtype.

        Raises
        ------
        ValueError
            If ``candidates`` is empty or a multi-fidelity item has an
            undeclared fidelity level.
        """
        cand_list = list(candidates)
        if not cand_list:
            raise ValueError("Cannot encode an empty candidate iterable.")
        return self._encode_inputs_with_fidelity(cand_list)

    # Internal helpers

    def _encode_inputs_with_fidelity(
        self, items: list[Candidate | Observation]
    ) -> torch.Tensor:
        """Encode items and append the fidelity value when active."""
        inputs = self._input_adapter(
            [item.x for item in items],
            device=self.device,
        ).to(device=self.device, dtype=self.dtype)
        if self._is_multi_fidelity:
            fidelities = torch.tensor(
                [
                    self._encode_fidelity_level(item.fidelity)
                    if item.fidelity is not None
                    else self.get_target_fidelity_value()
                    for item in items
                ],
                dtype=self.dtype,
                device=self.device,
            ).unsqueeze(-1)
            inputs = torch.cat([inputs, fidelities], dim=-1)
        return inputs

    def _encode_fidelity_level(self, fidelity_level: int) -> float:
        """Map a discrete fidelity level to the BoTorch-facing confidence value."""
        if fidelity_level not in self._fidelity_confidences:
            raise ValueError(
                "Missing fidelity confidence for "
                f"level {fidelity_level}. Call set_fidelity_confidences() with "
                "the oracle's confidence mapping before fitting or scoring a "
                "multi-fidelity DKL surrogate."
            )
        return float(self._fidelity_confidences[fidelity_level])

    def _remove_noise_prior(self) -> None:
        """Remove BoTorch LogNormalPrior from the GP noise and initialise to 0.1.

        The raw noise parameter can drift below zero during joint Adam training,
        causing a support validation error from the prior.  Initialising to 0.1
        also improves Cholesky conditioning early in training.
        """
        noise_covar = getattr(
            getattr(self.model, "likelihood", None), "noise_covar", None
        )
        if noise_covar is None:
            return
        noise_covar._priors.pop("noise_prior", None)
        with torch.no_grad():
            noise_covar.noise = 0.1

    def _set_train_mode(self) -> None:
        if self.model is not None:
            self.model.train()
        self._encoder.train()
        if hasattr(self, "_likelihood") and self._likelihood is not None:
            self._likelihood.train()

    def _set_eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()
        self._encoder.eval()
        if hasattr(self, "_likelihood") and self._likelihood is not None:
            self._likelihood.eval()


# ---------------------------------------------------------------------------
# Exact GP variant
# ---------------------------------------------------------------------------


class ExactDKLSurrogate(DeepKernelSurrogate):
    """DKL surrogate with an exact GP backed by BoTorch SingleTaskGP.

    The encoder is embedded inside an encoder kernel passed to SingleTaskGP as
    its covar_module, making all BoTorch acquisition functions work out of the
    box. Training jointly optimises encoder, GP kernel, and likelihood noise
    via Adam (ExactMarginalLogLikelihood + MLM loss).

    Parameters
    ----------
    encoder : torch.nn.Module
    input_adapter : InputAdapter
    training_params : object
    is_multi_fidelity : bool
    target_fidelity : int, optional
        Required when ``is_multi_fidelity=True``.
    standardize_outputs : bool
        Normalise GP outputs to mean 0 / variance 1.
    scale_inputs : bool
        Whether BoTorch should normalize the model-space inputs. Defaults to
        ``False`` because tokenized sequence inputs are not continuous features.
    """

    def __init__(
        self,
        encoder: Any,
        input_adapter: InputAdapter,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        standardize_outputs: bool = True,
        scale_inputs: bool = False,
    ) -> None:
        """Initialize an exact-GP DKL surrogate.

        Parameters
        ----------
        encoder : torch.nn.Module
            Feature encoder used inside the exact GP kernel.
        input_adapter : InputAdapter
            Callable converting domain values into model-space tensors.
        training_params : object
            DKL training settings, including the epoch count and learning rate.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidences to the GP inputs.
        target_fidelity : int, optional
            Fidelity level used for target-fidelity projections. Required when
            ``is_multi_fidelity`` is true.
        standardize_outputs : bool, default=True
            Whether to standardize regression targets before GP training.
        scale_inputs : bool, default=False
            Whether BoTorch should normalize model-space inputs.
        """
        super().__init__(
            encoder=encoder,
            input_adapter=input_adapter,
            training_params=training_params,
            is_multi_fidelity=is_multi_fidelity,
            target_fidelity=target_fidelity,
            scale_inputs=scale_inputs,
            standardize_outputs=standardize_outputs,
        )

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        gp_input_dim = self._encoder.latent_dim + (1 if self._is_multi_fidelity else 0)
        self.covar_module = EncoderKernel(
            encoder=self._encoder,
            base_kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=gp_input_dim)
            ),
            include_fidelity=self._is_multi_fidelity,
        )
        # Skip the abstract stub and call BoTorchGPSurrogate directly
        BoTorchGPSurrogate._build_model(self, train_X, train_Y)

    def _make_mll(self, num_data: int) -> ExactMarginalLogLikelihood:
        return ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def _gp_forward(self, model_X: torch.Tensor) -> Any:
        return self.model(model_X.to(device=self.device, dtype=self.dtype))

    def _make_optimizer(self) -> Adam:
        # model already contains the likelihood as a submodule
        return Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=self._training.lr,
        )


# ---------------------------------------------------------------------------
# Variational GP variant
# ---------------------------------------------------------------------------


class _VariationalDKLGP(gpytorch.models.ApproximateGP):
    """Sparse variational GP head used by :class:`VariationalDKLSurrogate`.

    Operates in **latent feature space** — it receives encoder output vectors,
    not raw token IDs.  The encoder is kept separate and called explicitly by
    the surrogate before invoking this GP.
    """

    def __init__(
        self,
        input_dim: int,
        num_inducing: int = 64,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        # The surrogate passes RuntimeContext.dtype; this fallback matches the
        # default runtime precision for standalone construction.
        inducing_points = torch.randn(
            num_inducing, input_dim, dtype=dtype, device=device
        )
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=input_dim)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class _VariationalBoTorchAdapter(Model):
    """BoTorch-compatible wrapper around the variational DKL GP.

    BoTorch acquisition functions (including qMF-MES) require a model that
    satisfies the :class:`botorch.models.model.Model` contract: it must
    implement :meth:`posterior`, :attr:`num_outputs`, and :attr:`batch_shape`.
    Plain GPyTorch ``ApproximateGP`` does not satisfy this contract; this
    adapter bridges the gap without modifying ``_VariationalDKLGP``.

    The adapter operates in **latent feature space**: ``X`` passed to
    :meth:`posterior` must already be encoded latent vectors (+ optional fidelity
    column), matching the input that ``_VariationalDKLGP.forward`` expects.

    Parameters
    ----------
    gp_model : _VariationalDKLGP
        The trained variational GP head.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The GP likelihood used for observation noise during prediction.
    """

    def __init__(
        self,
        gp_model: _VariationalDKLGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        """Initialize an adapter around a trained variational GP and likelihood.

        Parameters
        ----------
        gp_model : _VariationalDKLGP
            Sparse variational GP head operating on latent feature vectors.
        likelihood : gpytorch.likelihoods.GaussianLikelihood
            Likelihood used to include observation noise in posterior queries.
        """
        super().__init__()
        self._gp = gp_model
        self._likelihood = likelihood

    @property
    def num_outputs(self) -> int:
        """Return the number of modeled outputs."""
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        """Return the empty batch shape of the single-output GP."""
        return torch.Size([])

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Return a BoTorch-compatible GP posterior over latent-space inputs.

        Parameters
        ----------
        X : torch.Tensor
            Latent feature tensor of shape ``(..., d)`` where ``d`` is the
            encoder latent dimension (+ 1 for fidelity in multi-fidelity mode).
        observation_noise : bool, default=False
            If True, adds likelihood noise to the predictive variance.

        Returns
        -------
        GPyTorchPosterior
            BoTorch posterior backed by the variational GP distribution.
        """
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        self._gp.eval()
        self._likelihood.eval()
        dist = self._gp(X)
        if observation_noise:
            dist = self._likelihood(dist)
        return GPyTorchPosterior(distribution=dist)


class VariationalDKLSurrogate(DeepKernelSurrogate):
    """DKL surrogate with a sparse variational GP head — paper-faithful variant.

    Reproduces the architecture from the reference ``DeepKernelRegressor``:

    - Encoder and GP are **separate** components; encoding is explicit.
    - Training minimises ``VariationalELBO + MLM loss`` via Adam (ELBO, not MLL).
    - GP operates in **latent feature space** (encoder output + optional fidelity).
    - Compatible with all BoTorch acquisition functions via
      :class:`_VariationalBoTorchAdapter`, including qMF-MES.

    Parameters
    ----------
    encoder : torch.nn.Module
    input_adapter : InputAdapter
    training_params : object
    is_multi_fidelity : bool
        Append fidelity scalar to latent feature vectors.
    target_fidelity : int, optional
        Required when ``is_multi_fidelity=True``.
    num_inducing : int
        Number of variational inducing points.
    standardize_outputs : bool
        Normalise targets before training and denormalise predictions.
    """

    def __init__(
        self,
        encoder: Any,
        input_adapter: InputAdapter,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        num_inducing: int = 64,
        standardize_outputs: bool = True,
    ) -> None:
        """Initialize a sparse variational DKL surrogate.

        Parameters
        ----------
        encoder : torch.nn.Module
            Feature encoder whose output is modeled by the variational GP.
        input_adapter : InputAdapter
            Callable converting domain values into model-space tensors.
        training_params : object
            DKL training settings, including the epoch count and learning rate.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidences to latent features.
        target_fidelity : int, optional
            Fidelity level used for target-fidelity projections. Required when
            ``is_multi_fidelity`` is true.
        num_inducing : int, default=64
            Number of inducing points in the sparse variational GP.
        standardize_outputs : bool, default=True
            Whether to standardize regression targets before GP training.
        """
        self._num_inducing = num_inducing
        self._standardize_outputs = standardize_outputs
        self._gp_model: Optional[_VariationalDKLGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._botorch_adapter: Optional[_VariationalBoTorchAdapter] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        super().__init__(
            encoder=encoder,
            input_adapter=input_adapter,
            training_params=training_params,
            is_multi_fidelity=is_multi_fidelity,
            target_fidelity=target_fidelity,
            scale_inputs=False,
            standardize_outputs=False,  # handled manually via _prepare_targets
        )

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        gp_input_dim = self._encoder.latent_dim + (1 if self._is_multi_fidelity else 0)
        self._gp_model = _VariationalDKLGP(
            gp_input_dim,
            self._num_inducing,
            dtype=self.dtype,
            device=self.device,
        ).to(device=self.device, dtype=self.dtype)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self.device, dtype=self.dtype
        )
        self._botorch_adapter = _VariationalBoTorchAdapter(
            self._gp_model, self._likelihood
        )
        self.model = self._gp_model  # exposes is_fitted() / state-dict helpers

    def get_model(self) -> _VariationalBoTorchAdapter:
        """Return the BoTorch-compatible adapter wrapping the variational GP.

        Overrides :meth:`BoTorchGPSurrogate.get_model` so that acquisition
        functions receive a model that implements the BoTorch ``posterior()``
        interface and operates in latent feature space.

        Returns
        -------
        _VariationalBoTorchAdapter
            Fitted variational GP adapter for use by acquisition functions.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        """
        if self._botorch_adapter is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        return self._botorch_adapter

    def get_fidelity_dimension(self) -> Optional[int]:
        """Return the fidelity column index in latent feature space.

        Overrides the base method, which computes the index from ``train_X``
        shape (token space).  In the variational surrogate, acquisition
        functions receive **latent features** from :meth:`encode_candidates`,
        so the fidelity column sits at index ``latent_dim`` — the last column
        of the ``(latent_dim + 1)``-dimensional latent tensor.

        Returns
        -------
        int or None
            Fidelity column index in multi-fidelity mode, or ``None`` for
            single-fidelity surrogates.
        """
        if not self._is_multi_fidelity:
            return None
        return self._encoder.latent_dim

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Return encoder latent features (+ fidelity) for each candidate.

        Overrides the base input path. The variational GP head operates
        in latent feature space, so acquisition functions must receive encoded
        vectors rather than raw token IDs.  This ensures the candidate set
        produced by :class:`~activelearning.acquisition.botorch.candidate_set.TrainDataCandidateSetSpec`
        and the projection in MF-MES are consistent with the GP's input space.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates to encode.

        Returns
        -------
        torch.Tensor
            Batched latent features, with an optional fidelity confidence as
            the final column.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        """
        if self._gp_model is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        # Encode via the parent, then map inputs to latent features.
        input_X = super().encode_candidates(candidates).to(self.device)
        self._set_eval_mode()
        with torch.no_grad():
            return self._encode_with_fidelity(input_X)

    def _prepare_targets(self, train_Y: torch.Tensor) -> torch.Tensor:
        if self._standardize_outputs:
            self._y_mean = float(train_Y.mean())
            self._y_std = float(train_Y.std().nan_to_num(nan=1.0).clamp_min(1e-6))
            return (train_Y - self._y_mean) / self._y_std
        return train_Y

    def _make_mll(self, num_data: int) -> VariationalELBO:
        return VariationalELBO(self._likelihood, self._gp_model, num_data=num_data)

    def _gp_forward(self, model_X: torch.Tensor) -> Any:
        return self._gp_model(self._encode_with_fidelity(model_X))

    def _make_optimizer(self) -> Adam:
        return Adam(
            [
                parameter
                for parameter in (
                    list(self._encoder.parameters())
                    + list(self._gp_model.parameters())
                    + list(self._likelihood.parameters())
                )
                if parameter.requires_grad
            ],
            lr=self._training.lr,
        )

    def predict(self, candidates: Iterable[Candidate]) -> dict[str, Any]:
        """Predict target means and standard deviations for candidates.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates whose values are converted by the input adapter.

        Returns
        -------
        dict[str, Any]
            Dictionary containing CPU lists under ``"mean"`` and ``"std"``.
            Values are returned on the original, pre-standardization target
            scale.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        """
        if self._gp_model is None or self._likelihood is None:
            raise RuntimeError("Surrogate has not been fitted yet.")

        # encode_candidates() now returns latent features directly
        latent_X = self.encode_candidates(list(candidates)).to(self.device)
        with torch.no_grad():
            pred = self._likelihood(self._gp_model(latent_X))

        mean = pred.mean * self._y_std + self._y_mean
        std = pred.variance.sqrt() * self._y_std
        return {"mean": mean.cpu().tolist(), "std": std.cpu().tolist()}

    def _encode_with_fidelity(self, input_X: torch.Tensor) -> torch.Tensor:
        """Encode model inputs to latent features, appending fidelity if active."""
        if self._is_multi_fidelity:
            features = self._encoder(input_X[:, :-1])
            return torch.cat([features, input_X[:, -1:].to(features.dtype)], dim=-1)
        return self._encoder(input_X)

"""Deep Kernel Learning surrogates for SELFIES molecules optimisation.

Two variants are provided, both inheriting from :class:`BoTorchGPSurrogate`:

``ExactSelfiesDKLSurrogate``
    Uses BoTorch's ``SingleTaskGP`` with a :class:`SelfiesKernel` as the
    covariance module.  Compatible with **all BoTorch acquisition functions**.
    Training minimises ``ExactMarginalLogLikelihood + MLM loss`` via Adam.

``VariationalSelfiesDKLSurrogate``
    Uses a sparse variational GP head, following the reference
    ``DeepKernelMoleculeRegressor``.  Training minimises
    ``VariationalELBO + MLM loss`` via Adam.

**Multi-fidelity** is controlled via the ``is_multi_fidelity`` constructor
argument (and the ``is_multi_fidelity`` key in the YAML config). When enabled,
the surrogate appends the BoTorch-facing **fidelity confidence** from
``set_fidelity_confidences()`` as the last column of the feature tensor before
the GP. This keeps the fidelity coordinate in the continuous space used by
BoTorch's multi-fidelity helpers such as ``project_to_target_fidelity``.

Floating-point tensors in the SELFIES DKL stack follow ``RuntimeContext.dtype``
(``torch.float64`` by default). Token IDs remain integer tensors inside the
encoder/tokenizer path and are only cast to the runtime floating dtype where
BoTorch / GPyTorch expect floating inputs.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, Optional

import gpytorch
import torch
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from torch.optim import Adam

from activelearning.applications.molecules.selfies_transformer_encoder import (
    SelfiesTransformerEncoder,
)
from activelearning.applications.molecules.selfies_kernel import SelfiesKernel
from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Candidate, Observation


class SelfiesDeepKernelSurrogate(BoTorchGPSurrogate):
    """Base class for SELFIES DKL surrogates (template-method pattern).

    Handles tokenisation, MLM pre-training, and the shared Adam training loop.

    Not intended to be instantiated directly.  Use
    :class:`ExactSelfiesDKLSurrogate` or :class:`VariationalSelfiesDKLSurrogate`.

    Parameters
    ----------
    encoder : SelfiesTransformerEncoder
        The shared Transformer encoder jointly optimised with the GP.
    training_params : SelfiesTrainingConfig
        Training hyper-parameters (epochs, lr, mask_ratio, pretrain_epochs).
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
    * :meth:`_gp_forward` -- run the GP forward pass on token-ID tensors.
    * :meth:`_make_optimizer` -- return an Adam optimiser over all parameters.

    Subclasses **may** override:

    * :meth:`_prepare_targets` -- standardise targets before training.
    * :meth:`_set_train_mode` / :meth:`_set_eval_mode` -- train/eval toggling.
    * :meth:`predict` -- if the GP posterior is not a BoTorch posterior object.
    """

    def __init__(
        self,
        encoder: SelfiesTransformerEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        **botorch_kwargs: Any,
    ) -> None:
        if is_multi_fidelity and target_fidelity is None:
            raise ValueError(
                "target_fidelity must be set when is_multi_fidelity=True. "
                "Set it to the maximum fidelity level (e.g. max(fidelity_costs))."
            )
        self._encoder = encoder
        self._training = training_params

        self._target_fidelity_level = target_fidelity if is_multi_fidelity else None
        botorch_kwargs.setdefault("optimize_hyperparameters", False)
        botorch_kwargs.setdefault("is_multi_fidelity", is_multi_fidelity)
        super().__init__(**botorch_kwargs)

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Move the DKL stack onto the shared runtime device and dtype."""
        super().bind_runtime_context(runtime_context)
        self._apply_runtime_context()

    def _apply_runtime_context(self) -> None:
        """Apply the currently bound runtime device/dtype to all learnable modules."""
        self._encoder = self._encoder.to(device=self.device, dtype=self.dtype)
        if self.model is not None:
            self.model = self.model.to(device=self.device, dtype=self.dtype)
        if hasattr(self, "_gp_model") and self._gp_model is not None:
            self._gp_model = self._gp_model.to(device=self.device, dtype=self.dtype)
        if hasattr(self, "_likelihood") and self._likelihood is not None:
            self._likelihood = self._likelihood.to(device=self.device, dtype=self.dtype)
        if hasattr(self, "_botorch_adapter") and self._botorch_adapter is not None:
            self._botorch_adapter = self._botorch_adapter.to(
                device=self.device, dtype=self.dtype
            )

    def fit(self, observations: Iterable[Observation]) -> None:
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
        """DKL always retrains from scratch -- the AL loop will call fit()."""
        return False

    def get_target_fidelity_value(self) -> float | None:
        """Return the encoded target fidelity value used by BoTorch.

        Overrides :meth:`BoTorchGPSurrogate.get_target_fidelity_value` so the
        configured target fidelity level is converted through the active
        confidence mapping before BoTorch uses it.
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
    def _gp_forward(self, token_X: torch.Tensor) -> Any:
        """Run the GP forward pass and return the output distribution."""

    @abstractmethod
    def _make_optimizer(self) -> Adam:
        """Return an Adam optimiser over all trainable parameters."""

    def _prepare_targets(self, train_Y: torch.Tensor) -> torch.Tensor:
        """Optional hook for target standardisation. No-op by default."""
        return train_Y

    # Shared training loop

    def _joint_train(self, mll: Any) -> None:
        """Run the joint MLM + GP Adam training loop."""
        optimizer = self._make_optimizer()
        all_params = [p for group in optimizer.param_groups for p in group["params"]]
        train_X = self._train_X.to(device=self.device, dtype=self.dtype)
        targets = self._train_Y.squeeze(-1).to(device=self.device, dtype=self.dtype)
        # MLM only sees token IDs -- strip the fidelity column when present
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
            mlm_loss = self._encoder.mlm_loss(mlm_tokens, self._training.mask_ratio)
            # Extra Cholesky jitter stabilises early training when embeddings are similar
            with gpytorch.settings.cholesky_jitter(1e-1):
                gp_loss = -mll(self._gp_forward(train_X), targets)
            (mlm_loss + gp_loss).backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

        self._set_eval_mode()

    # Overrides: tokenise SELFIES strings

    def _parse_observations(
        self, observations: Iterable[Observation]
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Tokenise SELFIES strings to runtime-dtype token-ID tensors."""
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("Cannot parse an empty observation iterable.")
        tokens = self._tokenize_with_fidelity(obs_list)
        train_Y = torch.as_tensor([o.y for o in obs_list], dtype=self.dtype).unsqueeze(
            -1
        )
        return tokens, train_Y, self._is_multi_fidelity

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Tokenise candidates to runtime-dtype token-ID tensors.

        Appends the encoded fidelity confidence as the last column when
        ``is_multi_fidelity=True``.
        """
        cand_list = list(candidates)
        if not cand_list:
            raise ValueError("Cannot encode an empty candidate iterable.")
        return self._tokenize_with_fidelity(cand_list)

    # Internal helpers

    def _tokenize_with_fidelity(
        self, items: list[Candidate | Observation]
    ) -> torch.Tensor:
        """Tokenise items and append the encoded fidelity value when active."""
        strings = [self._extract_molecule_string(item) for item in items]
        tokens = self._tokenize_strings(strings)
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
            tokens = torch.cat([tokens, fidelities], dim=-1)
        return tokens

    def _encode_fidelity_level(self, fidelity_level: int) -> float:
        """Map a discrete fidelity level to the BoTorch-facing confidence value."""
        if fidelity_level not in self._fidelity_confidences:
            raise ValueError(
                "Missing fidelity confidence for "
                f"level {fidelity_level}. Call set_fidelity_confidences() with "
                "the oracle's confidence mapping before fitting or scoring a "
                "multi-fidelity SELFIES surrogate."
            )
        return float(self._fidelity_confidences[fidelity_level])

    def _tokenize_strings(self, strings: list[str]) -> torch.Tensor:
        return self._encoder.tokenizer.batch_from_selfies(
            strings, max_mol_tokens=self._encoder.max_mol_tokens, device=self.device
        ).to(dtype=self.dtype)

    def _extract_molecule_string(self, item: Candidate | Observation) -> str:
        """Extract the SELFIES string from a candidate or observation."""
        if isinstance(item.x, str):
            return item.x
        raise ValueError(
            f"Expected item.x to be a SELFIES string, got {type(item.x).__name__}."
        )

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


class ExactSelfiesDKLSurrogate(SelfiesDeepKernelSurrogate):
    """DKL surrogate with an exact GP backed by BoTorch SingleTaskGP.

    The encoder is embedded inside a SelfiesKernel passed to SingleTaskGP as
    its covar_module, making all BoTorch acquisition functions work out of the
    box.  Training jointly optimises encoder, GP kernel, and likelihood noise
    via Adam (ExactMarginalLogLikelihood + MLM loss).

    Parameters
    ----------
    encoder : SelfiesTransformerEncoder
    training_params : SelfiesTrainingConfig
    is_multi_fidelity : bool
    target_fidelity : int, optional
        Required when ``is_multi_fidelity=True``.
    standardize_outputs : bool
        Normalise GP outputs to mean 0 / variance 1.
    """

    def __init__(
        self,
        encoder: SelfiesTransformerEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        standardize_outputs: bool = True,
    ) -> None:
        super().__init__(
            encoder=encoder,
            training_params=training_params,
            is_multi_fidelity=is_multi_fidelity,
            target_fidelity=target_fidelity,
            scale_inputs=False,  # token IDs must not be normalised
            standardize_outputs=standardize_outputs,
        )

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        gp_input_dim = self._encoder.latent_dim + (1 if self._is_multi_fidelity else 0)
        self.covar_module = SelfiesKernel(
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

    def _gp_forward(self, token_X: torch.Tensor) -> Any:
        return self.model(token_X.to(device=self.device, dtype=self.dtype))

    def _make_optimizer(self) -> Adam:
        # model already contains the likelihood as a submodule
        return Adam(self.model.parameters(), lr=self._training.lr)


# ---------------------------------------------------------------------------
# Variational GP variant
# ---------------------------------------------------------------------------


class _VariationalDKLGP(gpytorch.models.ApproximateGP):
    """Sparse variational GP head used by VariationalSelfiesDKLSurrogate.

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
        super().__init__()
        self._gp = gp_model
        self._likelihood = likelihood

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
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
        """
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        self._gp.eval()
        self._likelihood.eval()
        dist = self._gp(X)
        if observation_noise:
            dist = self._likelihood(dist)
        return GPyTorchPosterior(distribution=dist)


class VariationalSelfiesDKLSurrogate(SelfiesDeepKernelSurrogate):
    """DKL surrogate with a sparse variational GP head — paper-faithful variant.

    Reproduces the architecture from the reference ``DeepKernelMoleculeRegressor``:

    - Encoder and GP are **separate** components; encoding is explicit.
    - Training minimises ``VariationalELBO + MLM loss`` via Adam (ELBO, not MLL).
    - GP operates in **latent feature space** (encoder output + optional fidelity).
    - Compatible with all BoTorch acquisition functions via
      :class:`_VariationalBoTorchAdapter`, including qMF-MES.

    Parameters
    ----------
    encoder : SelfiesTransformerEncoder
    training_params : SelfiesTrainingConfig
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
        encoder: SelfiesTransformerEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        num_inducing: int = 64,
        standardize_outputs: bool = True,
    ) -> None:
        self._num_inducing = num_inducing
        self._standardize_outputs = standardize_outputs
        self._gp_model: Optional[_VariationalDKLGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._botorch_adapter: Optional[_VariationalBoTorchAdapter] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        super().__init__(
            encoder=encoder,
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
        """
        if not self._is_multi_fidelity:
            return None
        return self._encoder.latent_dim

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Return encoder latent features (+ fidelity) for each candidate.

        Overrides the base tokenisation path.  The variational GP head operates
        in latent feature space, so acquisition functions must receive encoded
        vectors rather than raw token IDs.  This ensures the candidate set
        produced by :class:`~activelearning.acquisition.botorch.candidate_set.TrainDataCandidateSetSpec`
        and the projection in MF-MES are consistent with the GP's input space.
        """
        if self._gp_model is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        # Tokenise via parent, then encode to latent features.
        token_X = super().encode_candidates(candidates).to(self.device)
        self._set_eval_mode()
        with torch.no_grad():
            return self._encode_with_fidelity(token_X)

    def _prepare_targets(self, train_Y: torch.Tensor) -> torch.Tensor:
        if self._standardize_outputs:
            self._y_mean = float(train_Y.mean())
            self._y_std = float(train_Y.std().nan_to_num(nan=1.0).clamp_min(1e-6))
            return (train_Y - self._y_mean) / self._y_std
        return train_Y

    def _make_mll(self, num_data: int) -> VariationalELBO:
        return VariationalELBO(self._likelihood, self._gp_model, num_data=num_data)

    def _gp_forward(self, token_X: torch.Tensor) -> Any:
        return self._gp_model(self._encode_with_fidelity(token_X))

    def _make_optimizer(self) -> Adam:
        return Adam(
            list(self._encoder.parameters())
            + list(self._gp_model.parameters())
            + list(self._likelihood.parameters()),
            lr=self._training.lr,
        )

    def predict(self, candidates: Iterable[Candidate]) -> dict[str, Any]:
        """Predict mean and std, denormalised to the original target scale."""
        if self._gp_model is None or self._likelihood is None:
            raise RuntimeError("Surrogate has not been fitted yet.")

        # encode_candidates() now returns latent features directly
        latent_X = self.encode_candidates(list(candidates)).to(self.device)
        with torch.no_grad():
            pred = self._likelihood(self._gp_model(latent_X))

        mean = pred.mean * self._y_std + self._y_mean
        std = pred.variance.sqrt() * self._y_std
        return {"mean": mean.cpu().tolist(), "std": std.cpu().tolist()}

    def _encode_with_fidelity(self, token_X: torch.Tensor) -> torch.Tensor:
        """Encode token IDs → latent features, appending fidelity when active."""
        if self._is_multi_fidelity:
            features = self._encoder(token_X[:, :-1].long())
            return torch.cat([features, token_X[:, -1:].to(features.dtype)], dim=-1)
        return self._encoder(token_X.long())

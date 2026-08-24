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

The input representation is supplied through ``encoder.prepare_inputs()``,
which converts domain objects to model-space tensors before the same encoder is
re-applied inside the GP or kernel. Floating-point tensors follow
``RuntimeContext.dtype`` (``torch.float64`` by default).
"""

from __future__ import annotations

from abc import abstractmethod
from operator import index
from typing import Any, Iterable, Optional
from warnings import warn

import gpytorch
import torch
from torch.optim import Adam

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.encoder import LatentEncoder
from activelearning.utils.types import Candidate, Observation


class DeepKernelSurrogate(BoTorchGPSurrogate):
    """Base class for exact and variational deep-kernel surrogates.

    The class owns the representation-independent training and multi-fidelity
    logic. ``encoder.prepare_inputs()`` is the raw-input boundary: it converts
    the ``x`` values carried by candidates and observations into a batched
    tensor accepted by the encoder and GP.

    Not intended to be instantiated directly. Use :class:`ExactDKLSurrogate`
    or :class:`VariationalDKLSurrogate`.

    Parameters
    ----------
    encoder : LatentEncoder
        Feature encoder jointly optimised with the GP where its parameters
        are trainable. It must expose a callable ``prepare_inputs()`` method
        and an integer ``latent_dim`` attribute.
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
        encoder: LatentEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        **botorch_kwargs: Any,
    ) -> None:
        """Initialize a representation-independent DKL surrogate.

        Parameters
        ----------
        encoder : LatentEncoder
            Feature encoder jointly optimized with the GP when its parameters
            are trainable. It must implement ``prepare_inputs()`` for raw
            domain values and expose an integer ``latent_dim``.
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
            If the encoder does not expose the required DKL input contract.
        """
        if is_multi_fidelity and target_fidelity is None:
            raise ValueError(
                "target_fidelity must be set when is_multi_fidelity=True. "
                "Set it to the maximum fidelity level (e.g. max(fidelity_costs))."
            )
        if not callable(getattr(encoder, "prepare_inputs", None)):
            raise TypeError(
                "encoder must define a callable prepare_inputs(values, *, device) "
                "method."
            )
        latent_dim = getattr(encoder, "latent_dim", None)
        if isinstance(latent_dim, bool):
            raise TypeError("encoder must define an integer latent_dim attribute.")
        try:
            usable_latent_dim = index(latent_dim)
        except TypeError as error:
            raise TypeError(
                "encoder must define an integer latent_dim attribute."
            ) from error
        if usable_latent_dim < 1:
            raise TypeError("encoder.latent_dim must be a positive integer.")
        self._encoder = encoder
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

        Returns
        -------
        None
            The surrogate is updated in place.
        """
        super().bind_runtime_context(runtime_context)
        self._apply_runtime_context()

    def _apply_runtime_context(self) -> None:
        """Apply the currently bound runtime device/dtype to all learnable modules."""
        # Runtime precision applies to trainable DKL components; frozen
        # pretrained encoders may restore their checkpoint dtype below.
        self._encoder = self._encoder.to(device=self.device, dtype=self.dtype)
        for module in self._runtime_modules():
            module.to(device=self.device, dtype=self.dtype)
        restore_backbone_dtype = getattr(self._encoder, "restore_backbone_dtype", None)
        if restore_backbone_dtype is not None:
            restore_backbone_dtype()

    def _runtime_modules(self) -> tuple[torch.nn.Module, ...]:
        """Return modules that need the active runtime device and dtype."""
        return (self.model,) if self.model is not None else ()

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the encoder and GP jointly to a collection of observations.

        Parameters
        ----------
        observations : Iterable[Observation]
            Observations whose ``x`` values are converted by
            ``encoder.prepare_inputs()`` and whose ``y`` values are used as
            regression targets. An empty iterable leaves the surrogate
            unchanged.

        Returns
        -------
        None
            The fitted model is stored on the surrogate in place.
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
        has_mlm_loss = self._has_mlm_loss
        mlm_tokens = None
        if has_mlm_loss:
            # Auxiliary masked-token objectives consume integer token IDs;
            # ordinary DKL encoders never enter this branch.
            mlm_tokens = (
                train_X[:, :-1].long() if self._is_multi_fidelity else train_X.long()
            )
            self._warn_if_mlm_configuration_is_ignored()
            self._run_mlm_pretraining(optimizer, all_params, mlm_tokens)

        if not has_mlm_loss:
            self._warn_if_mlm_configuration_is_ignored()

        for _ in range(self._training.epochs):
            self._set_train_mode()
            optimizer.zero_grad()
            mlm_loss = None
            if mlm_tokens is not None:
                mlm_loss = self._encoder.mlm_loss(mlm_tokens, self._training.mask_ratio)
            # Extra jitter stabilises early training when embeddings are similar.
            with gpytorch.settings.cholesky_jitter(1e-1):
                gp_loss = -mll(self._gp_forward(train_X), targets)
            (gp_loss if mlm_loss is None else mlm_loss + gp_loss).backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

        self._set_eval_mode()

    def _run_mlm_pretraining(
        self,
        optimizer: Adam,
        all_params: list[torch.nn.Parameter],
        mlm_tokens: torch.Tensor,
    ) -> None:
        """Run the optional masked-language-model warm-up phase."""
        for _ in range(self._training.pretrain_epochs):
            self._set_train_mode()
            optimizer.zero_grad()
            self._encoder.mlm_loss(mlm_tokens, self._training.mask_ratio).backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

    def _warn_if_mlm_configuration_is_ignored(self) -> None:
        """Warn when explicit MLM settings target an encoder without MLM loss."""
        if self._has_mlm_loss:
            return
        fields_set = getattr(self._training, "model_fields_set", set())
        explicitly_configured = (
            ({"mask_ratio", "pretrain_epochs"} & set(fields_set))
            or self._training.pretrain_epochs > 0
            or self._training.mask_ratio != 0.125
        )
        if explicitly_configured:
            warn(
                "The configured encoder does not expose mlm_loss; "
                "mask_ratio and pretrain_epochs will be ignored.",
                UserWarning,
                stacklevel=3,
            )

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
        inputs = self._encoder.prepare_inputs(
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
        for module in self._runtime_modules():
            module.train()
        self._encoder.train()

    def _set_eval_mode(self) -> None:
        for module in self._runtime_modules():
            module.eval()
        self._encoder.eval()

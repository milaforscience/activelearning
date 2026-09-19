"""Variational DKL surrogate implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import gpytorch
import torch
from gpytorch.mlls import VariationalELBO
from torch.optim import Adam

from activelearning.surrogate.dkl.surrogate import DeepKernelSurrogate
from activelearning.surrogate.encoder import LatentEncoder
from activelearning.surrogate.variational_gp import (
    _VariationalBoTorchAdapter,
    _VariationalGP,
)
from activelearning.utils.types import Candidate


class VariationalDKLSurrogate(DeepKernelSurrogate):
    """DKL surrogate with a sparse variational GP head.

    The encoder and sparse GP are separate components:

    - Encoder and GP are **separate** components; encoding is explicit.
    - Training minimises ``VariationalELBO + MLM loss`` via Adam (ELBO, not MLL).
    - GP operates in **latent feature space** (encoder output + optional fidelity).
    - Compatible with all BoTorch acquisition functions via
      :class:`_VariationalBoTorchAdapter`, including qMF-MES.
    """

    def __init__(
        self,
        encoder: LatentEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        num_inducing: int = 64,
        standardize_outputs: bool = True,
        initial_likelihood_noise: float = 0.1,
    ) -> None:
        """Initialize a sparse variational DKL surrogate.

        Parameters
        ----------
        encoder : LatentEncoder
            Feature encoder whose output is modeled by the variational GP.
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
        initial_likelihood_noise : float, default=0.1
            Positive initial Gaussian likelihood noise.
        """
        if initial_likelihood_noise <= 0.0:
            raise ValueError("initial_likelihood_noise must be positive.")
        self._num_inducing = num_inducing
        self._standardize_outputs = standardize_outputs
        self._initial_likelihood_noise = initial_likelihood_noise
        self._gp_model: Optional[_VariationalGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._botorch_adapter: Optional[_VariationalBoTorchAdapter] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fit_train_X: torch.Tensor | None = None
        self._fit_train_Y: torch.Tensor | None = None
        self._validation_X: torch.Tensor | None = None
        self._validation_Y: torch.Tensor | None = None
        self._fit_train_indices = torch.empty(0, dtype=torch.long)
        self._validation_indices = torch.empty(0, dtype=torch.long)
        self._epochs_trained = 0
        self._validation_losses: list[float] = []
        self._best_validation_loss = float("inf")
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
        self._gp_model = _VariationalGP(
            gp_input_dim,
            self._num_inducing,
            dtype=self.dtype,
            device=self.device,
        ).to(device=self.device, dtype=self.dtype)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self.device, dtype=self.dtype
        )
        with torch.no_grad():
            self._likelihood.noise = self._initial_likelihood_noise
        self._botorch_adapter = _VariationalBoTorchAdapter(
            self._gp_model, self._likelihood
        )
        self.model = self._gp_model  # exposes is_fitted() / state-dict helpers

    def _runtime_modules(self) -> tuple[torch.nn.Module, ...]:
        """Return variational GP modules that follow the runtime context."""
        return tuple(
            module
            for module in (
                self._gp_model,
                self._likelihood,
                self._botorch_adapter,
            )
            if module is not None
        )

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
        vectors rather than raw token IDs. This keeps the candidate set and
        the projection in MF-MES consistent with the GP's input space.

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
            betas=self._training.betas,
        )

    def _split_training_data(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministically split parsed observations into fit and validation data."""
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError(
                "train_X and train_Y must contain the same number of rows."
            )
        count = train_X.shape[0]
        validation_count = int(count * self._training.validation_fraction)
        validation_count = min(validation_count, max(0, count - 1))
        seed = (
            self._training.validation_seed
            if self._training.validation_seed is not None
            else self.runtime_context.seed
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(count, generator=generator)
        self._validation_indices = permutation[:validation_count]
        self._fit_train_indices = permutation[validation_count:]
        fit_indices = self._fit_train_indices.to(device=train_X.device)
        validation_indices = self._validation_indices.to(device=train_X.device)
        fit_X = train_X[fit_indices]
        fit_Y = train_Y[fit_indices]
        validation_X = train_X[validation_indices]
        validation_Y = train_Y[validation_indices]
        return fit_X, fit_Y, validation_X, validation_Y

    def _iter_batch_indices(
        self,
        count: int,
        *,
        epoch: int,
    ) -> Iterable[torch.Tensor]:
        """Yield a deterministic shuffled partition of training-row indices."""
        batch_size = self._training.batch_size or count
        seed = (
            self._training.validation_seed
            if self._training.validation_seed is not None
            else self.runtime_context.seed
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        indices = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            yield indices[start : start + batch_size]

    def _snapshot_trainable_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Capture trainable encoder, GP, and likelihood parameters."""
        return {
            "encoder": {
                name: parameter.detach().clone()
                for name, parameter in self._encoder.named_parameters()
                if parameter.requires_grad
            },
            "model": {
                name: parameter.detach().clone()
                for name, parameter in self._gp_model.named_parameters()
                if parameter.requires_grad
            },
            "likelihood": {
                name: parameter.detach().clone()
                for name, parameter in self._likelihood.named_parameters()
                if parameter.requires_grad
            },
        }

    def _restore_trainable_state(
        self,
        state: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Restore a state captured by :meth:`_snapshot_trainable_state`."""
        modules = (
            ("encoder", self._encoder),
            ("model", self._gp_model),
            ("likelihood", self._likelihood),
        )
        for key, module in modules:
            saved = state[key]
            current = dict(module.named_parameters())
            with torch.no_grad():
                for name, value in saved.items():
                    current[name].copy_(value)

    def _validation_loss(self, mll: VariationalELBO) -> float:
        """Evaluate negative variational ELBO on the held-out split."""
        if self._validation_X is None or self._validation_X.shape[0] == 0:
            return float("inf")
        self._set_eval_mode()
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-1):
            loss = -mll(
                self._gp_forward(self._validation_X),
                self._validation_Y.squeeze(-1),
            )
        return float(loss.item())

    def _joint_train(self, mll: VariationalELBO) -> None:
        """Train the variational GP with optional validation and mini-batches."""
        assert self._train_X is not None
        assert self._train_Y is not None
        assert self._gp_model is not None
        assert self._likelihood is not None

        (
            self._fit_train_X,
            self._fit_train_Y,
            self._validation_X,
            self._validation_Y,
        ) = self._split_training_data(self._train_X, self._train_Y)
        mll = self._make_mll(self._fit_train_X.shape[0])
        optimizer = self._make_optimizer()
        all_params = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self._epochs_trained = 0
        self._validation_losses = []
        self._best_validation_loss = float("inf")
        best_state: dict[str, dict[str, torch.Tensor]] | None = None
        stale_epochs = 0

        for epoch in range(self._training.epochs):
            self._set_train_mode()
            for batch_indices in self._iter_batch_indices(
                self._fit_train_X.shape[0],
                epoch=epoch,
            ):
                batch_indices = batch_indices.to(device=self.device)
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-1):
                    loss = -mll(
                        self._gp_forward(self._fit_train_X[batch_indices]),
                        self._fit_train_Y[batch_indices].squeeze(-1),
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

            self._epochs_trained += 1
            if self._validation_X.shape[0] > 0:
                validation_loss = self._validation_loss(mll)
                self._validation_losses.append(validation_loss)
                if validation_loss < self._best_validation_loss:
                    self._best_validation_loss = validation_loss
                    best_state = self._snapshot_trainable_state()
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if (
                        self._training.early_stopping_patience is not None
                        and stale_epochs >= self._training.early_stopping_patience
                    ):
                        break

        if best_state is not None:
            self._restore_trainable_state(best_state)
        self._set_eval_mode()

    def _remove_noise_prior(self) -> None:
        """Remove the Gaussian likelihood noise prior and preserve its setting."""
        assert self._likelihood is not None
        noise_covar = self._likelihood.noise_covar
        noise_covar._priors.pop("noise_prior", None)
        with torch.no_grad():
            noise_covar.noise = self._initial_likelihood_noise

    def predict(self, candidates: Iterable[Candidate]) -> dict[str, Any]:
        """Predict target means and standard deviations for candidates.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates whose values are converted by
            ``encoder.prepare_inputs()``.

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

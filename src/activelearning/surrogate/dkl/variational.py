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
        """
        self._num_inducing = num_inducing
        self._standardize_outputs = standardize_outputs
        self._gp_model: Optional[_VariationalGP] = None
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
        self._gp_model = _VariationalGP(
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
        )

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

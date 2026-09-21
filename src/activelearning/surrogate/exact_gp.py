"""Exact Gaussian process over fixed feature representations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.utils.types import Candidate, Observation


class ExactGPSurrogate(BoTorchGPSurrogate):
    """Exact BoTorch GP fitted on fixed, non-trainable encoder features.

    Raw candidate values are mapped through ``encoder.encode`` before the
    inherited BoTorch model construction, hyperparameter fitting, and
    prediction. In multi-fidelity mode the fidelity confidence is appended as
    the last input column, as in :class:`BoTorchGPSurrogate`.
    """

    def __init__(
        self,
        encoder: FixedEncoder,
        is_multi_fidelity: bool = False,
        target_fidelity: int | None = None,
        scale_inputs: bool = True,
        standardize_outputs: bool = True,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a fixed-feature exact GP surrogate.

        Parameters
        ----------
        encoder : FixedEncoder
            Non-trainable mapping from raw values to fixed-width features.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidence to each feature row.
        target_fidelity : int, optional
            Discrete target fidelity used by multi-fidelity acquisitions.
        scale_inputs : bool, default=True
            Whether BoTorch normalizes the feature columns.
        standardize_outputs : bool, default=True
            Whether BoTorch standardizes regression targets.
        fit_kwargs : dict[str, Any], optional
            Keyword arguments passed to ``fit_gpytorch_mll``.
        """
        if is_multi_fidelity and target_fidelity is None:
            raise ValueError("target_fidelity must be set when is_multi_fidelity=True.")
        if encoder.feature_dim < 1:
            raise ValueError("encoder.feature_dim must be positive.")

        self._encoder = encoder
        self._target_fidelity_level = target_fidelity if is_multi_fidelity else None
        super().__init__(
            scale_inputs=scale_inputs,
            standardize_outputs=standardize_outputs,
            fit_kwargs=fit_kwargs,
            is_multi_fidelity=is_multi_fidelity,
        )

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind runtime settings to the surrogate and its fixed encoder."""
        super().bind_runtime_context(runtime_context)
        self._encoder.bind_runtime_context(runtime_context)

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Encode candidates into fixed feature space with optional fidelity."""
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("Cannot encode an empty candidate iterable.")
        return self._encode_items(candidate_list)

    def get_target_fidelity_value(self) -> float | None:
        """Return the encoded confidence for the configured target fidelity."""
        if not self._is_multi_fidelity or self._target_fidelity_level is None:
            return None
        return self._encode_fidelity_level(self._target_fidelity_level)

    def _parse_observations(
        self,
        observations: Iterable[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observations into fixed-feature training tensors."""
        observation_list = list(observations)
        if not observation_list:
            raise ValueError("Cannot parse an empty observation iterable.")
        train_X = self._encode_items(observation_list)
        train_Y = torch.as_tensor(
            [observation.y for observation in observation_list],
            dtype=train_X.dtype,
            device=train_X.device,
        ).unsqueeze(-1)
        return train_X, train_Y

    def _encode_items(
        self,
        items: Sequence[Candidate | Observation],
    ) -> torch.Tensor:
        """Encode fixed features and append fidelity confidence when enabled."""
        features = self._encoder.encode(
            [item.x for item in items],
            device=self.device,
        ).to(device=self.device, dtype=self.dtype)
        expected_shape = (len(items), self._encoder.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "Fixed encoder output must have shape "
                f"{expected_shape}, got {tuple(features.shape)}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("Fixed encoder output contains non-finite values.")
        if not self._is_multi_fidelity:
            return features

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
        return torch.cat([features, fidelities], dim=-1)

    def _encode_fidelity_level(self, fidelity_level: int) -> float:
        """Map a discrete fidelity id to its configured confidence."""
        if fidelity_level not in self._fidelity_confidences:
            raise ValueError(f"Missing fidelity confidence for level {fidelity_level}.")
        return float(self._fidelity_confidences[fidelity_level])

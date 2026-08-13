"""Regression tests for representation-independent DKL surrogates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.dkl.dkl_surrogate import (
    ExactDKLSurrogate,
    VariationalDKLSurrogate,
)
from activelearning.utils.types import Candidate, Observation


class _NumericEncoder(nn.Module):
    """Small continuous encoder used to exercise the generic DKL boundary."""

    latent_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, self.latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project a batch of continuous inputs to latent features."""
        return self.projection(inputs)


class _NumericInputAdapter:
    """Convert numeric candidate values to a batched floating-point tensor."""

    def __call__(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Return numeric inputs without sequence-specific preprocessing."""
        return torch.as_tensor(values, dtype=torch.float64, device=device)


@pytest.mark.parametrize(
    "surrogate_type",
    [ExactDKLSurrogate, VariationalDKLSurrogate],
)
def test_dkl_accepts_non_molecular_inputs(surrogate_type: type) -> None:
    """Both DKL variants should train through the generic input adapter."""
    torch.manual_seed(0)
    surrogate_kwargs: dict[str, Any] = {}
    if surrogate_type is VariationalDKLSurrogate:
        surrogate_kwargs["num_inducing"] = 2

    surrogate = surrogate_type(
        encoder=_NumericEncoder(),
        input_adapter=_NumericInputAdapter(),
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        **surrogate_kwargs,
    )
    observations = [
        Observation(x=[0.0, 0.0], y=0.0),
        Observation(x=[1.0, 1.0], y=1.0),
    ]

    surrogate.fit(observations)
    result = surrogate.predict([Candidate(x=[0.0, 0.0]), Candidate(x=[1.0, 1.0])])

    assert surrogate.is_fitted()
    assert len(result["mean"]) == 2
    assert len(result["std"]) == 2

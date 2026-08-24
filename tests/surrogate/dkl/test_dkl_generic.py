"""Regression tests for representation-independent DKL surrogates."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.dkl import (
    ExactDKLSurrogate,
    VariationalDKLSurrogate,
)
from activelearning.surrogate.encoder import LatentEncoder
from activelearning.utils.types import Candidate, Observation


class _NumericEncoder(LatentEncoder):
    """Small continuous encoder used to exercise the generic DKL boundary."""

    latent_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, self.latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project a batch of continuous inputs to latent features."""
        return self.projection(inputs)


class _MissingPrepareInputsEncoder(nn.Module):
    """Encoder-like module missing the raw-input preparation contract."""

    latent_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, self.latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project a batch of continuous inputs to latent features."""
        return self.projection(inputs)


class _MissingLatentDimEncoder(LatentEncoder):
    """Encoder missing the required latent-dimension contract."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project a batch of continuous inputs to latent features."""
        return self.projection(inputs)


@pytest.mark.parametrize(
    "surrogate_type",
    [ExactDKLSurrogate, VariationalDKLSurrogate],
)
def test_dkl_accepts_non_molecular_inputs(surrogate_type: type) -> None:
    """Both DKL variants should train through the generic encoder contract."""
    torch.manual_seed(0)
    surrogate_kwargs: dict[str, Any] = {}
    if surrogate_type is VariationalDKLSurrogate:
        surrogate_kwargs["num_inducing"] = 2

    surrogate = surrogate_type(
        encoder=_NumericEncoder(),
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


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (_MissingPrepareInputsEncoder(), "prepare_inputs"),
        (_MissingLatentDimEncoder(), "latent_dim"),
    ],
)
def test_dkl_rejects_encoders_missing_required_contract(
    encoder: nn.Module,
    message: str,
) -> None:
    """DKL surrogates must validate the shared encoder input contract."""
    with pytest.raises(TypeError, match=message):
        ExactDKLSurrogate(
            encoder=encoder,
            training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        )

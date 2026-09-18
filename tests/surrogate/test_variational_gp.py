"""Tests for the fixed-feature variational GP surrogate."""

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.config import VariationalGPTrainingConfig
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.variational_gp import VariationalGPSurrogate
from activelearning.utils.types import Candidate, Observation


class NumericFixedEncoder(FixedEncoder):
    """Encode scalar inputs as two deterministic numeric features."""

    feature_dim = 2

    def encode(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return each scalar and its square as fixed features."""
        inputs = torch.as_tensor(values, dtype=self.dtype, device=device)
        return torch.stack((inputs, inputs.square()), dim=-1)


def build_surrogate(
    *,
    is_multi_fidelity: bool = False,
    target_fidelity: int | None = None,
) -> VariationalGPSurrogate:
    """Build a small variational surrogate suitable for unit tests."""
    return VariationalGPSurrogate(
        encoder=NumericFixedEncoder(),
        training_params=VariationalGPTrainingConfig(epochs=1, lr=1e-2),
        is_multi_fidelity=is_multi_fidelity,
        target_fidelity=target_fidelity,
        num_inducing=3,
    )


def test_fit_predict_and_botorch_posterior() -> None:
    """Fit fixed features and expose finite predictions through BoTorch."""
    surrogate = build_surrogate()
    surrogate.bind_runtime_context(
        RuntimeContext(device=torch.device("cpu"), dtype=torch.float32)
    )
    surrogate.fit(
        [
            Observation(x=0.0, y=0.0),
            Observation(x=1.0, y=1.0),
            Observation(x=2.0, y=4.0),
        ]
    )

    candidates = [Candidate(x=0.5), Candidate(x=1.5)]
    encoded = surrogate.encode_candidates(candidates)
    predictions = surrogate.predict(candidates)
    posterior = surrogate.get_model().posterior(encoded)

    assert encoded.shape == (2, 2)
    assert encoded.dtype == torch.float32
    assert len(predictions["mean"]) == 2
    assert len(predictions["std"]) == 2
    assert torch.isfinite(torch.tensor(predictions["mean"])).all()
    assert torch.isfinite(torch.tensor(predictions["std"])).all()
    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == (2, 1)


def test_multi_fidelity_encoding_and_target_projection() -> None:
    """Append configured fidelity confidences in the target projection column."""
    surrogate = build_surrogate(is_multi_fidelity=True, target_fidelity=2)
    surrogate.set_fidelity_confidences({1: 0.4, 2: 1.0})
    surrogate.fit(
        [
            Observation(x=0.0, y=0.0, fidelity=1),
            Observation(x=1.0, y=0.8, fidelity=1),
            Observation(x=1.0, y=1.0, fidelity=2),
        ]
    )

    encoded = surrogate.encode_candidates(
        [Candidate(x=0.5, fidelity=1), Candidate(x=1.5, fidelity=2)]
    )

    assert surrogate.get_fidelity_dimension() == 2
    assert surrogate.get_target_fidelity_value() == pytest.approx(1.0)
    assert encoded.shape == (2, 3)
    assert encoded[:, -1].tolist() == pytest.approx([0.4, 1.0])


def test_state_dict_round_trip_restores_predictions() -> None:
    """Restore trained GP and output-scaling state into a fresh surrogate."""
    observations = [
        Observation(x=0.0, y=1.0),
        Observation(x=1.0, y=2.0),
        Observation(x=2.0, y=5.0),
    ]
    candidates = [Candidate(x=0.5), Candidate(x=1.5)]
    original = build_surrogate()
    original.fit(observations)
    state_dict = original.get_state_dict()
    assert state_dict is not None

    restored = build_surrogate()
    restored.load_state_dict(state_dict)
    restored.fit(observations)

    assert restored.predict(candidates) == pytest.approx(original.predict(candidates))

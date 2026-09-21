"""Tests for the fixed-feature exact GP surrogate."""

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.exact_gp import ExactGPSurrogate
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
) -> ExactGPSurrogate:
    """Build a small exact GP surrogate on a float64 CPU runtime."""
    surrogate = ExactGPSurrogate(
        encoder=NumericFixedEncoder(),
        is_multi_fidelity=is_multi_fidelity,
        target_fidelity=target_fidelity,
    )
    surrogate.bind_runtime_context(
        RuntimeContext(device=torch.device("cpu"), dtype=torch.float64)
    )
    return surrogate


def test_fit_predict_and_botorch_posterior() -> None:
    """Fit fixed features and expose finite predictions through BoTorch."""
    surrogate = build_surrogate()
    surrogate.fit(
        [
            Observation(x=0.0, y=0.0),
            Observation(x=1.0, y=1.0),
            Observation(x=2.0, y=4.0),
            Observation(x=3.0, y=9.0),
        ]
    )

    candidates = [Candidate(x=0.5), Candidate(x=1.5)]
    encoded = surrogate.encode_candidates(candidates)
    predictions = surrogate.predict(candidates)
    posterior = surrogate.get_model().posterior(encoded)

    assert encoded.shape == (2, 2)
    assert torch.isfinite(torch.tensor(predictions["mean"])).all()
    assert torch.isfinite(torch.tensor(predictions["std"])).all()
    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == (2, 1)
    assert not surrogate.updates_from_latest()


def test_multi_fidelity_encoding_and_target_projection() -> None:
    """Append configured fidelity confidences as the last model column."""
    surrogate = build_surrogate(is_multi_fidelity=True, target_fidelity=2)
    surrogate.set_fidelity_confidences({1: 0.4, 2: 1.0})
    surrogate.fit(
        [
            Observation(x=0.0, y=0.0, fidelity=1),
            Observation(x=1.0, y=0.8, fidelity=1),
            Observation(x=1.0, y=1.0, fidelity=2),
            Observation(x=2.0, y=4.0, fidelity=2),
        ]
    )

    encoded = surrogate.encode_candidates(
        [Candidate(x=0.5, fidelity=1), Candidate(x=1.5, fidelity=2)]
    )

    assert surrogate.get_fidelity_dimension() == 2
    assert surrogate.get_target_fidelity_value() == pytest.approx(1.0)
    assert encoded.shape == (2, 3)
    assert encoded[:, -1].tolist() == pytest.approx([0.4, 1.0])
    assert len(surrogate.predict([Candidate(x=0.5, fidelity=2)])["mean"]) == 1


def test_config_resolves_target_fidelity() -> None:
    """Resolve multi-fidelity mode and target fidelity from oracle confidences."""
    from activelearning.surrogate.config import ExactGPSurrogateConfig

    config = ExactGPSurrogateConfig.model_construct(
        type="ExactGPSurrogate",
        encoder=None,
        target_fidelity=None,
        scale_inputs=True,
        standardize_outputs=True,
        fit_kwargs={},
    )
    resolved = config.resolve_fidelity_confidences({1: 0.2, 3: 1.0})

    assert resolved.is_multi_fidelity
    assert resolved.target_fidelity == 3

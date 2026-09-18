"""Focused tests for the single-fidelity lower-bound MES wrapper."""

import math

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from activelearning.acquisition.botorch.botorch_single_fidelity import (
    QLowerBoundMaxValueEntropy,
)
from activelearning.acquisition.botorch.candidate_set import TrainDataCandidateSetSpec
from activelearning.acquisition.config import (
    AcquisitionConfig,
    QLowerBoundMaxValueEntropyConfig,
)
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Candidate, Observation


@pytest.fixture()
def observations() -> list[Observation]:
    """Return observations for a small single-fidelity GP."""
    return [
        Observation(x=[1.0, 2.0], y=5.0),
        Observation(x=[3.0, 4.0], y=7.0),
        Observation(x=[5.0, 6.0], y=9.0),
    ]


@pytest.fixture()
def fitted_surrogate(observations: list[Observation]) -> BoTorchGPSurrogate:
    """Fit a local BoTorch surrogate without external model downloads."""
    surrogate = BoTorchGPSurrogate()
    surrogate.fit(observations)
    return surrogate


@pytest.fixture()
def candidates() -> list[Candidate]:
    """Return singleton candidates for q-batch scoring."""
    return [Candidate(x=[2.0, 3.0]), Candidate(x=[4.0, 5.0])]


@pytest.fixture()
def multi_fidelity_observations() -> list[Observation]:
    """Return observations for checking the existing MF warning semantics."""
    return [
        Observation(x=[1.0, 2.0], y=5.0, fidelity=0),
        Observation(x=[3.0, 4.0], y=7.0, fidelity=1),
        Observation(x=[5.0, 6.0], y=9.0, fidelity=1),
        Observation(x=[1.0, 2.0], y=4.5, fidelity=0),
    ]


@pytest.fixture()
def fitted_mf_surrogate(
    multi_fidelity_observations: list[Observation],
) -> BoTorchGPSurrogate:
    """Fit a local multi-fidelity BoTorch surrogate."""
    surrogate = BoTorchGPSurrogate(is_multi_fidelity=True)
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)
    return surrogate


def test_scores_singletons_with_candidate_set(
    fitted_surrogate: BoTorchGPSurrogate,
    observations: list[Observation],
    candidates: list[Candidate],
) -> None:
    """The q-batch wrapper remains usable by singleton-scoring samplers."""
    acquisition = QLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(),
        num_mv_samples=5,
    )
    acquisition.update(fitted_surrogate, observations)

    scores = acquisition.score(candidates)

    assert acquisition.supports_singleton_scoring is True
    assert acquisition.supports_batch_scoring is False
    assert acquisition._botorch_acqf is not None
    assert acquisition._botorch_acqf.num_mv_samples == 5
    assert len(scores) == len(candidates)
    assert all(math.isfinite(score) and score >= 0.0 for score in scores)


def test_preserves_single_fidelity_multi_fidelity_warning(
    fitted_mf_surrogate: BoTorchGPSurrogate,
    multi_fidelity_observations: list[Observation],
) -> None:
    """A single-fidelity wrapper keeps the base class MF warning behavior."""
    acquisition = QLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec()
    )

    with pytest.warns(UserWarning, match="not a multi-fidelity acquisition"):
        acquisition.update(fitted_mf_surrogate, multi_fidelity_observations)

    assert acquisition._supports_multi_fidelity is False


def test_forwards_maximize_to_botorch(
    fitted_surrogate: BoTorchGPSurrogate,
    observations: list[Observation],
) -> None:
    """The objective direction is forwarded to qLowerBoundMaxValueEntropy."""
    acquisition = QLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(),
        maximize=False,
    )
    acquisition.update(fitted_surrogate, observations)

    assert acquisition.maximize is False
    assert acquisition._botorch_acqf is not None
    assert acquisition._botorch_acqf.maximize is False
    assert acquisition._botorch_acqf.weight == -1.0


def test_clamps_negative_information_gain() -> None:
    """Numerical negative information-gain estimates are projected to zero."""
    acquisition = QLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec()
    )
    acquisition._botorch_acqf = lambda X: torch.tensor(  # type: ignore[method-assign]
        [-1.0e-12, -0.5, 0.25],
        dtype=X.dtype,
        device=X.device,
    )

    scores = acquisition._score_encoded(torch.zeros(3, 1, 1, dtype=torch.float64))

    assert scores == [0.0, 0.0, 0.25]


def test_rejects_nonpositive_max_value_samples() -> None:
    """The wrapper validates BoTorch's maximum-value sample count."""
    with pytest.raises(ValueError, match="num_mv_samples must be > 0"):
        QLowerBoundMaxValueEntropy(
            candidate_set_spec=TrainDataCandidateSetSpec(),
            num_mv_samples=0,
        )


def test_config_registry_builds_single_fidelity_mes() -> None:
    """The registry parses and builds the single-fidelity MES configuration."""
    config = TypeAdapter(AcquisitionConfig).validate_python(
        {
            "type": "QLowerBoundMaxValueEntropy",
            "candidate_set_spec": {"type": "TrainDataCandidateSetSpec"},
            "num_mv_samples": 7,
            "maximize": False,
        }
    )

    assert isinstance(config, QLowerBoundMaxValueEntropyConfig)
    acquisition = config.build()
    assert isinstance(acquisition, QLowerBoundMaxValueEntropy)
    assert acquisition.maximize is False
    assert acquisition._num_mv_samples == 7


def test_config_rejects_nonpositive_max_value_samples() -> None:
    """The Pydantic config validates the maximum-value sample count."""
    with pytest.raises(ValidationError, match="greater than 0"):
        TypeAdapter(AcquisitionConfig).validate_python(
            {
                "type": "QLowerBoundMaxValueEntropy",
                "candidate_set_spec": {"type": "TrainDataCandidateSetSpec"},
                "num_mv_samples": 0,
            }
        )

"""Tests for acquisition config discriminators and builders."""

from pydantic import TypeAdapter

from activelearning.acquisition.botorch.botorch_single_fidelity import (
    QLowerBoundMaxValueEntropy,
    QMaxValueEntropy,
)
from activelearning.acquisition.config import AcquisitionConfig


def test_qmax_value_entropy_config_builds() -> None:
    """QMaxValueEntropy should parse and build from the discriminated config union."""
    config = TypeAdapter(AcquisitionConfig).validate_python(
        {
            "type": "QMaxValueEntropy",
            "candidate_set_spec": {"type": "TrainDataCandidateSetSpec"},
            "num_fantasies": 2,
            "num_mv_samples": 5,
            "num_y_samples": 16,
        }
    )

    assert isinstance(config.build(), QMaxValueEntropy)


def test_qlower_bound_max_value_entropy_config_builds() -> None:
    """QLowerBoundMaxValueEntropy should parse and build from the config union."""
    config = TypeAdapter(AcquisitionConfig).validate_python(
        {
            "type": "QLowerBoundMaxValueEntropy",
            "candidate_set_spec": {"type": "TrainDataCandidateSetSpec"},
            "num_mv_samples": 5,
        }
    )

    assert isinstance(config.build(), QLowerBoundMaxValueEntropy)

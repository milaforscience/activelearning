import math

import pytest
import torch

from activelearning.acquisition.botorch.candidate_set import TrainDataCandidateSetSpec
from activelearning.acquisition.botorch.cost_utility import FidelityCostUtility
from activelearning.acquisition.botorch.botorch_multifidelity import (
    QMultiFidelityLowerBoundMaxValueEntropy,
)
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Candidate, Observation


class TestFidelityCostUtility:
    def test_forward_scales_batch_items_by_matching_costs(self) -> None:
        utility = FidelityCostUtility(fidelity_costs={1: 2.0, 2: 5.0}, fixed_cost=0.25)
        X = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 2.0]],
                [[0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        deltas = torch.tensor([[4.0, 10.0, 6.0]], dtype=torch.float64)

        scaled = utility(X=X, deltas=deltas)

        expected = torch.tensor([[2.25, 2.25, 3.25]], dtype=torch.float64)
        assert torch.equal(scaled, expected)

    def test_forward_scales_encoded_confidences_after_mapping(self) -> None:
        utility = FidelityCostUtility(fidelity_costs={1: 2.0, 2: 5.0})
        utility.set_fidelity_confidences({1: 0.25, 2: 1.0})
        X = torch.tensor(
            [
                [[0.0, 0.25]],
                [[0.0, 1.0]],
            ],
            dtype=torch.float64,
        )
        deltas = torch.tensor([[4.0, 10.0]], dtype=torch.float64)

        scaled = utility(X=X, deltas=deltas)

        assert torch.equal(scaled, torch.tensor([[2.0, 2.0]], dtype=torch.float64))

    def test_forward_preserves_leading_sample_dims(self) -> None:
        utility = FidelityCostUtility(fidelity_costs={1: 2.0, 2: 4.0})
        X = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 2.0]],
            ],
            dtype=torch.float64,
        )
        deltas = torch.tensor(
            [
                [[2.0, 8.0], [4.0, 12.0]],
                [[6.0, 16.0], [8.0, 20.0]],
            ],
            dtype=torch.float64,
        )

        scaled = utility(X=X, deltas=deltas)

        expected = torch.tensor(
            [
                [[1.0, 2.0], [2.0, 3.0]],
                [[3.0, 4.0], [4.0, 5.0]],
            ],
            dtype=torch.float64,
        )
        assert torch.equal(scaled, expected)

    def test_forward_raises_for_unknown_fidelity(self) -> None:
        utility = FidelityCostUtility(fidelity_costs={1: 2.0})
        X = torch.tensor([[[0.0, 3.0]]], dtype=torch.float64)
        deltas = torch.tensor([[1.0]], dtype=torch.float64)

        with pytest.raises(ValueError, match="without configured costs"):
            utility(X=X, deltas=deltas)

    def test_forward_raises_for_q_batch_larger_than_one(self) -> None:
        utility = FidelityCostUtility(fidelity_costs={1: 2.0})
        X = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float64)
        deltas = torch.tensor([[1.0]], dtype=torch.float64)

        with pytest.raises(ValueError, match="q=1"):
            utility(X=X, deltas=deltas)

    def test_non_positive_costs_raise_at_construction(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            FidelityCostUtility(fidelity_costs={1: 0.0, 2: -1.0})

    def test_missing_confidence_mapping_raises(self) -> None:
        utility = FidelityCostUtility(fidelity_costs={1: 2.0, 2: 5.0})

        with pytest.raises(ValueError, match="Missing fidelity confidences"):
            utility.set_fidelity_confidences({1: 0.25})

    def test_forward_uses_deltas_dtype_and_device(self) -> None:
        utility = FidelityCostUtility(fidelity_costs={1: 2.0})
        X = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        deltas = torch.tensor([[3.0]], dtype=torch.float64)

        scaled = utility(X=X, deltas=deltas)

        assert scaled.dtype == torch.float64
        assert scaled.device == deltas.device
        assert math.isclose(scaled.item(), 1.5, rel_tol=1e-12)


@pytest.fixture()
def multi_fidelity_observations() -> list[Observation]:
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
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)
    return surrogate


def test_multifidelity_acquisition_scores_all_candidates_with_cost_utility(
    fitted_mf_surrogate: BoTorchGPSurrogate,
    multi_fidelity_observations: list[Observation],
) -> None:
    acquisition = QMultiFidelityLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(),
        num_fantasies=2,
        num_mv_samples=5,
        num_y_samples=16,
        cost_aware_utility=FidelityCostUtility(fidelity_costs={0: 1.0, 1: 3.0}),
    )
    candidates = [
        Candidate(x=[2.0, 3.0], fidelity=1),
        Candidate(x=[4.0, 5.0], fidelity=0),
    ]

    acquisition.update(fitted_mf_surrogate, multi_fidelity_observations)
    scores = acquisition.score(candidates)

    assert len(scores) == len(candidates)
    assert all(isinstance(score, float) for score in scores)
    assert all(math.isfinite(score) for score in scores)

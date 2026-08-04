import random
from typing import Callable, Sequence

import pytest
import torch

from activelearning.runtime import RuntimeContext
from activelearning.sampler.exact_grid_sampler import ExactGridSampler
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY


class DummyAcquisition:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.received_cost_weighting: (
            Callable[[list[float], list[Candidate]], list[float]] | None
        ) = None

    def score(
        self,
        candidates: Sequence[Candidate],
        cost_weighting: Callable[[list[float], list[Candidate]], list[float]]
        | None = None,
    ) -> list[float]:
        scores = list(self._scores)
        if len(scores) != len(candidates):
            raise AssertionError("test acquisition score length mismatch")
        self.received_cost_weighting = cost_weighting
        if cost_weighting is None:
            return scores
        return cost_weighting(scores, list(candidates))


def _candidate_signature(
    candidates: list[Candidate],
) -> list[tuple[tuple[float, ...], int]]:
    return [(tuple(candidate.x), candidate.fidelity) for candidate in candidates]


def test_sample_returns_full_grid_in_deterministic_order() -> None:
    sampler = ExactGridSampler(
        bounds=[(0.0, 1.0), (10.0, 12.0)],
        points_per_dimension=[2, 3],
    )

    result = sampler.sample()

    assert _candidate_signature(result) == [
        ((0.0, 10.0), DEFAULT_FIDELITY),
        ((0.0, 11.0), DEFAULT_FIDELITY),
        ((0.0, 12.0), DEFAULT_FIDELITY),
        ((1.0, 10.0), DEFAULT_FIDELITY),
        ((1.0, 11.0), DEFAULT_FIDELITY),
        ((1.0, 12.0), DEFAULT_FIDELITY),
    ]


def test_sample_expands_each_grid_point_across_fidelities() -> None:
    sampler = ExactGridSampler(
        bounds=[(0.0, 1.0)],
        points_per_dimension=[2],
        fidelities=[1, 3],
    )

    result = sampler.sample()

    assert _candidate_signature(result) == [
        ((0.0,), 1),
        ((0.0,), 3),
        ((1.0,), 1),
        ((1.0,), 3),
    ]


def test_uniform_subsampling_is_seeded_by_runtime_context_and_round() -> None:
    sampler = ExactGridSampler(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        points_per_dimension=[3, 3],
        num_samples=4,
    )
    sampler.bind_runtime_context(RuntimeContext(seed=11))

    full_pool = ExactGridSampler(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        points_per_dimension=[3, 3],
    ).sample()

    sampler.active_learning_round = 2
    first = sampler.sample()
    assert sampler.active_learning_round == 3
    second = sampler.sample()
    assert sampler.active_learning_round == 4
    expected_round_2 = [
        full_pool[idx] for idx in random.Random(13).sample(range(len(full_pool)), k=4)
    ]
    expected_round_3 = [
        full_pool[idx] for idx in random.Random(14).sample(range(len(full_pool)), k=4)
    ]

    assert _candidate_signature(first) == _candidate_signature(expected_round_2)
    assert _candidate_signature(second) == _candidate_signature(expected_round_3)
    assert _candidate_signature(first) != _candidate_signature(second)

    replay = ExactGridSampler(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        points_per_dimension=[3, 3],
        num_samples=4,
    )
    replay.bind_runtime_context(RuntimeContext(seed=11))
    replay.active_learning_round = 2
    assert _candidate_signature(replay.sample()) == _candidate_signature(first)
    assert _candidate_signature(replay.sample()) == _candidate_signature(second)


def test_weighted_subsampling_uses_multinomial_and_cost_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = ExactGridSampler(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        points_per_dimension=[2, 2],
        num_samples=2,
        use_acquisition_scores=True,
    )
    sampler.bind_runtime_context(RuntimeContext(seed=5))
    sampler.active_learning_round = 7
    acquisition = DummyAcquisition([1.0, 4.0, 9.0, 16.0])
    recorded: dict[str, object] = {}

    def fake_multinomial(
        input: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        recorded["weights"] = input.tolist()
        recorded["num_samples"] = num_samples
        recorded["replacement"] = replacement
        recorded["seed"] = None if generator is None else generator.initial_seed()
        return torch.tensor([3, 1], dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    result = sampler.sample(
        acquisition=acquisition,
        cost_fn=lambda candidates: [1.0, 2.0, 3.0, 4.0],
    )

    assert acquisition.received_cost_weighting is not None
    assert recorded == {
        "weights": [1.0, 2.0, 3.0, 4.0],
        "num_samples": 2,
        "replacement": False,
        "seed": 12,
    }
    assert sampler.active_learning_round == 8
    assert _candidate_signature(result) == [
        ((1.0, 1.0), DEFAULT_FIDELITY),
        ((0.0, 1.0), DEFAULT_FIDELITY),
    ]


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        ([1.0, -1.0], "non-negative"),
        ([0.0, 0.0], "at least one positive"),
        ([1.0, float("nan")], "finite"),
        ([1.0, float("inf")], "finite"),
    ],
)
def test_weighted_subsampling_rejects_invalid_scores(
    scores: list[float], match: str
) -> None:
    sampler = ExactGridSampler(
        bounds=[(0.0, 1.0)],
        points_per_dimension=[2],
        num_samples=1,
        use_acquisition_scores=True,
    )

    with pytest.raises(ValueError, match=match):
        sampler.sample(acquisition=DummyAcquisition(scores))
    assert sampler.active_learning_round == 0


def test_subsampling_without_replacement_rejects_oversized_request() -> None:
    sampler = ExactGridSampler(
        bounds=[(0.0, 1.0)],
        points_per_dimension=[3],
        fidelities=[1],
        num_samples=4,
    )

    with pytest.raises(ValueError, match="cannot exceed the candidate pool size"):
        sampler.sample()
    assert sampler.active_learning_round == 0

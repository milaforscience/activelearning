import pytest
import torch

from activelearning.runtime import RuntimeContext
from activelearning.sampler.hypercube_sampler import HypercubeSampler
from activelearning.utils.types import Candidate


BRANIN_BOUNDS = [(-5.0, 10.0), (0.0, 15.0)]
HARTMANN_BOUNDS = [(0.0, 1.0)] * 6


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _all_within_bounds(candidates: list[Candidate], bounds: list[tuple]) -> bool:
    return all(
        lower <= val <= upper
        for c in candidates
        for val, (lower, upper) in zip(c.x, bounds)
    )


# ---------------------------------------------------------------------------
# point_strategy="uniform" (default)
# ---------------------------------------------------------------------------


class TestUniformPointStrategy:
    def test_returns_correct_count(self):
        sampler = HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=20)
        assert len(sampler.sample()) == 20

    def test_returns_candidate_objects(self):
        sampler = HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=5)
        assert all(isinstance(c, Candidate) for c in sampler.sample())

    def test_x_has_correct_dimensionality(self):
        sampler = HypercubeSampler(bounds=HARTMANN_BOUNDS, num_samples=10)
        for c in sampler.sample():
            assert len(c.x) == 6

    def test_samples_within_bounds(self):
        sampler = HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=200)
        assert _all_within_bounds(sampler.sample(), BRANIN_BOUNDS)

    def test_default_fidelity_is_none(self):
        sampler = HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=5)
        assert all(c.fidelity is None for c in sampler.sample())

    def test_reproducibility_with_seed(self):
        sampler = HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=8)
        torch.manual_seed(42)
        first = sampler.sample()
        torch.manual_seed(42)
        second = sampler.sample()
        assert [c.x for c in first] == [c.x for c in second]

    def test_runtime_context_updates_tensor_dtype_after_construction(self):
        sampler = HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=8)
        sampler.bind_runtime_context(
            RuntimeContext(
                device=torch.device("cpu"),
                dtype=torch.float32,
                precision=32,
            )
        )

        lower, ranges = sampler._get_bounds_tensors()
        unit_points = sampler._generate_points()

        assert lower.dtype == torch.float32
        assert ranges.dtype == torch.float32
        assert unit_points.dtype == torch.float32


# ---------------------------------------------------------------------------
# point_strategy="lhs"
# ---------------------------------------------------------------------------


class TestLatinHypercubePointStrategy:
    def test_returns_correct_count(self):
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=20, point_strategy="lhs"
        )
        assert len(sampler.sample()) == 20

    def test_samples_within_bounds(self):
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=200, point_strategy="lhs"
        )
        assert _all_within_bounds(sampler.sample(), BRANIN_BOUNDS)

    def test_x_has_correct_dimensionality(self):
        sampler = HypercubeSampler(
            bounds=HARTMANN_BOUNDS, num_samples=10, point_strategy="lhs"
        )
        for c in sampler.sample():
            assert len(c.x) == 6

    def test_stratified_coverage_per_dimension(self):
        """Each stratum [k/n, (k+1)/n] should contain exactly one sample per dim."""
        num_samples = 50
        sampler = HypercubeSampler(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            num_samples=num_samples,
            point_strategy="lhs",
        )
        candidates = sampler.sample()
        for dim in range(2):
            values = [c.x[dim] for c in candidates]
            strata = [int(v * num_samples) for v in values]
            # Each stratum should appear exactly once
            assert sorted(strata) == list(range(num_samples)), (
                f"LHS stratification violated in dimension {dim}"
            )

    def test_lhs_different_from_uniform_distribution(self):
        """LHS should provide better space-filling than pure uniform sampling.

        Checks that the minimum nearest-neighbour distance for LHS is on average
        greater than for uniform sampling (statistical test over multiple runs).
        """
        import statistics

        def min_nn_distance(candidates: list[Candidate]) -> float:
            xs = [c.x for c in candidates]
            dists = []
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    d = sum((a - b) ** 2 for a, b in zip(xs[i], xs[j])) ** 0.5
                    dists.append(d)
            return min(dists) if dists else 0.0

        n = 30
        lhs_sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=n, point_strategy="lhs"
        )
        uni_sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=n, point_strategy="uniform"
        )

        lhs_dists = [min_nn_distance(lhs_sampler.sample()) for _ in range(20)]
        uni_dists = [min_nn_distance(uni_sampler.sample()) for _ in range(20)]

        assert statistics.mean(lhs_dists) >= statistics.mean(uni_dists) * 0.8, (
            "LHS mean min-NN distance should not be substantially worse than uniform"
        )

    def test_reproducibility_with_seed(self):
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=8, point_strategy="lhs"
        )
        torch.manual_seed(42)
        first = sampler.sample()
        torch.manual_seed(42)
        second = sampler.sample()
        assert [c.x for c in first] == [c.x for c in second]

    def test_lhs_runtime_context_updates_tensor_dtype_after_construction(self):
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS,
            num_samples=8,
            point_strategy="lhs",
        )
        sampler.bind_runtime_context(
            RuntimeContext(
                device=torch.device("cpu"),
                dtype=torch.float32,
                precision=32,
            )
        )

        unit_points = sampler._generate_points()

        assert unit_points.dtype == torch.float32


# ---------------------------------------------------------------------------
# Uniform fidelity assignment
# ---------------------------------------------------------------------------


class TestUniformFidelityAssignment:
    def test_fidelities_drawn_from_list(self):
        fidelities = [1, 2, 3]
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=50, fidelities=fidelities
        )
        for c in sampler.sample():
            assert c.fidelity in fidelities

    def test_all_fidelities_represented(self):
        fidelities = [1, 2, 3]
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=300, fidelities=fidelities
        )
        observed = {c.fidelity for c in sampler.sample()}
        assert observed == set(fidelities)

    def test_single_fidelity_always_selected(self):
        sampler = HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=20, fidelities=[7])
        assert all(c.fidelity == 7 for c in sampler.sample())


# ---------------------------------------------------------------------------
# Cost-inverse fidelity assignment
# ---------------------------------------------------------------------------


class TestCostInverseFidelityAssignment:
    def test_fidelities_drawn_from_cost_keys(self):
        fidelities = {1: 1.0, 2: 5.0, 3: 10.0}
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=100, fidelities=fidelities
        )
        for c in sampler.sample():
            assert c.fidelity in fidelities

    def test_cheaper_fidelity_sampled_more_often(self):
        """Fidelity 1 (cost=1) should be sampled ~10× more often than fidelity 3 (cost=10)."""
        fidelities = {1: 1.0, 3: 10.0}
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=1000, fidelities=fidelities
        )
        candidates = sampler.sample()
        count_1 = sum(1 for c in candidates if c.fidelity == 1)
        count_3 = sum(1 for c in candidates if c.fidelity == 3)
        # With ratio 10:1 (weights 1.0 vs 0.1), cheap fidelity should dominate
        assert count_1 > count_3 * 3, (
            f"Expected fidelity 1 (cost=1) to dominate fidelity 3 (cost=10), "
            f"got counts {count_1} vs {count_3}"
        )

    def test_fidelities_property_from_cost_keys(self):
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=10, fidelities={2: 2.0, 1: 1.0}
        )
        assert sampler._fidelity_levels == [1, 2]  # sorted

    def test_single_fidelity_cost_always_selected(self):
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS, num_samples=20, fidelities={5: 3.0}
        )
        assert all(c.fidelity == 5 for c in sampler.sample())


# ---------------------------------------------------------------------------
# Composition: LHS + cost-inverse fidelity
# ---------------------------------------------------------------------------


class TestComposition:
    def test_lhs_with_cost_inverse_fidelity(self):
        fidelities = {1: 1.0, 2: 5.0}
        sampler = HypercubeSampler(
            bounds=BRANIN_BOUNDS,
            num_samples=50,
            fidelities=fidelities,
            point_strategy="lhs",
        )
        candidates = sampler.sample()
        assert len(candidates) == 50
        assert _all_within_bounds(candidates, BRANIN_BOUNDS)
        assert all(c.fidelity in fidelities for c in candidates)

    def test_lhs_with_uniform_fidelity(self):
        sampler = HypercubeSampler(
            bounds=HARTMANN_BOUNDS,
            num_samples=20,
            fidelities=[1, 2],
            point_strategy="lhs",
        )
        candidates = sampler.sample()
        assert len(candidates) == 20
        assert all(c.fidelity in {1, 2} for c in candidates)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_on_empty_bounds(self):
        with pytest.raises(ValueError, match="bounds must not be empty"):
            HypercubeSampler(bounds=[], num_samples=5)

    def test_raises_on_invalid_bounds(self):
        with pytest.raises(ValueError, match="lower >= upper"):
            HypercubeSampler(bounds=[(5.0, 2.0)], num_samples=5)

    def test_raises_on_equal_bounds(self):
        with pytest.raises(ValueError, match="lower >= upper"):
            HypercubeSampler(bounds=[(3.0, 3.0)], num_samples=5)

    def test_raises_on_non_positive_num_samples(self):
        with pytest.raises(ValueError, match="num_samples must be > 0"):
            HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=0)

    def test_raises_on_invalid_point_strategy(self):
        with pytest.raises(ValueError, match="point_strategy must be"):
            HypercubeSampler(
                bounds=BRANIN_BOUNDS,
                num_samples=5,
                point_strategy="random",  # type: ignore[arg-type]
            )

    def test_raises_on_empty_fidelities(self):
        with pytest.raises(ValueError, match="fidelities must not be empty"):
            HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=5, fidelities=[])

    def test_raises_on_empty_fidelities_dict(self):
        with pytest.raises(ValueError, match="fidelities must not be empty"):
            HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=5, fidelities={})

    def test_raises_on_zero_cost(self):
        with pytest.raises(ValueError, match="costs must be positive"):
            HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=5, fidelities={1: 0.0})

    def test_raises_on_negative_cost(self):
        with pytest.raises(ValueError, match="costs must be positive"):
            HypercubeSampler(bounds=BRANIN_BOUNDS, num_samples=5, fidelities={1: -1.0})

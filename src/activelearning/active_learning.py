from collections import defaultdict

from activelearning.acquisition.acquisition import Acquisition
from activelearning.dataset.dataset import Dataset
from activelearning.oracle.oracle import Oracle
from activelearning.sampler.sampler import Sampler
from activelearning.selector.selector import Selector
from activelearning.surrogate.surrogate import Surrogate


def group_samples_by_fidelity(samples):
    """Groups samples by their fidelity level."""
    grouped_samples = defaultdict(list)
    for sample in samples:
        fidelity = getattr(sample, "fidelity", None)
        grouped_samples[fidelity].append(sample)
    return dict(grouped_samples)


def get_best_candidates(dataset: Dataset, k: int = 1):
    """Return the top-k observations by y value from a dataset.

    This assumes observations have a scalar 'y' attribute that supports comparison.
    """
    observations = dataset.get_observations()
    if not observations:
        return []
    # filter out None values if necessary, or assume y is valid
    valid_obs = [o for o in observations if o.y is not None]
    sorted_records = sorted(valid_obs, key=lambda r: r.y, reverse=True)
    return sorted_records[:k]


def active_learning(
    dataset: Dataset,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracles: dict[int, Oracle],
    budget: float,
):
    num_iterations = 0
    current_cost = 0.0

    while current_cost < budget:
        observations = dataset.get_observations()
        surrogate.fit(observations)
        acquisition.update(surrogate)

        # Sampler can use observations to avoid re-sampling known points
        samples = sampler.sample(acquisition=acquisition, observations=observations)

        # Pass acquisition to selector to allow it to compute scores if needed
        selected_samples = selector(samples, acquisition=acquisition)

        samples_by_fidelity = group_samples_by_fidelity(selected_samples)

        samples_added_this_iter = False

        for fidelity, samples in samples_by_fidelity.items():
            if fidelity not in oracles:
                continue

            oracle = oracles[fidelity]
            cost = oracle.get_cost(samples)
            if current_cost + cost > budget:
                continue

            scores = oracle.query(samples)
            dataset.add_samples(samples, scores)
            current_cost += cost
            samples_added_this_iter = True

        if not samples_added_this_iter:
            # Prevent infinite loop if we can't afford any new samples
            break

        num_iterations += 1

    return dataset, current_cost, num_iterations

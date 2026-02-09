from collections import defaultdict
from typing import Any, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.dataset.dataset import Dataset
from activelearning.oracle.oracle import Oracle
from activelearning.sampler.sampler import Sampler
from activelearning.selector.selector import Selector
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


def group_samples_by_fidelity(
    samples: Sequence[Candidate],
) -> dict[Any, list[Candidate]]:
    """Group samples by their fidelity level.

    Args:
        samples: Sequence of samples with optional fidelity attribute.

    Returns:
        Dictionary mapping fidelity levels to lists of samples.
    """
    grouped_samples = defaultdict(list)
    for sample in samples:
        fidelity = getattr(sample, "fidelity", None)
        grouped_samples[fidelity].append(sample)
    return dict(grouped_samples)


def get_best_candidates(dataset: Dataset, k: int = 1) -> list[Observation]:
    """Return the top-k observations by y value from a dataset.

    Args:
        dataset: Dataset containing observations.
        k: Number of top observations to return.

    Returns:
        List of top-k observations sorted by y value (descending).
        Returns empty list if no observations exist.

    Note:
        Filters out observations with None y values.
        Assumes y values support comparison operations.
    """
    observations = dataset.get_observations()
    if not observations:
        return []
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
) -> tuple[Dataset, float, int]:
    """Execute the active learning loop with budget constraints.

    Args:
        dataset: Dataset for storing and retrieving observations.
        surrogate: Surrogate model to fit on observations and predict on candidates.
        acquisition: Acquisition function to score candidate utility.
        sampler: Sampler to propose candidate subsets.
        selector: Selector to choose final candidates from sampled pool.
        oracles: Mapping of fidelity levels to oracle instances.
        budget: Maximum allowed cost for querying oracles.

    Returns:
        Tuple containing:
            - Updated dataset with new observations
            - Total cost incurred
            - Number of iterations completed
    """
    current_cost = 0.0
    num_iterations = 0

    while current_cost < budget:
        # Fit surrogate on current dataset observations and update acquisition function
        observations = dataset.get_observations()
        surrogate.fit(observations)
        acquisition.update(surrogate)

        # Sampler can use acquisition for scoring candidates and observations to avoid re-sampling
        samples = sampler.sample(acquisition=acquisition, observations=observations)

        # Pass acquisition to selector to allow it to compute scores if needed
        selected_samples = selector(samples, acquisition=acquisition)
        samples_by_fidelity = group_samples_by_fidelity(selected_samples)

        samples_added_this_iter = False
        for fidelity, samples in samples_by_fidelity.items():
            if fidelity not in oracles:
                continue

            # Check if we can afford to query this oracle with the number of samples
            oracle = oracles[fidelity]
            cost = oracle.get_cost(samples)
            if current_cost + cost > budget:
                continue

            # Query oracle and update dataset with new observations
            scores = oracle.query(samples)
            dataset.add_samples(samples, scores)
            current_cost += cost
            samples_added_this_iter = True

        # Prevent infinite loop if we can't afford any new samples
        if not samples_added_this_iter:
            break

        num_iterations += 1

    return dataset, current_cost, num_iterations

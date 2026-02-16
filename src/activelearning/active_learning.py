from collections import defaultdict
from typing import Any, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.dataset import Dataset
from activelearning.oracle.oracle import Oracle
from activelearning.sampler.sampler import Sampler
from activelearning.selector.selector import Selector
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate


def group_samples_by_fidelity(
    samples: Sequence[Candidate],
) -> dict[Any, list[Candidate]]:
    """Group samples by their fidelity level.

    Useful for multi-fidelity active learning where different fidelity
    levels may have different oracles with different costs.

    Args:
        samples: Sequence of samples with optional fidelity attribute.

    Returns:
        Dictionary mapping fidelity levels (including None) to lists of samples.
    """
    grouped_samples = defaultdict(list)
    for sample in samples:
        fidelity = getattr(sample, "fidelity", None)
        grouped_samples[fidelity].append(sample)
    return dict(grouped_samples)


def active_learning(
    dataset: Dataset,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracles: dict[int, Oracle],
    budget: Budget,
) -> tuple[Dataset, float, int]:
    """Execute the active learning loop with budget constraints.

    Iteratively: (1) fits surrogate on current data, (2) samples candidates,
    (3) selects candidates to label, (4) queries oracle(s) and adds observations.
    Stops when budget is exhausted or no affordable candidates remain.

    Args:
        dataset: Dataset for storing and retrieving observations.
        surrogate: Surrogate model to fit on observations.
        acquisition: Acquisition function to score candidate utility.
            Must be compatible with the surrogate (see acquisition.update() docs).
        sampler: Sampler to propose candidate subsets.
        selector: Selector to choose final candidates from sampled pool.
        oracles: Mapping of fidelity levels to oracle instances.
            Use {None: oracle} for single-fidelity scenarios.
        budget: Budget object managing allocation and consumption.

    Returns:
        Tuple containing:
            - Updated dataset with new observations
            - Total cost incurred across all queries
            - Number of active learning rounds completed

    Note:
        The loop terminates early if no candidates can be afforded within
        the remaining budget to prevent infinite loops.
    """
    initial_budget = budget.available_budget
    num_rounds = 0

    while budget.available_budget > 0:
        # Note: get_observations_iterable() is called multiple times to avoid
        # consuming the same iterable across multiple consumers

        # Fit surrogate on current dataset observations and update acquisition function
        surrogate.fit(dataset.get_observations_iterable())
        acquisition.update(surrogate, dataset.get_observations_iterable())

        # Sampler can use acquisition for scoring candidates and observations to avoid re-sampling
        samples = sampler.sample(
            acquisition=acquisition, observations=dataset.get_observations_iterable()
        )

        # Get round budget and pass to selector along with cost function
        round_budget = budget.get_round_budget(num_rounds)

        # Pass acquisition, cost_fn, and round budget to selector for cost-aware selection
        selected_samples = selector(
            samples,
            acquisition=acquisition,
            cost_fn=None,  # Will be set per-fidelity below
            round_budget=round_budget,
        )
        samples_by_fidelity = group_samples_by_fidelity(selected_samples)

        samples_added_this_iter = False
        for fidelity, fidelity_samples in samples_by_fidelity.items():
            if fidelity not in oracles:
                continue

            oracle = oracles[fidelity]

            # Query oracle - it will consume budget internally
            try:
                new_observations = oracle.query(fidelity_samples, budget)
                dataset.add_observations(new_observations)
                samples_added_this_iter = True
            except ValueError:
                # Budget exhausted during query - stop this fidelity
                continue

        # Prevent infinite loop if we can't afford any new samples
        if not samples_added_this_iter:
            break

        num_rounds += 1

    total_cost = initial_budget - budget.available_budget
    return dataset, total_cost, num_rounds

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


def active_learning(
    dataset: Dataset,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracles: dict[int, Oracle],
    budget: float,
    top_k: int,
):
    num_iterations = 0
    current_cost = 0.0

    while current_cost < budget:
        observations = dataset.get_observations()
        surrogate.fit(observations)
        acquisition.update(surrogate)
        samples = sampler.sample()
        selected_samples = selector(samples)

        samples_by_fidelity = group_samples_by_fidelity(selected_samples)
        for fidelity, samples in samples_by_fidelity.items():
            oracle = oracles[fidelity]
            cost = oracle.get_cost(samples)
            if current_cost + cost > budget:
                continue
            scores = oracle.query(samples)
            dataset.add_samples(samples, scores)
            current_cost += cost
        num_iterations += 1

    best_candidates = dataset.get_top_k(k=top_k)
    return best_candidates, current_cost, num_iterations


# Assumptions:
# 1) Multi-fidelity setting: each candidate has an object x and a fidelity level m
# 2) The surrogate model is a multi-fidelity surrogate that can be trained on the dataset and can
#    make predictions for any fidelity level

# Extensions to consider:
# - Pool-based active learning
# - Parallelization of oracle queries

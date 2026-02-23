from activelearning.acquisition.acquisition import Acquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.dataset import Dataset
from activelearning.oracle.oracle import Oracle
from activelearning.sampler.sampler import Sampler
from activelearning.selector.selector import Selector
from activelearning.surrogate.surrogate import Surrogate


def active_learning(
    dataset: Dataset,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracle: Oracle,
    budget: Budget,
) -> tuple[Dataset, float, int]:
    """Execute the active learning loop with budget constraints.

    Iteratively: (1) fits surrogate on current data, (2) samples candidates,
    (3) selects candidates to label, (4) queries oracle and adds observations.
    Stops when budget is exhausted or no affordable candidates remain.

    Parameters
    ----------
    dataset : Dataset
        Dataset for storing and retrieving observations.
    surrogate : Surrogate
        Surrogate model to fit on observations.
    acquisition : Acquisition
        Acquisition function to score candidate utility.
        Must be compatible with the surrogate (see acquisition.update() docs).
    sampler : Sampler
        Sampler to propose candidate subsets.
    selector : Selector
        Selector to choose final candidates from sampled pool.
    oracle : Oracle
        Oracle instance that handles all fidelity levels internally.
    budget : Budget
        Budget object managing allocation and consumption.

    Returns
    -------
    result : tuple[Dataset, float, int]
        Tuple containing:
            - Updated dataset with new observations
            - Total cost incurred across all queries
            - Number of active learning rounds completed

    Notes
    -----
        The loop terminates early if no candidates can be afforded within
        the remaining budget to prevent infinite loops.
    """
    initial_budget = budget.available_budget
    num_rounds = 0

    # Set fidelity confidences in surrogate before starting the loop
    surrogate.set_fidelity_confidences(oracle.get_fidelity_confidences())

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
            cost_fn=oracle.get_costs,
            round_budget=round_budget,
        )

        # No candidates selected for this round; terminate to avoid stalling.
        if not selected_samples:
            break

        # Query oracle to obtain total cost for the samples
        costs = oracle.get_costs(selected_samples)
        total_cost = sum(costs)

        # Check if we can afford this query before consuming budget
        if not budget.can_afford(total_cost):
            # Budget exhausted - stop iteration
            break

        # Consume budget and query oracle for new observations, then add to dataset
        budget.consume(total_cost)
        new_observations = oracle.query(selected_samples)
        dataset.add_observations(new_observations)

        num_rounds += 1

    total_cost = initial_budget - budget.available_budget
    return dataset, total_cost, num_rounds

from activelearning.acquisition.acquisition import Acquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.dataset import Dataset
from activelearning.oracle.oracle import Oracle
from activelearning.runtime import (
    DEFAULT_RUNTIME_CONTEXT,
    RuntimeContext,
    bind_runtime_context,
)
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
    runtime_context: RuntimeContext | None = None,
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
    runtime_context : RuntimeContext, optional
        Shared runtime settings propagated to runtime-aware components. If
        omitted, components fall back to the default context. If the context
        contains a logger, the loop records per-round metrics through it.

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
    resolved_runtime_context = runtime_context or DEFAULT_RUNTIME_CONTEXT
    logger = resolved_runtime_context.logger

    bind_runtime_context(
        [dataset, surrogate, acquisition, sampler, selector, oracle, budget],
        resolved_runtime_context,
    )

    initial_budget = budget.available_budget
    num_rounds = 0

    # Propagate oracle fidelity confidences to the surrogate before the loop.
    # Surrogates that don't use fidelity metadata safely ignore this (no-op default).
    surrogate.set_fidelity_confidences(oracle.get_fidelity_confidences())

    while budget.available_budget > 0:
        # Call once per round so all consumers share the same consistent epoch view.
        # Implementations must guarantee the returned iterable supports multiple
        # iterations with the same sequence (see Dataset.get_observations_iterable).
        observations = dataset.get_observations_iterable()

        # Dispatch surrogate update based on its declared strategy:
        # - updates_from_latest() True  → incremental update on new observations only
        # - updates_from_latest() False → full refit using the shared round iterable,
        #   guaranteeing the surrogate sees the same consistent data as acquisition/sampler.
        if surrogate.updates_from_latest():
            surrogate.update(dataset.get_latest_observations_iterable())
        else:
            surrogate.fit(observations)

        # Only couple the acquisition to the surrogate once it has been fitted.
        # Before fitting, acquisition falls back to its unfitted behaviour (e.g.
        # returning zero scores), enabling random candidate selection on cold start.
        if surrogate.is_fitted():
            acquisition.update(surrogate, observations)

        # Sampler can use acquisition for scoring candidates and observations to avoid re-sampling
        samples = sampler.sample(acquisition=acquisition, observations=observations)

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

        if logger is not None:
            logger.log_metric("round", num_rounds)
            logger.log_metric("num_new_samples", len(selected_samples))
            logger.log_metric("round_cost", total_cost)
            logger.log_metric("total_cost", initial_budget - budget.available_budget)
            logger.log_metric("budget_remaining", budget.available_budget)
            logger.log_step(num_rounds)

    total_cost = initial_budget - budget.available_budget

    if logger is not None:
        logger.end()

    return dataset, total_cost, num_rounds

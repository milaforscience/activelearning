from activelearning.acquisition.acquisition import Acquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.dataset import Dataset
from activelearning.logger.logger import Logger
from activelearning.oracle.oracle import Oracle
from activelearning.run_writer import RunWriter
from activelearning.runtime import (
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
    run_writer: RunWriter | None = None,
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
        omitted, components receive a fresh context with default values. If the
        context contains a logger, the loop records per-round metrics through it.
    run_writer : RunWriter, optional
        Structured run logger used to persist run-start metadata, per-round
        artifacts, and the final run summary.

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
    resolved_runtime_context = runtime_context or RuntimeContext()
    logger = resolved_runtime_context.logger

    bind_runtime_context(
        [dataset, surrogate, acquisition, sampler, selector, oracle, budget],
        resolved_runtime_context,
    )

    initial_budget = budget.available_budget
    num_rounds = 0
    resolved_runtime_context.active_learning_round = 0
    _start_run_logging(
        run_writer=run_writer,
        dataset=dataset,
        initial_budget=initial_budget,
    )

    # Propagate oracle fidelity confidences to the surrogate before the loop.
    # Surrogates that don't use fidelity metadata safely ignore this (no-op default).
    surrogate.set_fidelity_confidences(oracle.get_fidelity_confidences())

    while budget.available_budget > 0:
        resolved_runtime_context.active_learning_round = num_rounds
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
        sample_scores = acquisition.score(samples)

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
        selected_scores = acquisition.score(selected_samples)

        # Query oracle to obtain total cost for the samples
        costs = oracle.get_costs(selected_samples)
        total_cost = sum(costs)

        # Check if we can afford this query before consuming budget
        if not budget.can_afford(total_cost):
            # Budget exhausted - stop iteration
            break

        # Consume budget, query the oracle, and store the new observations.
        budget.consume(total_cost)
        new_observations = oracle.query(selected_samples)
        dataset.add_observations(new_observations)

        num_rounds += 1
        _log_completed_round(
            logger=logger,
            run_writer=run_writer,
            round_index=num_rounds,
            sampled_candidates=samples,
            sampled_scores=sample_scores,
            selected_candidates=selected_samples,
            selected_scores=selected_scores,
            selected_costs=costs,
            observations=new_observations,
            cumulative_cost=initial_budget - budget.available_budget,
            remaining_budget=budget.available_budget,
            num_new_samples=len(selected_samples),
            round_cost=total_cost,
        )

    total_cost = initial_budget - budget.available_budget

    _finish_run_logging(
        logger=logger,
        run_writer=run_writer,
        num_rounds=num_rounds,
        total_cost=total_cost,
        budget_remaining=budget.available_budget,
    )

    return dataset, total_cost, num_rounds


def _start_run_logging(
    *,
    run_writer: RunWriter | None,
    dataset: Dataset,
    initial_budget: float,
) -> None:
    """Write run-start logging artifacts when optional logging is enabled."""

    if run_writer is None:
        return

    start_metadata = {"initial_budget": initial_budget}
    initial_data_metadata = dict(start_metadata.get("initial_data", {}))
    initial_data_metadata["initial_observations"] = list(
        dataset.get_observations_iterable()
    )
    start_metadata["initial_data"] = initial_data_metadata
    run_writer.start_run(start_metadata)


def _log_completed_round(
    *,
    logger: Logger | None,
    run_writer: RunWriter | None,
    round_index: int,
    sampled_candidates,
    sampled_scores,
    selected_candidates,
    selected_scores,
    selected_costs,
    observations,
    cumulative_cost: float,
    remaining_budget: float,
    num_new_samples: int,
    round_cost: float,
) -> None:
    """Log one completed round to the configured logging backends."""

    if run_writer is not None:
        run_writer.record_round(
            round_index=round_index,
            sampled_candidates=sampled_candidates,
            sampled_scores=sampled_scores,
            selected_candidates=selected_candidates,
            selected_scores=selected_scores,
            selected_costs=selected_costs,
            observations=observations,
            cumulative_cost=cumulative_cost,
            remaining_budget=remaining_budget,
        )

    if logger is not None:
        logger.log_metric("round", round_index)
        logger.log_metric("num_new_samples", num_new_samples)
        logger.log_metric("round_cost", round_cost)
        logger.log_metric("total_cost", cumulative_cost)
        logger.log_metric("budget_remaining", remaining_budget)
        logger.log_step(round_index)


def _finish_run_logging(
    *,
    logger: Logger | None,
    run_writer: RunWriter | None,
    num_rounds: int,
    total_cost: float,
    budget_remaining: float,
) -> None:
    """Finalize the configured logging backends."""

    if logger is not None:
        logger.end()
    if run_writer is not None:
        run_writer.end_run(
            {
                "num_rounds": num_rounds,
                "total_cost": total_cost,
                "budget_remaining": budget_remaining,
            }
        )

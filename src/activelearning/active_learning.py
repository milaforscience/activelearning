import logging
import time
from collections.abc import Sequence
from dataclasses import replace

from activelearning.acquisition.acquisition import Acquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.dataset import Dataset
from activelearning.logger.logger import Logger
from activelearning.monitoring.diagnostics_config import DiagnosticsConfig
from activelearning.monitoring.orchestration import (
    collect_round_diagnostics,
    record_completed_round,
)
from activelearning.oracle.oracle import Oracle
from activelearning.monitoring.profiling import profile_operation
from activelearning.monitoring.run_writer import RoundRecord, RunWriter
from activelearning.runtime import (
    DEFAULT_RUNTIME_CONTEXT,
    RuntimeContext,
    bind_runtime_context,
)
from activelearning.sampler.sampler import Sampler
from activelearning.selector.selector import Selector
from activelearning.surrogate.plotting import PredictionPanel
from activelearning.surrogate.surrogate import MultiFidelitySurrogate, Surrogate
from activelearning.utils.types import (
    Candidate,
    Observation,
    candidate_inputs_match,
    filter_finite_target_observations,
)

_logger = logging.getLogger(__name__)


def _initialize_surrogate_fidelities(
    surrogate: Surrogate,
    oracle: Oracle,
) -> None:
    """Initialize a multi-fidelity surrogate from the oracle's metadata.

    Raises
    ------
    ValueError
        If a multi-fidelity oracle is paired with an unsupported surrogate.
    """
    fidelity_confidences = oracle.get_fidelity_confidences()
    if len(fidelity_confidences) == 1:
        return
    if not isinstance(surrogate, MultiFidelitySurrogate):
        raise ValueError(
            f"{type(surrogate).__name__} does not support multi-fidelity "
            "oracles. Use a MultiFidelitySurrogate implementation."
        )
    surrogate.set_fidelity_confidences(fidelity_confidences)


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
    diagnostics_config: DiagnosticsConfig = DiagnosticsConfig(),
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
        contains a logger, components and the loop can submit live telemetry
        through it.
    run_writer : RunWriter, optional
        Durable structured run writer used to persist run-start metadata,
        completed round records and artifacts, and the final run summary.
        It operates independently of the logger.
    diagnostics_config : DiagnosticsConfig, optional
        Controls optional diagnostic enrichment for configured sinks. Core round
        metrics and profiling remain active when diagnostics are disabled; with
        no logger or run writer, diagnostics are skipped.

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
    _initialize_surrogate_fidelities(surrogate, oracle)

    initial_budget = budget.available_budget
    num_rounds = 0
    surrogate_prequential_history: tuple[PredictionPanel, ...] = ()

    run_started = time.perf_counter()

    # Validate that every reachable round can afford at least one oracle query.
    # Catches misconfigured schedules (e.g. sigmoid with too-slow start) that
    # would silently terminate the experiment.
    budget.validate_schedule(min_query_cost=oracle.get_min_query_cost())
    _start_run_logging(
        run_writer=run_writer,
        dataset=dataset,
        initial_budget=initial_budget,
    )

    while budget.available_budget > 0 and (
        budget.max_rounds is None or num_rounds < budget.max_rounds
    ):
        round_started = time.perf_counter()
        profiling: dict[str, float] = {}

        # Call once per round so all consumers share the same consistent epoch view.
        # Implementations must guarantee the returned iterable supports multiple
        # iterations with the same sequence (see Dataset.get_observations_iterable).
        with profile_operation(profiling, "dataset/get_observations"):
            observations = list(dataset.get_observations_iterable())

        # Dispatch surrogate update based on its declared strategy:
        # - updates_from_latest() True  -> incremental update on new observations only
        # - updates_from_latest() False -> full refit using the shared round iterable,
        #   guaranteeing the surrogate sees the same consistent data as acquisition/sampler.
        if surrogate.updates_from_latest():
            with profile_operation(profiling, "surrogate/update"):
                surrogate.update(dataset.get_latest_observations_iterable())
        else:
            with profile_operation(profiling, "surrogate/fit"):
                surrogate.fit(observations)

        # Only couple the acquisition to the surrogate once it has been fitted.
        # Before fitting, acquisition falls back to its unfitted behaviour (e.g.
        # returning zero scores), enabling random candidate selection on cold start.
        if surrogate.is_fitted():
            with profile_operation(profiling, "acquisition/update"):
                acquisition.update(surrogate, observations)

        # Let the sampler build its candidate pool from the current acquisition,
        # observations, and oracle cost model.
        with profile_operation(profiling, "sampler/sample"):
            samples = sampler.sample(
                acquisition=acquisition,
                observations=observations,
                cost_fn=oracle.get_costs,
            )

        diagnostics_enabled = diagnostics_config.enabled and (
            logger is not None or run_writer is not None
        )
        # Get the current round budget and pass the same oracle cost model to
        # the selector for ranking/filtering.
        with profile_operation(profiling, "budget/get_round_budget"):
            round_budget = budget.get_round_budget(num_rounds)

        with profile_operation(profiling, "selector/select"):
            selected_samples = list(
                selector(
                    samples,
                    acquisition=acquisition,
                    cost_fn=oracle.get_costs,
                    round_budget=round_budget,
                )
            )

        # No candidates selected for this round; terminate to avoid stalling.
        if not selected_samples:
            break

        # Query oracle to obtain total cost for the samples.
        with profile_operation(profiling, "oracle/get_costs"):
            costs = oracle.get_costs(selected_samples)
        total_cost = sum(costs)

        # Check if we can afford this query before consuming budget.
        with profile_operation(profiling, "budget/can_afford"):
            can_afford = budget.can_afford(total_cost)
        if not can_afford:
            # Budget exhausted - stop iteration.
            break

        # Consume budget and query oracle for new observations.
        # Filter out any observations with invalid targets (None, NaN, or infinite)
        # before adding to the dataset.
        # Budget is consumed regardless: a failed evaluation still costs compute time.
        with profile_operation(profiling, "budget/consume"):
            budget.consume(total_cost)

        with profile_operation(profiling, "oracle/query"):
            new_observations = list(oracle.query(selected_samples))
        _validate_oracle_results(selected_samples, new_observations)

        with profile_operation(profiling, "oracle/filter_observations"):
            valid_observations = filter_finite_target_observations(new_observations)
        num_dropped = len(new_observations) - len(valid_observations)
        if num_dropped > 0:
            _logger.warning(
                "Dropped %d/%d oracle observation(s) with invalid targets "
                "(None, NaN, or infinite). Budget was already consumed.",
                num_dropped,
                len(new_observations),
            )

        with profile_operation(profiling, "dataset/add_observations"):
            dataset.add_observations(valid_observations)
        observations_after = list(dataset.get_observations_iterable())
        total_observations = len(observations_after)

        num_rounds += 1
        metrics: dict[str, int | float] = {
            "active_learning/round": num_rounds,
            "active_learning/samples/proposed": len(samples),
            "active_learning/samples/selected": len(selected_samples),
            "active_learning/observations/new": len(valid_observations),
            "active_learning/observations/dropped": num_dropped,
            "active_learning/observations/total": total_observations,
            "active_learning/cost/round": total_cost,
            "active_learning/cost/cumulative": initial_budget - budget.available_budget,
            "active_learning/budget/round": round_budget,
            "active_learning/budget/remaining": budget.available_budget,
        }

        preliminary_record = RoundRecord(
            round_index=num_rounds,
            observations_before=observations,
            observations_after=observations_after,
            sampled_candidates=samples,
            selected_candidates=selected_samples,
            selected_costs=costs,
            queried_observations=new_observations,
            valid_observations=valid_observations,
            round_budget=round_budget,
            initial_budget=initial_budget,
            cumulative_cost=initial_budget - budget.available_budget,
            remaining_budget=budget.available_budget,
            metrics=metrics,
            profiling=profiling,
            diagnostics={},
        )
        with profile_operation(profiling, "diagnostics/total"):
            (
                diagnostic_metrics,
                diagnostic_figures,
                surrogate_prequential_history,
            ) = collect_round_diagnostics(
                record=preliminary_record,
                surrogate=surrogate,
                acquisition=acquisition,
                sampler=sampler,
                selector=selector,
                oracle=oracle,
                dataset=dataset,
                budget=budget,
                enabled=diagnostics_enabled,
                include_figures=(num_rounds % diagnostics_config.figure_interval == 0),
                max_points=diagnostics_config.max_points,
                prequential_history=surrogate_prequential_history,
            )
        profiling["profiling/round/total_s"] = time.perf_counter() - round_started
        record = replace(
            preliminary_record,
            profiling=dict(profiling),
            diagnostics=diagnostic_metrics,
        )

        record_completed_round(
            logger=logger,
            run_writer=run_writer,
            record=record,
            figures=diagnostic_figures,
        )

    total_cost = initial_budget - budget.available_budget
    elapsed_time_s = time.perf_counter() - run_started
    num_observations = len(dataset.get_observations_iterable())

    _finish_run_logging(
        logger=logger,
        run_writer=run_writer,
        num_rounds=num_rounds,
        total_cost=total_cost,
        budget_remaining=budget.available_budget,
        elapsed_time_s=elapsed_time_s,
        num_observations=num_observations,
    )

    return dataset, total_cost, num_rounds


def _start_run_logging(
    *,
    run_writer: RunWriter | None,
    dataset: Dataset,
    initial_budget: float,
) -> None:
    """Write run-start artifacts when a run writer is configured."""
    if run_writer is None:
        return

    run_writer.start_run(
        {
            "initial_budget": initial_budget,
            "initial_data": {
                "initial_observations": list(dataset.get_observations_iterable())
            },
        }
    )


def _validate_oracle_results(
    candidates: Sequence[Candidate],
    observations: Sequence[Observation],
) -> None:
    """Validate positional correspondence between oracle inputs and outputs."""
    if len(candidates) != len(observations):
        raise ValueError(
            "Oracle must return one observation for each selected candidate."
        )
    for candidate, observation in zip(candidates, observations):
        if not isinstance(observation, Observation):
            raise ValueError("Oracle must return Observation objects.")
        if candidate.fidelity != observation.fidelity:
            raise ValueError("Oracle observations must preserve candidate fidelity.")
        if candidate_inputs_match(candidate, observation) is False:
            raise ValueError(
                "Oracle observations must preserve candidate input identity."
            )


def _finish_run_logging(
    *,
    logger: Logger | None,
    run_writer: RunWriter | None,
    num_rounds: int,
    total_cost: float,
    budget_remaining: float,
    elapsed_time_s: float,
    num_observations: int,
) -> None:
    """Finalize the configured monitoring sinks."""
    summary = {
        "num_rounds": num_rounds,
        "total_cost": total_cost,
        "budget_remaining": budget_remaining,
        "elapsed_time_s": elapsed_time_s,
        "num_observations": num_observations,
    }
    try:
        if run_writer is not None:
            run_writer.end_run(summary)
    finally:
        if logger is not None:
            logger.end()

"""Bridge completed-round monitoring producers to independent output sinks."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from activelearning.acquisition.acquisition import Acquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.dataset import Dataset
from activelearning.monitoring.diagnostics import (
    append_prequential_panel,
    budget_diagnostics,
    dataset_diagnostics,
    oracle_diagnostics,
    sampler_diagnostics,
    selection_score_diagnostics,
    surrogate_diagnostics,
)
from activelearning.logger.logger import Logger
from activelearning.monitoring.keys import validate_log_key
from activelearning.monitoring.run_writer import RoundRecord, RunWriter
from activelearning.oracle.oracle import Oracle
from activelearning.sampler.sampler import Sampler
from activelearning.selector.selector import Selector
from activelearning.surrogate.plotting import PredictionPanel
from activelearning.surrogate.surrogate import Surrogate

_logger = logging.getLogger(__name__)


def collect_round_diagnostics(
    *,
    record: RoundRecord,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracle: Oracle,
    dataset: Dataset,
    budget: Budget,
    enabled: bool,
    include_figures: bool,
    max_points: int,
    prequential_history: Sequence[PredictionPanel],
) -> tuple[dict[str, int | float], dict[str, Figure], tuple[PredictionPanel, ...]]:
    """Collect optional reusable and component diagnostics for one completed round.

    This function reads without modifying the completed round data. It does not
    fit models, query oracles, or rescore candidates for monitoring. Even when
    diagnostics are disabled, it clears temporary selector and component data
    so values cannot appear in a later round.
    """
    metrics: dict[str, int | float] = {}
    figures: dict[str, Figure] = {}
    updated_prequential_history = tuple(prequential_history)
    if enabled:
        try:
            surrogate_metrics, surrogate_figures, current_panel = surrogate_diagnostics(
                surrogate,
                record,
                max_points=max_points,
                include_figures=include_figures,
                prequential_history=prequential_history,
            )
        except Exception:
            _logger.warning(
                "Could not collect surrogate diagnostics for round %d.",
                record.round_index,
                exc_info=True,
            )
            metrics["diagnostics/failures/surrogate"] = 1
        else:
            _validate_component_namespace("surrogate", surrogate_metrics)
            _validate_component_namespace("surrogate", surrogate_figures)
            _merge_unique(metrics, surrogate_metrics)
            _merge_unique(figures, surrogate_figures)
            updated_prequential_history = append_prequential_panel(
                prequential_history,
                current_panel,
                max_points=max_points,
            )

        diagnostic_calls = (
            ("sampler", lambda: sampler_diagnostics(record, max_points=max_points)),
            ("oracle", lambda: oracle_diagnostics(record)),
            ("dataset", lambda: dataset_diagnostics(record)),
            ("budget", lambda: budget_diagnostics(record)),
        )
        for component_name, diagnostic_call in diagnostic_calls:
            try:
                component_metrics, component_figures = diagnostic_call()
            except Exception:
                _logger.warning(
                    "Could not collect %s diagnostics for round %d.",
                    component_name,
                    record.round_index,
                    exc_info=True,
                )
                metrics[f"diagnostics/failures/{component_name}"] = 1
                continue
            _validate_component_namespace(component_name, component_metrics)
            _validate_component_namespace(component_name, component_figures)
            _merge_unique(metrics, component_metrics)
            _merge_unique(figures, component_figures)

    score_drain = getattr(selector, "drain_selection_scores", None)
    if callable(score_drain):
        try:
            selection_scores = score_drain()
            if enabled:
                score_metrics, score_figures = selection_score_diagnostics(
                    selection_scores,
                    include_figures=include_figures,
                    max_points=max_points,
                )
            else:
                score_metrics, score_figures = {}, {}
        except Exception:
            _logger.warning(
                "Could not collect selector score diagnostics for round %d.",
                record.round_index,
                exc_info=True,
            )
            if enabled:
                metrics["diagnostics/failures/selector"] = 1
        else:
            if enabled:
                score_metric_groups = _partition_score_payload(score_metrics)
                score_figure_groups = _partition_score_payload(score_figures)
                for component_name, component_metrics in score_metric_groups.items():
                    _validate_component_namespace(component_name, component_metrics)
                    _merge_unique(metrics, component_metrics)
                for component_name, component_figures in score_figure_groups.items():
                    _validate_component_namespace(component_name, component_figures)
                    _merge_unique(figures, component_figures)
            else:
                _close_figures(score_figures.values())

    for component_name, component in (
        ("dataset", dataset),
        ("surrogate", surrogate),
        ("acquisition", acquisition),
        ("sampler", sampler),
        ("selector", selector),
        ("oracle", oracle),
        ("budget", budget),
    ):
        drain = getattr(component, "drain_round_diagnostics", None)
        if not callable(drain):
            continue
        try:
            component_metrics, component_figures = drain(
                include_figures=enabled and include_figures,
                max_points=max_points,
            )
        except Exception:
            _logger.warning(
                "Could not collect %s implementation diagnostics for round %d.",
                component_name,
                record.round_index,
                exc_info=True,
            )
            if enabled:
                metrics[f"diagnostics/failures/{component_name}"] = 1
            continue
        if enabled:
            _validate_component_namespace(component_name, component_metrics)
            _validate_component_namespace(component_name, component_figures)
            _merge_unique(metrics, component_metrics)
            _merge_unique(figures, component_figures)
        else:
            _close_figures(component_figures.values())
    return metrics, figures, updated_prequential_history


def record_completed_round(
    *,
    logger: Logger | None,
    run_writer: RunWriter | None,
    record: RoundRecord,
    figures: Mapping[str, Figure],
) -> None:
    """Fan one validated completed-round payload out to configured sinks.

    The writer persists the record and figures first. The logger then receives
    the same core metrics, profiling, diagnostics, and figures before one round
    step is committed. Figures are closed after both sinks have consumed them.
    """
    _validate_log_namespace(record.metrics)
    _validate_log_namespace(record.profiling)
    _validate_log_namespace(record.diagnostics)
    _validate_log_namespace(figures)
    try:
        for key, figure in figures.items():
            figure.set_label(key)

        if run_writer is not None:
            run_writer.record_round(record, figures)

        if logger is not None:
            for key, value in record.metrics.items():
                logger.log_metric(key, value)
            for key, value in record.profiling.items():
                logger.log_metric(key, value)
            for key, value in record.diagnostics.items():
                logger.log_metric(key, value)
            for key, figure in figures.items():
                logger.log_figure(key, figure)
            logger.log_step(record.round_index)
    finally:
        _close_figures(figures.values())


def _merge_unique(target: dict[str, object], incoming: Mapping[str, object]) -> None:
    """Merge mappings without silently overwriting diagnostic keys."""
    duplicates = target.keys() & incoming.keys()
    if duplicates:
        raise ValueError(f"Duplicate diagnostic keys: {sorted(duplicates)}")
    target.update(incoming)


def _partition_score_payload(
    values: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Partition score diagnostics before validating their component namespaces."""
    grouped = {"acquisition": {}, "selector": {}}
    invalid_keys = []
    for key, value in values.items():
        if key.startswith("acquisition/"):
            grouped["acquisition"][key] = value
        elif key.startswith("selector/"):
            grouped["selector"][key] = value
        else:
            invalid_keys.append(key)
    if invalid_keys:
        raise ValueError(
            "Selection score diagnostics must start with 'acquisition/' or "
            f"'selector/': {invalid_keys}"
        )
    return grouped


def _validate_component_namespace(
    component_name: str,
    values: Mapping[str, object],
) -> None:
    """Require diagnostic payloads to use their component-first namespace."""
    expected_prefix = f"{component_name}/"
    invalid_keys = [
        key
        for key in values
        if not isinstance(key, str) or not key.startswith(expected_prefix)
    ]
    if invalid_keys:
        raise ValueError(
            f"{component_name} diagnostics must start with {expected_prefix!r}: "
            f"{[str(key) for key in invalid_keys]}"
        )


def _validate_log_namespace(values: Mapping[str, object]) -> None:
    """Require every tracker-bound value to use a known slash namespace."""
    invalid_keys = []
    for key in values:
        try:
            validate_log_key(key)
        except ValueError:
            invalid_keys.append(key)
    if invalid_keys:
        raise ValueError(f"Invalid log namespaces: {sorted(invalid_keys)}")


def _close_figures(figures: Sequence[Figure]) -> None:
    """Close each unique Matplotlib figure after all configured sinks consume it."""
    closed_figure_ids: set[int] = set()
    for figure in figures:
        if id(figure) in closed_figure_ids:
            continue
        closed_figure_ids.add(id(figure))
        plt.close(figure)

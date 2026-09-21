"""Pure, reusable diagnostics for completed active-learning rounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING, Any

import torch
from matplotlib.figure import Figure

from activelearning.surrogate.plotting import (
    PredictionPanel,
    build_predicted_vs_observed_figure,
)
from activelearning.surrogate.surrogate import Surrogate
from activelearning.selector.selector import SelectionScores
from activelearning.utils.types import (
    Candidate,
    candidate_identity,
    candidate_inputs_match,
)

if TYPE_CHECKING:
    from activelearning.monitoring.run_writer import RoundRecord


def surrogate_diagnostics(
    surrogate: Surrogate,
    record: "RoundRecord",
    *,
    max_points: int,
    include_figures: bool,
    prequential_history: Sequence[PredictionPanel] = (),
) -> tuple[dict[str, int | float], dict[str, Figure], PredictionPanel | None]:
    """Compute leakage-free current-round and rolling surrogate quality.

    The current round is predicted before its oracle observations are added to
    the next model update. Predictions for scalar targets are evaluated in
    bounded batches, while the returned metrics cover every finite target in
    the round. The supplied history is never modified.
    """
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    if not surrogate.is_fitted():
        return {}, {}, None

    current_panel = _prospective_prediction_panel(
        surrogate,
        record,
        max_points=max_points,
    )
    metrics: dict[str, int | float] = {}
    if current_panel is not None:
        _add_prediction_metrics(
            metrics,
            "surrogate/general/held_out/current_round",
            current_panel,
        )

    updated_history = append_prequential_panel(
        prequential_history,
        current_panel,
        max_points=max_points,
    )
    rolling_panel = updated_history[0] if updated_history else None
    if rolling_panel is not None:
        _add_prediction_metrics(
            metrics,
            "surrogate/general/held_out/rolling",
            rolling_panel,
        )

    figures: dict[str, Figure] = {}
    if include_figures:
        figure_panels: list[PredictionPanel] = []
        if current_panel is not None:
            figure_panels.append(current_panel)
        if rolling_panel is not None and prequential_history:
            figure_panels.append(rolling_panel)
        figure = build_predicted_vs_observed_figure(
            figure_panels,
            max_points=max_points,
        )
        if figure is not None:
            figures["surrogate/general/predicted_vs_observed"] = figure
    return metrics, figures, current_panel


def sampler_diagnostics(
    record: "RoundRecord",
    *,
    max_points: int,
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize candidate duplication, overlap, and fidelity proportions."""
    _ = max_points
    candidates = record.sampled_candidates
    metrics: dict[str, int | float] = {}
    candidate_keys = [candidate_identity(candidate) for candidate in candidates]
    if candidates and all(key is not None for key in candidate_keys):
        known_candidate_keys = [key for key in candidate_keys if key is not None]
        unique_keys = set(known_candidate_keys)
        metrics["sampler/general/duplicate_fraction"] = 1.0 - len(unique_keys) / len(
            known_candidate_keys
        )
        observed_keys = [
            candidate_identity(observation)
            for observation in record.observations_before
        ]
        if all(key is not None for key in observed_keys):
            known_observed_keys = {key for key in observed_keys if key is not None}
            metrics["sampler/general/observed_overlap_fraction"] = sum(
                key in known_observed_keys for key in known_candidate_keys
            ) / len(known_candidate_keys)
    if candidates:
        for fidelity in sorted({candidate.fidelity for candidate in candidates}):
            metrics[f"sampler/general/fidelity_{fidelity}/fraction"] = sum(
                candidate.fidelity == fidelity for candidate in candidates
            ) / len(candidates)
    return metrics, {}


def oracle_diagnostics(
    record: "RoundRecord",
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize oracle failures and query cost by fidelity."""
    if len(record.selected_candidates) != len(record.selected_costs):
        raise ValueError("Selected candidates and costs must have matching lengths.")
    if len(record.selected_candidates) != len(record.queried_observations):
        raise ValueError(
            "Selected candidates and observations must have matching lengths."
        )
    queried = len(record.queried_observations)
    invalid = queried - len(record.valid_observations)
    metrics: dict[str, int | float] = {}
    if queried:
        metrics["oracle/general/failure_rate"] = invalid / queried
    for fidelity in sorted(
        {candidate.fidelity for candidate in record.selected_candidates}
    ):
        fidelity_costs = [
            cost
            for candidate, cost in zip(
                record.selected_candidates,
                record.selected_costs,
            )
            if candidate.fidelity == fidelity
        ]
        metrics[f"oracle/general/fidelity_{fidelity}/queried"] = len(fidelity_costs)
        metrics[f"oracle/general/fidelity_{fidelity}/cost_total"] = float(
            sum(fidelity_costs)
        )
    return metrics, {}


def dataset_diagnostics(
    record: "RoundRecord",
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize observation count and best scalar target by fidelity."""
    metrics: dict[str, int | float] = {}
    for fidelity in sorted(
        {observation.fidelity for observation in record.observations_after}
    ):
        fidelity_observations = [
            observation
            for observation in record.observations_after
            if observation.fidelity == fidelity
        ]
        metrics[f"dataset/general/fidelity_{fidelity}/count"] = len(
            fidelity_observations
        )
        targets = [
            target
            for observation in fidelity_observations
            if (target := _finite_scalar(observation.y)) is not None
        ]
        if targets:
            metrics[f"dataset/general/fidelity_{fidelity}/target_best"] = max(targets)
    return metrics, {}


def budget_diagnostics(
    record: "RoundRecord",
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize round and cumulative budget utilization."""
    spend = float(sum(record.selected_costs))
    metrics: dict[str, int | float] = {}
    if record.round_budget > 0:
        metrics["budget/general/round/utilization"] = spend / record.round_budget
    if record.initial_budget > 0:
        metrics["budget/general/cumulative/utilization"] = (
            record.cumulative_cost / record.initial_budget
        )
    return metrics, {}


def selection_score_diagnostics(
    scores: SelectionScores | None,
    *,
    include_figures: bool,
    max_points: int,
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize the scores used by a completed built-in selection.

    The snapshot contains the full pool scores and the indices selected from
    that pool. It is consumed without rescoring the acquisition function.
    Non-finite values are excluded from summaries and figures, while selection
    itself may still have used them (for example, an infinite zero-cost ratio).
    """
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    if scores is None:
        return {}, {}
    if len(scores.acquisition_scores) != len(scores.ranking_scores):
        raise ValueError("Acquisition and ranking scores must have matching lengths.")
    if any(
        index < 0 or index >= len(scores.acquisition_scores)
        for index in scores.selected_indices
    ):
        raise ValueError("Selected score indices must refer to the candidate pool.")

    selected_acquisition_scores = [
        scores.acquisition_scores[index] for index in scores.selected_indices
    ]
    selected_ranking_scores = [
        scores.ranking_scores[index] for index in scores.selected_indices
    ]
    sampled_acquisition_scores = _finite_values(scores.acquisition_scores)
    selected_acquisition_scores = _finite_values(selected_acquisition_scores)
    sampled_ranking_scores = _finite_values(scores.ranking_scores)
    selected_ranking_scores = _finite_values(selected_ranking_scores)

    metrics: dict[str, int | float] = {}
    _merge_diagnostic_metrics(
        metrics,
        _summary("acquisition/general/sampled", sampled_acquisition_scores),
    )
    _merge_diagnostic_metrics(
        metrics,
        _summary("acquisition/general/selected", selected_acquisition_scores),
    )
    _merge_diagnostic_metrics(
        metrics,
        _summary(
            "selector/general/ranking",
            sampled_ranking_scores,
            include_count=False,
        ),
    )
    _merge_diagnostic_metrics(
        metrics,
        _summary(
            "selector/general/selected_ranking",
            selected_ranking_scores,
            include_count=False,
            include_std=False,
        ),
    )

    figures: dict[str, Figure] = {}
    if include_figures:
        distributions = []
        if sampled_acquisition_scores and selected_acquisition_scores:
            distributions.append(
                (
                    "Acquisition scores",
                    sampled_acquisition_scores,
                    selected_acquisition_scores,
                )
            )
        if sampled_ranking_scores and selected_ranking_scores:
            distributions.append(
                (
                    "Ranking scores",
                    sampled_ranking_scores,
                    selected_ranking_scores,
                )
            )
        figure = _score_distribution_figure(
            distributions,
            max_points=max_points,
        )
        if figure is not None:
            figures["acquisition/general/score_distribution"] = figure
    return metrics, figures


def append_prequential_panel(
    history: Sequence[PredictionPanel],
    panel: PredictionPanel | None,
    *,
    max_points: int,
) -> tuple[PredictionPanel, ...]:
    """Append a panel and retain at most ``max_points`` newest rows."""
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    panels = [*history]
    if panel is not None:
        panels.append(panel)
    combined = _combine_prediction_panels(panels)
    if combined is None:
        return ()
    bounded_rows = list(
        zip(
            combined.targets,
            combined.means,
            combined.standard_deviations or [None] * len(combined.targets),
            combined.fidelities,
        )
    )[-max_points:]
    standard_deviations = (
        None
        if combined.standard_deviations is None
        else tuple(row[2] for row in bounded_rows if row[2] is not None)
    )
    if standard_deviations is not None and len(standard_deviations) != len(
        bounded_rows
    ):
        standard_deviations = None
    return (
        PredictionPanel(
            title="Held-out rolling",
            targets=tuple(row[0] for row in bounded_rows),
            means=tuple(row[1] for row in bounded_rows),
            standard_deviations=standard_deviations,
            fidelities=tuple(row[3] for row in bounded_rows),
        ),
    )


def _prospective_prediction_panel(
    surrogate: Surrogate,
    record: "RoundRecord",
    *,
    max_points: int,
) -> PredictionPanel | None:
    """Predict selected candidates against their newly queried targets."""
    if len(record.selected_candidates) != len(record.queried_observations):
        raise ValueError(
            "Selected candidates and observations must have matching lengths."
        )
    pairs: list[tuple[Candidate, float]] = []
    for candidate, observation in zip(
        record.selected_candidates,
        record.queried_observations,
    ):
        if candidate.fidelity != observation.fidelity:
            raise ValueError("Oracle observations must preserve candidate fidelity.")
        if candidate_inputs_match(candidate, observation) is False:
            raise ValueError(
                "Oracle observations must preserve candidate input identity."
            )
        target = _finite_scalar(observation.y)
        if target is not None:
            pairs.append((candidate, target))
    if not pairs:
        return None

    means: list[float] = []
    standard_deviations: list[float] | None = []
    for start in range(0, len(pairs), max_points):
        batch = pairs[start : start + max_points]
        prediction = _predict(surrogate, [candidate for candidate, _ in batch])
        if prediction is None:
            return None
        batch_means, batch_standard_deviations = prediction
        means.extend(batch_means)
        if batch_standard_deviations is None:
            standard_deviations = None
        elif standard_deviations is not None:
            standard_deviations.extend(batch_standard_deviations)
    return PredictionPanel(
        title="Held-out current round",
        targets=tuple(target for _, target in pairs),
        means=tuple(means),
        standard_deviations=(
            None if standard_deviations is None else tuple(standard_deviations)
        ),
        fidelities=tuple(candidate.fidelity for candidate, _ in pairs),
    )


def _predict(
    surrogate: Surrogate,
    candidates: Sequence[Candidate],
) -> tuple[list[float], list[float] | None] | None:
    """Return aligned, finite mean and optional standard-deviation predictions."""
    try:
        prediction = surrogate.predict(candidates)
    except (NotImplementedError, TypeError, ValueError, RuntimeError):
        return None
    if not isinstance(prediction, Mapping):
        return None
    means = _prediction_values(prediction.get("mean"), len(candidates))
    if means is None:
        return None
    standard_deviations = _prediction_values(prediction.get("std"), len(candidates))
    if standard_deviations is not None and any(
        value < 0 for value in standard_deviations
    ):
        standard_deviations = None
    return means, standard_deviations


def _prediction_values(values: Any, expected_length: int) -> list[float] | None:
    """Coerce one aligned prediction output to finite scalar values."""
    if values is None:
        return None
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        try:
            converted = [_finite_scalar(value) for value in values]
        except TypeError:
            return None
        if len(converted) != expected_length or any(
            value is None for value in converted
        ):
            return None
        return [value for value in converted if value is not None]
    if tensor.numel() != expected_length or not bool(torch.isfinite(tensor).all()):
        return None
    return [float(value) for value in tensor.tolist()]


def _add_prediction_metrics(
    metrics: dict[str, int | float],
    prefix: str,
    panel: PredictionPanel,
) -> None:
    """Add aligned prediction quality metrics for one diagnostics panel."""
    residuals = [mean - target for mean, target in zip(panel.means, panel.targets)]
    metrics[f"{prefix}/count"] = len(panel.targets)
    metrics[f"{prefix}/rmse"] = (
        _mean([residual * residual for residual in residuals]) ** 0.5
    )
    metrics[f"{prefix}/mae"] = _mean([abs(residual) for residual in residuals])
    metrics[f"{prefix}/bias"] = _mean(residuals)
    target_mean = _mean(panel.targets)
    total_sum_squares = sum((target - target_mean) ** 2 for target in panel.targets)
    if len(panel.targets) > 1 and total_sum_squares > 0:
        metrics[f"{prefix}/r2"] = (
            1.0 - sum(residual * residual for residual in residuals) / total_sum_squares
        )
    if panel.standard_deviations is not None:
        metrics[f"{prefix}/std_mean"] = _mean(panel.standard_deviations)
        metrics[f"{prefix}/coverage_95"] = sum(
            abs(residual) <= 1.96 * standard_deviation
            for residual, standard_deviation in zip(
                residuals,
                panel.standard_deviations,
            )
        ) / len(residuals)


def _finite_scalar(value: Any) -> float | None:
    """Return a finite scalar conversion, or ``None`` when unavailable."""
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
        return None
    return float(tensor.reshape(()).item())


def _finite_values(values: Sequence[Any]) -> list[float]:
    """Return finite scalar values, omitting unavailable values."""
    return [
        finite_value
        for value in values
        if (finite_value := _finite_scalar(value)) is not None
    ]


def _summary(
    prefix: str,
    values: Sequence[float],
    *,
    include_count: bool = True,
    include_std: bool = True,
) -> dict[str, int | float]:
    """Return standard summary statistics for a non-empty value sequence."""
    if not values:
        return {}
    mean = _mean(values)
    metrics: dict[str, int | float] = {}
    if include_count:
        metrics[f"{prefix}/count"] = len(values)
    metrics[f"{prefix}/mean"] = mean
    if include_std:
        metrics[f"{prefix}/std"] = (
            sum((value - mean) ** 2 for value in values) / len(values)
        ) ** 0.5
    metrics[f"{prefix}/min"] = min(values)
    metrics[f"{prefix}/max"] = max(values)
    return metrics


def _histogram_bin_edges(
    sampled: Sequence[float],
    selected: Sequence[float],
) -> list[float]:
    """Return common histogram edges for sampled and selected values."""
    values = [*sampled, *selected]
    if not values:
        return []
    value_min = min(values)
    value_max = max(values)
    if value_min == value_max:
        padding = max(abs(value_min) * 0.05, 0.5)
        return [value_min - padding, value_max + padding]
    bin_count = max(5, min(30, math.ceil(math.sqrt(len(values)))))
    step = (value_max - value_min) / bin_count
    return [value_min + step * index for index in range(bin_count + 1)]


def _score_distribution_figure(
    distributions: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    *,
    max_points: int,
) -> Figure | None:
    """Build shared-bin sampled-versus-selected score histograms."""
    if not distributions:
        return None
    figure = Figure(figsize=(7.0 * len(distributions), 4.5))
    for axis_index, (title, sampled, selected) in enumerate(distributions, start=1):
        axis = figure.add_subplot(1, len(distributions), axis_index)
        axis.hist(
            _bounded_values(sampled, max_points),
            bins=_histogram_bin_edges(sampled, selected),
            alpha=0.65,
            label="sampled",
        )
        axis.hist(
            _bounded_values(selected, max_points),
            bins=_histogram_bin_edges(sampled, selected),
            alpha=0.65,
            label="selected",
        )
        axis.set_title(title)
        axis.set_xlabel("Score")
        axis.set_ylabel("Count")
        axis.legend()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Acquisition and selector score distributions")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def _bounded_values(values: Sequence[float], max_points: int) -> list[float]:
    """Return a deterministic subset of values for plotting."""
    if len(values) <= max_points:
        return list(values)
    if max_points == 1:
        return [values[0]]
    indices = [
        index * (len(values) - 1) // (max_points - 1) for index in range(max_points)
    ]
    return [values[index] for index in indices]


def _merge_diagnostic_metrics(
    target: dict[str, int | float],
    incoming: Mapping[str, int | float],
) -> None:
    """Merge generated score metrics without silently overwriting keys."""
    duplicates = target.keys() & incoming.keys()
    if duplicates:
        raise ValueError(f"Duplicate diagnostic keys: {sorted(duplicates)}")
    target.update(incoming)


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty numeric sequence."""
    return float(sum(values) / len(values))


def _combine_prediction_panels(
    panels: Sequence[PredictionPanel],
) -> PredictionPanel | None:
    """Combine prequential prediction panels for rolling error metrics."""
    if not panels:
        return None
    standard_deviations: list[float] | None = []
    for panel in panels:
        if panel.standard_deviations is None:
            standard_deviations = None
            break
        standard_deviations.extend(panel.standard_deviations)
    return PredictionPanel(
        title="Held-out rolling",
        targets=tuple(target for panel in panels for target in panel.targets),
        means=tuple(mean for panel in panels for mean in panel.means),
        standard_deviations=(
            None if standard_deviations is None else tuple(standard_deviations)
        ),
        fidelities=tuple(fidelity for panel in panels for fidelity in panel.fidelities),
    )

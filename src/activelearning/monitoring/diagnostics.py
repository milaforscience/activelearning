"""Pure, reusable diagnostics for completed active-learning rounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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


_PREDICTION_BATCH_SIZE = 4096


@dataclass(frozen=True)
class PrequentialHistory:
    """Round-level prediction chunks and exact rolling metric accumulators."""

    panels: tuple[PredictionPanel, ...] = ()
    count: int = 0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    residual_sum: float = 0.0
    target_mean: float = 0.0
    target_m2: float = 0.0
    standard_deviation_sum: float | None = 0.0
    coverage_count: int | None = 0


def surrogate_diagnostics(
    surrogate: Surrogate,
    record: "RoundRecord",
    *,
    include_figures: bool,
    prequential_history: PrequentialHistory = PrequentialHistory(),
) -> tuple[dict[str, int | float], dict[str, Figure], PrequentialHistory]:
    """Compute leakage-free current-round and rolling surrogate quality.

    The current round is predicted before its oracle observations are added to
    the next model update. Metrics and rendered panels cover every finite target
    in the round. Historical values remain in round-level chunks while rolling
    metrics use sufficient statistics instead of cumulative value tuples.
    """
    if not surrogate.is_fitted():
        return {}, {}, prequential_history

    current_panel = _prospective_prediction_panel(
        surrogate,
        record,
    )
    metrics: dict[str, int | float] = {}
    if current_panel is not None:
        _add_prediction_metrics(
            metrics,
            "surrogate/general/held_out/current_round",
            current_panel,
        )

    updated_history = _append_prequential_panel(
        prequential_history,
        current_panel,
    )
    if updated_history.count:
        _add_rolling_prediction_metrics(
            metrics,
            "surrogate/general/held_out/rolling",
            updated_history,
        )

    figures: dict[str, Figure] = {}
    if include_figures:
        figure_panels: list[PredictionPanel] = []
        if current_panel is not None:
            figure_panels.append(current_panel)
        if prequential_history.panels:
            figure_panels.extend(updated_history.panels)
        figure = build_predicted_vs_observed_figure(
            figure_panels,
        )
        if figure is not None:
            figures["surrogate/general/predicted_vs_observed"] = figure
    return metrics, figures, updated_history


def sampler_diagnostics(
    record: "RoundRecord",
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize candidate duplication, overlap, and fidelity proportions."""
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
            for target in _finite_scalar_values(
                [observation.y for observation in fidelity_observations]
            )
            if target is not None
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
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize the scores used by a completed built-in selection.

    The snapshot contains the full pool scores and the indices selected from
    that pool. It is consumed without rescoring the acquisition function.
    Non-finite values are excluded from summaries and figures, while selection
    itself may still have used them (for example, an infinite zero-cost ratio).
    """
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
        )
        if figure is not None:
            figures["acquisition/general/score_distribution"] = figure
    return metrics, figures


def _append_prequential_panel(
    history: PrequentialHistory,
    panel: PredictionPanel | None,
) -> PrequentialHistory:
    """Append one round chunk and update rolling sufficient statistics."""
    if panel is None or not panel.targets:
        return history

    targets = torch.as_tensor(panel.targets, dtype=torch.float64)
    means = torch.as_tensor(panel.means, dtype=torch.float64)
    residuals = means - targets
    panel_count = len(panel.targets)
    panel_target_mean = float(targets.mean().item())
    panel_target_m2 = float((targets - panel_target_mean).square().sum().item())
    if history.count:
        mean_delta = panel_target_mean - history.target_mean
        total_count = history.count + panel_count
        target_mean = history.target_mean + mean_delta * panel_count / total_count
        target_m2 = (
            history.target_m2
            + panel_target_m2
            + mean_delta**2 * history.count * panel_count / total_count
        )
    else:
        total_count = panel_count
        target_mean = panel_target_mean
        target_m2 = panel_target_m2

    if history.standard_deviation_sum is None or panel.standard_deviations is None:
        standard_deviation_sum = None
        coverage_count = None
    else:
        standard_deviations = torch.as_tensor(
            panel.standard_deviations,
            dtype=torch.float64,
        )
        if history.coverage_count is None:
            raise ValueError("Coverage count is missing for a finite deviation sum.")
        standard_deviation_sum = history.standard_deviation_sum + float(
            standard_deviations.sum().item()
        )
        coverage_count = history.coverage_count + int(
            (residuals.abs() <= 1.96 * standard_deviations).sum().item()
        )

    rolling_panel = replace(panel, title="Held-out rolling")
    return PrequentialHistory(
        panels=(*history.panels, rolling_panel),
        count=total_count,
        squared_error_sum=history.squared_error_sum
        + float(residuals.square().sum().item()),
        absolute_error_sum=history.absolute_error_sum
        + float(residuals.abs().sum().item()),
        residual_sum=history.residual_sum + float(residuals.sum().item()),
        target_mean=target_mean,
        target_m2=target_m2,
        standard_deviation_sum=standard_deviation_sum,
        coverage_count=coverage_count,
    )


def _prospective_prediction_panel(
    surrogate: Surrogate,
    record: "RoundRecord",
) -> PredictionPanel | None:
    """Predict selected candidates against their newly queried targets."""
    if len(record.selected_candidates) != len(record.queried_observations):
        raise ValueError(
            "Selected candidates and observations must have matching lengths."
        )
    candidates: list[Candidate] = []
    target_values: list[Any] = []
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
        candidates.append(candidate)
        target_values.append(observation.y)
    finite_targets = _finite_scalar_values(target_values)
    pairs: list[tuple[Candidate, float]] = [
        (candidate, target)
        for candidate, target in zip(candidates, finite_targets)
        if target is not None
    ]
    if not pairs:
        return None

    prediction = _predict(surrogate, [candidate for candidate, _ in pairs])
    if prediction is None:
        return None
    means, standard_deviations = prediction
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
    """Return aligned predictions while bounding one model call's batch size."""
    means: list[float] = []
    standard_deviations: list[float] | None = []
    for start in range(0, len(candidates), _PREDICTION_BATCH_SIZE):
        batch = candidates[start : start + _PREDICTION_BATCH_SIZE]
        try:
            prediction = surrogate.predict(batch)
        except (NotImplementedError, TypeError, ValueError, RuntimeError):
            return None
        if not isinstance(prediction, Mapping):
            return None
        batch_means = _prediction_values(prediction.get("mean"), len(batch))
        if batch_means is None:
            return None
        means.extend(batch_means)
        batch_standard_deviations = _prediction_values(
            prediction.get("std"),
            len(batch),
        )
        if (
            batch_standard_deviations is None
            or any(value < 0 for value in batch_standard_deviations)
        ):
            standard_deviations = None
        elif standard_deviations is not None:
            standard_deviations.extend(batch_standard_deviations)
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
    target_values = torch.as_tensor(panel.targets, dtype=torch.float64)
    mean_values = torch.as_tensor(panel.means, dtype=torch.float64)
    residuals = mean_values - target_values
    metrics[f"{prefix}/count"] = len(panel.targets)
    metrics[f"{prefix}/rmse"] = (
        float(torch.mean(residuals.square()).sqrt().item())
    )
    metrics[f"{prefix}/mae"] = float(residuals.abs().mean().item())
    metrics[f"{prefix}/bias"] = float(residuals.mean().item())
    target_mean = float(target_values.mean().item())
    total_sum_squares = float((target_values - target_mean).square().sum().item())
    if len(panel.targets) > 1 and total_sum_squares > 0:
        metrics[f"{prefix}/r2"] = (
            1.0 - float(residuals.square().sum().item()) / total_sum_squares
        )
    if panel.standard_deviations is not None:
        standard_deviations = torch.as_tensor(
            panel.standard_deviations,
            dtype=torch.float64,
        )
        metrics[f"{prefix}/std_mean"] = float(standard_deviations.mean().item())
        metrics[f"{prefix}/coverage_95"] = float(
            (residuals.abs() <= 1.96 * standard_deviations)
            .to(torch.float64)
            .mean()
            .item()
        )


def _add_rolling_prediction_metrics(
    metrics: dict[str, int | float],
    prefix: str,
    history: PrequentialHistory,
) -> None:
    """Add rolling prediction metrics from exact accumulated statistics."""
    count = history.count
    metrics[f"{prefix}/count"] = count
    metrics[f"{prefix}/rmse"] = math.sqrt(history.squared_error_sum / count)
    metrics[f"{prefix}/mae"] = history.absolute_error_sum / count
    metrics[f"{prefix}/bias"] = history.residual_sum / count
    if count > 1 and history.target_m2 > 0:
        metrics[f"{prefix}/r2"] = (
            1.0 - history.squared_error_sum / history.target_m2
        )
    if history.standard_deviation_sum is not None:
        metrics[f"{prefix}/std_mean"] = history.standard_deviation_sum / count
        metrics[f"{prefix}/coverage_95"] = history.coverage_count / count


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
    tensor = _scalar_tensor(values)
    if tensor is not None:
        return [float(value) for value in tensor[torch.isfinite(tensor)].tolist()]
    return [
        finite_value
        for value in values
        if (finite_value := _finite_scalar(value)) is not None
    ]


def _finite_scalar_values(values: Sequence[Any]) -> list[float | None]:
    """Return aligned finite scalar conversions, preserving invalid entries."""
    tensor = _scalar_tensor(values)
    if tensor is not None:
        scalar_values = tensor.tolist()
        finite_mask = torch.isfinite(tensor).tolist()
        return [
            float(value) if is_finite else None
            for value, is_finite in zip(scalar_values, finite_mask)
        ]
    return [_finite_scalar(value) for value in values]


def _scalar_tensor(values: Sequence[Any]) -> torch.Tensor | None:
    """Convert a sequence of scalar-like values in one operation when possible."""
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    if tensor.numel() != len(values):
        return None
    return tensor.reshape(-1)


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty numeric sequence."""
    return float(sum(values) / len(values))


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
) -> Figure | None:
    """Build shared-bin sampled-versus-selected score histograms."""
    if not distributions:
        return None
    figure = Figure(figsize=(7.0 * len(distributions), 4.5))
    for axis_index, (title, sampled, selected) in enumerate(distributions, start=1):
        axis = figure.add_subplot(1, len(distributions), axis_index)
        bins = _histogram_bin_edges(sampled, selected)
        axis.hist(
            sampled,
            bins=bins,
            alpha=0.65,
            label="sampled",
        )
        axis.hist(
            selected,
            bins=bins,
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


def _merge_diagnostic_metrics(
    target: dict[str, int | float],
    incoming: Mapping[str, int | float],
) -> None:
    """Merge generated score metrics without silently overwriting keys."""
    duplicates = target.keys() & incoming.keys()
    if duplicates:
        raise ValueError(f"Duplicate diagnostic keys: {sorted(duplicates)}")
    target.update(incoming)

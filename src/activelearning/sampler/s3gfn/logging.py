"""Runtime logging helpers for the S3-GFN sampler."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from matplotlib.figure import Figure

from activelearning.sampler.s3gfn.replay_buffer import ReplayBuffer
from activelearning.utils.types import Candidate

_logger = logging.getLogger("activelearning.sampler.s3gfn.sampler")


@dataclass
class _RoundMetrics:
    """Accumulate S3-GFN training and generation metrics for one round."""

    generated_counts: list[int] = field(default_factory=list)
    valid_counts: list[int] = field(default_factory=list)
    synthesizable_counts: list[int] = field(default_factory=list)
    online_rtb_losses: list[float | None] = field(default_factory=list)
    replay_losses: list[float | None] = field(default_factory=list)
    auxiliary_losses: list[float | None] = field(default_factory=list)
    log_z_values: list[float] = field(default_factory=list)
    raw_reward_scores: list[float] = field(default_factory=list)
    raw_reward_means: list[float] = field(default_factory=list)
    raw_reward_maxes: list[float] = field(default_factory=list)
    training_step_durations_s: list[float] = field(default_factory=list)
    online_updates: int = 0
    replay_updates: int = 0
    generation_attempts: int = 0
    generation_invalid: int = 0
    generation_duplicates: int = 0
    final_candidates: tuple[Candidate, ...] = ()
    positive_buffer_size: int = 0
    negative_buffer_size: int = 0
    training_duration_s: float | None = None
    generation_duration_s: float | None = None

    def record_training_step(
        self,
        *,
        generated_count: int,
        valid_count: int,
        synthesizable_count: int,
        online_loss: float | None,
        replay_loss: float | None,
        auxiliary_loss: float | None,
        log_z: float,
        raw_reward_scores: Sequence[float],
    ) -> None:
        """Record one training step and its generated-batch statistics."""
        self.generated_counts.append(generated_count)
        self.valid_counts.append(valid_count)
        self.synthesizable_counts.append(synthesizable_count)
        self.log_z_values.append(log_z)
        reward_values = [float(value) for value in raw_reward_scores]
        self.raw_reward_scores.extend(reward_values)
        if reward_values:
            self.raw_reward_means.append(_mean(reward_values))
            self.raw_reward_maxes.append(max(reward_values))
        self.online_rtb_losses.append(online_loss)
        if online_loss is not None:
            self.online_updates += 1
        self.replay_losses.append(replay_loss)
        if replay_loss is not None:
            self.replay_updates += 1
        self.auxiliary_losses.append(auxiliary_loss)

    def record_generation_batch(
        self,
        *,
        attempts: int,
        invalid_count: int,
        duplicate_count: int,
    ) -> None:
        """Record rejection counts from one final-generation batch."""
        self.generation_attempts += attempts
        self.generation_invalid += invalid_count
        self.generation_duplicates += duplicate_count

    def record_final_candidates(self, candidates: Sequence[Candidate]) -> None:
        """Store the candidates returned by the completed round."""
        self.final_candidates = tuple(candidates)


class S3GFNLoggingMixin:
    """Provide runtime logging for :class:`S3GFNSampler`."""

    @property
    def round_metrics(self) -> _RoundMetrics:
        """Return the accumulator for the current sampling round."""
        return self._round_metrics

    def _log_training_progress(
        self,
        *,
        step_number: int,
        progress_interval: int,
        generated_count: int,
        valid_count: int,
        synthesizable_count: int,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
    ) -> None:
        """Log periodic training counts."""
        if not (
            step_number == 1
            or step_number % progress_interval == 0
            or step_number == self.n_train_steps
        ):
            return
        _logger.info(
            "S3-GFN training step %d/%d: generated=%d, valid=%d, "
            "invalid=%d, synthesizable=%d, positive_buffer=%d, "
            "negative_buffer=%d.",
            step_number,
            self.n_train_steps,
            generated_count,
            valid_count,
            generated_count - valid_count,
            synthesizable_count,
            len(positive_buffer),
            len(negative_buffer) if negative_buffer is not None else 0,
        )

    def drain_round_diagnostics(
        self,
        *,
        include_figures: bool,
        max_points: int,
    ) -> tuple[dict[str, float | int], dict[str, Figure]]:
        """Return and clear the S3-GFN diagnostics for the current AL round."""
        round_metrics = self.round_metrics
        try:
            if not _has_round_data(round_metrics):
                return {}, {}
            generated_total = sum(round_metrics.generated_counts)
            valid_total = sum(round_metrics.valid_counts)
            synthesizable_total = sum(round_metrics.synthesizable_counts)
            final_candidate_count = len(round_metrics.final_candidates)
            generation_attempts = round_metrics.generation_attempts
            metrics: dict[str, float | int] = {
                "sampler/s3gfn/train/online_updates": round_metrics.online_updates,
                "sampler/s3gfn/train/replay_updates": round_metrics.replay_updates,
                "sampler/s3gfn/train/generated_total": generated_total,
                "sampler/s3gfn/train/valid_total": valid_total,
                "sampler/s3gfn/train/synthesizable_total": synthesizable_total,
                "sampler/s3gfn/train/validity_rate": _safe_ratio(
                    valid_total, generated_total
                ),
                "sampler/s3gfn/train/synthesizable_rate": _safe_ratio(
                    synthesizable_total, valid_total
                ),
                "sampler/s3gfn/train/positive_buffer": round_metrics.positive_buffer_size,
                "sampler/s3gfn/train/negative_buffer": round_metrics.negative_buffer_size,
                "sampler/s3gfn/generation/attempts": generation_attempts,
                "sampler/s3gfn/generation/yield": _safe_ratio(
                    final_candidate_count, generation_attempts
                ),
                "sampler/s3gfn/generation/invalid_rate": _safe_ratio(
                    round_metrics.generation_invalid, generation_attempts
                ),
                "sampler/s3gfn/generation/duplicate_rate": _safe_ratio(
                    round_metrics.generation_duplicates, generation_attempts
                ),
            }
            online_losses = _present_values(round_metrics.online_rtb_losses)
            replay_losses = _present_values(round_metrics.replay_losses)
            contrastive_losses = _present_values(round_metrics.auxiliary_losses)
            if online_losses:
                metrics["sampler/s3gfn/train/online_rtb_loss_mean"] = _mean(
                    online_losses
                )
                metrics["sampler/s3gfn/train/online_rtb_loss_final"] = online_losses[-1]
            if replay_losses:
                metrics["sampler/s3gfn/train/replay_loss_mean"] = _mean(replay_losses)
            if contrastive_losses:
                metrics["sampler/s3gfn/train/contrastive_loss_mean"] = _mean(
                    contrastive_losses
                )
                metrics["sampler/s3gfn/train/contrastive_loss_final"] = (
                    contrastive_losses[-1]
                )
            if round_metrics.log_z_values:
                metrics["sampler/s3gfn/train/log_z_final"] = round_metrics.log_z_values[
                    -1
                ]
            if round_metrics.raw_reward_scores:
                metrics["sampler/s3gfn/reward/raw_mean"] = _mean(
                    round_metrics.raw_reward_scores
                )
                metrics["sampler/s3gfn/reward/raw_max"] = max(
                    round_metrics.raw_reward_scores
                )
            if round_metrics.training_duration_s is not None:
                metrics["sampler/s3gfn/train/duration_s"] = float(
                    round_metrics.training_duration_s
                )
            if round_metrics.generation_duration_s is not None:
                metrics["sampler/s3gfn/generation/duration_s"] = float(
                    round_metrics.generation_duration_s
                )

            figures: dict[str, Figure] = {}
            if include_figures:
                for key, figure in (
                    (
                        "sampler/s3gfn/training_losses",
                        _build_training_losses_figure(round_metrics, max_points),
                    ),
                    (
                        "sampler/s3gfn/log_z",
                        _build_log_z_figure(round_metrics, max_points),
                    ),
                    (
                        "sampler/s3gfn/reward/trajectory",
                        _build_reward_figure(round_metrics, max_points),
                    ),
                ):
                    if figure is not None:
                        figures[key] = figure
            return metrics, figures
        finally:
            self._round_metrics = _RoundMetrics()


def _present_values(values: Sequence[float | None]) -> list[float]:
    """Return recorded numeric values while omitting unavailable updates."""
    return [float(value) for value in values if value is not None]


def _has_round_data(metrics: _RoundMetrics) -> bool:
    """Return whether an accumulator contains data from a completed round."""
    return bool(
        metrics.generated_counts
        or metrics.final_candidates
        or metrics.generation_attempts
        or metrics.training_duration_s is not None
        or metrics.generation_duration_s is not None
    )


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty numeric sequence."""
    return float(sum(values) / len(values))


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a ratio as a plain float, using zero for an empty denominator."""
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _build_training_losses_figure(
    metrics: _RoundMetrics,
    max_points: int,
) -> Figure | None:
    """Build a loss trajectory figure for the recorded training steps."""
    series = (
        ("online RTB", metrics.online_rtb_losses),
        ("replay", metrics.replay_losses),
        ("contrastive", metrics.auxiliary_losses),
    )
    if not any(value is not None for _, values in series for value in values):
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    for label, values in series:
        points = [
            (step, float(value))
            for step, value in enumerate(values, start=1)
            if value is not None
        ]
        points = _bounded(points, max_points)
        if points:
            axis.plot(
                [step for step, _ in points],
                [value for _, value in points],
                label=label,
            )
    axis.set_xlabel("Training step")
    axis.set_ylabel("S3-GFN training loss")
    axis.set_title("S3-GFN training losses")
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _build_log_z_figure(metrics: _RoundMetrics, max_points: int) -> Figure | None:
    """Build a log-normalizer trajectory figure for the recorded steps."""
    if not metrics.log_z_values:
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    points = _bounded(
        list(enumerate(metrics.log_z_values, start=1)),
        max_points,
    )
    axis.plot([step for step, _ in points], [value for _, value in points])
    axis.set_xlabel("Training step")
    axis.set_ylabel("S3-GFN log Z")
    axis.set_title("Sampler S3-GFN: log Z")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _build_reward_figure(metrics: _RoundMetrics, max_points: int) -> Figure | None:
    """Build a raw-reward trajectory figure for recorded training batches."""
    if not metrics.raw_reward_means:
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    mean_points = _bounded(
        list(enumerate(metrics.raw_reward_means, start=1)),
        max_points,
    )
    axis.plot(
        [step for step, _ in mean_points],
        [value for _, value in mean_points],
        label="mean",
    )
    if metrics.raw_reward_maxes:
        max_points_values = _bounded(
            list(enumerate(metrics.raw_reward_maxes, start=1)),
            max_points,
        )
        axis.plot(
            [step for step, _ in max_points_values],
            [value for _, value in max_points_values],
            label="max",
        )
    axis.set_xlabel("Training step")
    axis.set_ylabel("S3-GFN raw reward")
    axis.set_title("Sampler S3-GFN: raw reward trajectory")
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _bounded(
    values: Sequence[tuple[int, float]], max_points: int
) -> list[tuple[int, float]]:
    """Return a deterministic evenly-spaced plot subset."""
    if len(values) <= max_points:
        return list(values)
    if max_points == 1:
        return [values[0]]
    indices = [
        index * (len(values) - 1) // (max_points - 1) for index in range(max_points)
    ]
    return [values[index] for index in indices]

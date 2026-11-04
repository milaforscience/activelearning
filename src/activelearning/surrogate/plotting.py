"""Plotting utilities shared by active-learning surrogate implementations."""

from dataclasses import dataclass
from typing import Sequence

from matplotlib.figure import Figure


@dataclass(frozen=True)
class PredictionPanel:
    """Aligned observed and predicted values for one diagnostic panel."""

    title: str
    targets: tuple[float, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...] | None
    fidelities: tuple[int, ...]

    def __post_init__(self) -> None:
        """Normalize sequence fields and validate their alignment."""
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "means", tuple(self.means))
        if self.standard_deviations is not None:
            object.__setattr__(
                self,
                "standard_deviations",
                tuple(self.standard_deviations),
            )
        object.__setattr__(self, "fidelities", tuple(self.fidelities))
        lengths = {len(self.targets), len(self.means), len(self.fidelities)}
        if self.standard_deviations is not None:
            lengths.add(len(self.standard_deviations))
        if len(lengths) != 1:
            raise ValueError("Prediction panel values must have matching lengths.")

    def bounded(self, max_points: int) -> "PredictionPanel":
        """Return a deterministic subset suitable for rendering."""
        if max_points < 1:
            raise ValueError("max_points must be at least 1.")
        if len(self.targets) <= max_points:
            return self
        if max_points == 1:
            indices = [0]
        else:
            indices = [
                index * (len(self.targets) - 1) // (max_points - 1)
                for index in range(max_points)
            ]
        standard_deviations = (
            None
            if self.standard_deviations is None
            else tuple(self.standard_deviations[index] for index in indices)
        )
        return PredictionPanel(
            title=self.title,
            targets=tuple(self.targets[index] for index in indices),
            means=tuple(self.means[index] for index in indices),
            standard_deviations=standard_deviations,
            fidelities=tuple(self.fidelities[index] for index in indices),
        )


def build_predicted_vs_observed_figure(
    panels: Sequence[PredictionPanel],
    *,
    max_points: int = 1000,
) -> Figure | None:
    """Build one or more fidelity-grouped observed-versus-predicted panels."""
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    valid_panels = [panel.bounded(max_points) for panel in panels if panel.targets]
    if not valid_panels:
        return None

    identity_min, identity_max = _shared_axis_limits(valid_panels)
    figure = Figure(figsize=(7.0 * len(valid_panels), 5.0))
    for panel_index, panel in enumerate(valid_panels, start=1):
        axis = figure.add_subplot(1, len(valid_panels), panel_index)
        axis.plot(
            [identity_min, identity_max],
            [identity_min, identity_max],
            linestyle="--",
            color="black",
            label="identity",
        )

        for fidelity in sorted(set(panel.fidelities)):
            indices = [
                index
                for index, candidate_fidelity in enumerate(panel.fidelities)
                if candidate_fidelity == fidelity
            ]
            fidelity_targets = [panel.targets[index] for index in indices]
            fidelity_means = [panel.means[index] for index in indices]
            label = f"fidelity {fidelity}"
            if panel.standard_deviations is None:
                axis.scatter(fidelity_targets, fidelity_means, label=label)
            else:
                axis.errorbar(
                    fidelity_targets,
                    fidelity_means,
                    yerr=[panel.standard_deviations[index] for index in indices],
                    fmt="o",
                    capsize=2,
                    label=label,
                )

        axis.set_xlim(identity_min, identity_max)
        axis.set_ylim(identity_min, identity_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Observed target")
        axis.set_ylabel("Predicted mean")
        axis.set_title(panel.title)
        axis.legend()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.suptitle("Surrogate diagnostics: held-out predictions")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _shared_axis_limits(panels: Sequence[PredictionPanel]) -> tuple[float, float]:
    """Return common square-axis limits covering every rendered panel."""
    values: list[float] = []
    for panel in panels:
        values.extend(panel.targets)
        values.extend(panel.means)
        if panel.standard_deviations is not None:
            values.extend(
                mean - standard_deviation
                for mean, standard_deviation in zip(
                    panel.means, panel.standard_deviations
                )
            )
            values.extend(
                mean + standard_deviation
                for mean, standard_deviation in zip(
                    panel.means, panel.standard_deviations
                )
            )

    value_min = min(values)
    value_max = max(values)
    padding = (
        max(abs(value_min) * 0.05, 0.05)
        if value_min == value_max
        else (value_max - value_min) * 0.05
    )
    return value_min - padding, value_max + padding

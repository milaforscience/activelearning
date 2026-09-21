"""Plotting utilities shared by active-learning surrogate implementations."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch


_PREDICTION_GRID_SIZE = 64


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


def build_predicted_vs_observed_figure(
    panels: Sequence[PredictionPanel],
) -> Figure | None:
    """Build bounded exact fidelity-grouped observed-versus-predicted panels."""
    valid_panels = [
        panel for panel in panels if _finite_prediction_rows(panel)[0].size
    ]
    if not valid_panels:
        return None
    panel_groups: dict[str, list[PredictionPanel]] = {}
    for panel in valid_panels:
        panel_groups.setdefault(panel.title, []).append(panel)

    identity_min, identity_max = _shared_axis_limits(valid_panels)
    bin_edges = np.linspace(
        identity_min,
        identity_max,
        _PREDICTION_GRID_SIZE + 1,
    )
    figure = Figure(figsize=(7.0 * len(panel_groups), 5.0))
    for panel_index, (title, panel_group) in enumerate(
        panel_groups.items(),
        start=1,
    ):
        axis = figure.add_subplot(1, len(panel_groups), panel_index)
        (identity_line,) = axis.plot(
            [identity_min, identity_max],
            [identity_min, identity_max],
            linestyle="--",
            color="black",
            label="identity",
        )

        fidelities = {
            fidelity
            for panel in panel_group
            for fidelity in _finite_prediction_rows(panel)[2].tolist()
        }
        legend_handles = [identity_line]
        fidelity_colors = colormaps["tab10"]
        for fidelity_index, fidelity in enumerate(sorted(fidelities)):
            counts = np.zeros(
                (_PREDICTION_GRID_SIZE, _PREDICTION_GRID_SIZE),
                dtype=float,
            )
            for panel in panel_group:
                targets, means, panel_fidelities = _finite_prediction_rows(panel)
                fidelity_mask = panel_fidelities == fidelity
                chunk_counts, _, _ = np.histogram2d(
                    targets[fidelity_mask],
                    means[fidelity_mask],
                    bins=(bin_edges, bin_edges),
                )
                counts += chunk_counts
            color = fidelity_colors(fidelity_index % fidelity_colors.N)
            colormap = LinearSegmentedColormap.from_list(
                f"fidelity_{fidelity_index}",
                ("white", color),
            )
            count_grid = np.ma.masked_equal(counts.T, 0)
            axis.pcolormesh(
                bin_edges,
                bin_edges,
                count_grid,
                shading="auto",
                cmap=colormap,
                norm=LogNorm(vmin=1, vmax=max(2, int(counts.max()))),
                alpha=0.85,
            )
            legend_handles.append(
                Patch(
                    facecolor=color,
                    alpha=0.85,
                    label=f"fidelity {fidelity}",
                )
            )

        axis.set_xlim(identity_min, identity_max)
        axis.set_ylim(identity_min, identity_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Observed target")
        axis.set_ylabel("Predicted mean")
        axis.set_title(title)
        axis.legend(handles=legend_handles)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.suptitle("Surrogate diagnostics: held-out predictions")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _shared_axis_limits(panels: Sequence[PredictionPanel]) -> tuple[float, float]:
    """Return common square-axis limits covering every rendered panel."""
    value_min = float("inf")
    value_max = float("-inf")
    for panel in panels:
        targets, means, _ = _finite_prediction_rows(panel)
        value_min = min(value_min, float(targets.min()), float(means.min()))
        value_max = max(value_max, float(targets.max()), float(means.max()))

    padding = (
        max(abs(value_min) * 0.05, 0.05)
        if value_min == value_max
        else (value_max - value_min) * 0.05
    )
    return value_min - padding, value_max + padding


def _finite_prediction_rows(
    panel: PredictionPanel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite target/mean pairs and their aligned fidelities."""
    targets = np.asarray(panel.targets, dtype=float)
    means = np.asarray(panel.means, dtype=float)
    fidelities = np.asarray(panel.fidelities)
    finite = np.isfinite(targets) & np.isfinite(means)
    return targets[finite], means[finite], fidelities[finite]

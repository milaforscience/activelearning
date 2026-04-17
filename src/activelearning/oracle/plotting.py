from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from activelearning.utils.types import Candidate


def build_augmented_2d_landscape_figure(
    evaluator: Callable[[torch.Tensor], torch.Tensor],
    candidates: Sequence[Candidate],
    bounds: Sequence[tuple[float, float]],
    fidelity_confidences: Mapping[int, float],
    supported_fidelities: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    title: str,
    axis_labels: tuple[str, str] = ("x1", "x2"),
    colorbar_label: str = "Oracle score",
    landscape_fidelity: float | None = None,
    grid_size: int = 220,
    filled_levels: int = 60,
    line_levels: int = 18,
    colormap: str = "viridis",
    minima: Sequence[tuple[float, float]] | None = None,
) -> Figure:
    """Render a generic 2-D augmented-function landscape with queried candidates.

    The surface is evaluated on a rectangular 2-D grid, typically at the highest
    configured fidelity unless ``landscape_fidelity`` is provided explicitly.
    """
    if len(bounds) != 2:
        raise ValueError(
            "2-D landscape plotting expects exactly two bounds: one per axis."
        )

    (x1_min, x1_max), (x2_min, x2_max) = bounds
    x1_values = np.linspace(x1_min, x1_max, grid_size)
    x2_values = np.linspace(x2_min, x2_max, grid_size)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

    target_fidelity = (
        max(fidelity_confidences.values())
        if landscape_fidelity is None
        else landscape_fidelity
    )
    model_inputs = np.column_stack(
        (
            x1_grid.ravel(),
            x2_grid.ravel(),
            np.full(x1_grid.size, target_fidelity),
        )
    )

    with torch.no_grad():
        landscape_tensor = torch.as_tensor(
            model_inputs,
            dtype=dtype,
            device=device,
        )
        landscape = (
            evaluator(landscape_tensor).detach().cpu().numpy().reshape(x1_grid.shape)
        )

    figure, axis = plt.subplots(figsize=(8, 6))
    contour = axis.contourf(
        x1_grid,
        x2_grid,
        landscape,
        levels=filled_levels,
        cmap=colormap,
    )
    axis.contour(
        x1_grid,
        x2_grid,
        landscape,
        levels=line_levels,
        colors="white",
        linewidths=0.35,
        alpha=0.35,
    )
    figure.colorbar(contour, ax=axis, label=colorbar_label)

    fidelity_colors = plt.get_cmap("tab10", max(len(supported_fidelities), 1))
    for index, fidelity in enumerate(supported_fidelities):
        fidelity_points = [
            _extract_candidate_coordinates(candidate)
            for candidate in candidates
            if candidate.fidelity == fidelity
        ]
        if not fidelity_points:
            continue
        xs, ys = zip(*fidelity_points)
        axis.scatter(
            xs,
            ys,
            s=80,
            color=fidelity_colors(index),
            edgecolors="white",
            linewidths=0.8,
            label=f"Fidelity {fidelity}",
        )

    axis.set_xlim(x1_min, x1_max)
    axis.set_ylim(x2_min, x2_max)
    axis.set_xlabel(axis_labels[0])
    axis.set_ylabel(axis_labels[1])
    axis.set_title(title)
    if minima:
        min_xs, min_ys = zip(*minima)
        axis.scatter(
            min_xs,
            min_ys,
            s=120,
            color="red",
            marker="x",
            linewidths=2.0,
            zorder=5,
            label="Minima",
        )
    if candidates or minima:
        axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def _extract_candidate_coordinates(candidate: Candidate) -> tuple[float, float]:
    """Normalize a Branin candidate into plottable x/y coordinates."""
    coordinates = torch.as_tensor(candidate.x).reshape(-1)
    if coordinates.numel() != 2:
        raise ValueError("2-D landscape plotting expects two-dimensional candidates.")
    return float(coordinates[0].item()), float(coordinates[1].item())

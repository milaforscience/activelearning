import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from activelearning.surrogate.plotting import (
    PredictionPanel,
    build_predicted_vs_observed_figure,
)


def test_prediction_panel_groups_fidelities() -> None:
    """The diagnostic should plot predictions and group points by fidelity."""
    panel = PredictionPanel(
        title="held-out",
        targets=(1.0, 2.0),
        means=(1.1, 1.9),
        standard_deviations=None,
        fidelities=(0, 1),
    )

    figure = build_predicted_vs_observed_figure([panel])

    assert isinstance(figure, Figure)
    labels = {text.get_text() for text in figure.axes[0].get_legend().get_texts()}
    assert labels == {"identity", "fidelity 0", "fidelity 1"}
    plt.close(figure)


def test_build_predicted_vs_observed_figure_returns_none_for_empty_data() -> None:
    """The diagnostic should skip empty panels."""
    panel = PredictionPanel(
        title="empty",
        targets=(),
        means=(),
        standard_deviations=None,
        fidelities=(),
    )

    assert build_predicted_vs_observed_figure([panel]) is None


def test_build_predicted_vs_observed_figure_retains_large_panels() -> None:
    """The count grids retain every observation with bounded artist sizes."""
    targets = tuple(float(index % 1000) for index in range(100_000))
    panel = PredictionPanel(
        title="large panel",
        targets=targets,
        means=tuple(target + 0.1 for target in targets),
        standard_deviations=None,
        fidelities=tuple(index % 2 for index in range(100_000)),
    )

    figure = build_predicted_vs_observed_figure([panel])

    assert figure is not None
    grids = figure.axes[0].collections
    assert len(grids) == 2
    assert all(len(grid.get_array()) <= 64 * 64 for grid in grids)
    assert sum(np.ma.sum(grid.get_array()) for grid in grids) == 100_000
    plt.close(figure)


def test_build_predicted_vs_observed_figure_shares_compatible_axes() -> None:
    """Every panel should use the same limits and equal data-unit scaling."""
    figure = build_predicted_vs_observed_figure(
        [
            PredictionPanel(
                title="small range",
                targets=(0.0, 1.0),
                means=(0.1, 0.9),
                standard_deviations=(0.05, 0.05),
                fidelities=(0, 0),
            ),
            PredictionPanel(
                title="large range",
                targets=(10.0, 20.0),
                means=(12.0, 18.0),
                standard_deviations=(1.0, 1.0),
                fidelities=(1, 1),
            ),
        ],
    )

    assert figure is not None
    first_axis, second_axis = figure.axes
    assert first_axis.get_xlim() == second_axis.get_xlim()
    assert first_axis.get_ylim() == second_axis.get_ylim()
    assert first_axis.get_xlim() == first_axis.get_ylim()
    assert first_axis.get_aspect() == 1.0
    assert second_axis.get_aspect() == 1.0
    plt.close(figure)

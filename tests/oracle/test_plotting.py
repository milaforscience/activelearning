import torch

from activelearning.oracle.plotting import build_augmented_2d_landscape_figure
from activelearning.utils.types import Candidate


def test_build_augmented_2d_landscape_figure_supports_custom_metadata():
    def evaluator(inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, 0] - 0.5 * inputs[:, 1] + inputs[:, 2]

    figure = build_augmented_2d_landscape_figure(
        evaluator=evaluator,
        candidates=[Candidate(x=[0.25, 0.75], fidelity=1)],
        bounds=((0.0, 1.0), (-1.0, 2.0)),
        fidelity_confidences={1: 0.4, 2: 1.0},
        supported_fidelities=[1, 2],
        dtype=torch.float64,
        device=torch.device("cpu"),
        title="Custom landscape",
        axis_labels=("u", "v"),
        colorbar_label="Custom score",
        landscape_fidelity=0.4,
        grid_size=24,
        filled_levels=12,
        line_levels=6,
    )

    main_axis = figure.axes[0]
    colorbar_axis = figure.axes[1]

    assert main_axis.get_title() == "Custom landscape"
    assert main_axis.get_xlabel() == "u"
    assert main_axis.get_ylabel() == "v"
    assert colorbar_axis.get_ylabel() == "Custom score"
    assert tuple(round(value, 3) for value in main_axis.get_xlim()) == (0.0, 1.0)
    assert tuple(round(value, 3) for value in main_axis.get_ylim()) == (-1.0, 2.0)

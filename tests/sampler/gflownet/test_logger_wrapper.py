from unittest.mock import Mock, patch

import torch
from matplotlib.figure import Figure

from activelearning.sampler.gflownet.logger_wrapper import (
    RuntimeGFlowNetLoggerWrapper,
)


def _wrapper() -> RuntimeGFlowNetLoggerWrapper:
    """Build a wrapper instance without initializing an upstream run."""
    wrapper = RuntimeGFlowNetLoggerWrapper.__new__(RuntimeGFlowNetLoggerWrapper)
    wrapper._last_step = None
    wrapper._pending_metrics = {}
    wrapper._pending_figures = {}
    wrapper.context = "validation"
    return wrapper


def test_format_key_adds_namespace_and_normalizes_spaces() -> None:
    """Runtime keys should identify the sampler and preserve logger context."""
    wrapper = _wrapper()

    assert wrapper._format_key("train rewards mean", use_context=True) == (
        "sampler/gflownet/validation/train_rewards_mean"
    )
    assert wrapper._format_key("summary/loss", use_context=False) == (
        "sampler/gflownet/summary/gflownet_loss"
    )
    assert wrapper._format_key("loss", use_context=True) == (
        "sampler/gflownet/validation/gflownet_loss"
    )


def test_log_metrics_buffer_normalized_values_without_advancing_round_step() -> None:
    """Buffered metrics should use the namespace and retain native step locally."""
    wrapper = _wrapper()

    with patch(
        "activelearning.sampler.gflownet.logger_wrapper.GFlowNetLogger.log_metrics"
    ) as upstream_log_metrics:
        wrapper.log_metrics(
            {"loss": torch.tensor(0.5)},
            step=7,
            use_context=True,
        )

    upstream_log_metrics.assert_called_once_with(
        {"sampler/gflownet/validation/gflownet_loss": torch.tensor(0.5)},
        step=7,
        use_context=False,
    )
    assert wrapper._last_step == 7
    metrics, figures = wrapper.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    )
    assert metrics == {"sampler/gflownet/validation/gflownet_loss": 0.5}
    assert figures == {}
    assert wrapper.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    ) == ({}, {})


def test_log_time_uses_timing_namespace() -> None:
    """Upstream timing names should remain identifiable in the pending output."""
    wrapper = _wrapper()
    wrapper.do = Mock(times=True)

    with patch(
        "activelearning.sampler.gflownet.logger_wrapper.GFlowNetLogger.log_metrics"
    ):
        wrapper.log_time({"train": 1.25}, use_context=True)

    metrics, _ = wrapper.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    )
    assert metrics == {"sampler/gflownet/validation/timing/train": 1.25}


def test_log_plots_retains_component_first_figure_names() -> None:
    """GFlowNet figure names should never be emitted as bare upstream keys."""
    wrapper = _wrapper()
    figure = Figure()

    with patch(
        "activelearning.sampler.gflownet.logger_wrapper.GFlowNetLogger.log_plots"
    ) as upstream_log_plots:
        wrapper.log_plots({"loss": figure}, step=7, use_context=True)

    upstream_log_plots.assert_called_once_with(
        {"sampler/gflownet/validation/gflownet_loss": figure},
        step=7,
        use_context=False,
    )
    _, figures = wrapper.drain_round_diagnostics(
        include_figures=True,
        max_points=1000,
    )
    assert figures == {"sampler/gflownet/validation/gflownet_loss": figure}


def test_log_histogram_namespaces_upstream_key() -> None:
    """Histogram names should identify the GFlowNet sampler before forwarding."""
    wrapper = _wrapper()

    with patch(
        "activelearning.sampler.gflownet.logger_wrapper.GFlowNetLogger.log_histogram"
    ) as upstream_log_histogram:
        wrapper.log_histogram("loss", [0.1, 0.2], step=7, use_context=True)

    upstream_log_histogram.assert_called_once_with(
        "sampler/gflownet/validation/gflownet_loss",
        [0.1, 0.2],
        step=7,
        use_context=False,
    )


def test_log_summary_uses_sampler_namespace_without_context() -> None:
    """Summary metrics should not inherit the upstream validation context."""
    wrapper = _wrapper()
    wrapper._last_step = 7

    with patch(
        "activelearning.sampler.gflownet.logger_wrapper.GFlowNetLogger.log_summary"
    ) as upstream_log_summary:
        wrapper.log_summary({"loss": torch.tensor(0.5)})

    upstream_log_summary.assert_called_once_with(
        {"sampler/gflownet/summary/gflownet_loss": torch.tensor(0.5)}
    )
    metrics, _ = wrapper.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    )
    assert metrics == {"sampler/gflownet/summary/gflownet_loss": 0.5}

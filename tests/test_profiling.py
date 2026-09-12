"""Tests for active-learning operation timing."""

import pytest

import activelearning.monitoring.profiling as profiling_module
from activelearning.monitoring.profiling import profile_operation


def test_profile_operation_records_elapsed_time(monkeypatch) -> None:
    """A completed operation records its elapsed time under the profiling key."""
    clock = iter((10.0, 12.5))
    monkeypatch.setattr(
        profiling_module.time,
        "perf_counter",
        lambda: next(clock),
    )
    metrics: dict[str, float] = {}
    completed: list[bool] = []

    with profile_operation(metrics, "test/action"):
        completed.append(True)

    assert completed == [True]
    assert metrics == {"profiling/test/action_s": 2.5}


def test_profile_operation_does_not_record_failed_operation(monkeypatch) -> None:
    """A failed operation propagates without recording an incomplete duration."""
    clock = iter((10.0, 12.5))
    monkeypatch.setattr(
        profiling_module.time,
        "perf_counter",
        lambda: next(clock),
    )
    metrics: dict[str, float] = {}

    with pytest.raises(RuntimeError, match="expected failure"):
        with profile_operation(metrics, "test/action"):
            raise RuntimeError("expected failure")

    assert metrics == {}

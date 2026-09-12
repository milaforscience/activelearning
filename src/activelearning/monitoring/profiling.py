"""Context-managed timing for active-learning operations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def profile_operation(metrics: dict[str, float], label: str) -> Iterator[None]:
    """Record the elapsed time of a successfully completed operation."""
    started = time.perf_counter()
    yield
    metrics[f"profiling/{label}_s"] = time.perf_counter() - started

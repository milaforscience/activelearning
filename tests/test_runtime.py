from unittest.mock import Mock

import numpy as np
import torch
import random

from activelearning.runtime import RuntimeConfig, RuntimeContext
from activelearning.utils.seeding import set_global_seed


def test_runtime_context_stores_runtime_values():
    """RuntimeContext should store directly provided runtime values."""
    logger = Mock()

    context = RuntimeContext(
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=logger,
        seed=123,
    )

    assert context.logger is logger
    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert context.seed == 123


def test_runtime_config_build_context_includes_seed() -> None:
    """RuntimeConfig should propagate seed into the materialized context."""
    config = RuntimeConfig(device="cpu", precision=32, seed=7)

    context = config.build_context(logger=None)

    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert context.seed == 7


def test_set_global_seed_reseeds_python_numpy_and_torch() -> None:
    """Resetting the same seed should reproduce Python, NumPy, and Torch draws."""
    set_global_seed(42)
    first_values = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )

    set_global_seed(42)
    second_values = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )

    assert second_values == first_values

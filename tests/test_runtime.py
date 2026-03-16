from unittest.mock import Mock

import torch

from activelearning.runtime import RuntimeContext


def test_runtime_context_stores_runtime_values():
    """RuntimeContext should store directly provided runtime values."""
    logger = Mock()

    context = RuntimeContext(
        device=torch.device("cpu"),
        dtype=torch.float32,
        precision=32,
        logger=logger,
    )

    assert context.logger is logger
    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert context.precision == 32

"""Faster state copies for ``gflownet.utils.batch.Batch``.

``Batch.add_to_batch`` copies every state and parent with
``gflownet.utils.common.copy``, which falls back to ``copy.deepcopy`` for any
non-tensor state. Composite environments (``SetFix``, ``Stack``) use dict
states, so every transition goes through the generic ``deepcopy`` machinery,
which was ~40% of GFlowNet sampling time in the xTB IP/EA benchmark.
"""

from copy import deepcopy
from typing import Any

import gflownet.utils.batch as _gflownet_batch
import torch


def copy_state(x: Any) -> Any:
    """Copy a GFlowNet state, cloning tensors and recursing into containers.

    Gives the same result as ``gflownet.utils.common.copy`` for states built
    from tensors, plain dicts, lists, tuples and immutable scalars. Any other
    object, including subclasses of those containers, is copied with
    ``deepcopy``.

    Parameters
    ----------
    x : Any
        The state to copy.

    Returns
    -------
    Any
        An independent copy of ``x``.
    """
    if torch.is_tensor(x):
        return x.clone().detach()
    kind = type(x)
    if kind is dict:
        return {k: copy_state(v) for k, v in x.items()}
    if kind is list:
        return [copy_state(v) for v in x]
    if kind is tuple:
        return tuple(copy_state(v) for v in x)
    if x is None or kind in (bool, int, float, str):
        return x
    return deepcopy(x)


def install_fast_batch_copy() -> None:
    """Make ``gflownet.utils.batch`` use :func:`copy_state` for its copies."""
    _gflownet_batch.copy = copy_state

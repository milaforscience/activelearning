"""Shared utilities for the GFlowNet sampler package."""

from typing import Any, Union

import torch

from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.utils.types import Candidate

# Accepted input shapes for states_proxy: 2-D tensor, list of tensors,
# list of plain sequences (single-fidelity), or list of dicts (multi-fidelity).
_StatesProxy = Union[torch.Tensor, list]


def proxy_states_to_candidates(
    states_proxy: _StatesProxy,
    env: Any,
    fidelity_map: list[int] | None = None,
) -> list[Candidate]:
    """Convert proxy-format states to :class:`~activelearning.utils.types.Candidate` objects.

    This is the single place that handles all shapes returned by
    ``env.states2proxy``. Callers (the GFlowNet sampler and the acquisition
    proxy) should use this instead of duplicating the conversion logic.

    For multi-fidelity envs (:class:`~activelearning.sampler.gflownet.\
multi_fidelity_env_wrapper.MultiFidelityGFlowNetEnvWrapperBase`),
    ``states2proxy`` returns a list of dict-like objects keyed by sub-env
    index. ``env.idx_base_env`` and ``env.idx_fidelity`` locate the coordinate
    tensor and the fidelity scalar respectively.

    The GFlowNet ``Choice`` env uses **1-based** fidelity indices: the source
    state is ``0`` (uncommitted), and choosing option ``i`` produces state ``i``
    (so options ``1..N`` for ``n_options=N``). Use ``fidelity_map`` to translate
    these raw indices to the domain-specific fidelity values expected by the
    oracle. For example, ``fidelity_map=[5, 10, 15]`` maps raw index
    ``1 → 5``, ``2 → 10``, ``3 → 15``; ``fidelity_map=[1, 2, 3]`` is the
    identity mapping used when oracle keys start at 1.

    For single-fidelity envs, ``states2proxy`` may return:

    - a 2-D tensor ``[N, D]``
    - a list of 1-D tensors (stacked internally)
    - a list of plain sequences

    Parameters
    ----------
    states_proxy : torch.Tensor or list
        Output of ``env.states2proxy(states)``. Must be a ``torch.Tensor``
        or a ``list``; any other type raises ``TypeError``.
    env : GFlowNetEnv
        The environment that produced ``states_proxy``. Detected as
        multi-fidelity when it is a
        :class:`~activelearning.sampler.gflownet.multi_fidelity_env_wrapper.MultiFidelityGFlowNetEnvWrapperBase`
        instance; otherwise treated as single-fidelity.
    fidelity_map : list[int] or None
        Maps 1-based ``Choice`` env states to domain fidelity values. Raw
        index ``i`` (``1..N``) is translated to ``fidelity_map[i - 1]``.
        When ``None``, the raw index is stamped directly. Ignored for
        single-fidelity envs.

    Returns
    -------
    list[Candidate]
        One :class:`~activelearning.utils.types.Candidate` per state.
        Multi-fidelity candidates carry a non-``None`` ``fidelity`` field.

    Raises
    ------
    TypeError
        If ``states_proxy`` is neither a ``torch.Tensor`` nor a ``list``.
    """
    if not isinstance(states_proxy, (list, torch.Tensor)):
        raise TypeError(
            f"states_proxy must be a list or torch.Tensor, got {type(states_proxy).__name__}. "
            "This likely indicates a bug in the calling code."
        )
    if len(states_proxy) == 0:
        return []

    if isinstance(env, MultiFidelityGFlowNetEnvWrapperBase):
        idx_base = env.idx_base_env
        idx_fid = env.idx_fidelity
        candidates = []
        for state_proxy in states_proxy:
            base = state_proxy[idx_base]
            fid_raw = state_proxy[idx_fid]
            base_vals = (
                base.detach().cpu().tolist() if torch.is_tensor(base) else list(base)
            )
            raw_index = int(
                fid_raw[0].item() if torch.is_tensor(fid_raw) else fid_raw[0]
            )
            # Choice env: raw_index is 1-based (1..N); convert to 0-based for fidelity_map.
            fidelity = (
                fidelity_map[raw_index - 1] if fidelity_map is not None else raw_index
            )
            candidates.append(Candidate(x=tuple(base_vals), fidelity=fidelity))
        return candidates

    # Single-fidelity: normalize the three possible shapes to a list of tuples.
    if torch.is_tensor(states_proxy):
        rows = states_proxy.detach().cpu().tolist()
    elif torch.is_tensor(states_proxy[0]):
        rows = torch.stack(states_proxy).detach().cpu().tolist()
    else:
        rows = [list(s) for s in states_proxy]

    return [Candidate(x=tuple(row)) for row in rows]

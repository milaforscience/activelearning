"""Shared utilities for the GFlowNet sampler package."""

from typing import Any

import torch

from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.utils.types import Candidate


def proxy_states_to_candidates(states_proxy: Any, env: Any) -> list[Candidate]:
    """Convert proxy-format states to :class:`~activelearning.utils.types.Candidate` objects.

    This is the single place that handles all shapes returned by
    ``env.states2proxy``. Callers (the GFlowNet sampler and the acquisition
    proxy) should use this instead of duplicating the conversion logic.

    For multi-fidelity envs (:class:`~activelearning.sampler.gflownet.\
multi_fidelity_env_wrapper.MultiFidelityGFlowNetEnvWrapperBase`),
    ``states2proxy`` returns a list of dict-like objects keyed by sub-env
    index. ``env.idx_base_env`` and ``env.idx_fidelity`` locate the coordinate
    tensor and the fidelity scalar respectively.

    For single-fidelity envs, ``states2proxy`` may return:
    - a 2-D tensor ``[N, D]``
    - a list of 1-D tensors (stacked internally)
    - a list of plain sequences

    Parameters
    ----------
    states_proxy : tensor, list, or list of dicts
        Output of ``env.states2proxy(states)``.
    env : GFlowNetEnv
        The environment that produced ``states_proxy``.

    Returns
    -------
    list[Candidate]
        One :class:`~activelearning.utils.types.Candidate` per state.
        Multi-fidelity candidates carry a non-``None`` ``fidelity`` field.
    """
    if not isinstance(states_proxy, (list, torch.Tensor)) or len(states_proxy) == 0:
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
            fidelity = int(
                fid_raw[0].item() if torch.is_tensor(fid_raw) else fid_raw[0]
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

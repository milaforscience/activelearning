"""Shared utilities for the GFlowNet sampler package."""

from collections.abc import Mapping, Sequence
from typing import Any, Union

import torch

from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY

# Accepted input shapes for states_proxy: 2-D tensor, list of tensors,
# list of plain sequences (single-fidelity), or list of dicts (multi-fidelity).
_StatesProxy = Union[torch.Tensor, list]


def _is_non_string_sequence(value: Any) -> bool:
    """True for list/tuple-like values, but not text."""
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _normalize_proxy_value(value: Any) -> Any:
    """Convert a proxy value to a stable Python representation."""
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.tolist()
        except TypeError:
            pass
    if _is_non_string_sequence(value):
        return tuple(value)
    return value


def _single_fidelity_proxy_values(proxy_states: Any) -> list[Any]:
    """Normalize single-fidelity proxy outputs to one proxy value per candidate."""
    if proxy_states is None:
        return []

    if torch.is_tensor(proxy_states):
        proxy_values = proxy_states.detach().cpu().tolist()
        return proxy_values if proxy_states.ndim > 1 else [proxy_values]

    proxy_states = _normalize_proxy_value(proxy_states)
    if isinstance(proxy_states, Mapping):
        return [proxy_states]
    if not _is_non_string_sequence(proxy_states):
        return [proxy_states]

    proxy_values = list(proxy_states)
    if not proxy_values:
        return []

    if all(
        isinstance(proxy_value, Mapping)
        or isinstance(proxy_value, (str, bytes, bytearray))
        or _is_non_string_sequence(_normalize_proxy_value(proxy_value))
        or torch.is_tensor(proxy_value)
        for proxy_value in proxy_values
    ):
        return [_normalize_proxy_value(proxy_value) for proxy_value in proxy_values]

    return [proxy_values]


def proxy_states_to_candidates(
    states_proxy: _StatesProxy,
    env: Any,
    fidelity_map: Sequence[int] = (DEFAULT_FIDELITY,),
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
    - a list of plain sequences or strings
    - a list of dicts (Mapping)
    - a single sequence or Mapping (treated as one candidate)

    Parameters
    ----------
    states_proxy : torch.Tensor, list, tuple, or Mapping
        Output of ``env.states2proxy(states)``. Strings, bytes, and ``None``
        are invalid and raise ``TypeError``.
    env : GFlowNetEnv
        The environment that produced ``states_proxy``. Detected as
        multi-fidelity when it is a
        :class:`~activelearning.sampler.gflownet.multi_fidelity_env_wrapper.MultiFidelityGFlowNetEnvWrapperBase`
        instance; otherwise treated as single-fidelity.
    fidelity_map : Sequence[int]
        Maps 1-based ``Choice`` env states to domain fidelity values. Raw
        index ``i`` (``1..N``) is translated to ``fidelity_map[i - 1]``.
        The first value is stamped on candidates from single-fidelity
        environments.

    Returns
    -------
    list[Candidate]
        One :class:`~activelearning.utils.types.Candidate` per state.
        Multi-fidelity candidates carry a fidelity from ``fidelity_map``;
        single-fidelity candidates carry its first value.

    Raises
    ------
    TypeError
        If ``states_proxy`` is a string, bytes, or any other non-collection type.
    """
    if isinstance(states_proxy, (str, bytes, bytearray)) or not isinstance(
        states_proxy, (torch.Tensor, Sequence, Mapping)
    ):
        raise TypeError(
            f"states_proxy must be a list, tuple, torch.Tensor, or Mapping, "
            f"got {type(states_proxy).__name__}. "
            "This likely indicates a bug in the calling code."
        )
    if len(states_proxy) == 0:
        return []

    if isinstance(env, MultiFidelityGFlowNetEnvWrapperBase):
        idx_base = env.idx_base_env
        idx_fid = env.idx_fidelity
        candidates: list[Candidate] = []
        for state_proxy in states_proxy:
            fid_raw = state_proxy[idx_fid]
            raw_index = int(
                fid_raw[0].item() if torch.is_tensor(fid_raw) else fid_raw[0]
            )
            # Choice env: raw_index is 1-based (1..N); convert to 0-based for fidelity_map.
            candidates.append(
                Candidate(
                    x=_normalize_proxy_value(state_proxy[idx_base]),
                    fidelity=fidelity_map[raw_index - 1],
                )
            )
        return candidates

    return [
        Candidate(x=_normalize_proxy_value(proxy_value), fidelity=fidelity_map[0])
        for proxy_value in _single_fidelity_proxy_values(states_proxy)
    ]

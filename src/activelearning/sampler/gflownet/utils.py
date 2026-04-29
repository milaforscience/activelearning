"""Shared utilities for the GFlowNet sampler package."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.utils.types import Candidate


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


def _single_fidelity_proxy_values(proxy_coords: Any) -> list[Any]:
    """Normalize single-fidelity proxy outputs to one proxy value per candidate."""
    if proxy_coords is None:
        return []

    if torch.is_tensor(proxy_coords):
        proxy_values = proxy_coords.detach().cpu().tolist()
        return proxy_values if proxy_coords.ndim > 1 else [proxy_values]

    proxy_coords = _normalize_proxy_value(proxy_coords)
    if isinstance(proxy_coords, Mapping):
        return [proxy_coords]
    if not _is_non_string_sequence(proxy_coords):
        return [proxy_coords]

    proxy_values = list(proxy_coords)
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


def proxy_states_to_candidates(proxy_coords: Any, env: Any) -> list[Candidate]:
    """Convert ``env.states2proxy(...)`` output to candidates."""
    if isinstance(env, MultiFidelityGFlowNetEnvWrapperBase):
        if not _is_non_string_sequence(proxy_coords):
            return []
        idx_base = env.idx_base_env
        idx_fid = env.idx_fidelity

        candidates: list[Candidate] = []
        for proxy_state in proxy_coords:
            fidelity = _normalize_proxy_value(proxy_state[idx_fid])
            if _is_non_string_sequence(fidelity):
                if len(fidelity) == 0:
                    raise ValueError("Fidelity proxy value must not be empty.")
                fidelity = fidelity[0]

            candidates.append(
                Candidate(
                    x=_normalize_proxy_value(proxy_state[idx_base]),
                    fidelity=int(fidelity),
                )
            )
        return candidates

    return [
        Candidate(x=_normalize_proxy_value(proxy_value))
        for proxy_value in _single_fidelity_proxy_values(proxy_coords)
    ]

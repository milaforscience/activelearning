"""Tests for molecule GFlowNet environments."""

from __future__ import annotations

import torch
from gflownet.envs.sequences.base import SequenceBase

from activelearning_molecules.samplers.gflownet_envs import SelfiesSmiles


def _env() -> SelfiesSmiles:
    """Build a small CPU SELFIES environment."""
    return SelfiesSmiles(max_length=8, device="cpu", float_precision=32)


def test_states2proxy_returns_canonical_smiles() -> None:
    """SELFIES states are decoded and canonicalized to SMILES."""
    env = _env()
    state = env.readable2state("[C] [=C] [C] [=C] [C] [=C] [Ring1] [=Branch1]")

    assert env.states2proxy([state]) == ["c1ccccc1"]


def test_states2proxy_keeps_undecodable_molecules() -> None:
    """States without a valid molecule fall back to their raw decoded text."""
    env = _env()
    state = env.readable2state("[Ring1]")

    assert env.states2proxy([state]) == [""]


def _states(env: SelfiesSmiles) -> list:
    """Return representative states: empty, partial, full and pad-in-the-middle."""
    pad, full = env.pad_idx, env.max_length
    seqs = [[pad] * full, [1] * full]
    for filled in range(1, full):
        seqs.append(
            [1 + (i % env.n_tokens) for i in range(filled)] + [pad] * (full - filled)
        )
    seqs.append([1, 2, pad, 3] + [pad] * (full - 4))
    return [torch.tensor(seq, dtype=torch.long) for seq in seqs]


def test_get_seq_length_matches_sequence_base() -> None:
    """The fast override agrees with the upstream implementation everywhere."""
    env = _env()
    for state in _states(env):
        expected = int(SequenceBase._get_seq_length(env, state))
        got = env._get_seq_length(state)

        assert isinstance(got, int)
        assert got == expected


def test_forward_mask_matches_sequence_base() -> None:
    """The fast forward-mask override agrees with the upstream implementation."""
    env = _env()
    for state in _states(env):
        for done in (False, True):
            expected = SequenceBase.get_mask_invalid_actions_forward(env, state, done)

            assert env.get_mask_invalid_actions_forward(state, done) == expected

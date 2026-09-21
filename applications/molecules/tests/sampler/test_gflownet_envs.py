"""Tests for molecule GFlowNet environments."""

from __future__ import annotations

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

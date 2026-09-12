from __future__ import annotations

import pytest
import torch

import activelearning_molecules.samplers.s3gfn.replay_buffer as replay_module


class _FakeChem:
    @staticmethod
    def MolFromSmiles(smiles: str):
        return None if smiles == "invalid" else smiles

    @staticmethod
    def MolToSmiles(molecule, isomericSmiles=False):
        return molecule


class _FakeAllChem:
    @staticmethod
    def GetMorganFingerprintAsBitVect(molecule, radius, nBits):
        return frozenset(molecule)


class _FakeDataStructs:
    @staticmethod
    def BulkTanimotoSimilarity(fingerprint, fingerprints):
        values = []
        for other in fingerprints:
            union = len(fingerprint | other)
            values.append(len(fingerprint & other) / union if union else 1.0)
        return values


def _fake_rdkit():
    return _FakeChem, _FakeDataStructs, _FakeAllChem


def test_positive_buffer_replaces_a_similar_lower_reward(monkeypatch):
    monkeypatch.setattr(replay_module, "require_rdkit", _fake_rdkit)
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        similarity_threshold=0.75,
        policy="reward",
        seed=1,
    )
    tokens = torch.tensor([1, 2, 0])

    assert buffer.add_batch(tokens.unsqueeze(0), ["CC"], [1.0]) == 1
    assert buffer.add_batch(tokens.unsqueeze(0), ["CCC"], [0.5]) == 0
    assert buffer.add_batch(tokens.unsqueeze(0), ["CCC"], [2.0]) == 1
    assert [entry.smiles for entry in buffer.entries] == ["CCC"]


def test_negative_buffer_is_fifo_and_deduplicated():
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        policy="fifo",
    )
    tokens = torch.tensor([1, 2, 0])

    assert buffer.add_batch(tokens.unsqueeze(0), ["a"]) == 1
    assert buffer.add_batch(tokens.unsqueeze(0), ["a"]) == 0
    assert buffer.add_batch(tokens.unsqueeze(0), ["b"]) == 1
    assert buffer.add_batch(tokens.unsqueeze(0), ["c"]) == 1
    assert [entry.smiles for entry in buffer.entries] == ["b", "c"]


def test_fifo_buffer_warns_when_similarity_threshold_is_configured():
    with pytest.warns(UserWarning, match="ignored when policy='fifo'"):
        replay_module.ReplayBuffer(
            pad_token_id=0,
            policy="fifo",
            similarity_threshold=0.5,
        )


def test_replay_sample_uses_requested_reward_dtype():
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        policy="fifo",
    )
    tokens = torch.tensor([1, 2, 0])
    buffer.add_batch(tokens.unsqueeze(0), ["a"], [1.0])

    batch = buffer.sample(count=1, device="cpu", dtype=torch.float64)

    assert batch.reward_scores.dtype == torch.float64


def test_replay_round_trips_terminal_fidelity_indices():
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        policy="fifo",
    )
    tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])

    assert (
        buffer.add_batch(
            tokens,
            ["a", "b"],
            [1.0, 2.0],
            fidelity_indices=torch.tensor([0, 1]),
        )
        == 2
    )

    batch = buffer.sample(count=2, device="cpu")

    assert batch.fidelity_indices is not None
    assert dict(zip(batch.smiles, batch.fidelity_indices.tolist())) == {
        "a": 0,
        "b": 1,
    }

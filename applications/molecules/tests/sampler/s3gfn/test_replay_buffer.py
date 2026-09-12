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


# Single-fidelity behaviour must not depend on how the shared fidelity action
# is represented, so these tests cover both ``None`` and a constant index.
_CONSTANT_FIDELITIES = pytest.mark.parametrize("fidelity", [None, 0])


def _constant_fidelity(fidelity: int | None) -> list[int] | None:
    return None if fidelity is None else [fidelity]


@_CONSTANT_FIDELITIES
def test_positive_buffer_replaces_a_similar_lower_reward(monkeypatch, fidelity):
    monkeypatch.setattr(replay_module, "require_rdkit", _fake_rdkit)
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        similarity_threshold=0.75,
        policy="reward",
        seed=1,
    )
    tokens = torch.tensor([1, 2, 0]).unsqueeze(0)
    fidelities = _constant_fidelity(fidelity)

    assert buffer.add_batch(tokens, ["CC"], [1.0], fidelity_indices=fidelities) == 1
    assert buffer.add_batch(tokens, ["CCC"], [0.5], fidelity_indices=fidelities) == 0
    assert buffer.add_batch(tokens, ["CCC"], [2.0], fidelity_indices=fidelities) == 1
    assert buffer.add_batch(tokens, ["CCC"], [3.0], fidelity_indices=fidelities) == 0
    assert [entry.smiles for entry in buffer.entries] == ["CCC"]


@_CONSTANT_FIDELITIES
def test_negative_buffer_is_fifo_and_deduplicated(fidelity):
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        policy="fifo",
    )
    tokens = torch.tensor([1, 2, 0]).unsqueeze(0)
    fidelities = _constant_fidelity(fidelity)

    assert buffer.add_batch(tokens, ["a"], fidelity_indices=fidelities) == 1
    assert buffer.add_batch(tokens, ["a"], fidelity_indices=fidelities) == 0
    assert buffer.add_batch(tokens, ["b"], fidelity_indices=fidelities) == 1
    assert buffer.add_batch(tokens, ["c"], fidelity_indices=fidelities) == 1
    assert [entry.smiles for entry in buffer.entries] == ["b", "c"]


@pytest.mark.parametrize("policy", ["reward", "fifo"])
def test_buffer_keeps_same_molecule_at_each_fidelity(monkeypatch, policy):
    monkeypatch.setattr(replay_module, "require_rdkit", _fake_rdkit)
    buffer = replay_module.ReplayBuffer(pad_token_id=0, capacity=4, policy=policy)
    tokens = torch.tensor([1, 2, 0]).unsqueeze(0)

    assert buffer.add_batch(tokens, ["CC"], [1.0], fidelity_indices=[0]) == 1
    assert buffer.add_batch(tokens, ["CC"], [3.0], fidelity_indices=[1]) == 1
    assert buffer.add_batch(tokens, ["CC"], [5.0], fidelity_indices=[1]) == 0
    assert [entry.key for entry in buffer.entries] == [("CC", 0), ("CC", 1)]


def test_positive_buffer_similarity_is_compared_within_fidelity(monkeypatch):
    monkeypatch.setattr(replay_module, "require_rdkit", _fake_rdkit)
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=4,
        similarity_threshold=0.75,
        policy="reward",
    )
    tokens = torch.tensor([1, 2, 0]).unsqueeze(0)

    assert buffer.add_batch(tokens, ["CC"], [1.0], fidelity_indices=[0]) == 1
    # A similar molecule at another fidelity is kept alongside, not replacing.
    assert buffer.add_batch(tokens, ["CCC"], [2.0], fidelity_indices=[1]) == 1
    # At the same fidelity, the higher-reward similar molecule replaces.
    assert buffer.add_batch(tokens, ["CCC"], [2.0], fidelity_indices=[0]) == 1
    assert [entry.key for entry in buffer.entries] == [("CCC", 0), ("CCC", 1)]


def test_full_positive_buffer_evicts_global_lowest_reward(monkeypatch):
    monkeypatch.setattr(replay_module, "require_rdkit", _fake_rdkit)
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        similarity_threshold=0.75,
        policy="reward",
    )
    tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])

    assert (
        buffer.add_batch(tokens, ["CC", "N"], [1.0, 2.0], fidelity_indices=[0, 1]) == 2
    )
    assert buffer.add_batch(tokens[:1], ["O"], [1.5], fidelity_indices=[1]) == 1
    assert [entry.key for entry in buffer.entries] == [("O", 1), ("N", 1)]


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


def test_reward_priorities_use_float32_without_positive_epsilon(monkeypatch):
    monkeypatch.setattr(replay_module, "require_rdkit", _fake_rdkit)
    recorded: dict[str, torch.Tensor] = {}

    def recording_multinomial(
        weights,
        num_samples,
        replacement=False,
        *,
        generator=None,
    ):
        del replacement, generator
        recorded["weights"] = weights.clone()
        return torch.arange(num_samples)

    monkeypatch.setattr(replay_module.torch, "multinomial", recording_multinomial)
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        policy="reward",
    )
    tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    buffer.add_batch(tokens, ["a", "b"], [0.001, 0.002])

    batch = buffer.sample(
        count=2,
        device="cpu",
        dtype=torch.bfloat16,
        reward_prioritized=True,
    )

    weights = recorded["weights"]
    assert weights.dtype == torch.float32
    assert float(weights[1] / weights[0]) == pytest.approx(2.0)
    assert batch.reward_scores.dtype == torch.bfloat16


def test_reward_priorities_shift_nonpositive_values_to_positive_weights(monkeypatch):
    monkeypatch.setattr(replay_module, "require_rdkit", _fake_rdkit)
    recorded: dict[str, torch.Tensor] = {}

    def recording_multinomial(
        weights,
        num_samples,
        replacement=False,
        *,
        generator=None,
    ):
        del replacement, generator
        recorded["weights"] = weights.clone()
        return torch.arange(num_samples)

    monkeypatch.setattr(replay_module.torch, "multinomial", recording_multinomial)
    buffer = replay_module.ReplayBuffer(
        pad_token_id=0,
        capacity=2,
        policy="reward",
    )
    tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    buffer.add_batch(tokens, ["a", "b"], [-1.0, 0.0])

    buffer.sample(count=2, device="cpu", reward_prioritized=True)

    weights = recorded["weights"]
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)


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

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
import pytest

import activelearning.sampler.s3gfn.sampler as sampler_module


class _FakeChem:
    @staticmethod
    def MolFromSmiles(smiles: str):
        return None if smiles == "invalid" else smiles

    @staticmethod
    def MolToSmiles(molecule, canonical=True):
        return molecule


class _FakeAcquisition:
    supports_singleton_scoring = True

    def __init__(self) -> None:
        self.seen_fidelities: list[int] = []

    def score(self, candidates, cost_weighting=None):
        self.seen_fidelities.extend(candidate.fidelity for candidate in candidates)
        scores = [float(candidate.fidelity) for candidate in candidates]
        if cost_weighting is not None:
            return cost_weighting(scores, candidates)
        return scores


class _FakeBatchAcquisition:
    supports_singleton_scoring = False


class _FakeModel:
    pad_token_id = 0
    prior = nn.Linear(1, 1)
    policy = nn.Linear(1, 1)

    def __init__(self) -> None:
        self.generated = False

    def encode_smiles(self, smiles, max_length=None):
        return torch.ones((len(smiles), 3), dtype=torch.long)

    def generate(self, count, max_length, temperature):
        assert count == 2
        assert max_length == 8
        assert temperature == 1.0
        if self.generated:
            return SimpleNamespace(
                smiles=(), input_ids=torch.empty((0, 0), dtype=torch.long)
            )
        self.generated = True
        return SimpleNamespace(
            smiles=("CC", "CO"),
            input_ids=torch.ones((2, 3), dtype=torch.long),
            fidelity_indices=torch.tensor([1, 0], dtype=torch.long),
        )

    def policy_eval(self):
        pass


class _FakeSynthesizability:
    def __init__(self, threshold):
        assert threshold == 4.0

    @staticmethod
    def classify_batch(smiles):
        return [True] * len(smiles)


def test_sampler_returns_canonical_smiles_with_conditionally_sampled_fidelities(
    monkeypatch,
):
    fake_model = _FakeModel()
    acquisition = _FakeAcquisition()
    sampler = sampler_module.S3GFNSampler(
        n_samples=2,
        fidelities=[1, 2],
        max_length=8,
        batch_size=2,
        n_train_steps=1,
        aux_coefficient=0.0,
    )
    sampler._new_round_model = lambda: fake_model
    sampler._train_round = lambda **kwargs: None
    monkeypatch.setattr(
        sampler_module,
        "require_rdkit",
        lambda: (_FakeChem, None, None),
    )
    monkeypatch.setattr(
        sampler_module,
        "SAScoreSynthesizability",
        _FakeSynthesizability,
    )

    candidates = sampler.sample(
        acquisition=acquisition,
        cost_fn=lambda candidates: [
            float(candidate.fidelity) for candidate in candidates
        ],
    )

    assert len(candidates) == 2
    assert {candidate.x for candidate in candidates} == {"CC", "CO"}
    assert [candidate.fidelity for candidate in candidates] == [2, 1]
    assert acquisition.seen_fidelities == []


def test_prepare_batch_scores_the_selected_fidelity(monkeypatch):
    fake_model = _FakeModel()
    acquisition = _FakeAcquisition()
    sampler = sampler_module.S3GFNSampler(
        n_samples=2,
        fidelities=[1, 2],
        max_length=8,
        batch_size=2,
        n_train_steps=1,
        aux_coefficient=0.0,
    )
    monkeypatch.setattr(
        sampler_module,
        "SAScoreSynthesizability",
        _FakeSynthesizability,
    )
    monkeypatch.setattr(
        sampler_module,
        "require_rdkit",
        lambda: (_FakeChem, None, None),
    )

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC", "CO"),
        fidelity_indices=torch.tensor([1, 0]),
        synthesizability=_FakeSynthesizability(threshold=4.0),
        molecule_chem=_FakeChem,
        acquisition=acquisition,
        cost_fn=None,
    )

    assert prepared.reward_scores.tolist() == [2.0, 1.0]
    assert acquisition.seen_fidelities == [2, 1]


def test_single_fidelity_preparation_omits_terminal_action(monkeypatch):
    fake_model = _FakeModel()
    sampler = sampler_module.S3GFNSampler(
        n_samples=1,
        fidelities=[7],
        max_length=8,
        batch_size=1,
        n_train_steps=1,
        aux_coefficient=0.0,
    )
    monkeypatch.setattr(
        sampler_module,
        "require_rdkit",
        lambda: (_FakeChem, None, None),
    )

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC",),
        fidelity_indices=None,
        synthesizability=_FakeSynthesizability(threshold=4.0),
        molecule_chem=_FakeChem,
        acquisition=_FakeAcquisition(),
        cost_fn=None,
    )

    assert prepared.fidelity_indices is None
    assert prepared.reward_scores.tolist() == [7.0]


def test_sampler_rejects_batch_only_acquisitions():
    sampler = sampler_module.S3GFNSampler(
        n_samples=1,
        fidelities=[1],
        n_train_steps=1,
    )

    with pytest.raises(ValueError, match="singleton scoring"):
        sampler.sample(acquisition=_FakeBatchAcquisition())


def test_canonicalization_rejects_disconnected_molecules():
    assert (
        sampler_module._canonicalize_to_smiles(
            "C.C",
            molecule_chem=_FakeChem,
        )
        is None
    )

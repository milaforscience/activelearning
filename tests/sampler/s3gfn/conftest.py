"""Shared fixtures and test doubles for the S3-GFN sampler tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from torch import nn

import activelearning.sampler.s3gfn.sampler as sampler_module


class FakeTokenizer:
    """Right-padded tokenizer stub with distinct padding and EOS ids."""

    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"


class FakeChem:
    """Minimal RDKit ``Chem`` stand-in treating SMILES as their own molecule."""

    @staticmethod
    def MolFromSmiles(smiles: str):
        """Return ``None`` for the reserved invalid SMILES, else the input."""
        return None if smiles == "invalid" else smiles

    @staticmethod
    def MolToSmiles(molecule, canonical=True, isomericSmiles=False):
        """Return the molecule unchanged, standing in for canonicalization."""
        return molecule


class FakeSynthesizability:
    """Synthesizability stub that accepts every molecule."""

    def __init__(self, threshold: float = 4.0) -> None:
        self.threshold = threshold

    @staticmethod
    def classify_batch(smiles) -> list[bool]:
        """Classify every molecule as synthesizable."""
        return [True] * len(smiles)


class FakeAcquisition:
    """Singleton-scoring acquisition returning each candidate's fidelity."""

    supports_singleton_scoring = True

    def __init__(self) -> None:
        self.seen_fidelities: list[int] = []

    def score(self, candidates, cost_weighting=None):
        """Score candidates by fidelity, recording what was scored."""
        self.seen_fidelities.extend(candidate.fidelity for candidate in candidates)
        scores = [float(candidate.fidelity) for candidate in candidates]
        if cost_weighting is not None:
            return cost_weighting(scores, candidates)
        return scores


class FakeModel:
    """Policy stub yielding one fixed batch, then exhausting."""

    pad_token_id = 0
    prior = nn.Linear(1, 1)
    policy = nn.Linear(1, 1)

    def __init__(self) -> None:
        self.generated = False
        self.generate_calls: list[dict[str, Any]] = []

    def encode_smiles(self, smiles):
        """Encode every molecule as an identical token row."""
        return torch.ones((len(smiles), 3), dtype=torch.long)

    def generate(self, count, max_length, temperature):
        """Return two molecules on the first call and nothing afterwards."""
        self.generate_calls.append(
            {
                "count": count,
                "max_length": max_length,
                "temperature": temperature,
            }
        )
        if self.generated:
            return SimpleNamespace(
                smiles=(),
                input_ids=torch.empty((0, 0), dtype=torch.long),
                fidelity_indices=None,
            )
        self.generated = True
        return SimpleNamespace(
            smiles=("CC", "CO"),
            input_ids=torch.ones((2, 3), dtype=torch.long),
            fidelity_indices=torch.tensor([1, 0], dtype=torch.long),
        )

    def policy_eval(self):
        """Match the model interface; evaluation mode is a no-op here."""


@pytest.fixture
def fake_acquisition() -> FakeAcquisition:
    """Return a fresh singleton-scoring acquisition stub."""
    return FakeAcquisition()


@pytest.fixture
def fake_model() -> FakeModel:
    """Return a fresh policy stub for round-level sampler tests."""
    return FakeModel()


@pytest.fixture
def make_sampler() -> Callable[..., Any]:
    """Return a factory building an ``S3GFNSampler`` with small test settings.

    The defaults keep rounds tiny so tests stay fast; pass keyword overrides
    for any setting a test actually exercises.
    """

    def factory(**overrides: Any):
        settings: dict[str, Any] = {
            "n_samples": 2,
            "fidelities": [1, 2],
            "max_length": 8,
            "batch_size": 2,
            "n_train_steps": 1,
            "aux_coefficient": 0.0,
        }
        settings.update(overrides)
        return sampler_module.S3GFNSampler(**settings)

    return factory


@pytest.fixture
def make_replay_buffer() -> Callable[..., Any]:
    """Return a factory building small FIFO replay buffers."""

    def factory(capacity: int = 4, policy: str = "fifo", pad_token_id: int = 0):
        return sampler_module.ReplayBuffer(
            pad_token_id=pad_token_id,
            capacity=capacity,
            policy=policy,
        )

    return factory


@pytest.fixture
def patch_molecule_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the optional RDKit and SA-score dependencies with stubs."""
    monkeypatch.setattr(
        sampler_module,
        "require_rdkit",
        lambda: (FakeChem, None, None),
    )
    monkeypatch.setattr(
        sampler_module,
        "SAScoreSynthesizability",
        FakeSynthesizability,
    )

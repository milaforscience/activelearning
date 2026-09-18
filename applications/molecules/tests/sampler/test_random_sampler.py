"""Focused tests for the random molecular sampler."""

from __future__ import annotations

import random

import pytest
from pydantic import TypeAdapter, ValidationError

from activelearning.config_registry import create_config_registry
from activelearning.sampler.config import SamplerConfig
from activelearning_molecules.config_catalogs import CONFIG_CATALOGS
from activelearning_molecules.samplers import random_sampler as sampler_module
from activelearning_molecules.samplers.config import RandomMoleculeSamplerConfig


class _FakeChem:
    """Minimal chemistry stand-in for canonicalization tests."""

    @staticmethod
    def MolFromSmiles(smiles: str) -> str | None:
        """Reject the reserved invalid string."""
        return None if smiles == "invalid" else smiles

    @staticmethod
    def MolToSmiles(
        molecule: str,
        canonical: bool = True,
        isomericSmiles: bool = False,
    ) -> str:
        """Return the fake molecule unchanged."""
        del canonical, isomericSmiles
        return molecule


class _FakeSelfies:
    """Decoder mapping test tokens to SMILES strings."""

    _decoded = {
        "invalid-token": "invalid",
        "duplicate-a": "CC",
        "duplicate-b": "CC",
        "unique": "CO",
    }

    @classmethod
    def decoder(cls, value: str) -> str:
        """Decode a token from the test mapping."""
        return cls._decoded[value]


@pytest.fixture
def patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch optional chemistry and SELFIES dependencies."""
    monkeypatch.setattr(
        sampler_module,
        "require_rdkit",
        lambda: (_FakeChem, None, None),
    )
    monkeypatch.setattr(sampler_module, "_require_selfies", lambda: _FakeSelfies)


def test_random_sampler_returns_canonical_unique_smiles_and_uniform_fidelities(
    patch_dependencies: None,
) -> None:
    """Invalid and duplicate decodes are discarded before fidelity assignment."""
    sampler = sampler_module.RandomMoleculeSampler(
        n_samples=2,
        fidelities=[1, 2, 3],
        min_length=1,
        max_length=1,
        max_generation_attempts=4,
        selfies_vocab=["token"],
        seed=7,
    )
    generated = iter(["invalid-token", "duplicate-a", "duplicate-b", "unique"])
    sampler._sample_selfies = lambda rng: next(generated)

    candidates = sampler.sample()

    assert [candidate.x for candidate in candidates] == ["CC", "CO"]
    assert all(candidate.fidelity in {1, 2, 3} for candidate in candidates)


def test_random_sampler_fails_after_bounded_invalid_duplicate_exhaustion(
    patch_dependencies: None,
) -> None:
    """The sampler reports bounded exhaustion instead of looping forever."""
    sampler = sampler_module.RandomMoleculeSampler(
        n_samples=2,
        fidelities=[1],
        min_length=1,
        max_length=1,
        max_generation_attempts=3,
        selfies_vocab=["token"],
    )
    generated = iter(["invalid-token", "duplicate-a", "duplicate-b"])
    sampler._sample_selfies = lambda rng: next(generated)

    with pytest.raises(RuntimeError, match="max_generation_attempts|bounded"):
        sampler.sample()


def test_random_sampler_samples_lengths_and_tokens_uniformly() -> None:
    """Each random SELFIES draw uses the configured inclusive support."""
    sampler = sampler_module.RandomMoleculeSampler(
        n_samples=1,
        fidelities=[1],
        min_length=2,
        max_length=4,
        selfies_vocab=["[C]", "[O]"],
    )
    rng = random.Random(11)

    draws = [sampler._sample_selfies(rng) for _ in range(100)]

    assert all(2 <= len(draw) // 3 <= 4 for draw in draws)
    assert all(draw.replace("[C]", "").replace("[O]", "") == "" for draw in draws)


def test_random_sampler_seeds_each_round_deterministically(
    patch_dependencies: None,
) -> None:
    """Samplers with the same seed reproduce the same first round."""
    first = sampler_module.RandomMoleculeSampler(
        n_samples=2,
        fidelities=[1, 2],
        min_length=1,
        max_length=1,
        max_generation_attempts=100,
        selfies_vocab=["duplicate-a", "unique"],
        seed=19,
    )
    second = sampler_module.RandomMoleculeSampler(
        n_samples=2,
        fidelities=[1, 2],
        min_length=1,
        max_length=1,
        max_generation_attempts=100,
        selfies_vocab=["duplicate-a", "unique"],
        seed=19,
    )

    first_candidates = first.sample()
    second_candidates = second.sample()

    assert first_candidates == second_candidates
    assert first._round_index == second._round_index == 1


def test_random_sampler_config_is_registered_with_smiles_output() -> None:
    """The application catalog exposes the random sampler contract."""
    registry = create_config_registry({"activelearning-molecules": CONFIG_CATALOGS})
    config = TypeAdapter(SamplerConfig).validate_python(
        {
            "type": "RandomMoleculeSampler",
            "n_samples": 3,
            "fidelities": [1, 2],
            "min_length": 2,
            "max_length": 5,
            "max_generation_attempts": 17,
            "seed": 9,
        },
        context={"config_registry": registry},
    )

    assert isinstance(config, RandomMoleculeSamplerConfig)
    assert config.output_representation == "smiles"
    assert config.max_generation_attempts == 17
    assert isinstance(config.build(), sampler_module.RandomMoleculeSampler)


def test_random_sampler_config_rejects_reversed_length_range() -> None:
    """A maximum SELFIES length below the minimum is invalid."""
    with pytest.raises(ValidationError, match="max_length"):
        RandomMoleculeSamplerConfig(n_samples=1, min_length=5, max_length=4)

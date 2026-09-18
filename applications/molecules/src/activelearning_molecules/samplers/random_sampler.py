"""Uniform random molecular sampler."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation
from activelearning.utils.warnings import warn_ignored_args
from activelearning_molecules.constants import SELFIES_VOCAB_SMALL
from activelearning_molecules._optional import missing_molecules_dependency_error
from activelearning_molecules.samplers.molecule_utils import (
    canonicalize_connected_smiles,
)
from activelearning_molecules.samplers.s3gfn._optional import require_rdkit


def _require_selfies() -> Any:
    """Return the SELFIES module or raise an actionable dependency error."""
    try:
        import selfies as sf
    except ImportError as error:
        raise missing_molecules_dependency_error(
            "RandomMoleculeSampler", error
        ) from error
    return sf


class RandomMoleculeSampler(Sampler):
    """Generate unique canonical molecules from uniformly random SELFIES.

    A SELFIES length is sampled uniformly from the configured inclusive range,
    then each token is sampled uniformly from :data:`SELFIES_VOCAB_SMALL`.
    Decoded strings are canonicalized with RDKit and disconnected, invalid, or
    duplicate molecules are discarded.
    """

    def __init__(
        self,
        n_samples: int,
        fidelities: Sequence[int],
        *,
        min_length: int = 1,
        max_length: int = 64,
        max_generation_attempts: int = 100_000,
        seed: int = 42,
        selfies_vocab: Sequence[str] = SELFIES_VOCAB_SMALL,
        max_attempts: int | None = None,
    ) -> None:
        """Initialize the random molecule sampler.

        Parameters
        ----------
        n_samples : int
            Number of unique candidates returned by each sampling call.
        fidelities : Sequence[int]
            Fidelity levels assigned uniformly to returned candidates.
        min_length : int, optional
            Minimum sampled SELFIES length, inclusive.
        max_length : int, optional
            Maximum sampled SELFIES length, inclusive.
        max_generation_attempts : int, optional
            Maximum number of SELFIES draws allowed per sampling call.
        seed : int, optional
            Base seed. The active-learning round index is added before drawing.
        selfies_vocab : Sequence[str], optional
            SELFIES token vocabulary sampled uniformly.

        Raises
        ------
        ValueError
            If a count, range, seed, vocabulary, or fidelity list is invalid.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if not fidelities:
            raise ValueError("fidelities must contain at least one level.")
        if len(set(fidelities)) != len(fidelities):
            raise ValueError("fidelities must not contain duplicates.")
        if min_length <= 0:
            raise ValueError("min_length must be positive.")
        if max_length < min_length:
            raise ValueError("max_length must be greater than or equal to min_length.")
        if max_attempts is not None:
            if (
                max_generation_attempts != 100_000
                and max_generation_attempts != max_attempts
            ):
                raise ValueError(
                    "max_generation_attempts and max_attempts must agree when "
                    "both are provided."
                )
            max_generation_attempts = max_attempts
        if max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive.")
        if seed < 0:
            raise ValueError("seed must be nonnegative.")
        if not selfies_vocab:
            raise ValueError("selfies_vocab must contain at least one token.")
        if any(not isinstance(token, str) or not token for token in selfies_vocab):
            raise ValueError("selfies_vocab must contain non-empty strings.")

        self.n_samples = int(n_samples)
        self.fidelities = tuple(int(fidelity) for fidelity in fidelities)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.max_generation_attempts = int(max_generation_attempts)
        # Retain the old attribute name for callers that inspect sampler state.
        self.max_attempts = self.max_generation_attempts
        self.seed = int(seed)
        self.selfies_vocab = tuple(selfies_vocab)
        self._round_index = 0

    def sample(
        self,
        acquisition: Any | None = None,
        observations: Iterable[Observation] | None = None,
        cost_fn: Callable[[Sequence[Candidate]], list[float]] | None = None,
    ) -> list[Candidate]:
        """Return unique canonical molecules with uniformly sampled fidelities.

        Parameters
        ----------
        acquisition : Any, optional
            Unused; accepted for sampler interface compatibility.
        observations : Iterable[Observation], optional
            Unused; accepted for sampler interface compatibility.
        cost_fn : callable, optional
            Unused; accepted for sampler interface compatibility.

        Returns
        -------
        list[Candidate]
            Exactly ``n_samples`` unique canonical connected molecules.

        Raises
        ------
        RuntimeError
            If ``max_generation_attempts`` draws do not produce enough unique
            molecules.
        """
        warn_ignored_args(
            self,
            acquisition=acquisition,
            observations=observations,
            cost_fn=cost_fn,
        )

        round_seed = self.seed + self._round_index
        rng = random.Random(round_seed)
        molecule_chem, _, _ = require_rdkit()
        selfies_module = _require_selfies()
        candidates: list[Candidate] = []
        seen_smiles: set[str] = set()

        for _ in range(1, self.max_generation_attempts + 1):
            selfies_string = self._sample_selfies(rng)
            try:
                decoded_smiles = selfies_module.decoder(selfies_string)
            except Exception:
                continue
            canonical = canonicalize_connected_smiles(
                decoded_smiles,
                molecule_chem=molecule_chem,
            )
            if canonical is None or canonical in seen_smiles:
                continue
            seen_smiles.add(canonical)
            candidates.append(Candidate(x=canonical))
            if len(candidates) == self.n_samples:
                break

        if len(candidates) != self.n_samples:
            raise RuntimeError(
                "RandomMoleculeSampler exhausted its bounded generation budget: "
                f"generated {len(candidates)} valid unique molecules after "
                f"{self.max_generation_attempts} attempts; requested {self.n_samples}."
            )

        fidelity_values = self._sample_fidelities(
            random.Random(round_seed),
            count=len(candidates),
        )
        candidates = [
            Candidate(x=candidate.x, fidelity=fidelity)
            for candidate, fidelity in zip(candidates, fidelity_values)
        ]
        self._round_index += 1
        return candidates

    def _sample_selfies(self, rng: random.Random) -> str:
        """Draw one SELFIES string with uniform length and token choices."""
        length = rng.randint(self.min_length, self.max_length)
        return "".join(rng.choice(self.selfies_vocab) for _ in range(length))

    def _sample_fidelities(self, rng: random.Random, *, count: int) -> list[int]:
        """Draw candidate fidelities uniformly from the configured levels."""
        return [rng.choice(self.fidelities) for _ in range(count)]


__all__ = ["RandomMoleculeSampler"]

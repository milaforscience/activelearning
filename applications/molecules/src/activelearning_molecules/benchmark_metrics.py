"""Metrics and molecule normalization for the xTB IP/EA benchmark."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from activelearning_molecules.samplers.molecule_utils import (
    canonicalize_connected_smiles,
)


def canonicalize_smiles(value: object) -> str | None:
    """Return a connected canonical SMILES string, or ``None`` if invalid."""
    if not isinstance(value, str):
        return None
    return canonicalize_connected_smiles(value, molecule_chem=Chem)


def finite_fidelity_three_scores(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Extract finite fidelity-3 scores keyed by canonical molecule.

    Duplicate finite values must agree exactly. This catches corrupted run
    artifacts instead of silently allowing the same molecule to affect a
    checkpoint twice.
    """
    scores: dict[str, float] = {}
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        try:
            fidelity = int(observation.get("fidelity", 1))
            value = float(observation["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if fidelity != 3 or not math.isfinite(value):
            continue
        smiles = canonicalize_smiles(observation.get("x"))
        if smiles is None:
            continue
        previous = scores.get(smiles)
        if previous is not None and previous != value:
            raise ValueError(
                f"Conflicting finite fidelity-3 values for molecule {smiles!r}: "
                f"{previous} and {value}."
            )
        scores[smiles] = value
    return scores


def top_k_score_and_diversity(
    scores: Mapping[str, float],
    *,
    k: int = 100,
) -> dict[str, float | int | None]:
    """Compute mean top-k score and Morgan/Tanimoto diversity.

    Ties are resolved by canonical SMILES to make output deterministic.
    Diversity is ``1 - mean(pairwise Tanimoto similarity)`` and is undefined
    for fewer than two selected molecules.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    normalized_scores: dict[str, float] = {}
    for raw_smiles, raw_score in scores.items():
        smiles = canonicalize_smiles(raw_smiles)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if smiles is None or not math.isfinite(score):
            continue
        previous = normalized_scores.get(smiles)
        if previous is not None and previous != score:
            raise ValueError(
                f"Conflicting scores for molecule {smiles!r}: {previous} and {score}."
            )
        normalized_scores[smiles] = score

    finite_scores = list(normalized_scores.items())
    top_scores = sorted(finite_scores, key=lambda item: (-item[1], item[0]))[:k]
    count = len(top_scores)
    if count == 0:
        return {
            "mean_score": None,
            "diversity": None,
            "top_k_count": 0,
        }

    fingerprints = _fingerprints([smiles for smiles, _ in top_scores])
    diversity: float | None
    if len(fingerprints) < 2:
        diversity = None
    else:
        similarities = [
            DataStructs.TanimotoSimilarity(fingerprints[index], fingerprints[other])
            for index in range(len(fingerprints))
            for other in range(index)
        ]
        diversity = 1.0 - sum(similarities) / len(similarities)

    return {
        "mean_score": sum(score for _, score in top_scores) / count,
        "diversity": diversity,
        "top_k_count": count,
    }


def _fingerprints(smiles: Sequence[str]) -> list[Any]:
    """Build radius-2, 2048-bit Morgan fingerprints."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints: list[Any] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Cannot fingerprint invalid SMILES {value!r}.")
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


__all__ = [
    "canonicalize_smiles",
    "finite_fidelity_three_scores",
    "top_k_score_and_diversity",
]

"""Utilities shared by molecular samplers."""

from __future__ import annotations

from typing import Any


def canonicalize_connected_smiles(
    smiles: str,
    *,
    molecule_chem: Any,
) -> str | None:
    """Return a canonical connected SMILES string or ``None``.

    Parameters
    ----------
    smiles : str
        Candidate SMILES text.
    molecule_chem : Any
        RDKit-compatible ``Chem`` module used for parsing and canonicalizing.

    Returns
    -------
    str or None
        Canonical non-empty connected SMILES, or ``None`` when the input is
        invalid, empty, or contains multiple disconnected components.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        molecule = molecule_chem.MolFromSmiles(smiles.strip())
    except (TypeError, ValueError, RuntimeError):
        return None
    if molecule is None:
        return None

    try:
        canonical = molecule_chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=False,
        )
    except (TypeError, ValueError, RuntimeError):
        return None
    if not isinstance(canonical, str) or not canonical or "." in canonical:
        return None
    return canonical


__all__ = ["canonicalize_connected_smiles"]

"""Molecule visualization helpers for xTB oracle queries."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import selfies as sf
from matplotlib.figure import Figure
from rdkit import Chem, rdBase
from rdkit.Chem import Draw

from activelearning.utils.types import Candidate, Observation


def build_xtb_query_molecule_figure(
    candidates: Sequence[Candidate],
    observations: Sequence[Observation],
    *,
    task: str,
    mol_repr: str,
    limit: int = 25,
    mols_per_row: int = 5,
    sub_img_size: tuple[int, int] = (260, 220),
) -> Figure:
    """Render queried xTB molecules as a capped annotated RDKit 2-D grid.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        Queried candidates, in the same order as ``observations``.
    observations : Sequence[Observation]
        Oracle observations produced for ``candidates``.
    task : str
        ``"ea"`` or ``"ip"``; used in panel labels and the figure title.
    mol_repr : str
        Input molecular representation, either ``"selfies"`` or ``"smiles"``.
    limit : int, optional
        Maximum number of molecules to display. Displayed records are ranked by
        observed score descending, and only the top ``limit`` entries are kept
        when a query exceeds this cap. Defaults to 25.
    mols_per_row : int, optional
        Number of molecules per RDKit grid row. Defaults to 5.
    sub_img_size : tuple[int, int], optional
        Width and height, in pixels, for each RDKit grid cell.

    Returns
    -------
    Figure
        A Matplotlib figure ready for the repository logger backends.
    """
    if len(candidates) != len(observations):
        raise ValueError(
            "Molecule visualization expects matching candidates and observations."
        )
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if mols_per_row < 1:
        raise ValueError("mols_per_row must be at least 1.")

    indexed_records = list(enumerate(zip(candidates, observations)))
    if not indexed_records:
        return _build_empty_query_figure(task)

    ranked_records = sorted(
        indexed_records,
        key=lambda item: _sort_score(item[1][1].y),
        reverse=True,
    )
    selected_records = ranked_records[:limit]

    molecules: list[Chem.Mol] = []
    legends: list[str] = []
    for original_index, (candidate, observation) in selected_records:
        molecule, legend = _build_molecule_panel(
            candidate=candidate,
            observation=observation,
            original_index=original_index,
            task=task,
            mol_repr=mol_repr,
        )
        molecules.append(molecule)
        legends.append(legend)

    grid_image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        useSVG=False,
    )
    width_px, height_px = grid_image.size
    figure, axis = plt.subplots(
        figsize=(max(6.0, width_px / 140.0), max(3.0, height_px / 140.0 + 0.5))
    )
    axis.imshow(grid_image)
    axis.axis("off")
    axis.set_title(_figure_title(task, len(selected_records), len(indexed_records)))
    figure.tight_layout()
    return figure


def _build_empty_query_figure(task: str) -> Figure:
    """Return a small placeholder figure for an empty query batch."""
    figure, axis = plt.subplots(figsize=(6.0, 2.0))
    axis.text(
        0.5,
        0.5,
        f"No xTB {task.upper()} molecules queried.",
        ha="center",
        va="center",
    )
    axis.axis("off")
    figure.tight_layout()
    return figure


def _build_molecule_panel(
    *,
    candidate: Candidate,
    observation: Observation,
    original_index: int,
    task: str,
    mol_repr: str,
) -> tuple[Chem.Mol, str]:
    """Build a drawable RDKit molecule and legend for one query record."""
    raw_molecule = _extract_raw_molecule(candidate, observation)
    smiles, invalid_reason = _decode_molecule(raw_molecule, mol_repr)
    molecule = _mol_from_smiles(smiles) if smiles is not None else None
    if molecule is None:
        molecule = Chem.Mol()
        identifier = f"invalid: {invalid_reason or 'invalid SMILES'}"
    else:
        identifier = smiles

    legend = _build_legend(
        original_index=original_index,
        observation=observation,
        task=task,
        identifier=identifier,
    )
    return molecule, legend


def _extract_raw_molecule(candidate: Candidate, observation: Observation) -> str:
    """Extract the raw molecule string carried by a candidate/observation pair."""
    if isinstance(candidate.x, str):
        return candidate.x
    if candidate.metadata is not None and "raw" in candidate.metadata:
        return str(candidate.metadata["raw"])
    if isinstance(observation.x, str):
        return observation.x
    if observation.metadata is not None and "raw" in observation.metadata:
        return str(observation.metadata["raw"])
    return str(candidate.x)


def _decode_molecule(molecule: str, mol_repr: str) -> tuple[str | None, str | None]:
    """Decode a molecule to SMILES, returning an invalid reason on failure."""
    if mol_repr == "smiles":
        return molecule, None
    if mol_repr != "selfies":
        return None, f"unsupported representation {mol_repr!r}"

    try:
        smiles = sf.decoder(molecule)
    except sf.DecoderError as exc:
        return None, str(exc)
    if smiles == "":
        return None, "empty molecule"
    return smiles, None


def _mol_from_smiles(smiles: str | None) -> Chem.Mol | None:
    """Build an RDKit molecule from SMILES without emitting RDKit diagnostics."""
    if smiles is None:
        return None
    try:
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(smiles)
    except (TypeError, ValueError):
        return None


def _build_legend(
    *,
    original_index: int,
    observation: Observation,
    task: str,
    identifier: str,
) -> str:
    """Format the molecule panel legend."""
    fidelity = observation.fidelity if observation.fidelity is not None else "none"
    header = (
        f"#{original_index + 1} {task.upper()}@fid={fidelity}: "
        f"{_format_score(observation.y)}"
    )
    return f"{header}\n{_shorten_identifier(identifier)}"


def _shorten_identifier(identifier: str, width: int = 42) -> str:
    """Return a compact molecule identifier suitable for RDKit grid legends."""
    return textwrap.shorten(identifier, width=width, placeholder="...")


def _format_score(score: Any) -> str:
    """Format an oracle score as eV when numeric."""
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        return str(score)
    if math.isnan(score_value):
        return "nan"
    if math.isinf(score_value):
        return str(score_value)
    return f"{score_value:.3f} eV"


def _sort_score(score: Any) -> float:
    """Return a finite score for ranking, pushing invalid scores to the end."""
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        return -math.inf
    if not math.isfinite(score_value):
        return -math.inf
    return score_value


def _figure_title(task: str, displayed: int, total: int) -> str:
    """Build the figure title, including cap information when relevant."""
    title = f"xTB {task.upper()} queried molecules"
    if displayed != total:
        title = f"{title} (top {displayed} of {total})"
    return title

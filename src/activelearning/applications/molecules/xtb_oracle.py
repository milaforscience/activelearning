"""Subprocess-based multi-fidelity oracle for ionisation potential / electron affinity.

Uses the ``xtb`` binary (GFN2-xTB) via subprocess, exactly as described in the
MF-GFN paper.  The binary must be available in the system PATH.

Fidelity levels
---------------
1 → RDKit/MMFF geometry + ``xtb --gfn 2 --vip``/``--vea`` (vertical, no geometry opt)
2 → RDKit/MMFF geometry + ``xtb --gfn 2 --opt`` (neutral opt) + vertical IP/EA
3 → RDKit/MMFF + neutral xtb opt + ionic xtb opt → adiabatic IP/EA from
    ``TOTAL ENERGY`` fields in the optimisation logs

Notes
-----
Install xtb: ``conda install -c conda-forge xtb``
or download the standalone binary from https://github.com/grimme-lab/xtb/releases.

Dependencies
------------
uv sync --extra molecules   # includes selfies, rdkit
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt

from activelearning.applications.molecules._optional import (
    missing_molecules_dependency_error,
)

try:
    import selfies as sf
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem
except ImportError as error:  # pragma: no cover - exercised via subprocess test
    raise missing_molecules_dependency_error("XTBIPEAOracle", error) from error

from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.utils.types import Candidate, Observation
from activelearning.applications.molecules.plotting import (
    build_xtb_query_molecule_figure,
)

# xTB reports energies in atomic units (Hartree), but ionization potential and
# electron affinity are conventionally expressed in eV. This factor is the
# CODATA 2018 internationally recommended value of one Hartree in eV
# (27.211386245… eV), truncated to six significant figures.
_HARTREE_TO_EV: float = 27.2114

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers (SELFIES/SMILES → XYZ file)
# ---------------------------------------------------------------------------


@dataclass
class ConformerConfig:
    """RDKit conformer generation settings.

    Parameters
    ----------
    num_conformers : int
        Default number of conformers to generate before selecting the
        lowest-energy starting geometry.
    max_attempts : int
        Maximum embedding attempts.
    random_coords : bool
        Use random coordinates as starting points.
    prune_rms_thresh : float
        RMSD threshold below which duplicate conformers are pruned.
    """

    num_conformers: int = 2
    max_attempts: int = 100
    random_coords: bool = True
    prune_rms_thresh: float = 1.5

    def __post_init__(self) -> None:
        if self.num_conformers < 1:
            raise ValueError("num_conformers must be at least 1.")


def hartree_to_ev(hartree: float) -> float:
    """Convert energy from Hartree to electronvolt."""
    return hartree * _HARTREE_TO_EV


def _decode_to_smiles(molecule: str, mol_repr: str = "selfies") -> str:
    """Decode a molecules string to SMILES.

    Parameters
    ----------
    molecule : str
        Molecule in the specified representation.
    mol_repr : str
        ``"selfies"`` or ``"smiles"``.

    Returns
    -------
    str
        SMILES string. May be empty when a syntactically valid SELFIES string
        collapses to the empty molecule during decoding.
    """
    if mol_repr == "selfies":
        smiles = sf.decoder(molecule)
    elif mol_repr == "smiles":
        smiles = molecule
    else:
        raise ValueError(f"Unsupported molecular representation: {mol_repr!r}")
    return smiles


def _write_best_rdkit_xyz(
    smiles: str,
    xyz_path: Path,
    conformer_cfg: ConformerConfig,
    ff: str = "mmff",
) -> Path:
    """Generate RDKit conformers, optimise with a force field, and write the
    lowest-energy conformer to an XYZ file.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecules.
    xyz_path : Path
        Destination path for the XYZ file.
    conformer_cfg : ConformerConfig
        Settings for RDKit conformer generation.
    ff : str
        Force field: ``"mmff"`` or ``"uff"``.

    Returns
    -------
    Path
        The written XYZ file path (same as ``xyz_path``).
    """
    # RDKit emits atom-typing and sanitization diagnostics directly to stderr.
    # Suppress those here and surface compact Python exceptions instead.
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        Chem.SanitizeMol(mol)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol_h,
            numConfs=conformer_cfg.num_conformers,
            pruneRmsThresh=conformer_cfg.prune_rms_thresh,
            maxAttempts=conformer_cfg.max_attempts,
            useRandomCoords=conformer_cfg.random_coords,
        )

    num_conformers_generated = mol_h.GetNumConformers()
    if num_conformers_generated == 0:
        raise RuntimeError(f"RDKit failed to generate conformers for {smiles!r}")

    ff_name = ff.lower()
    energies: List[float] = []
    if ff_name == "mmff":
        with rdBase.BlockLogs():
            has_params = AllChem.MMFFHasAllMoleculeParams(mol_h)
            mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if not has_params or mp is None:
            raise RuntimeError(f"MMFF parameters unavailable for {smiles!r}")
        for conf_id in range(num_conformers_generated):
            with rdBase.BlockLogs():
                AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=1000)
                force_field = AllChem.MMFFGetMoleculeForceField(
                    mol_h, mp, confId=conf_id
                )
            if force_field is None:
                raise RuntimeError(
                    f"MMFF force field construction failed for {smiles!r}"
                )
            energies.append(force_field.CalcEnergy())
    elif ff_name == "uff":
        with rdBase.BlockLogs():
            has_params = AllChem.UFFHasAllMoleculeParams(mol_h)
        if not has_params:
            raise RuntimeError(f"UFF parameters unavailable for {smiles!r}")
        for conf_id in range(num_conformers_generated):
            with rdBase.BlockLogs():
                AllChem.UFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=1000)
                force_field = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)
            if force_field is None:
                raise RuntimeError(
                    f"UFF force field construction failed for {smiles!r}"
                )
            energies.append(force_field.CalcEnergy())
    else:
        raise ValueError(f"Unsupported force field: {ff!r}")

    best_conf = int(min(range(len(energies)), key=lambda i: energies[i]))
    Chem.MolToXYZFile(mol_h, str(xyz_path), confId=best_conf)
    return xyz_path


# ---------------------------------------------------------------------------
# xtb subprocess helpers
# ---------------------------------------------------------------------------


def _run_xtb(
    xyz_path: Path,
    args: Sequence[str],
    output_path: Path,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run ``xtb <xyz_path> <args...>`` and capture output to a file."""
    command = ["xtb", str(xyz_path), *args]
    try:
        with open(output_path, "w") as fh:
            result = subprocess.run(
                command,
                cwd=str(cwd) if cwd is not None else None,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
    except FileNotFoundError as error:
        raise RuntimeError(
            "xtb executable not found. Install xtb and ensure it is available on PATH."
        ) from error

    if result.returncode != 0:
        output_tail = output_path.read_text(encoding="utf-8", errors="replace")[-500:]
        raise RuntimeError(
            f"xTB command failed with return code {result.returncode}: "
            f"{' '.join(command)}\nOutput tail: {output_tail!r}"
        )
    return result


def _run_xtb_optimize(
    xyz_path: Path,
    gfn_version: int,
    charge: Optional[int] = None,
) -> tuple[Path, Path]:
    """Run ``xtb --opt`` geometry optimisation in ``xyz_path.parent``.

    xtb writes ``xtbopt.xyz`` (converged) or ``xtblast.xyz`` (not converged)
    to the working directory.  This function renames whichever exists to a
    stable name and returns ``(optimised_xyz, log_path)``.

    Returns
    -------
    optimised_xyz : Path
        Path to the optimised XYZ geometry.
    log_path : Path
        Path to the xtb output log.
    """
    workdir = xyz_path.parent
    log_path = workdir / f"{xyz_path.stem}_opt.out"

    args: List[str] = ["--gfn", str(gfn_version), "--opt"]
    if charge is not None:
        args.extend(["--chrg", str(charge)])

    _run_xtb(xyz_path=xyz_path, args=args, output_path=log_path, cwd=workdir)

    for candidate_name in ("xtbopt.xyz", "xtblast.xyz"):
        candidate = workdir / candidate_name
        if candidate.exists():
            final_xyz = workdir / f"{xyz_path.stem}_xtbopt.xyz"
            shutil.move(str(candidate), str(final_xyz))
            return final_xyz, log_path

    # Optimisation produced no new geometry file — fall back to input
    return xyz_path, log_path


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


def _parse_vertical_ipea(output_text: str, task: str) -> float:
    """Extract the vertical IP or EA from xtb ``--vip``/``--vea`` output.

    Looks for lines of the form::

        delta SCC EA (eV):    2.3456
        delta SCC IP (eV):    9.8765

    Parameters
    ----------
    output_text : str
        Full text of the xtb output file.
    task : str
        ``"ea"`` or ``"ip"``.

    Returns
    -------
    float
        The parsed value in eV.
    """
    numeric_pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    if task == "ea":
        pattern = rf"delta SCC EA \(eV\):\s*{numeric_pattern}"
    elif task == "ip":
        pattern = rf"delta SCC IP \(eV\):\s*{numeric_pattern}"
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    match = re.search(pattern, output_text)
    if match is None:
        raise RuntimeError(
            f"Could not parse XTB {task.upper()} from output.\n"
            f"Output snippet: {output_text[-500:]!r}"
        )
    return float(match.group(1))


def _parse_total_energy(output_text: str) -> float:
    """Extract the last ``TOTAL ENERGY`` value (Hartree) from xtb output.

    xtb prints multiple ``TOTAL ENERGY`` lines during an optimisation; the
    last one corresponds to the final geometry.

    Parameters
    ----------
    output_text : str
        Full text of the xtb output file.

    Returns
    -------
    float
        Energy in Hartree.
    """
    matches = re.findall(
        r"TOTAL ENERGY\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        output_text,
    )
    if not matches:
        raise RuntimeError(
            f"Could not find TOTAL ENERGY in XTB output.\n"
            f"Output snippet: {output_text[-500:]!r}"
        )
    return float(matches[-1])


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


class XTBIPEAOracle(MultiFidelityOracle):
    """Multi-fidelity oracle that evaluates molecular IP/EA using the xtb binary.

    Implements the three-level fidelity ladder from the MF-GFN paper using
    ``xtb --gfn 2`` via subprocess:

    Fidelity 1
        RDKit/MMFF geometry → ``xtb --gfn 2 --vip``/``--vea`` (vertical, no opt)
    Fidelity 2
        RDKit/MMFF geometry → ``xtb --gfn 2 --opt`` (neutral opt) → vertical IP/EA
    Fidelity 3
        Fidelity-2 neutral geometry → ionic opt → adiabatic IP/EA from
        ``TOTAL ENERGY`` differences in the optimisation logs, minus
        the empirical ``correction_factor``.

    Handles both the *DKL path* (``candidate.x`` is a SELFIES string) and the
    *pre-embed path* (``candidate.x`` is a tensor, original string in
    ``candidate.metadata["raw"]``).

    Parameters
    ----------
    task : str
        ``"ea"`` (electron affinity) or ``"ip"`` (ionisation potential).
    fidelity_costs : dict[int, float]
        Computational cost per sample for each fidelity level.
    fidelity_confidences : dict[int, float], optional
        Confidence value in ``[0, 1]``.  Defaults to costs normalised by max.
    gfn_version : int
        GFN-xTB parametrisation passed to ``--gfn`` (default: 2).
    ff : str
        RDKit force field for initial 3-D geometry: ``"mmff"`` or ``"uff"``.
    correction_factor : float
        Empirical correction subtracted from adiabatic IP/EA (eV).
        Default 4.8455 matches the MF-GFN paper (GFN2-xTB).
    conformer_cfg : ConformerConfig, optional
        Global/default RDKit conformer generation settings.
    per_fidelity_num_conformers : dict[int, int], optional
        Optional per-fidelity override for ``conformer_cfg.num_conformers``.
        Unlisted fidelities fall back to the global/default setting.
    mol_repr : str
        Input molecules representation: ``"selfies"`` or ``"smiles"``.
    negate_score : bool, optional
        Whether to negate the raw EA/IP value before returning it as the active-
        learning objective. This is useful for lower-is-better properties such
        as IP when the acquisition loop is formulated as a maximization problem.
        Defaults to ``True`` for IP and ``False`` for EA when omitted.
    log_molecule_visualizations : bool
        Whether to log a 2-D RDKit grid of queried molecules when a runtime
        logger is bound.
    molecule_visualization_limit : int
        Maximum number of queried molecules to include in each logged grid.
        Large query batches are ranked by score and capped to this limit.
    Notes
    -----
    SELFIES strings that decode to the empty molecule and molecules that fail
    during RDKit/MMFF or xtb processing return ``NaN``. The dataset layer
    filters these failed evaluations before surrogate fitting, matching the
    original MF-GFN behavior where geometry-construction failures were treated
    as invalid molecules.
    """

    def __init__(
        self,
        task: str,
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]] = None,
        gfn_version: int = 2,
        ff: str = "mmff",
        correction_factor: float = 4.8455,
        conformer_cfg: Optional[ConformerConfig] = None,
        per_fidelity_num_conformers: Optional[dict[int, int]] = None,
        mol_repr: str = "selfies",
        negate_score: bool | None = None,
        log_molecule_visualizations: bool = False,
        molecule_visualization_limit: int = 25,
    ) -> None:
        if task not in {"ea", "ip"}:
            raise ValueError(f"task must be 'ea' or 'ip', got {task!r}")
        if not fidelity_costs:
            raise ValueError("fidelity_costs must define at least one fidelity.")
        if molecule_visualization_limit < 1:
            raise ValueError("molecule_visualization_limit must be at least 1.")

        self._task = task
        self._gfn_version = gfn_version
        self._ff = ff
        self._correction_factor = correction_factor
        self._conformer_cfg = conformer_cfg or ConformerConfig()
        self._per_fidelity_num_conformers = self._validate_per_fidelity_num_conformers(
            per_fidelity_num_conformers, fidelity_costs
        )
        self._mol_repr = mol_repr
        self._negate_score = task == "ip" if negate_score is None else negate_score
        self.log_molecule_visualizations = log_molecule_visualizations
        self._molecule_visualization_limit = molecule_visualization_limit

        confidences = self._resolve_fidelity_confidences(
            fidelity_costs, fidelity_confidences
        )

        fidelity_configs: dict[int, dict[str, Any]] = {
            fid: {
                "cost_per_sample": fidelity_costs[fid],
                "fidelity_confidence": confidences[fid],
                "score_fn": lambda mol_str, _fid=fid: self._objective_score(
                    mol_str, _fid
                ),
            }
            for fid in fidelity_costs
        }
        super().__init__(fidelity_configs)

    # ------------------------------------------------------------------
    # MultiFidelityOracle override
    # ------------------------------------------------------------------

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        """Evaluate candidates with xtb at the fidelity specified per-candidate."""
        observations: list[Observation] = []
        for candidate in candidates:
            fidelity = self._validate_candidate_fidelity(
                candidate, self.fidelity_configs
            )
            mol_str = self._extract_molecule_string(candidate)
            score = self._objective_score(mol_str, fidelity)
            observations.append(
                Observation(
                    x=candidate.x,
                    y=score,
                    fidelity=fidelity,
                    metadata=candidate.metadata,
                )
            )
        if self.log_molecule_visualizations:
            self._log_query_molecule_visualizations(candidates, observations)
        return observations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_molecule_string(self, candidate: Candidate) -> str:
        """Return the molecules string from a candidate."""
        if isinstance(candidate.x, str):
            return candidate.x
        if candidate.metadata is not None and "raw" in candidate.metadata:
            return str(candidate.metadata["raw"])
        raise ValueError(
            "Cannot extract molecules string: candidate.x is not a string and "
            "candidate.metadata does not contain a 'raw' key."
        )

    def _log_query_molecule_visualizations(
        self,
        candidates: Sequence[Candidate],
        observations: Sequence[Observation],
    ) -> None:
        """Log a capped RDKit grid of queried molecules, if a logger is bound."""
        if self.logger is None:
            return

        figure = build_xtb_query_molecule_figure(
            candidates=candidates,
            observations=observations,
            task=self._task,
            mol_repr=self._mol_repr,
            limit=self._molecule_visualization_limit,
        )
        try:
            self.logger.log_figure(f"xtb_{self._task}_query_molecules", figure)
        finally:
            plt.close(figure)

    def _xtb_score(self, molecule: str, fidelity: int) -> float:
        """Evaluate a single molecules at the given fidelity level.

        Follows the MF-GFN paper fidelity ladder:
        1 → vertical score on MMFF geometry
        2 → vertical score on xtb-optimised neutral geometry
        3 → adiabatic score (neutral opt + ionic opt)

        Parameters
        ----------
        molecule : str
            Molecule in ``self._mol_repr`` format.
        fidelity : int
            1, 2, or 3.

        Returns
        -------
        float
            IP or EA score in eV.
        """
        if fidelity not in {1, 2, 3}:
            raise ValueError(f"fidelity must be 1, 2, or 3, got {fidelity!r}")

        try:
            smiles = _decode_to_smiles(molecule, mol_repr=self._mol_repr)
            if smiles == "":
                logger.warning(
                    "Returning NaN for molecule %r at fidelity %d: decoded to the "
                    "empty molecule.",
                    molecule,
                    fidelity,
                )
                return float("nan")

            with TemporaryDirectory(prefix="xtb_mol_") as tmp:
                workdir = Path(tmp)
                conformer_cfg = self._conformer_cfg_for_fidelity(fidelity)
                neutral_xyz = _write_best_rdkit_xyz(
                    smiles=smiles,
                    xyz_path=workdir / "neutral_mmff.xyz",
                    conformer_cfg=conformer_cfg,
                    ff=self._ff,
                )

                if fidelity == 1:
                    return self._vertical_score(neutral_xyz)

                # Fidelity 2+: optimise the neutral geometry with xtb
                neutral_xtb_xyz, neutral_log = _run_xtb_optimize(
                    neutral_xyz, gfn_version=self._gfn_version
                )
                if fidelity == 2:
                    return self._vertical_score(neutral_xtb_xyz)

                # Fidelity 3: also optimise the ionic geometry
                ionic_charge = -1 if self._task == "ea" else 1
                ionic_xtb_xyz, ionic_log = _run_xtb_optimize(
                    neutral_xtb_xyz,
                    gfn_version=self._gfn_version,
                    charge=ionic_charge,
                )
                return self._adiabatic_score(neutral_log, ionic_log)
        except Exception as exc:
            logger.warning(
                "Returning NaN for molecule %r at fidelity %d: %s.",
                molecule,
                fidelity,
                exc,
            )
            logger.debug(
                "Detailed oracle failure for molecule %r at fidelity %d.",
                molecule,
                fidelity,
                exc_info=True,
            )
            return float("nan")

    def _objective_score(self, molecule: str, fidelity: int) -> float:
        """Return the active-learning objective value for one molecule query."""

        raw_score = self._xtb_score(molecule, fidelity)
        return -raw_score if self._negate_score else raw_score

    @staticmethod
    def _validate_per_fidelity_num_conformers(
        per_fidelity_num_conformers: Optional[dict[int, int]],
        fidelity_costs: dict[int, float],
    ) -> dict[int, int]:
        if per_fidelity_num_conformers is None:
            return {}
        unknown_fidelities = sorted(
            set(per_fidelity_num_conformers) - set(fidelity_costs)
        )
        if unknown_fidelities:
            raise ValueError(
                "per_fidelity_num_conformers contains unsupported fidelities: "
                f"{unknown_fidelities}."
            )
        invalid_counts = {
            fidelity: count
            for fidelity, count in per_fidelity_num_conformers.items()
            if count < 1
        }
        if invalid_counts:
            raise ValueError(
                "per_fidelity_num_conformers must contain positive integers. "
                f"Got: {invalid_counts}."
            )
        return dict(per_fidelity_num_conformers)

    @staticmethod
    def _resolve_fidelity_confidences(
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]],
    ) -> dict[int, float]:
        if fidelity_confidences is None:
            max_cost = max(fidelity_costs.values())
            return {fid: cost / max_cost for fid, cost in fidelity_costs.items()}

        missing_fidelities = sorted(set(fidelity_costs) - set(fidelity_confidences))
        extra_fidelities = sorted(set(fidelity_confidences) - set(fidelity_costs))
        if missing_fidelities or extra_fidelities:
            message_parts = []
            if missing_fidelities:
                message_parts.append(f"missing keys {missing_fidelities}")
            if extra_fidelities:
                message_parts.append(f"unexpected keys {extra_fidelities}")
            raise ValueError(
                "fidelity_confidences must have exactly the same fidelity keys as "
                f"fidelity_costs; got {' and '.join(message_parts)}."
            )
        return dict(fidelity_confidences)

    def _conformer_cfg_for_fidelity(self, fidelity: int) -> ConformerConfig:
        num_conformers = self._per_fidelity_num_conformers.get(fidelity)
        if num_conformers is None:
            return self._conformer_cfg
        return replace(self._conformer_cfg, num_conformers=num_conformers)

    def _vertical_score(self, xyz_path: Path) -> float:
        """Run ``xtb --vip``/``--vea`` and parse the result.

        Parameters
        ----------
        xyz_path : Path
            Path to the XYZ geometry file (in ``xyz_path.parent``).

        Returns
        -------
        float
            Vertical IP or EA in eV.
        """
        output_path = xyz_path.parent / f"{xyz_path.stem}_ipea.out"
        flag = "--vea" if self._task == "ea" else "--vip"
        _run_xtb(
            xyz_path=xyz_path,
            args=["--gfn", str(self._gfn_version), flag],
            output_path=output_path,
            cwd=xyz_path.parent,
        )
        return _parse_vertical_ipea(output_path.read_text(), task=self._task)

    def _adiabatic_score(self, neutral_log: Path, ionic_log: Path) -> float:
        """Compute adiabatic IP/EA from xtb geometry-optimisation logs.

        Extracts the final ``TOTAL ENERGY`` from each log, converts to eV,
        and subtracts the empirical correction factor.

        Parameters
        ----------
        neutral_log : Path
            Log from the neutral-geometry xtb optimisation.
        ionic_log : Path
            Log from the ionic-geometry xtb optimisation.

        Returns
        -------
        float
            Adiabatic IP or EA in eV, after correction.
        """
        neutral_energy = _parse_total_energy(neutral_log.read_text())
        ionic_energy = _parse_total_energy(ionic_log.read_text())
        if self._task == "ip":
            return (
                hartree_to_ev(ionic_energy - neutral_energy) - self._correction_factor
            )
        # EA: anion lower in energy → EA = E(neutral) - E(anion) > 0
        return hartree_to_ev(neutral_energy - ionic_energy) - self._correction_factor

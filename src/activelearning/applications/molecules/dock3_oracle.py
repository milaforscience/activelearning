"""DOCK3 molecular docking oracle.

Scores SMILES strings against a prepared receptor by running the DOCK3
toolchain (``ligbuild`` to build a ``.db2`` ligand bundle, then ``dock64`` to
dock it) and parsing the best pose energy out of ``OUTDOCK``.

The pipeline is ported from the standalone ``dock_smiles`` project and keeps
several non-obvious behaviours that are load bearing; see the individual
function docstrings before changing anything here.

Unlike :mod:`activelearning.applications.molecules.xtb_oracle`, this module
depends only on the standard library: the chemistry happens in external
binaries, so no optional ``molecules`` extra is required to import it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.utils.types import Candidate, Observation

logger = logging.getLogger(__name__)

DEFAULT_DOCKENV_SH = "/project/rrg-mailhoto/share/dockingpackages/dockenv.sh"
DEFAULT_DOCK64_EXE = "/project/rrg-mailhoto/share/dock64"

# dock64 at DEFAULT_DOCK64_EXE is DOCK 3.8 and rejects the 3.7 header that the
# shipped INDOCK templates are written with.
_INDOCK_VER_OLD = "DOCK 3.7 parameter"
_INDOCK_VER_NEW = "DOCK 3.8 parameter"
_INDOCK_LIGAND_PLACEHOLDER = "split_database_index"

# OUTDOCK format signals. The leading whitespace is part of the match.
_OUTDOCK_HEADER = "  mol#           id_num"
_OUTDOCK_END = "  we reached the end of the"
_OUTDOCK_FILE_MARKERS = (" close the file:", " open the file:")

# Longest tmp_dir prefix that still leaves AMSOL room to work; see
# _short_workdir_base for the rationale.
_MAX_SAFE_TMP_DIR_CHARS = 30


def _short_workdir_base() -> Path:
    """Return a short directory to use as the base for per-call workdirs.

    AMSOL (called by ligbuild) has a fixed-width Fortran path buffer of about
    80 characters. Once the per-protomer, per-tautomer and per-conformer
    subdirectories are appended underneath the workdir, the workdir prefix must
    stay around 25 characters or less. Beyond that AMSOL silently corrupts the
    build and leaves a degraded ``.db2``, producing a much weaker score (-16
    instead of -71 for the same molecule) with no error anywhere. Under
    ``$SLURM_TMPDIR`` (``/localscratch/<user>.<JOBID>.0/``, roughly 30
    characters) that threshold is already breached for many ligands.

    To keep the fast node-local NVMe scratch *and* a short path, expose
    ``$SLURM_TMPDIR`` through a short symlink at ``/tmp/d.<JOB_TAG>``. The
    alias is per-Slurm-job so concurrent jobs on a node never fight over the
    symlink target.

    Returns
    -------
    Path
        Directory under which per-call workdirs should be created.
    """
    job_tag = os.environ.get("SLURM_JOB_ID") or os.environ.get("RUN_ID") or "cli"
    alias = Path(f"/tmp/d.{job_tag}")
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if not slurm_tmpdir:
        alias.mkdir(parents=True, exist_ok=True)
        return alias

    target = Path(slurm_tmpdir)
    try:
        if alias.is_symlink():
            if alias.resolve() != target.resolve():
                alias.unlink()
                alias.symlink_to(target, target_is_directory=True)
        elif not alias.exists():
            alias.symlink_to(target, target_is_directory=True)
    except OSError:
        # /tmp not writable, or a race with a sibling process. Fall back to a
        # plain directory; AMSOL still gets a short prefix, we just lose the
        # NVMe scratch.
        alias.mkdir(parents=True, exist_ok=True)
    return alias


def _warmup_dockenv(
    *,
    dockenv_sh: str,
    ligbuild_exe: str,
    timeout: int = 600,
) -> None:
    """Source ``dockenv.sh`` once and probe that ligbuild and openeye load.

    On a fresh compute node the first ``source dockenv.sh`` triggers a ``pip
    install`` of ``build_3d_dock_py`` and its OpenEye/Pyro4 dependencies into a
    node-local venv. Running this serially before any parallel ligbuild call
    eliminates the concurrent-pip-install race that otherwise corrupts the venv
    and makes every subsequent ligbuild fail.

    Best effort by design: failures are logged, never raised, so nodes with a
    healthy pre-built environment still proceed and the per-call ligbuild
    surfaces the real error if there is one.

    Parameters
    ----------
    dockenv_sh : str
        Path to the ``dockenv.sh`` environment bootstrap script.
    ligbuild_exe : str
        Name or path of the ligbuild executable to probe for.
    timeout : int, default=600
        Seconds to wait for the probe before giving up.
    """
    probe_cmd = (
        f'source "{dockenv_sh}" && '
        f'command -v "{ligbuild_exe}" >/dev/null && '
        'python -c "import openeye, Pyro4" 2>&1'
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", probe_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "dockenv warmup timed out after %ds (dockenv_sh=%s). Continuing; "
            "per-call ligbuild may fail if the venv is not provisioned.",
            timeout,
            dockenv_sh,
        )
        return
    except Exception as error:
        logger.warning("dockenv warmup raised %s: %s", type(error).__name__, error)
        return

    if result.returncode == 0:
        logger.info("dockenv warmup OK (dockenv_sh=%s)", dockenv_sh)
        return

    tail = ((result.stdout or "") + (result.stderr or ""))[-1500:]
    logger.warning(
        "dockenv warmup probe exited rc=%d. Continuing - per-call ligbuild "
        "will retry. Last output:\n%s",
        result.returncode,
        tail,
    )


def _run_ligbuild(
    smi_file: Path,
    out_dir: Path,
    *,
    dockenv_sh: str,
    ligbuild_exe: str,
    timeout: int,
    ligbuild_timeout: int,
) -> Path:
    """Run ligbuild on ``smi_file`` and return the ``.tgz`` bundle it produced.

    Parameters
    ----------
    smi_file : Path
        One-line ``.smi`` file holding ``"<smiles> <name>"``.
    out_dir : Path
        Directory ligbuild writes its bundle into. Its parent is used as the
        working directory so ``db2_outputs/`` and ``db2_archives/`` stay
        isolated per call.
    dockenv_sh : str
        Path to the ``dockenv.sh`` environment bootstrap script.
    ligbuild_exe : str
        Name or path of the ligbuild executable.
    timeout : int
        Wall-clock seconds allowed for the ligbuild subprocess.
    ligbuild_timeout : int
        Internal per-protomer timeout hint written into ``custom_parms.json``.

    Returns
    -------
    Path
        Path of the produced ``.tgz`` bundle.

    Raises
    ------
    RuntimeError
        If ligbuild produced no ``.tgz`` anywhere under the working directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # verbose=1 makes build_db2 print its errors instead of silently returning
    # None, which is the difference between a diagnosable failure and a mystery.
    custom_parms_path = out_dir.parent / "custom_parms.json"
    custom_parms_path.write_text(
        json.dumps({"verbose": 1, "timeout": ligbuild_timeout})
    )

    cmd = (
        f'source "{dockenv_sh}" && '
        f'{ligbuild_exe} "{smi_file}" "{out_dir}" "{custom_parms_path}"'
    )
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(out_dir.parent),
    )

    # Deliberately not checking the return code: ligbuild commonly exits rc=1
    # because its shutil.rmtree("db2_outputs") cleanup races in parallel jobs,
    # *after* the .tgz has already been produced and copied into out_dir. Look
    # for the artifact instead, and only fail if it is truly absent.
    tgz_files = (
        list(out_dir.glob("*.tgz"))
        or list((out_dir / "db2_archives").glob("*.tgz"))
        or list((out_dir.parent / "db2_archives").glob("*.tgz"))
        or list(out_dir.parent.rglob("*.tgz"))
    )
    if not tgz_files:
        raise RuntimeError(
            f"ligbuild produced no .tgz under {out_dir.parent} "
            f"(rc={result.returncode}, ligbuild_timeout={ligbuild_timeout}s, "
            f"subprocess_timeout={timeout}s).\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return tgz_files[0]


def _extract_db2(tgz_path: Path, extract_dir: Path) -> Path:
    """Extract a ligbuild bundle and return the first ``.db2`` file inside it.

    Parameters
    ----------
    tgz_path : Path
        The ``.tgz`` bundle produced by ligbuild.
    extract_dir : Path
        Directory to extract into.

    Returns
    -------
    Path
        Path of the extracted ``.db2`` file.

    Raises
    ------
    RuntimeError
        If the archive contained no ``.db2`` file.
    """
    extract_kwargs: dict[str, Any] = {}
    if sys.version_info >= (3, 12):
        extract_kwargs["filter"] = "data"
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_dir, **extract_kwargs)

    # The bundle nests the db2 in a subdirectory (bundle_lig_000/lig.db2), so
    # search recursively rather than assuming a depth.
    db2_files = list(extract_dir.rglob("*.db2"))
    if not db2_files:
        raise RuntimeError(f"No .db2 file found after extracting {tgz_path}")
    return db2_files[0]


def _build_indock(indock_template: Path, db2_path: Path, dest_dir: Path) -> Path:
    """Patch the INDOCK template for this ligand and write it into ``dest_dir``.

    Parameters
    ----------
    indock_template : Path
        The INDOCK template shipped alongside the receptor dockfiles.
    db2_path : Path
        The ligand ``.db2`` file this run should dock.
    dest_dir : Path
        Directory to write the patched INDOCK into. This is also dock64's
        working directory, so the ligand path is made relative to it when
        possible.

    Returns
    -------
    Path
        Path of the written ``INDOCK_run`` file.
    """
    text = indock_template.read_text()
    text = text.replace(_INDOCK_VER_OLD, _INDOCK_VER_NEW)

    try:
        db2_ref: Path = db2_path.relative_to(dest_dir)
    except ValueError:
        # db2 lives outside dest_dir (the usual case, since it is extracted a
        # level up); an absolute path works just as well for dock64.
        db2_ref = db2_path
    text = text.replace(_INDOCK_LIGAND_PLACEHOLDER, str(db2_ref))

    out = dest_dir / "INDOCK_run"
    out.write_text(text)
    return out


def _run_dock64(
    work_dir: Path,
    indock_path: Path,
    *,
    dock64_exe: str,
    dockfiles_dir: Path,
    timeout: int,
) -> None:
    """Run dock64 in ``work_dir`` against a private copy of the dockfiles.

    The dockfiles must be a **real copy** at ``work_dir.parent/dockfiles``, not
    a symlink: the large Fortran binary grid files (``.phi``, ``.bmp``,
    ``.vdw``, ``.desolv``) read incorrectly through symlinks on networked and
    Lustre filesystems, which silently mis-scores every molecule with no error.
    Placing them as a sibling of ``work_dir`` is what makes the
    ``../dockfiles/...`` references inside INDOCK resolve.

    Parameters
    ----------
    work_dir : Path
        Directory dock64 runs in; receives ``OUTDOCK`` and the ``test.*``
        outputs.
    indock_path : Path
        The patched INDOCK file inside ``work_dir``.
    dock64_exe : str
        Path to the dock64 binary, copied into ``work_dir`` before running.
    dockfiles_dir : Path
        Receptor grid directory to copy.
    timeout : int
        Wall-clock seconds allowed for the dock64 subprocess.

    Raises
    ------
    RuntimeError
        If dock64 produced no ``OUTDOCK`` file.
    """
    dock64_dest = work_dir / "dock64"
    if not dock64_dest.exists():
        shutil.copy2(dock64_exe, str(dock64_dest))
        dock64_dest.chmod(dock64_dest.stat().st_mode | 0o111)

    dockfiles_copy = work_dir.parent / "dockfiles"
    if not dockfiles_copy.exists():
        shutil.copytree(str(dockfiles_dir), str(dockfiles_copy))

    result = subprocess.run(
        ["./dock64", indock_path.name],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
        timeout=timeout,
    )

    # dock64 exits non-zero even on success (it raises an ieee_inexact signal),
    # so the return code carries no information. Presence of OUTDOCK is the
    # real success signal.
    if not (work_dir / "OUTDOCK").exists():
        raise RuntimeError(
            f"dock64 produced no OUTDOCK.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


def _parse_outdock_score(outdock_path: Path) -> Optional[float]:
    """Return the best (most negative) total score from an ``OUTDOCK`` file.

    Pose lines have 21 whitespace-separated fields, 0-indexed as::

        0=mol#   1=id_num  2=flexiblecode  3=matched  4=nscored  5=time
        6=hac    7=setnum  8=matnum        9=rank    10=charge  11=elect
       12=gist  13=vdW    14=psol         15=asol    16=tStrain 17=mStrain
       18=rec_d 19=r_hyd  20=Total

    **Fortran field-width overflow.** When Total does not fit its fixed-width
    field, dock64 writes ``**********`` instead of a number. The format string
    has no separator between ``r_hyd`` and ``Total``, so the two merge into one
    token (``0.00**********``), yielding 20 fields instead of 21. In that case
    Total is recomputed from the components::

        Total = elect + gist + vdW + psol + asol + rec_d + r_hyd

    ``charge``, ``tStrain`` and ``mStrain`` are intentionally excluded; this was
    cross-checked against aspirin, where the remaining components sum to -17.26.

    Parameters
    ----------
    outdock_path : Path
        Path of the ``OUTDOCK`` file written by dock64.

    Returns
    -------
    float or None
        Best total score found, or ``None`` if the file is unreadable or no
        pose was placed.
    """
    try:
        lines = outdock_path.read_text().splitlines()
    except Exception:
        return None

    best: Optional[float] = None
    found_start = False
    skip_next = False

    for line in lines:
        if skip_next:
            # The line after an open/close marker is a bare filename.
            skip_next = False
            continue
        if line.startswith(_OUTDOCK_END):
            break
        if line.startswith(_OUTDOCK_FILE_MARKERS):
            skip_next = True
            continue
        if line.startswith(_OUTDOCK_HEADER):
            found_start = True
            continue
        if not found_start:
            continue

        parts = line.split()
        total: Optional[float] = None

        if len(parts) == 21:
            try:
                total = float(parts[20])
            except ValueError:
                pass
        elif len(parts) == 20 and "*" in parts[-1]:
            try:
                total = (
                    float(parts[11])  # elect
                    + float(parts[12])  # gist
                    + float(parts[13])  # vdW
                    + float(parts[14])  # psol
                    + float(parts[15])  # asol
                    + float(parts[18])  # rec_d
                    + float(parts[19].rstrip("*"))  # r_hyd
                )
            except (ValueError, IndexError):
                pass

        if total is not None and (best is None or total < best):
            best = total

    return best


def _classify_failure(stage: str, message: str) -> str:
    """Return a compact, aggregation-friendly label for a failure.

    Parameters
    ----------
    stage : str
        Pipeline stage that failed: ``"ligbuild"``, ``"db2_extract"``,
        ``"dock64"`` or ``"outdock_parse"``.
    message : str
        The exception message to classify.

    Returns
    -------
    str
        A short snake_case failure label.
    """
    text = message.lower()
    if stage == "ligbuild":
        if "timeout reached for" in text and ".db2" in text:
            return "ligbuild_db2_timeout"
        if "timed out" in text or "timeoutexpired" in text:
            return "ligbuild_subprocess_timeout"
        if "error in build_db2" in text and "list index out of range" in text:
            return "ligbuild_build_db2_index_error"
        if "protomer builds failed" in text:
            return "ligbuild_protomer_build_failed"
        if "produced no .tgz" in text:
            return "ligbuild_no_tgz"
        return "ligbuild_failed"
    if stage == "db2_extract":
        return "db2_extract_failed"
    if stage == "dock64":
        if "produced no outdock" in text:
            return "dock64_no_outdock"
        if "timed out" in text or "timeoutexpired" in text:
            return "dock64_timeout"
        return "dock64_failed"
    if stage == "outdock_parse":
        return "dock64_no_pose_or_score"
    return f"{stage}_failed"


class Dock3Oracle(MultiFidelityOracle):
    """Single-fidelity oracle scoring SMILES with the DOCK3 toolchain.

    Each ``candidate.x`` must be a SMILES string. Per molecule the oracle
    writes a ``.smi`` file, runs ``ligbuild`` to build a ``.db2`` ligand bundle,
    runs ``dock64`` against the receptor dockfiles, and parses the best pose
    energy out of ``OUTDOCK``.

    DOCK3 has no natural fidelity ladder, so exactly one fidelity level must be
    declared. Cheaper molecular oracles can be combined with this one at other
    fidelity levels through
    :class:`~activelearning.oracle.composite_oracle.CompositeOracle`.

    Parameters
    ----------
    indock_template : str or Path
        INDOCK template shipped with the receptor dockfiles.
    dockfiles_dir : str or Path
        Directory of prepared receptor grid files.
    fidelity_costs : dict[int, float]
        Cost per sample. Must declare exactly one fidelity level.
    fidelity_confidences : dict[int, float], optional
        Confidence in ``[0, 1]`` per fidelity. Defaults to costs normalized by
        the maximum cost.
    dockenv_sh : str or Path, optional
        Environment bootstrap script sourced before ligbuild. Defaults to
        :data:`DEFAULT_DOCKENV_SH`.
    dock64_exe : str or Path, optional
        Path to the dock64 binary. Defaults to :data:`DEFAULT_DOCK64_EXE`.
    ligbuild_exe : str
        Name or path of the ligbuild executable, resolved from ``$PATH`` after
        sourcing ``dockenv_sh``.
    tmp_dir : str or Path, optional
        Directory for per-call workdirs, preserved for debugging. When omitted,
        ephemeral workdirs are created under a short alias and removed after
        each call. Keep any explicit value short; see
        :func:`_short_workdir_base`.
    timeout : int, default=300
        Wall-clock seconds allowed for *each* of the ligbuild and dock64
        subprocesses. This, not ``ligbuild_timeout``, is what actually bounds a
        ligbuild run.
    ligbuild_timeout : int, default=150
        Internal per-protomer timeout hint passed to ligbuild through
        ``custom_parms.json``. Only meaningful when it is below ``timeout``.
    num_workers : int, optional
        Threads used to evaluate a query batch. ``None`` resolves to
        ``$SLURM_CPUS_PER_TASK``, else the CPU count, else 1. Defaults to 1
        (serial). The work happens in subprocesses, so threads scale well.
    negate_score : bool, default=True
        Whether to return ``-score``. DOCK3 scores are negative-is-better while
        the acquisition loop maximizes, so the default makes the strongest
        binder the highest-valued observation.
    warmup : bool, default=True
        Whether to probe the docking environment during construction. Leave
        enabled in production: the probe must run single-threaded before any
        parallel ligbuild call.

    Notes
    -----
    Molecules that fail at any stage score ``NaN``, and the dataset layer drops
    them before surrogate fitting. Budget is still consumed for them, since a
    failed docking run costs real compute. This differs from the standalone
    ``dock_smiles`` scripts, which use ``0.0`` as the failure sentinel; ``0.0``
    is also a legitimate (if weak) DOCK3 score, so it cannot be distinguished
    from a failure.
    """

    def __init__(
        self,
        indock_template: str | Path,
        dockfiles_dir: str | Path,
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]] = None,
        dockenv_sh: str | Path | None = None,
        dock64_exe: str | Path | None = None,
        ligbuild_exe: str = "ligbuild",
        tmp_dir: str | Path | None = None,
        timeout: int = 300,
        ligbuild_timeout: int = 150,
        num_workers: Optional[int] = 1,
        negate_score: bool = True,
        warmup: bool = True,
    ) -> None:
        """Initialize the DOCK3-backed molecular oracle.

        Raises
        ------
        FileNotFoundError
            If the INDOCK template or the dockfiles directory is missing.
        ValueError
            If no fidelity or more than one fidelity is declared, or if a
            timeout or worker count is not positive.
        """
        if not fidelity_costs:
            raise ValueError("fidelity_costs must declare exactly one fidelity.")
        if len(fidelity_costs) != 1:
            raise ValueError(
                "Dock3Oracle is single-fidelity and must declare exactly one "
                f"fidelity level; got {sorted(fidelity_costs)}. Compose it with "
                "other oracles through CompositeOracle for multi-fidelity runs."
            )
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        if ligbuild_timeout <= 0:
            raise ValueError(
                f"ligbuild_timeout must be positive, got {ligbuild_timeout!r}"
            )
        if num_workers is not None and num_workers < 1:
            raise ValueError(
                f"num_workers must be None or at least 1, got {num_workers!r}"
            )

        self._indock_template = Path(indock_template).resolve()
        self._dockfiles_dir = Path(dockfiles_dir).resolve()
        if not self._indock_template.exists():
            raise FileNotFoundError(
                f"INDOCK template not found: {self._indock_template}"
            )
        if not self._dockfiles_dir.is_dir():
            raise FileNotFoundError(
                f"dockfiles_dir not found or not a directory: {self._dockfiles_dir}"
            )

        self._dockenv_sh = DEFAULT_DOCKENV_SH if dockenv_sh is None else str(dockenv_sh)
        self._dock64_exe = DEFAULT_DOCK64_EXE if dock64_exe is None else str(dock64_exe)
        self._ligbuild_exe = ligbuild_exe
        self._timeout = timeout
        self._ligbuild_timeout = ligbuild_timeout
        self._negate_score = negate_score
        self._num_workers = self._resolve_num_workers(num_workers)

        if tmp_dir is None:
            self._tmp_dir: Optional[Path] = None
        else:
            self._tmp_dir = Path(tmp_dir).resolve()
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
            if len(str(self._tmp_dir)) > _MAX_SAFE_TMP_DIR_CHARS:
                logger.warning(
                    "tmp_dir=%r is %d chars; ligbuild/AMSOL may silently corrupt "
                    "builds once total paths exceed ~80 chars. Prefer a tmp_dir "
                    "under /tmp/ (<=25 chars) for reliable docking scores.",
                    str(self._tmp_dir),
                    len(str(self._tmp_dir)),
                )

        if warmup:
            _warmup_dockenv(
                dockenv_sh=self._dockenv_sh,
                ligbuild_exe=self._ligbuild_exe,
            )

        confidences = self._resolve_fidelity_confidences(
            fidelity_costs, fidelity_confidences
        )
        fidelity_configs: dict[int, dict[str, Any]] = {
            fid: {
                "cost_per_sample": fidelity_costs[fid],
                "fidelity_confidence": confidences[fid],
                "score_fn": self._objective_score,
            }
            for fid in fidelity_costs
        }
        super().__init__(fidelity_configs)

    # ------------------------------------------------------------------
    # MultiFidelityOracle override
    # ------------------------------------------------------------------

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        """Dock each candidate and return one observation per candidate.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates to evaluate. Each must carry a supported ``fidelity``
            and a SMILES string in ``candidate.x``.

        Returns
        -------
        list[Observation]
            One observation per candidate, preserving input order. ``y`` is
            ``NaN`` for molecules that failed to build or dock. Each
            observation's ``metadata`` carries the candidate's metadata plus
            ``dock3_raw_score`` and ``dock3_failure_reason``.

        Raises
        ------
        ValueError
            If a candidate has an unsupported fidelity or a non-string ``x``.
        """
        # Validate everything up front and serially, so programmer errors raise
        # instead of being swallowed as a per-molecule NaN inside a worker.
        prepared = [
            (
                self._validate_candidate_fidelity(candidate, self.fidelity_configs),
                self._extract_smiles(candidate),
            )
            for candidate in candidates
        ]
        if not prepared:
            return []

        smiles_list = [smiles for _, smiles in prepared]
        if self._num_workers == 1 or len(smiles_list) == 1:
            results = [self._safe_dock3_score(smiles) for smiles in smiles_list]
        else:
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                # executor.map preserves input order.
                results = list(executor.map(self._safe_dock3_score, smiles_list))

        observations: list[Observation] = []
        for candidate, (fidelity, _), (raw, reason) in zip(
            candidates, prepared, results
        ):
            observations.append(
                Observation(
                    x=candidate.x,
                    y=self._apply_sign(raw),
                    fidelity=fidelity,
                    metadata={
                        **(candidate.metadata or {}),
                        "dock3_raw_score": raw,
                        "dock3_failure_reason": reason,
                    },
                )
            )
        return observations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_num_workers(num_workers: Optional[int]) -> int:
        """Resolve the worker count, defaulting to the Slurm CPU allocation.

        Parameters
        ----------
        num_workers : int or None
            Requested worker count, or ``None`` to auto-detect.

        Returns
        -------
        int
            Concrete number of worker threads to use.
        """
        if num_workers is not None:
            return num_workers
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            try:
                return max(1, int(slurm_cpus))
            except ValueError:
                logger.warning("Ignoring unparsable SLURM_CPUS_PER_TASK=%r", slurm_cpus)
        return os.cpu_count() or 1

    @staticmethod
    def _resolve_fidelity_confidences(
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]],
    ) -> dict[int, float]:
        """Resolve confidences, defaulting to cost-normalized fractions.

        Parameters
        ----------
        fidelity_costs : dict[int, float]
            Cost per sample for each fidelity level.
        fidelity_confidences : dict[int, float] or None
            Explicit confidence values in ``[0, 1]``. When ``None``, each
            fidelity's confidence is set to ``cost / max_cost``.

        Returns
        -------
        dict[int, float]
            Confidence value for every fidelity key in ``fidelity_costs``.

        Raises
        ------
        ValueError
            If the provided keys do not match those of ``fidelity_costs``.
        """
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

    @staticmethod
    def _extract_smiles(candidate: Candidate) -> str:
        """Return the SMILES string carried by a candidate.

        Parameters
        ----------
        candidate : Candidate
            The candidate to read ``x`` from.

        Returns
        -------
        str
            The candidate's SMILES string.

        Raises
        ------
        ValueError
            If ``candidate.x`` is not a string.
        """
        if not isinstance(candidate.x, str):
            raise ValueError(
                "Expected candidate.x to be a SMILES string, got "
                f"{type(candidate.x).__name__}."
            )
        return candidate.x

    def _apply_sign(self, raw: float) -> float:
        """Apply the maximization sign convention to a raw DOCK3 score.

        Parameters
        ----------
        raw : float
            Raw DOCK3 total score, or ``NaN`` for a failed evaluation.

        Returns
        -------
        float
            ``-raw`` when ``negate_score`` is set, else ``raw``. ``NaN``
            propagates unchanged.
        """
        return -raw if self._negate_score else raw

    def _objective_score(self, smiles: str) -> float:
        """Score a SMILES string, applying the sign convention.

        This is the ``score_fn`` stored in the fidelity config, used by the
        inherited :meth:`MultiFidelityOracle.query`.

        Parameters
        ----------
        smiles : str
            The molecule to dock.

        Returns
        -------
        float
            The objective value, or ``NaN`` if the molecule failed to dock.
        """
        raw, _ = self._safe_dock3_score(smiles)
        return self._apply_sign(raw)

    def _safe_dock3_score(self, smiles: str) -> tuple[float, Optional[str]]:
        """Dock a molecule, converting any unexpected failure into ``NaN``.

        Parameters
        ----------
        smiles : str
            The molecule to dock.

        Returns
        -------
        tuple[float, str or None]
            The raw DOCK3 score and a failure label, or ``(score, None)`` on
            success. The score is ``NaN`` whenever a label is present.
        """
        try:
            return self._dock3_score(smiles)
        except Exception as error:
            logger.warning(
                "Returning NaN for molecule %r: %s.",
                smiles,
                error,
            )
            logger.debug(
                "Detailed oracle failure for molecule %r.", smiles, exc_info=True
            )
            return float("nan"), _classify_failure("oracle", str(error))

    def _dock3_score(
        self, smiles: str, name: str = "lig"
    ) -> tuple[float, Optional[str]]:
        """Run the full DOCK3 pipeline for one molecule.

        Parameters
        ----------
        smiles : str
            The molecule to dock.
        name : str, default="lig"
            Ligand name used for the ``.smi`` file and bundle naming.

        Returns
        -------
        tuple[float, str or None]
            The raw DOCK3 total score and ``None`` on success, or ``NaN`` and a
            failure label from :func:`_classify_failure` on failure.
        """
        work_dir, auto_cleanup = self._make_workdir(smiles)
        try:
            smi_file = work_dir / f"{name}.smi"
            smi_file.write_text(f"{smiles} {name}\n")

            try:
                tgz_path = _run_ligbuild(
                    smi_file,
                    work_dir / "ligbuild_out",
                    dockenv_sh=self._dockenv_sh,
                    ligbuild_exe=self._ligbuild_exe,
                    timeout=self._timeout,
                    ligbuild_timeout=self._ligbuild_timeout,
                )
            except Exception as error:
                logger.warning(
                    "ligbuild failed for %r (workdir=%s): %s", smiles, work_dir, error
                )
                return float("nan"), _classify_failure("ligbuild", str(error))

            try:
                db2_path = _extract_db2(tgz_path, work_dir)
            except Exception as error:
                logger.warning(
                    "db2 extraction failed for %r (workdir=%s): %s",
                    smiles,
                    work_dir,
                    error,
                )
                return float("nan"), _classify_failure("db2_extract", str(error))

            # dock64 runs from run_dir so that the '../dockfiles/...' paths in
            # INDOCK resolve to this call's private dockfiles copy.
            run_dir = work_dir / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            indock_path = _build_indock(self._indock_template, db2_path, run_dir)

            try:
                _run_dock64(
                    run_dir,
                    indock_path,
                    dock64_exe=self._dock64_exe,
                    dockfiles_dir=self._dockfiles_dir,
                    timeout=self._timeout,
                )
            except Exception as error:
                logger.warning(
                    "dock64 failed for %r (workdir=%s): %s", smiles, work_dir, error
                )
                return float("nan"), _classify_failure("dock64", str(error))

            score = _parse_outdock_score(run_dir / "OUTDOCK")
            if score is None:
                return float("nan"), _classify_failure(
                    "outdock_parse", "no pose or score"
                )
            return score, None
        finally:
            if auto_cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _make_workdir(self, smiles: str) -> tuple[Path, bool]:
        """Create a private working directory for one docking call.

        Parameters
        ----------
        smiles : str
            The molecule being docked, used to name persistent workdirs.

        Returns
        -------
        tuple[Path, bool]
            The working directory and whether the caller should delete it.
        """
        if self._tmp_dir is None:
            base = _short_workdir_base()
            return Path(tempfile.mkdtemp(prefix="d", dir=str(base))), True

        # Persistent workdirs are for debugging, so make them identifiable.
        # mkdtemp still guarantees uniqueness across threads and processes.
        digest = f"{abs(hash(smiles)) % (10**8):08d}"
        work_dir = Path(
            tempfile.mkdtemp(prefix=f"dock3_{digest}_", dir=str(self._tmp_dir))
        )
        return work_dir, False

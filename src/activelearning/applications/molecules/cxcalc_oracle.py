"""ChemAxon cxcalc microspecies-distribution oracle.

Scores SMILES strings by the fraction of the molecule that carries a net
negative formal charge at a given pH, computed with ChemAxon's ``cxcalc
msdistr``. Against AmpC beta-lactamase this anionic character is a cheap,
genuinely informative proxy for docking: in a 10M-molecule reference screen the
mean DOCK3 score falls from about -32 for uncharged molecules to about -52 for
fully anionic ones.

The pipeline is ported from the standalone ``dock_smiles`` project, where it is
called the "AmpC noisy oracle" (``oracle_worker.sh`` and the lab's
``parse_msdistr.py``). Several non-obvious behaviours are load bearing; see the
individual function docstrings before changing anything here.

Observed values are an estimated probability of binding rather than the raw
anionic percentage, so that this oracle and
:mod:`activelearning.applications.molecules.dock3_oracle` report the same
quantity and can be composed into one multi-fidelity run. The conversion is a
deliberately coarse two-valued step; see :func:`anion_percent_to_probability`.

Unlike :mod:`activelearning.applications.molecules.xtb_oracle`, this module
needs only the standard library: the chemistry happens in an external binary,
so no optional ``molecules`` extra is required to import it.
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from activelearning.applications.molecules._parallel import resolve_num_workers
from activelearning.applications.molecules._subprocess import _run_subprocess
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.utils.types import Candidate, Observation

logger = logging.getLogger(__name__)

DEFAULT_CXCALC_EXE = "/project/rrg-mailhoto/share/software/freechem-19.15.r4/bin/cxcalc"

# cxcalc is an install4j-wrapped Java launcher, so a JVM must be on PATH. On the
# cluster that is a module load; see CxcalcOracle for how to override it.
DEFAULT_ENV_SETUP = "module load java"

# The SDF property tag holding each microspecies' proportion. Matched by prefix
# rather than by a fully formatted "DISTR[pH=7.4]" so that a non-default pH
# still parses regardless of how cxcalc renders the number.
_DISTR_TAG_PREFIX = "DISTR[pH="

# Net formal charges that count as "anionic". AmpC favours carboxylates, which
# are the -1 species; -2 covers di-acids.
_DEFAULT_ANIONIC_CHARGES = (-1, -2)

# Molecule used to prime the install4j cache during warmup. Any valid SMILES
# would do; ethanol matches the upstream launch script.
_WARMUP_SMILES = "CCO"


def anion_percent_to_probability(
    percent: float,
    *,
    threshold: float = 0.0,
    zero_probability: float = 0.0,
    nonzero_probability: float = 0.01,
) -> float:
    """Convert a raw anionic percentage into an estimated probability of binding.

    A deliberately coarse two-valued step: a molecule carrying no negative
    charge at the working pH is treated as a non-binder, and any molecule that
    does carry some is given a small uniform probability. It exists so this
    oracle reports the same quantity as
    :class:`~activelearning.applications.molecules.dock3_oracle.Dock3Oracle`,
    letting the two be composed into a single-target multi-fidelity run.

    ``threshold`` is worth tuning. In the 10M-molecule AmpC reference screen,
    75.4% of molecules are exactly ``0``, but a further 10.8% fall in
    ``(0, 0.5)`` and dock much more like the zeros (mean DOCK3 -34.70) than like
    the genuinely anionic (-44.61). The default of ``0.0`` keeps every trace of
    charge on the binder side; raising it to ``0.5`` or ``1.0`` moves that
    million-molecule band across.

    Parameters
    ----------
    percent : float
        Summed proportion of anionic microspecies, in percent. ``NaN`` for a
        failed evaluation.
    threshold : float, default=0.0
        Percentage above which a molecule counts as anionic. The comparison is
        strict, so a molecule exactly at the threshold is treated as neutral.
    zero_probability : float, default=0.0
        Probability reported for a molecule at or below ``threshold``.
    nonzero_probability : float, default=0.01
        Probability reported for a molecule above ``threshold``.

    Returns
    -------
    float
        The estimated probability of binding. ``NaN`` propagates unchanged, so
        that a molecule cxcalc could not evaluate stays distinguishable from
        one that is merely uncharged. Collapsing failures onto
        ``zero_probability`` would turn every environment failure into a
        confident "does not bind" for the majority class.
    """
    if not math.isfinite(percent):
        return float("nan")
    return nonzero_probability if percent > threshold else zero_probability


def _build_input_smi(path: Path, smiles_by_id: Sequence[tuple[str, str]]) -> None:
    """Write a cxcalc input file of ``"<smiles> <id>"`` lines.

    cxcalc keys its SDF output by the molecule *name*, which is the second
    whitespace-separated column of the input. Ids must therefore be unique
    within a file: a query batch can legitimately contain the same SMILES twice
    (a sampler is free to propose duplicates), so the caller supplies positional
    ids rather than deriving them from the molecule.

    Parameters
    ----------
    path : Path
        File to write.
    smiles_by_id : Sequence[tuple[str, str]]
        ``(id, smiles)`` pairs, in the order they should be written.
    """
    path.write_text("".join(f"{smiles} {mol_id}\n" for mol_id, smiles in smiles_by_id))


def _parse_msdistr_sdf(
    sdf_path: Path,
    *,
    anionic_charges: Iterable[int] = _DEFAULT_ANIONIC_CHARGES,
) -> dict[str, float]:
    """Sum the anionic microspecies proportions for each molecule in an SDF.

    ``cxcalc msdistr`` emits one SDF record per *microspecies*, not per
    molecule, with consecutive records sharing a title line when they belong to
    the same input molecule. This walks the file as a small state machine:

    - the first line of a record is its title, i.e. the molecule id;
    - ``M  CHG  n  atom1 chg1  atom2 chg2 ...`` lines carry atomic formal
      charges, and a microspecies' net charge is their sum (a record with no
      such line is neutral);
    - the line *after* a ``> <DISTR[pH=...]>`` tag holds the proportion;
    - ``$$$$`` ends the record, so the next line is the next title.

    Proportions whose net charge is in ``anionic_charges`` are accumulated per
    molecule; every molecule seen is present in the result, at ``0.0`` if none
    of its microspecies qualified.

    Three deliberate differences from the upstream ``parse_msdistr.py``: the
    property tag is matched by prefix so a non-default pH still parses; blank
    title lines are skipped (the production 10M run emitted at least one record
    with an empty id); and an unparsable proportion is skipped rather than
    raising, so one malformed record cannot fail a whole chunk.

    Parameters
    ----------
    sdf_path : Path
        The SDF written by ``cxcalc -o <file> msdistr``.
    anionic_charges : Iterable[int], optional
        Net formal charges that count as anionic.

    Returns
    -------
    dict[str, float]
        Molecule id to summed anionic proportion, in percent.
    """
    targets = set(anionic_charges)
    sums: dict[str, float] = {}
    current_id: Optional[str] = None
    current_charges: list[int] = []
    at_title = True
    read_proportion = False

    with sdf_path.open() as handle:
        for line in handle:
            if read_proportion:
                read_proportion = False
                try:
                    proportion = float(line.strip())
                except ValueError:
                    continue
                if current_id is not None and sum(current_charges) in targets:
                    sums[current_id] = sums.get(current_id, 0.0) + proportion
                continue

            if at_title:
                title = line.strip()
                # A blank title belongs to no molecule; keep parsing so the
                # records that follow still land on the right ids.
                current_id = title or None
                if current_id is not None:
                    sums.setdefault(current_id, 0.0)
                current_charges = []
                at_title = False
            elif line.startswith("M  CHG"):
                # MDL format: M  CHG  n  atom1 chg1  atom2 chg2 ...
                fields = line.split()
                try:
                    count = int(fields[2])
                    current_charges += [
                        int(fields[4 + 2 * index]) for index in range(count)
                    ]
                except (ValueError, IndexError):
                    continue
            elif line.startswith(">") and _DISTR_TAG_PREFIX in line:
                read_proportion = True
            elif line.startswith("$$$$"):
                at_title = True

    return sums


def _quote(argument: str) -> str:
    """Return a shell-quoted argument.

    Parameters
    ----------
    argument : str
        Argument to quote.

    Returns
    -------
    str
        The argument wrapped in double quotes with any embedded ones escaped.
    """
    escaped = argument.replace('"', '\\"')
    return f'"{escaped}"'


def _run_cxcalc(
    input_smi: Path,
    output_sdf: Path,
    *,
    cxcalc_exe: str,
    env_setup: Optional[str],
    ph: float,
    timeout: int,
) -> None:
    """Run ``cxcalc msdistr`` over one input file.

    Parameters
    ----------
    input_smi : Path
        Input file of ``"<smiles> <id>"`` lines.
    output_sdf : Path
        SDF path for cxcalc to write.
    cxcalc_exe : str
        Path to the cxcalc launcher.
    env_setup : str or None
        Shell command sourced before cxcalc, typically ``"module load java"``.
        ``None`` invokes the binary directly with no shell.
    ph : float
        pH at which to compute the microspecies distribution.
    timeout : int
        Wall-clock seconds allowed for the subprocess.

    Raises
    ------
    RuntimeError
        If cxcalc produced no output SDF, or an empty one.
    """
    args = [cxcalc_exe, "-o", str(output_sdf), "msdistr", "-H", str(ph), str(input_smi)]
    if env_setup:
        command = ["bash", "-c", f"{env_setup} && {' '.join(_quote(a) for a in args)}"]
    else:
        command = args

    result = _run_subprocess(command, timeout=timeout, cwd=input_smi.parent)

    # The install4j launcher's return code is unreliable, so the shell scripts
    # this is ported from test for a non-empty SDF instead. Do the same.
    if not output_sdf.exists() or output_sdf.stat().st_size == 0:
        tail = ((result.stdout or "") + (result.stderr or ""))[-1500:]
        raise RuntimeError(
            f"cxcalc produced no SDF at {output_sdf} (rc={result.returncode}). "
            f"Last output:\n{tail}"
        )


def _warmup_cxcalc(
    *,
    cxcalc_exe: str,
    env_setup: Optional[str],
    ph: float,
    timeout: int = 600,
) -> None:
    """Run one molecule through cxcalc to prime its cache and probe the environment.

    The lab's patched launcher writes its install4j cache under ``/tmp`` (see
    ``bin/cxcalc.README`` in the install, which documents the NFS ``$HOME``
    stampede this replaced). A cold cache hit concurrently by several workers is
    exactly the race the upstream ``launch_oracle.sh`` guards against with a
    warmup, so this must run single-threaded before any parallel call. It is
    also the cheapest place to surface a missing JVM or an expired license,
    which would otherwise turn every molecule into an undifferentiated ``NaN``.

    Best effort by design: failures are logged, never raised, so a node with a
    healthy environment still proceeds and the per-chunk call surfaces the real
    error if there is one.

    The probe deliberately goes through :func:`_run_cxcalc`, so it exercises the
    same shell form, the same binary and the same arguments as the real calls.
    A probe that differed could report success for an environment the real call
    never sees, which is worse than no probe at all.

    Parameters
    ----------
    cxcalc_exe : str
        Path to the cxcalc launcher.
    env_setup : str or None
        Shell command sourced before cxcalc.
    ph : float
        pH to probe with; matches the oracle's configured value.
    timeout : int, default=600
        Seconds to wait for the probe before giving up.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="cxcalc_warmup_") as tmp:
            work = Path(tmp)
            input_smi = work / "input.smi"
            _build_input_smi(input_smi, [("0", _WARMUP_SMILES)])
            _run_cxcalc(
                input_smi,
                work / "out.sdf",
                cxcalc_exe=cxcalc_exe,
                env_setup=env_setup,
                ph=ph,
                timeout=timeout,
            )
    except Exception as error:
        logger.warning(
            "cxcalc warmup failed (%s: %s). Continuing - per-chunk calls will "
            "retry, but every molecule will score NaN if the environment is "
            "genuinely broken. cxcalc_exe=%s, env_setup=%r.",
            type(error).__name__,
            error,
            cxcalc_exe,
            env_setup,
        )
        return
    logger.info("cxcalc warmup OK (cxcalc_exe=%s)", cxcalc_exe)


def _classify_failure(stage: str, message: str) -> str:
    """Return a compact, aggregation-friendly label for a failure.

    Parameters
    ----------
    stage : str
        Pipeline stage that failed: ``"cxcalc"``, ``"parse"``, ``"lookup"`` or
        ``"smiles"``.
    message : str
        The exception message to classify.

    Returns
    -------
    str
        A short snake_case failure label.
    """
    text = message.lower()
    if stage == "smiles":
        return "cxcalc_bad_smiles"
    if stage == "lookup":
        return "cxcalc_no_result"
    if stage == "parse":
        return "cxcalc_parse_failed"
    if stage == "cxcalc":
        if "timed out" in text or "timeoutexpired" in text:
            return "cxcalc_timeout"
        if "produced no sdf" in text:
            return "cxcalc_no_sdf"
        return "cxcalc_failed"
    return f"{stage}_failed"


class CxcalcOracle(MultiFidelityOracle):
    """Single-fidelity oracle scoring SMILES by anionic character at a given pH.

    Each ``candidate.x`` must be a SMILES string. A whole query batch is written
    to one input file and evaluated with a single ``cxcalc msdistr`` call, which
    is what makes this oracle cheap: cxcalc is a JVM application, so amortizing
    one start-up over many molecules matters far more than parallelism.

    The observed value is **not** the anionic percentage. It is converted to an
    estimated probability of binding by
    :func:`anion_percent_to_probability`, a two-valued step whose boundary and
    both output values are configurable. The raw percentage is kept in each
    observation's metadata. Reporting a probability is what lets this oracle
    share a multi-fidelity target with
    :class:`~activelearning.applications.molecules.dock3_oracle.Dock3Oracle`,
    which observes the same quantity through a fitted hit-rate model.

    cxcalc computes one property at one pH, so there is no fidelity ladder and
    exactly one fidelity level must be declared. Combine it with more expensive
    oracles at other levels through
    :class:`~activelearning.oracle.composite_oracle.CompositeOracle`.

    Parameters
    ----------
    fidelity_costs : dict[int, float]
        Cost per sample. Must declare exactly one fidelity level.
    fidelity_confidences : dict[int, float], optional
        Confidence in ``[0, 1]`` per fidelity. Defaults to costs normalized by
        the maximum cost, which for a single-fidelity oracle is always ``1.0``
        - set it explicitly when composing with another oracle.
    cxcalc_exe : str or Path, optional
        Path to the cxcalc launcher. Defaults to :data:`DEFAULT_CXCALC_EXE`.
    env_setup : str or None, default=:data:`DEFAULT_ENV_SETUP`
        Shell command run before cxcalc, to put a JVM on ``PATH``. The default
        relies on ``module`` being an exported shell function, which is how lmod
        makes itself visible to non-login shells; it is inherited from whatever
        shell launched the run. Pass ``None`` to invoke the binary directly when
        java is already loaded.
    ph : float, default=7.4
        pH at which the microspecies distribution is computed.
    anionic_charges : Sequence[int], optional
        Net formal charges counted as anionic. Defaults to ``(-1, -2)``.
    anion_percent_threshold : float, default=0.0
        Percentage strictly above which a molecule counts as anionic.
    zero_probability : float, default=0.0
        Probability reported at or below the threshold.
    nonzero_probability : float, default=0.01
        Probability reported above the threshold.
    timeout : int, default=600
        Wall-clock seconds allowed for each cxcalc subprocess.
    chunk_size : int, default=12500
        Maximum molecules per cxcalc invocation. Matches the production chunk
        size; a typical acquisition batch is a single chunk.
    num_workers : int, optional
        Threads used to evaluate chunks. Defaults to ``None``, which sizes the
        pool from the CPUs this job actually holds; see
        :func:`~activelearning.applications.molecules._parallel.resolve_num_workers`.
        Only relevant for batches larger than ``chunk_size``, since a batch that
        fits in one chunk is one subprocess.
    warmup : bool, default=True
        Whether to prime the cxcalc environment during construction. Leave
        enabled in production: the probe must run single-threaded before any
        parallel call.

    Notes
    -----
    Molecules that fail score ``NaN``, and the dataset layer drops them before
    surrogate fitting. Budget is still consumed for them, since a failed
    evaluation costs real compute. ``NaN`` rather than ``zero_probability``
    matters more here than for docking: with the default step, ``0.0`` is the
    value roughly three quarters of all molecules legitimately take, so it
    cannot double as a failure sentinel.

    cxcalc also drops molecules silently. The reference 10M-molecule run
    returned about 12,400 rows per 12,500-molecule chunk, so an id missing from
    the output is an expected per-molecule failure rather than a broken run.
    """

    def __init__(
        self,
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]] = None,
        cxcalc_exe: str | Path | None = None,
        env_setup: Optional[str] = DEFAULT_ENV_SETUP,
        ph: float = 7.4,
        anionic_charges: Sequence[int] = _DEFAULT_ANIONIC_CHARGES,
        anion_percent_threshold: float = 0.0,
        zero_probability: float = 0.0,
        nonzero_probability: float = 0.01,
        timeout: int = 600,
        chunk_size: int = 12500,
        num_workers: Optional[int] = None,
        warmup: bool = True,
    ) -> None:
        """Initialize the cxcalc-backed molecular oracle.

        Raises
        ------
        ValueError
            If no fidelity or more than one fidelity is declared, if a timeout,
            chunk size or worker count is not positive, if ``anionic_charges``
            is empty or holds non-integers, if ``ph`` is not finite, if either
            probability falls outside ``[0, 1]``, or if the threshold is
            negative.
        """
        if len(fidelity_costs) != 1:
            raise ValueError(
                "CxcalcOracle is single-fidelity and must declare exactly one "
                f"fidelity level; got {sorted(fidelity_costs)}. Compose it with "
                "other oracles through CompositeOracle for multi-fidelity runs."
            )
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size!r}")
        if num_workers is not None and num_workers < 1:
            raise ValueError(
                f"num_workers must be None or at least 1, got {num_workers!r}"
            )
        if not math.isfinite(ph):
            raise ValueError(f"ph must be a finite number, got {ph!r}")

        charges = tuple(anionic_charges)
        if not charges:
            raise ValueError("anionic_charges must declare at least one charge.")
        if any(isinstance(c, bool) or not isinstance(c, int) for c in charges):
            raise ValueError(f"anionic_charges must be integers, got {charges!r}")

        for name, probability in (
            ("zero_probability", zero_probability),
            ("nonzero_probability", nonzero_probability),
        ):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {probability!r}")
        if not math.isfinite(anion_percent_threshold) or anion_percent_threshold < 0:
            raise ValueError(
                "anion_percent_threshold must be a non-negative number, got "
                f"{anion_percent_threshold!r}"
            )

        self._cxcalc_exe = DEFAULT_CXCALC_EXE if cxcalc_exe is None else str(cxcalc_exe)
        self._env_setup = env_setup
        self._ph = float(ph)
        self._anionic_charges = charges
        self._anion_percent_threshold = float(anion_percent_threshold)
        self._zero_probability = float(zero_probability)
        self._nonzero_probability = float(nonzero_probability)
        self._timeout = timeout
        self._chunk_size = chunk_size
        self._num_workers = resolve_num_workers(num_workers)

        # Deliberately no existence check on cxcalc_exe: it may only resolve
        # once env_setup has run. The warmup is what surfaces a bad path.
        if warmup:
            _warmup_cxcalc(
                cxcalc_exe=self._cxcalc_exe,
                env_setup=self._env_setup,
                ph=self._ph,
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
        """Evaluate each candidate and return one observation per candidate.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates to evaluate. Each must carry a supported ``fidelity``
            and a SMILES string in ``candidate.x``.

        Returns
        -------
        list[Observation]
            One observation per candidate, preserving input order. ``y`` is
            ``NaN`` for molecules cxcalc failed to evaluate. Each observation's
            ``metadata`` carries the candidate's metadata plus
            ``cxcalc_anion_percent``, ``cxcalc_ph`` and
            ``cxcalc_failure_reason``.

        Raises
        ------
        ValueError
            If a candidate has an unsupported fidelity or a non-string ``x``.

        See Also
        --------
        _log_query_failure_summary : Aggregate outcomes logged for each batch.
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

        results = self._score_batch([smiles for _, smiles in prepared])

        observations = [
            Observation(
                x=candidate.x,
                y=self._probability(percent),
                fidelity=fidelity,
                metadata={
                    **(candidate.metadata or {}),
                    "cxcalc_anion_percent": percent,
                    "cxcalc_ph": self._ph,
                    "cxcalc_failure_reason": reason,
                },
            )
            for candidate, (fidelity, _), (percent, reason) in zip(
                candidates, prepared, results
            )
        ]
        self._log_query_failure_summary(results)
        return observations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_batch(
        self, smiles_list: Sequence[str]
    ) -> list[tuple[float, Optional[str]]]:
        """Evaluate a whole batch of SMILES, chunking and parallelising as configured.

        Molecules that cannot be written to a cxcalc input file at all are held
        out here rather than sent: cxcalc keys results by the second column of
        each line, so an embedded space would silently shift every id in the
        file and misalign the entire chunk.

        Parameters
        ----------
        smiles_list : Sequence[str]
            SMILES strings, in query order.

        Returns
        -------
        list[tuple[float, str or None]]
            Per-molecule ``(anion_percent, failure_reason)`` pairs, in the same
            order. The percentage is ``NaN`` whenever a reason is present.
        """
        results: list[tuple[float, Optional[str]]] = [(float("nan"), None)] * len(
            smiles_list
        )

        sendable: list[tuple[int, str]] = []
        for index, smiles in enumerate(smiles_list):
            if not smiles or any(char.isspace() for char in smiles):
                logger.warning(
                    "Returning NaN for molecule %r: empty, or contains "
                    "whitespace, which cxcalc would read as a name column.",
                    smiles,
                )
                results[index] = (float("nan"), _classify_failure("smiles", ""))
            else:
                sendable.append((index, smiles))

        chunks = [
            sendable[start : start + self._chunk_size]
            for start in range(0, len(sendable), self._chunk_size)
        ]
        if not chunks:
            return results

        if self._num_workers == 1 or len(chunks) == 1:
            chunk_results = [self._run_chunk(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                # executor.map preserves input order.
                chunk_results = list(executor.map(self._run_chunk, chunks))

        for chunk, (percentages, chunk_reason) in zip(chunks, chunk_results):
            for index, _ in chunk:
                if chunk_reason is not None:
                    results[index] = (float("nan"), chunk_reason)
                    continue
                percent = percentages.get(str(index))
                if percent is None:
                    # cxcalc silently drops roughly 1% of molecules; an absent
                    # id is a per-molecule failure, not a broken chunk.
                    results[index] = (
                        float("nan"),
                        _classify_failure("lookup", ""),
                    )
                else:
                    results[index] = (percent, None)
        return results

    def _run_chunk(
        self, chunk: Sequence[tuple[int, str]]
    ) -> tuple[dict[str, float], Optional[str]]:
        """Run one cxcalc invocation over a chunk of molecules.

        Parameters
        ----------
        chunk : Sequence[tuple[int, str]]
            ``(batch_index, smiles)`` pairs. The batch index doubles as the
            cxcalc molecule name, which guarantees uniqueness even when the
            same SMILES appears twice in a batch.

        Returns
        -------
        tuple[dict[str, float], str or None]
            Parsed id-to-percentage mapping and ``None`` on success, or an
            empty mapping and a failure label covering the whole chunk.
        """
        try:
            with tempfile.TemporaryDirectory(prefix="cxcalc_") as tmp:
                work = Path(tmp)
                input_smi = work / "input.smi"
                output_sdf = work / "out.sdf"
                _build_input_smi(
                    input_smi, [(str(index), smiles) for index, smiles in chunk]
                )
                _run_cxcalc(
                    input_smi,
                    output_sdf,
                    cxcalc_exe=self._cxcalc_exe,
                    env_setup=self._env_setup,
                    ph=self._ph,
                    timeout=self._timeout,
                )
                return (
                    _parse_msdistr_sdf(
                        output_sdf, anionic_charges=self._anionic_charges
                    ),
                    None,
                )
        except subprocess.TimeoutExpired as error:
            logger.warning(
                "cxcalc timed out on a chunk of %d molecules: %s", len(chunk), error
            )
            return {}, _classify_failure("cxcalc", "timed out")
        except Exception as error:
            logger.warning(
                "cxcalc failed on a chunk of %d molecules: %s", len(chunk), error
            )
            logger.debug("Detailed cxcalc chunk failure.", exc_info=True)
            return {}, _classify_failure("cxcalc", str(error))

    def _log_query_failure_summary(
        self, results: Sequence[tuple[float, Optional[str]]]
    ) -> None:
        """Log aggregate cxcalc outcomes for one query batch, if a logger is bound.

        Failed molecules score ``NaN`` and are dropped by the active-learning
        loop before they reach the dataset, so the per-observation
        ``cxcalc_failure_reason`` never survives the round. Without this summary
        the two situations that matter cannot be told apart: a sampler emitting
        molecules cxcalc cannot handle, and a compute node with no JVM or an
        expired license where every molecule fails identically. Both spend the
        full budget, since cost is charged before the query runs.

        Parameters
        ----------
        results : Sequence[tuple[float, str or None]]
            Per-molecule ``(anion_percent, failure_reason)`` pairs from the most
            recent :meth:`query` call.
        """
        if self.logger is None or not results:
            return

        reasons = Counter(reason for _, reason in results if reason is not None)
        total = len(results)
        succeeded = total - sum(reasons.values())

        self.logger.log_metric("cxcalc/queried", float(total))
        self.logger.log_metric("cxcalc/succeeded", float(succeeded))
        self.logger.log_metric("cxcalc/success_rate", succeeded / total)
        for reason, count in sorted(reasons.items()):
            self.logger.log_metric(f"cxcalc/failures/{reason}", float(count))

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

    def _probability(self, percent: float) -> float:
        """Convert a raw anionic percentage into an estimated probability of binding.

        Parameters
        ----------
        percent : float
            Summed anionic proportion in percent, or ``NaN`` for a failed
            evaluation.

        Returns
        -------
        float
            Probability in ``[0, 1]``. ``NaN`` propagates unchanged, so failed
            molecules stay failures rather than becoming non-binders.
        """
        return anion_percent_to_probability(
            percent,
            threshold=self._anion_percent_threshold,
            zero_probability=self._zero_probability,
            nonzero_probability=self._nonzero_probability,
        )

    def _objective_score(self, smiles: str) -> float:
        """Evaluate one SMILES string and return its probability of binding.

        This is the ``score_fn`` stored in the fidelity config, used by the
        inherited :meth:`MultiFidelityOracle.query`. :meth:`query` is overridden
        to batch instead, since one cxcalc call per molecule would pay a JVM
        start-up per molecule and throw away the oracle's whole cost advantage.

        Parameters
        ----------
        smiles : str
            The molecule to evaluate.

        Returns
        -------
        float
            The objective value, or ``NaN`` if the molecule failed.
        """
        percent, _ = self._score_batch([smiles])[0]
        return self._probability(percent)

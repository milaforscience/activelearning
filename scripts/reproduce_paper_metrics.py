"""Paper-facing metrics computed from recorded reproduction run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Literal, Sequence

from activelearning.applications.molecules._optional import (
    missing_molecules_dependency_error,
)

RunTaskGroup = Literal["synthetic", "molecules"]

RUN_MANIFEST_FILENAME = "run_manifest.json"
ROUND_HISTORY_FILENAME = "round_history.jsonl"

_DEFAULT_TASK_GROUP_BY_TASK: dict[str, RunTaskGroup] = {
    "branin": "synthetic",
    "hartmann": "synthetic",
    "molecules_ea": "molecules",
    "molecules_ip": "molecules",
}
_DEFAULT_TOP_K_BY_TASK: dict[str, int] = {
    "branin": 50,
    "hartmann": 10,
    "molecules_ea": 100,
    "molecules_ip": 100,
}
_DEFAULT_NEGATED_SCORE_TASKS = frozenset({"branin"})


@dataclass(frozen=True)
class RunDescriptor:
    """Resolved metadata for one recorded paper-reproduction run."""

    task_group: RunTaskGroup
    task: str
    method: str
    seed: int
    run_directory: Path
    manifest_path: Path
    round_history_path: Path
    highest_fidelity: int
    initial_data_mode: str
    total_active_learning_budget: float
    top_k: int
    negate_score: bool
    molecule_representation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly descriptor payload."""

        return {
            "task_group": self.task_group,
            "task": self.task,
            "method": self.method,
            "seed": self.seed,
            "run_directory": str(self.run_directory),
            "manifest_path": str(self.manifest_path),
            "round_history_path": str(self.round_history_path),
            "highest_fidelity": self.highest_fidelity,
            "initial_data_mode": self.initial_data_mode,
            "total_active_learning_budget": self.total_active_learning_budget,
            "top_k": self.top_k,
            "negate_score": self.negate_score,
            "molecule_representation": self.molecule_representation,
        }


@dataclass(frozen=True)
class RecordedRun:
    """Loaded manifest plus round history for one recorded run."""

    descriptor: RunDescriptor
    manifest: dict[str, Any]
    round_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HighFidelityTraceRow:
    """One recorded highest-fidelity observation aligned to the run budget."""

    task_group: RunTaskGroup
    task: str
    method: str
    seed: int
    round_index: int
    cumulative_budget: float
    budget_fraction_of_total_active_learning_budget: float
    source: Literal["initial_data", "active_learning_round"]
    observation_index: int
    fidelity: int
    target_value: float
    paper_score: float
    identifier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly trace row."""

        return {
            "task_group": self.task_group,
            "task": self.task,
            "method": self.method,
            "seed": self.seed,
            "round_index": self.round_index,
            "cumulative_budget": self.cumulative_budget,
            "budget_fraction_of_total_active_learning_budget": (
                self.budget_fraction_of_total_active_learning_budget
            ),
            "source": self.source,
            "observation_index": self.observation_index,
            "fidelity": self.fidelity,
            "target_value": self.target_value,
            "paper_score": self.paper_score,
            "identifier": self.identifier,
        }


@dataclass(frozen=True)
class SyntheticMetricRow:
    """Paper-facing synthetic metrics at one active-learning checkpoint."""

    task: str
    method: str
    seed: int
    round_index: int
    cumulative_budget: float
    budget_fraction_of_total_active_learning_budget: float
    high_fidelity_observation_count: int
    top_k: int
    top_k_observation_count: int
    mean_top_k_score: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly synthetic metric row."""

        return {
            "task_group": "synthetic",
            "task": self.task,
            "method": self.method,
            "seed": self.seed,
            "round_index": self.round_index,
            "cumulative_budget": self.cumulative_budget,
            "budget_fraction_of_total_active_learning_budget": (
                self.budget_fraction_of_total_active_learning_budget
            ),
            "high_fidelity_observation_count": self.high_fidelity_observation_count,
            "top_k": self.top_k,
            "top_k_observation_count": self.top_k_observation_count,
            "mean_top_k_score": self.mean_top_k_score,
        }


@dataclass(frozen=True)
class MoleculeMetricRow:
    """Paper-facing molecule metrics at one active-learning checkpoint."""

    task: str
    method: str
    seed: int
    round_index: int
    cumulative_budget: float
    budget_fraction_of_total_sf_gfn_budget: float
    high_fidelity_observation_count: int
    top_k: int
    top_k_observation_count: int
    mean_top_k_score: float | None
    mean_top_k_energy: float | None
    mean_pairwise_tanimoto_distance: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly molecule metric row."""

        return {
            "task_group": "molecules",
            "task": self.task,
            "method": self.method,
            "seed": self.seed,
            "round_index": self.round_index,
            "cumulative_budget": self.cumulative_budget,
            "budget_fraction_of_total_sf_gfn_budget": (
                self.budget_fraction_of_total_sf_gfn_budget
            ),
            "high_fidelity_observation_count": self.high_fidelity_observation_count,
            "top_k": self.top_k,
            "top_k_observation_count": self.top_k_observation_count,
            "mean_top_k_score": self.mean_top_k_score,
            "mean_top_k_energy": self.mean_top_k_energy,
            "mean_pairwise_tanimoto_distance": self.mean_pairwise_tanimoto_distance,
        }


@dataclass(frozen=True)
class SyntheticRunMetrics:
    """Per-run synthetic metric tables ready for later plotting."""

    run: RunDescriptor
    checkpoint_rows: tuple[SyntheticMetricRow, ...]
    high_fidelity_trace_rows: tuple[HighFidelityTraceRow, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly synthetic metrics payload."""

        return {
            "run": self.run.to_dict(),
            "checkpoint_rows": [row.to_dict() for row in self.checkpoint_rows],
            "high_fidelity_trace_rows": [
                row.to_dict() for row in self.high_fidelity_trace_rows
            ],
        }


@dataclass(frozen=True)
class MoleculeRunMetrics:
    """Per-run molecule metric tables ready for later plotting."""

    run: RunDescriptor
    checkpoint_rows: tuple[MoleculeMetricRow, ...]
    high_fidelity_trace_rows: tuple[HighFidelityTraceRow, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly molecule metrics payload."""

        return {
            "run": self.run.to_dict(),
            "checkpoint_rows": [row.to_dict() for row in self.checkpoint_rows],
            "high_fidelity_trace_rows": [
                row.to_dict() for row in self.high_fidelity_trace_rows
            ],
        }


RunMetrics = SyntheticRunMetrics | MoleculeRunMetrics


@dataclass(frozen=True)
class MetricCatalog:
    """Collection of per-run metric tables plus flattened aggregation rows."""

    synthetic_runs: tuple[SyntheticRunMetrics, ...]
    molecule_runs: tuple[MoleculeRunMetrics, ...]

    def synthetic_checkpoint_rows(self) -> tuple[SyntheticMetricRow, ...]:
        """Return all synthetic checkpoint rows across runs."""

        return tuple(
            row
            for run_metrics in self.synthetic_runs
            for row in run_metrics.checkpoint_rows
        )

    def synthetic_high_fidelity_rows(self) -> tuple[HighFidelityTraceRow, ...]:
        """Return all synthetic highest-fidelity trace rows across runs."""

        return tuple(
            row
            for run_metrics in self.synthetic_runs
            for row in run_metrics.high_fidelity_trace_rows
        )

    def molecule_checkpoint_rows(self) -> tuple[MoleculeMetricRow, ...]:
        """Return all molecule checkpoint rows across runs."""

        return tuple(
            row
            for run_metrics in self.molecule_runs
            for row in run_metrics.checkpoint_rows
        )

    def molecule_high_fidelity_rows(self) -> tuple[HighFidelityTraceRow, ...]:
        """Return all molecule highest-fidelity trace rows across runs."""

        return tuple(
            row
            for run_metrics in self.molecule_runs
            for row in run_metrics.high_fidelity_trace_rows
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly metrics catalog."""

        return {
            "schema_version": 1,
            "synthetic_runs": [run.to_dict() for run in self.synthetic_runs],
            "molecule_runs": [run.to_dict() for run in self.molecule_runs],
            "synthetic_checkpoint_rows": [
                row.to_dict() for row in self.synthetic_checkpoint_rows()
            ],
            "synthetic_high_fidelity_rows": [
                row.to_dict() for row in self.synthetic_high_fidelity_rows()
            ],
            "molecule_checkpoint_rows": [
                row.to_dict() for row in self.molecule_checkpoint_rows()
            ],
            "molecule_high_fidelity_rows": [
                row.to_dict() for row in self.molecule_high_fidelity_rows()
            ],
        }


@dataclass(frozen=True)
class _BudgetCheckpoint:
    """One budget checkpoint where paper-facing metrics should be evaluated."""

    round_index: int
    cumulative_budget: float
    normalized_budget: float


@dataclass(frozen=True)
class _RecordedObservation:
    """Internal normalized observation record used to compute metrics."""

    round_index: int
    cumulative_budget: float
    normalized_budget: float
    source: Literal["initial_data", "active_learning_round"]
    observation_index: int
    fidelity: int
    target_value: float
    paper_score: float
    identifier: str | None
    molecule_input: str | None


def load_recorded_run(path: Path) -> RecordedRun:
    """Load one recorded run from a run directory or manifest path."""

    manifest_path, round_history_path = _resolve_run_artifact_paths(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    round_history = tuple(_load_round_history(round_history_path))
    run_payload = _resolve_run_payload(manifest, manifest_path)
    reproduce_paper = _reproduce_paper_config(manifest)
    task = str(run_payload["task"])
    oracle_costs = _extract_oracle_costs(manifest.get("config", {})) or {3: 1.0}
    highest_fidelity = max(int(fidelity) for fidelity in oracle_costs)
    total_budget = _extract_total_active_learning_budget(manifest.get("config", {}))
    task_group = _resolve_task_group(
        task=task,
        task_group=run_payload.get("task_group") or reproduce_paper.get("task_group"),
    )
    molecule_representation = (
        _normalize_molecule_representation(
            reproduce_paper.get("molecule_representation")
        )
        if task_group == "molecules"
        else None
    )
    descriptor = RunDescriptor(
        task_group=task_group,
        task=task,
        method=str(run_payload["method"]),
        seed=int(run_payload["seed"]),
        run_directory=manifest_path.parent,
        manifest_path=manifest_path,
        round_history_path=round_history_path,
        highest_fidelity=highest_fidelity,
        initial_data_mode=_resolve_initial_data_mode(manifest),
        total_active_learning_budget=total_budget,
        top_k=_resolve_top_k(task=task, reproduce_paper=reproduce_paper),
        negate_score=_resolve_negate_score(task=task, reproduce_paper=reproduce_paper),
        molecule_representation=molecule_representation,
    )
    return RecordedRun(
        descriptor=descriptor,
        manifest=manifest,
        round_history=round_history,
    )


def _resolve_run_payload(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    """Recover run identity from recorded metadata or the recorded config."""

    run_payload = manifest.get("run", {})
    config = manifest.get("config", {})
    reproduce_paper = _reproduce_paper_config(manifest)
    task, method = _infer_task_and_method(manifest, manifest_path)
    if task is None or method is None:
        raise KeyError(
            "Recorded run is missing run.task/method metadata and no paper config path "
            "was captured in cli.config_files."
        )

    seed = run_payload.get("seed", config.get("runtime", {}).get("seed"))
    if seed is None:
        raise KeyError(
            "Recorded run is missing a seed in both run metadata and config.runtime.seed."
        )

    return {
        "task_group": _resolve_task_group(
            task=task,
            task_group=run_payload.get("task_group")
            or reproduce_paper.get("task_group"),
        ),
        "task": task,
        "method": method,
        "seed": int(seed),
    }


def _infer_task_and_method(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[str | None, str | None]:
    """Infer the paper experiment identity from the manifest or run directory."""

    run_payload = manifest.get("run", {})
    task = run_payload.get("task")
    method = run_payload.get("method")
    if task is not None and method is not None:
        return str(task), str(method)

    cli_metadata = manifest.get("cli", {})
    config_files = cli_metadata.get("config_files", [])
    inferred_task: str | None = None
    inferred_method: str | None = None
    for config_file in config_files:
        task_from_path, method_from_path = _infer_task_and_method_from_config_file(
            Path(str(config_file))
        )
        if task_from_path is not None:
            inferred_task = task_from_path
        if method_from_path is not None:
            inferred_method = method_from_path
        if inferred_task is not None and inferred_method is not None:
            return inferred_task, inferred_method

    inferred = _infer_task_and_method_from_run_directory(manifest_path.parent)
    if inferred is not None:
        return inferred
    return inferred_task, inferred_method


def _infer_task_and_method_from_config_file(
    config_path: Path,
) -> tuple[str | None, str | None]:
    """Parse layered paper-config paths to recover task and method identifiers."""

    parts = config_path.parts
    if len(parts) >= 2 and parts[-2] == "tasks":
        task = config_path.stem
        return (task, None) if task in _DEFAULT_TASK_GROUP_BY_TASK else (None, None)
    if len(parts) >= 3 and parts[-3] == "methods":
        return None, config_path.stem
    if len(parts) >= 2 and parts[-2] in _DEFAULT_TASK_GROUP_BY_TASK:
        return parts[-2], config_path.stem
    return None, None


def _infer_task_and_method_from_run_directory(
    run_directory: Path,
) -> tuple[str, str] | None:
    """Fall back to the paper output directory convention ``task/method/seed_*``."""

    if len(run_directory.parts) < 3:
        return None
    method = run_directory.parts[-2]
    task = run_directory.parts[-3]
    if task in _DEFAULT_TASK_GROUP_BY_TASK:
        return task, method
    return None


def _reproduce_paper_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved paper-reproduction metadata stored in config."""

    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return {}
    reproduce_paper = config.get("reproduce_paper", {})
    if not isinstance(reproduce_paper, dict):
        return {}
    return reproduce_paper


def _resolve_task_group(task: str, task_group: Any) -> RunTaskGroup:
    """Resolve the task group from recorded metadata with a small fallback table."""

    if task_group in {"synthetic", "molecules"}:
        return task_group
    return _DEFAULT_TASK_GROUP_BY_TASK[task]


def _resolve_top_k(task: str, reproduce_paper: dict[str, Any]) -> int:
    """Resolve the paper top-k value from config metadata or defaults."""

    if "top_k" in reproduce_paper:
        return int(reproduce_paper["top_k"])
    return _DEFAULT_TOP_K_BY_TASK[task]


def _resolve_negate_score(task: str, reproduce_paper: dict[str, Any]) -> bool:
    """Resolve whether paper scores should negate the recorded target values."""

    if "negate_score" in reproduce_paper:
        return bool(reproduce_paper["negate_score"])
    return task in _DEFAULT_NEGATED_SCORE_TASKS


def _extract_total_active_learning_budget(config: dict[str, Any]) -> float:
    """Read the total budget directly from the resolved config."""

    budget = config.get("budget", {})
    if not isinstance(budget, dict):
        return 0.0
    available_budget = budget.get("available_budget")
    return float(available_budget) if available_budget is not None else 0.0


def _extract_oracle_costs(config: dict[str, Any]) -> dict[int, float] | None:
    """Read oracle fidelity costs directly from the resolved config when present."""

    oracle = config.get("oracle", {})
    if not isinstance(oracle, dict):
        return None
    costs = oracle.get("fidelity_costs")
    if not isinstance(costs, dict):
        return None
    return {int(fidelity): float(cost) for fidelity, cost in costs.items()}


def _resolve_initial_data_mode(manifest: dict[str, Any]) -> str:
    """Resolve initial-data mode from legacy metadata or the recorded config."""

    initial_data = manifest.get("initial_data", {})
    if "mode" in initial_data:
        return str(initial_data["mode"])

    config_initial_data = (
        manifest.get("config", {}).get("dataset", {}).get("initial_data", {})
    )
    if "fidelity_column" in config_initial_data:
        return "multi_fidelity"
    if config_initial_data:
        return "single_fidelity"
    return "unknown"


def compute_run_metrics(path_or_run: Path | RecordedRun) -> RunMetrics:
    """Compute paper-facing metrics for one recorded run."""

    run = (
        path_or_run
        if isinstance(path_or_run, RecordedRun)
        else load_recorded_run(path_or_run)
    )
    recorded_observations = _collect_recorded_observations(run)
    high_fidelity_observations = tuple(
        observation
        for observation in recorded_observations
        if observation.fidelity == run.descriptor.highest_fidelity
    )
    high_fidelity_rows = tuple(
        _to_high_fidelity_trace_row(run.descriptor, observation)
        for observation in high_fidelity_observations
    )
    checkpoints = _build_budget_checkpoints(run)

    if run.descriptor.task_group == "synthetic":
        return SyntheticRunMetrics(
            run=run.descriptor,
            checkpoint_rows=_build_synthetic_checkpoint_rows(
                descriptor=run.descriptor,
                high_fidelity_observations=high_fidelity_observations,
                checkpoints=checkpoints,
            ),
            high_fidelity_trace_rows=high_fidelity_rows,
        )

    return MoleculeRunMetrics(
        run=run.descriptor,
        checkpoint_rows=_build_molecule_checkpoint_rows(
            descriptor=run.descriptor,
            high_fidelity_observations=high_fidelity_observations,
            checkpoints=checkpoints,
        ),
        high_fidelity_trace_rows=high_fidelity_rows,
    )


def collect_run_metrics(paths: Path | Sequence[Path]) -> MetricCatalog:
    """Collect per-run metric tables from one or more output roots."""

    input_paths = [paths] if isinstance(paths, Path) else list(paths)
    run_directories = _discover_run_directories(input_paths)
    synthetic_runs: list[SyntheticRunMetrics] = []
    molecule_runs: list[MoleculeRunMetrics] = []
    for run_directory in run_directories:
        run_metrics = compute_run_metrics(run_directory)
        if isinstance(run_metrics, SyntheticRunMetrics):
            synthetic_runs.append(run_metrics)
        else:
            molecule_runs.append(run_metrics)

    return MetricCatalog(
        synthetic_runs=tuple(
            sorted(
                synthetic_runs,
                key=lambda metrics: (
                    metrics.run.task,
                    metrics.run.method,
                    metrics.run.seed,
                ),
            )
        ),
        molecule_runs=tuple(
            sorted(
                molecule_runs,
                key=lambda metrics: (
                    metrics.run.task,
                    metrics.run.method,
                    metrics.run.seed,
                ),
            )
        ),
    )


def _resolve_run_artifact_paths(path: Path) -> tuple[Path, Path]:
    """Resolve manifest and round-history paths from a directory or manifest path."""

    resolved = path.resolve()
    manifest_path = (
        resolved
        if resolved.name == RUN_MANIFEST_FILENAME
        else resolved / RUN_MANIFEST_FILENAME
    )
    round_history_path = manifest_path.parent / ROUND_HISTORY_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found at {manifest_path!s}.")
    if not round_history_path.is_file():
        raise FileNotFoundError(f"Round history not found at {round_history_path!s}.")
    return manifest_path, round_history_path


def _load_round_history(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON records from a round-history JSONL file."""

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _discover_run_directories(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Discover concrete run directories from output roots or manifest paths."""

    run_directories: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.is_file():
            manifest_path, _ = _resolve_run_artifact_paths(resolved)
            run_directories.add(manifest_path.parent)
            continue
        if (resolved / RUN_MANIFEST_FILENAME).is_file():
            run_directories.add(resolved)
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"Metrics input path does not exist: {resolved!s}.")
        run_directories.update(
            manifest_path.parent
            for manifest_path in resolved.rglob(RUN_MANIFEST_FILENAME)
        )
    return tuple(sorted(run_directories))


def _collect_recorded_observations(
    run: RecordedRun,
) -> tuple[_RecordedObservation, ...]:
    """Normalize all recorded observations from one run for metric computation."""

    observations: list[_RecordedObservation] = []
    initial_payload = run.manifest.get("initial_data", {})
    initial_observations = initial_payload.get("initial_observations", [])
    for observation_index, observation in enumerate(initial_observations):
        observations.append(
            _build_recorded_observation(
                descriptor=run.descriptor,
                observation=observation,
                round_index=0,
                cumulative_budget=0.0,
                normalized_budget=0.0,
                source="initial_data",
                observation_index=observation_index,
            )
        )

    running_index = len(observations)
    for round_payload in run.round_history:
        round_index = int(round_payload.get("round_index", round_payload.get("round")))
        cumulative_budget = float(round_payload["cumulative_cost"])
        normalized_budget = _normalize_budget_fraction(
            cumulative_budget,
            run.descriptor.total_active_learning_budget,
        )
        round_observations = round_payload.get(
            "new_observations", round_payload.get("observations", [])
        )
        for observation in round_observations:
            observations.append(
                _build_recorded_observation(
                    descriptor=run.descriptor,
                    observation=observation,
                    round_index=round_index,
                    cumulative_budget=cumulative_budget,
                    normalized_budget=normalized_budget,
                    source="active_learning_round",
                    observation_index=running_index,
                )
            )
            running_index += 1
    return tuple(observations)


def _build_recorded_observation(
    *,
    descriptor: RunDescriptor,
    observation: dict[str, Any],
    round_index: int,
    cumulative_budget: float,
    normalized_budget: float,
    source: Literal["initial_data", "active_learning_round"],
    observation_index: int,
) -> _RecordedObservation:
    """Normalize one serialized observation payload for metrics."""

    metadata = observation.get("metadata") or {}
    fidelity = _normalize_recorded_fidelity(
        recorded_fidelity=observation.get("fidelity"),
        descriptor=descriptor,
    )
    target_value = float(observation["y"])
    paper_score = _paper_score(
        negate_score=descriptor.negate_score,
        target_value=target_value,
    )
    identifier = _coerce_identifier(metadata.get("identifier"))
    molecule_input = (
        _extract_molecule_input(observation=observation)
        if descriptor.task_group == "molecules"
        else None
    )
    return _RecordedObservation(
        round_index=round_index,
        cumulative_budget=cumulative_budget,
        normalized_budget=normalized_budget,
        source=source,
        observation_index=observation_index,
        fidelity=fidelity,
        target_value=target_value,
        paper_score=paper_score,
        identifier=identifier,
        molecule_input=molecule_input,
    )


def _normalize_recorded_fidelity(
    *,
    recorded_fidelity: Any,
    descriptor: RunDescriptor,
) -> int:
    """Resolve stripped single-fidelity observations back to the highest fidelity."""

    if recorded_fidelity is None:
        if descriptor.initial_data_mode != "single_fidelity":
            raise ValueError(
                "Recorded multi-fidelity observations must include explicit fidelity values."
            )
        return descriptor.highest_fidelity
    return int(recorded_fidelity)


def _extract_molecule_input(*, observation: dict[str, Any]) -> str:
    """Extract the recorded molecule string used for RDKit fingerprinting."""

    x_value = observation.get("x")
    if isinstance(x_value, str):
        return x_value
    metadata = observation.get("metadata") or {}
    raw_value = metadata.get("raw")
    if isinstance(raw_value, str):
        return raw_value
    raise ValueError(
        "Molecule metrics require observations to carry a string SELFIES/SMILES value."
    )


def _coerce_identifier(value: Any) -> str | None:
    """Return a compact optional identifier for exported trace rows."""

    if value is None:
        return None
    return str(value)


def _paper_score(*, negate_score: bool, target_value: float) -> float:
    """Convert a raw recorded target value into the paper-facing score."""

    return -target_value if negate_score else target_value


def _normalize_molecule_representation(representation: Any) -> str:
    """Normalize the molecule string representation label."""

    if representation is None:
        return "selfies"
    return str(representation).strip().lower()


def _normalize_budget_fraction(
    cumulative_budget: float,
    total_active_learning_budget: float,
) -> float:
    """Return the active-learning budget fraction, guarding against zero totals."""

    if total_active_learning_budget <= 0:
        return 0.0
    return cumulative_budget / total_active_learning_budget


def _build_budget_checkpoints(run: RecordedRun) -> tuple[_BudgetCheckpoint, ...]:
    """Return initial plus per-round budget checkpoints for one run."""

    checkpoints = [
        _BudgetCheckpoint(round_index=0, cumulative_budget=0.0, normalized_budget=0.0)
    ]
    checkpoints.extend(
        _BudgetCheckpoint(
            round_index=int(round_payload["round_index"]),
            cumulative_budget=float(round_payload["cumulative_cost"]),
            normalized_budget=_normalize_budget_fraction(
                float(round_payload["cumulative_cost"]),
                run.descriptor.total_active_learning_budget,
            ),
        )
        for round_payload in run.round_history
    )
    return tuple(checkpoints)


def _build_synthetic_checkpoint_rows(
    *,
    descriptor: RunDescriptor,
    high_fidelity_observations: Sequence[_RecordedObservation],
    checkpoints: Sequence[_BudgetCheckpoint],
) -> tuple[SyntheticMetricRow, ...]:
    """Build synthetic metric rows using the paper's task-specific top-k."""

    top_k = descriptor.top_k
    return tuple(
        SyntheticMetricRow(
            task=descriptor.task,
            method=descriptor.method,
            seed=descriptor.seed,
            round_index=checkpoint.round_index,
            cumulative_budget=checkpoint.cumulative_budget,
            budget_fraction_of_total_active_learning_budget=checkpoint.normalized_budget,
            high_fidelity_observation_count=len(
                _observations_up_to_round(
                    high_fidelity_observations, checkpoint.round_index
                )
            ),
            top_k=top_k,
            top_k_observation_count=len(
                _top_k_observations(
                    _observations_up_to_round(
                        high_fidelity_observations, checkpoint.round_index
                    ),
                    top_k,
                )
            ),
            mean_top_k_score=_mean_score(
                _top_k_observations(
                    _observations_up_to_round(
                        high_fidelity_observations, checkpoint.round_index
                    ),
                    top_k,
                )
            ),
        )
        for checkpoint in checkpoints
    )


def _build_molecule_checkpoint_rows(
    *,
    descriptor: RunDescriptor,
    high_fidelity_observations: Sequence[_RecordedObservation],
    checkpoints: Sequence[_BudgetCheckpoint],
) -> tuple[MoleculeMetricRow, ...]:
    """Build molecule metric rows using the paper's top-100 and Tanimoto metrics."""

    rows: list[MoleculeMetricRow] = []
    for checkpoint in checkpoints:
        observed = _observations_up_to_round(
            high_fidelity_observations, checkpoint.round_index
        )
        top_observations = _top_k_observations(observed, descriptor.top_k)
        rows.append(
            MoleculeMetricRow(
                task=descriptor.task,
                method=descriptor.method,
                seed=descriptor.seed,
                round_index=checkpoint.round_index,
                cumulative_budget=checkpoint.cumulative_budget,
                budget_fraction_of_total_sf_gfn_budget=checkpoint.normalized_budget,
                high_fidelity_observation_count=len(observed),
                top_k=descriptor.top_k,
                top_k_observation_count=len(top_observations),
                mean_top_k_score=_mean_score(top_observations),
                mean_top_k_energy=_mean_target_value(top_observations),
                mean_pairwise_tanimoto_distance=_mean_pairwise_tanimoto_distance(
                    observations=top_observations,
                    representation=descriptor.molecule_representation or "selfies",
                ),
            )
        )
    return tuple(rows)


def _observations_up_to_round(
    observations: Sequence[_RecordedObservation],
    round_index: int,
) -> tuple[_RecordedObservation, ...]:
    """Return observations available by the end of one round."""

    return tuple(
        observation
        for observation in observations
        if observation.round_index <= round_index
    )


def _top_k_observations(
    observations: Sequence[_RecordedObservation],
    top_k: int,
) -> tuple[_RecordedObservation, ...]:
    """Return the paper top-k set from the available highest-fidelity observations."""

    ranked = sorted(
        observations, key=lambda observation: observation.paper_score, reverse=True
    )
    return tuple(ranked[:top_k])


def _mean_score(observations: Sequence[_RecordedObservation]) -> float | None:
    """Return the mean paper score for a possibly short top-k prefix."""

    if not observations:
        return None
    return fmean(observation.paper_score for observation in observations)


def _mean_target_value(observations: Sequence[_RecordedObservation]) -> float | None:
    """Return the mean raw recorded target value for a possibly short top-k prefix."""

    if not observations:
        return None
    return fmean(observation.target_value for observation in observations)


def _mean_pairwise_tanimoto_distance(
    *,
    observations: Sequence[_RecordedObservation],
    representation: str,
) -> float | None:
    """Return the mean pairwise Tanimoto distance for the current top-k molecules."""

    if len(observations) < 2:
        return None

    try:
        import selfies as sf
        from rdkit import Chem, DataStructs, rdBase
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:  # pragma: no cover - depends on optional extras
        raise missing_molecules_dependency_error("Molecule metrics", error) from error

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for observation in observations:
        molecule_input = observation.molecule_input
        if molecule_input is None:
            raise ValueError("Molecule metrics require a recorded molecule string.")
        smiles = molecule_input
        if representation == "selfies":
            try:
                smiles = sf.decoder(molecule_input)
            except sf.DecoderError as error:
                raise ValueError(
                    f"Invalid SELFIES string {molecule_input!r} in recorded metrics input."
                ) from error
            if smiles == "":
                raise ValueError(
                    "Recorded SELFIES metric input decoded to an empty SMILES string."
                )
        elif representation != "smiles":
            raise ValueError(
                f"Unsupported molecule representation for metrics: {representation!r}."
            )

        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(
                f"Recorded molecule metrics input could not be parsed: {smiles!r}."
            )
        fingerprints.append(generator.GetFingerprint(molecule))

    distances = [
        1.0 - float(DataStructs.TanimotoSimilarity(left, right))
        for left, right in combinations(fingerprints, 2)
    ]
    return fmean(distances) if distances else None


def _to_high_fidelity_trace_row(
    descriptor: RunDescriptor,
    observation: _RecordedObservation,
) -> HighFidelityTraceRow:
    """Convert an internal highest-fidelity record to an exported trace row."""

    return HighFidelityTraceRow(
        task_group=descriptor.task_group,
        task=descriptor.task,
        method=descriptor.method,
        seed=descriptor.seed,
        round_index=observation.round_index,
        cumulative_budget=observation.cumulative_budget,
        budget_fraction_of_total_active_learning_budget=observation.normalized_budget,
        source=observation.source,
        observation_index=observation.observation_index,
        fidelity=observation.fidelity,
        target_value=observation.target_value,
        paper_score=observation.paper_score,
        identifier=observation.identifier,
    )

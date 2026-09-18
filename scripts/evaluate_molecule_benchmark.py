"""Evaluate molecule benchmark runs at xTB fidelity three."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from activelearning.utils.types import Candidate
from activelearning_molecules.benchmark_metrics import (
    canonicalize_smiles,
    finite_fidelity_three_scores,
    top_k_score_and_diversity,
)
from activelearning_molecules.oracles.config import XTBIPEAOracleConfig


CACHE_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
DEFAULT_TOP_K = 100
DEFAULT_BATCH_SIZE = 32
OracleFactory = Callable[[Mapping[str, Any]], Any]


def oracle_parameter_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Return a stable digest for a complete fidelity-3 oracle definition."""
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_run_directories(base_dir: Path) -> list[Path]:
    """Find run directories containing both benchmark manifest artifacts."""
    return sorted(
        path.parent
        for path in base_dir.rglob("run_manifest.json")
        if (path.parent / "round_history.jsonl").is_file()
    )


def evaluate_benchmark(
    base_dir: Path,
    output_dir: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retry_failures: bool = False,
    oracle_factory: OracleFactory | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all discovered runs and write versioned JSON/CSV artifacts."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    run_dirs = discover_run_directories(base_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No benchmark runs found under {base_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _EvaluationCache(
        output_dir / "fidelity3_cache.jsonl",
        retry_failures=retry_failures,
    )
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(
            _evaluate_run(
                run_dir,
                cache=cache,
                top_k=top_k,
                batch_size=batch_size,
                oracle_factory=oracle_factory,
            )
        )
    cache.flush()
    _write_metrics(output_dir, rows)
    return rows


def _evaluate_run(
    run_dir: Path,
    *,
    cache: "_EvaluationCache",
    top_k: int,
    batch_size: int,
    oracle_factory: OracleFactory | None,
) -> list[dict[str, Any]]:
    """Evaluate one run checkpoint by checkpoint."""
    manifest = _load_json(run_dir / "run_manifest.json")
    config = manifest.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"Run manifest {run_dir} has no mapping config.")
    oracle_parameters = _fidelity_three_oracle_parameters(config)
    oracle_fingerprint = oracle_parameter_fingerprint(oracle_parameters)
    oracle = (
        oracle_factory(oracle_parameters)
        if oracle_factory is not None
        else _build_oracle(oracle_parameters)
    )
    task = _run_value(config, "benchmark", "task") or _run_value(
        config, "oracle", "task"
    )
    method = _run_value(config, "benchmark", "method") or _method_from_path(run_dir)
    seed = _run_value(config, "runtime", "seed") or _seed_from_path(run_dir)
    if task is None or method is None or seed is None:
        raise ValueError(f"Could not identify task/method/seed for {run_dir}.")

    initial = _initial_observations(manifest)
    history_records = _load_history(run_dir / "round_history.jsonl")
    all_observations = [
        *initial,
        *[
            observation
            for record in history_records
            for observation in record["valid_observations"]
        ],
    ]
    recorded_scores = finite_fidelity_three_scores(all_observations)
    cumulative = list(initial)
    rescored_molecules: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in history_records:
        valid_observations = record["valid_observations"]
        cumulative.extend(valid_observations)
        normalized = _deduplicate_observations(cumulative)
        rescored_molecules.update(
            _evaluate_missing(
                normalized,
                recorded_scores=recorded_scores,
                task=str(task),
                oracle=oracle,
                cache=cache,
                oracle_fingerprint=oracle_fingerprint,
                batch_size=batch_size,
            )
        )
        scores = cache.successful_scores(
            task=str(task),
            oracle_fingerprint=oracle_fingerprint,
            smiles=normalized,
        )
        metric_values = top_k_score_and_diversity(scores, k=top_k)
        unique_count = len(normalized)
        fidelity_three_count = len(scores)
        rows.append(
            {
                "schema_version": METRICS_SCHEMA_VERSION,
                "run_dir": str(run_dir),
                "task": str(task),
                "method": str(method),
                "seed": int(seed),
                "round": int(record["round_index"]),
                "cumulative_acquisition_cost": float(record["cumulative_cost"]),
                "unique_molecule_count": unique_count,
                "fidelity_3_count": fidelity_three_count,
                "fidelity_3_fraction": (
                    fidelity_three_count / unique_count if unique_count else 0.0
                ),
                "mean_top_100_score": metric_values["mean_score"],
                "mean_top_100_diversity": metric_values["diversity"],
                "top_100_count": metric_values["top_k_count"],
                "additional_fidelity_3_rescoring_count": len(rescored_molecules),
                "additional_fidelity_3_rescoring_nominal_cost": (
                    len(rescored_molecules) * _fidelity_three_cost(oracle_parameters)
                ),
            }
        )
    return rows


def _evaluate_missing(
    normalized: Mapping[str, list[Mapping[str, Any]]],
    *,
    recorded_scores: Mapping[str, float],
    task: str,
    oracle: Any,
    cache: "_EvaluationCache",
    oracle_fingerprint: str,
    batch_size: int,
) -> set[str]:
    """Evaluate molecules without a cache entry, preserving recorded f3 values."""
    pending: list[str] = []
    for smiles in normalized:
        if smiles in recorded_scores:
            cache.record_success(
                task=task,
                smiles=smiles,
                oracle_fingerprint=oracle_fingerprint,
                value=recorded_scores[smiles],
                source="run",
            )
        elif cache.should_evaluate(task, smiles, oracle_fingerprint):
            pending.append(smiles)

    evaluated: set[str] = set()
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        evaluated.update(batch)
        candidates = [Candidate(x=smiles, fidelity=3) for smiles in batch]
        try:
            observations = list(oracle.query(candidates))
            if len(observations) != len(batch):
                raise RuntimeError(
                    f"Oracle returned {len(observations)} observations for "
                    f"{len(batch)} candidates."
                )
            for smiles, observation in zip(batch, observations, strict=True):
                value = _finite_value(observation.y)
                if value is None:
                    cache.record_failure(
                        task=task,
                        smiles=smiles,
                        oracle_fingerprint=oracle_fingerprint,
                        error="Oracle returned a non-finite fidelity-3 value.",
                    )
                else:
                    cache.record_success(
                        task=task,
                        smiles=smiles,
                        oracle_fingerprint=oracle_fingerprint,
                        value=value,
                        source="oracle",
                    )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            for smiles in batch:
                cache.record_failure(
                    task=task,
                    smiles=smiles,
                    oracle_fingerprint=oracle_fingerprint,
                    error=message,
                )
        cache.flush()
    return evaluated


def _load_history(path: Path) -> list[dict[str, Any]]:
    """Load and validate all round records from one JSONL history."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"Malformed round record at {path}:{line_number}.")
            observations = record.get("valid_observations", [])
            if not isinstance(observations, list):
                raise ValueError(
                    f"Malformed valid_observations at {path}:{line_number}."
                )
            try:
                round_index = int(record["round_index"])
                cumulative_cost = float(record["cumulative_cost"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Malformed round metadata at {path}:{line_number}."
                ) from error
            records.append(
                {
                    "round_index": round_index,
                    "cumulative_cost": cumulative_cost,
                    "valid_observations": observations,
                }
            )
    return records


def _deduplicate_observations(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Normalize observations and retain all values for each first-seen molecule."""
    normalized: dict[str, list[Mapping[str, Any]]] = {}
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        smiles = canonicalize_smiles(observation.get("x"))
        if smiles is None:
            continue
        normalized.setdefault(smiles, []).append(observation)
    return normalized


class _EvaluationCache:
    """Append-only cache with conflict validation for fidelity-3 values."""

    def __init__(self, path: Path, *, retry_failures: bool) -> None:
        self.path = path
        self.retry_failures = retry_failures
        self.entries: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self._load()

    def should_evaluate(self, task: str, smiles: str, fingerprint: str) -> bool:
        """Return whether this key needs a new oracle request."""
        entry = self.entries.get((task, smiles, fingerprint))
        return entry is None or (self.retry_failures and entry["status"] == "failure")

    def record_success(
        self,
        *,
        task: str,
        smiles: str,
        oracle_fingerprint: str,
        value: float,
        source: str,
    ) -> None:
        """Record a finite value, rejecting conflicting finite cache values."""
        key = (task, smiles, oracle_fingerprint)
        current = self.entries.get(key)
        if current is not None and current["status"] == "success":
            if float(current["value"]) != value:
                raise ValueError(f"Conflicting cache values for {key!r}.")
            return
        record = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "task": task,
            "smiles": smiles,
            "oracle_fingerprint": oracle_fingerprint,
            "status": "success",
            "value": value,
            "source": source,
            "timestamp": time.time(),
        }
        self.entries[key] = record
        self.pending.append(record)

    def record_failure(
        self,
        *,
        task: str,
        smiles: str,
        oracle_fingerprint: str,
        error: str,
    ) -> None:
        """Record an explicit failed oracle evaluation."""
        key = (task, smiles, oracle_fingerprint)
        current = self.entries.get(key)
        if current is not None and current["status"] == "success":
            return
        record = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "task": task,
            "smiles": smiles,
            "oracle_fingerprint": oracle_fingerprint,
            "status": "failure",
            "error": error,
            "timestamp": time.time(),
        }
        self.entries[key] = record
        self.pending.append(record)

    def successful_scores(
        self,
        *,
        task: str,
        oracle_fingerprint: str,
        smiles: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> dict[str, float]:
        """Return successful cache values for the current molecule set."""
        values: dict[str, float] = {}
        for molecule in smiles:
            entry = self.entries.get((task, molecule, oracle_fingerprint))
            if entry is not None and entry["status"] == "success":
                values[molecule] = float(entry["value"])
        return values

    def flush(self) -> None:
        """Append pending records to disk in one durable write."""
        if not self.pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in self.pending:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.pending.clear()

    def _load(self) -> None:
        """Load and validate all existing cache records."""
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema_version") != CACHE_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported cache schema at {self.path}:{line_number}."
                    )
                key = (
                    str(record["task"]),
                    str(record["smiles"]),
                    str(record["oracle_fingerprint"]),
                )
                status = record.get("status")
                if status not in {"success", "failure"}:
                    raise ValueError(
                        f"Invalid cache status at {self.path}:{line_number}."
                    )
                if status == "success" and _finite_value(record.get("value")) is None:
                    raise ValueError(
                        f"Non-finite cache value at {self.path}:{line_number}."
                    )
                previous = self.entries.get(key)
                if previous is not None and previous["status"] == "success":
                    if status == "success" and previous["value"] != record["value"]:
                        raise ValueError(f"Conflicting cache values for {key!r}.")
                    if status == "failure":
                        continue
                self.entries[key] = record


def _fidelity_three_oracle_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the exact SMILES/fidelity-3 oracle parameters from a run config."""
    raw_oracle = config.get("oracle")
    if not isinstance(raw_oracle, Mapping):
        raise ValueError("Run config is missing an oracle mapping.")
    oracle_config = XTBIPEAOracleConfig.model_validate(dict(raw_oracle))
    if 3 not in oracle_config.fidelity_costs:
        raise ValueError("Run oracle config must declare fidelity 3.")
    parameters = oracle_config.model_dump(mode="json")
    cost = float(oracle_config.fidelity_costs[3])
    parameters["fidelity_costs"] = {"3": cost}
    parameters["fidelity_confidences"] = None
    parameters["per_fidelity_num_conformers"] = {"3": 4}
    parameters["num_conformers"] = 4
    parameters["mol_repr"] = "smiles"
    parameters["log_molecule_visualizations"] = False
    parameters["molecule_visualization_limit"] = 1
    return parameters


def _fidelity_three_cost(parameters: Mapping[str, Any]) -> float:
    """Read the fidelity-three cost from an oracle parameter mapping."""
    costs = parameters.get("fidelity_costs")
    if not isinstance(costs, Mapping):
        raise ValueError("Oracle parameters have no fidelity costs.")
    for key, value in costs.items():
        if int(key) == 3:
            return float(value)
    raise ValueError("Oracle parameters have no fidelity-three cost.")


def _build_oracle(parameters: Mapping[str, Any]) -> Any:
    """Build the molecular oracle from serialized configuration."""
    config = XTBIPEAOracleConfig.model_validate(dict(parameters))
    return config.build()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a useful path-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest {path} must contain an object.")
    return value


def _initial_observations(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract initial observations from a run manifest."""
    initial_data = manifest.get("initial_data", {})
    if not isinstance(initial_data, Mapping):
        return []
    observations = initial_data.get("initial_observations", [])
    return [value for value in observations if isinstance(value, Mapping)]


def _run_value(config: Mapping[str, Any], section: str, key: str) -> Any:
    """Read a nested run metadata value when present."""
    value = config.get(section)
    return value.get(key) if isinstance(value, Mapping) else None


def _method_from_path(run_dir: Path) -> str | None:
    """Infer a method from the conventional output path."""
    return run_dir.parent.name if run_dir.name.startswith("seed_") else None


def _seed_from_path(run_dir: Path) -> int | None:
    """Infer a seed from a conventional ``seed_<n>`` directory."""
    if not run_dir.name.startswith("seed_"):
        return None
    try:
        return int(run_dir.name.removeprefix("seed_"))
    except ValueError:
        return None


def _finite_value(value: Any) -> float | None:
    """Convert a scalar to a finite float, otherwise return ``None``."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _write_metrics(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSON and CSV metric artifacts."""
    json_path = output_dir / "molecule_metrics.json"
    json_path.write_text(
        json.dumps(
            {"schema_version": METRICS_SCHEMA_VERSION, "rows": list(rows)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "molecule_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    """Build the evaluator CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--retry-failures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate benchmark runs from the command line."""
    args = _build_parser().parse_args(argv)
    output_dir = args.output or args.base_dir / "metrics"
    evaluate_benchmark(
        args.base_dir.expanduser().resolve(),
        output_dir.expanduser().resolve(),
        top_k=args.top_k,
        batch_size=args.batch_size,
        retry_failures=args.retry_failures,
    )


if __name__ == "__main__":
    main()

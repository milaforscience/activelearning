"""Durable monitoring sinks for structured active-learning run outputs."""

import csv
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from activelearning.monitoring.keys import validate_log_key
from activelearning.utils.types import Candidate, Observation


@dataclass(frozen=True)
class RoundRecord:
    """Immutable persistence-ready data from one completed active-learning round.

    The record groups round identity, observations before and after the round,
    sampled and selected candidates, oracle results, and budget state. Its
    ``metrics``, ``profiling``, and ``diagnostics`` mappings contain core round
    values, operational timings, and optional diagnostic enrichment,
    respectively. Raw selector and acquisition score arrays are temporary data
    used to calculate round diagnostics; they are deliberately not persisted.
    """

    round_index: int
    observations_before: Sequence[Observation]
    observations_after: Sequence[Observation]
    sampled_candidates: Sequence[Candidate]
    selected_candidates: Sequence[Candidate]
    selected_costs: Sequence[float]
    queried_observations: Sequence[Observation]
    valid_observations: Sequence[Observation]
    round_budget: float
    initial_budget: float
    cumulative_cost: float
    remaining_budget: float
    metrics: Mapping[str, int | float]
    profiling: Mapping[str, float]
    diagnostics: Mapping[str, int | float]


class RunWriter(ABC):
    """Durable structured-output sink for active-learning runs.

    The active-learning loop owns this interface and supplies one completed
    :class:`RoundRecord` per round. Writers persist reproducibility metadata,
    records, and optional figures independently of live :class:`Logger`
    telemetry; components never write directly to a run writer.
    """

    @abstractmethod
    def start_run(self, metadata: dict[str, Any]) -> None:
        """Write run metadata before the first active-learning round."""

    @abstractmethod
    def record_round(
        self,
        record: RoundRecord,
        figures: Mapping[str, Figure] | None = None,
    ) -> None:
        """Persist one completed round and optional diagnostic figures.

        Parameters
        ----------
        record : RoundRecord
            Immutable, persistence-ready data for one completed round.
        figures : Mapping[str, Figure] | None, optional
            Diagnostic figures keyed by their stable metric namespace.
        """

    @abstractmethod
    def end_run(self, summary: dict[str, Any]) -> None:
        """Write the final run summary."""


class JSONLinesRunWriter(RunWriter):
    """Write durable run outputs as JSON, JSONL, CSV, and local figure artifacts.

    The output directory contains ``run_manifest.json``, ``round_history.jsonl``,
    ``run_summary.json``, and ``experiment_log.csv``. Diagnostic figures are
    written below ``artifacts/`` using their monitoring namespaces.
    """

    def __init__(
        self,
        output_dir: Path,
        write_samples: bool = True,
        write_config: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.write_samples = write_samples
        self.write_config = write_config
        self.metadata = metadata or {}
        self._manifest: dict[str, Any] | None = None
        self._method: str | None = None
        self._seed: int | str | None = None
        self._best_objective_value: float | None = None
        self._experiment_rows: list[dict[str, int | float | str | None]] = []
        self._next_experiment_row_index = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rounds_path = self.output_dir / "round_history.jsonl"
        self._started = False

    def start_run(self, metadata: dict[str, Any]) -> None:
        """Write run metadata to disk."""
        self._rounds_path.write_text("", encoding="utf-8")
        self._manifest = _deep_merge(self.metadata, metadata)
        self._method = _resolve_method(self._manifest, self.output_dir)
        self._seed = _resolve_seed(self._manifest)
        persisted_manifest = dict(self._manifest)
        if not self.write_config:
            persisted_manifest.pop("config", None)
        self._write_json("run_manifest.json", persisted_manifest)
        self._write_experiment_log()
        self._started = True

    def record_round(
        self,
        record: RoundRecord,
        figures: Mapping[str, Figure] | None = None,
    ) -> None:
        """Append one round record to the JSONL history."""
        if not self._started:
            raise RuntimeError("Call start_run() before recording rounds.")
        for values in (record.metrics, record.profiling, record.diagnostics):
            for key in values:
                validate_log_key(key)
        for key in figures or {}:
            validate_log_key(key)

        artifact_paths = self._write_figures(
            round_index=record.round_index,
            figures=figures or {},
        )
        payload: dict[str, Any] = {
            "round_index": record.round_index,
            "selected_candidates": _jsonable(record.selected_candidates),
            "selected_costs": _jsonable(record.selected_costs),
            "queried_observations": _jsonable(record.queried_observations),
            "valid_observations": _jsonable(record.valid_observations),
            "round_budget": float(record.round_budget),
            "round_cost": float(sum(record.selected_costs)),
            "initial_budget": float(record.initial_budget),
            "cumulative_cost": float(record.cumulative_cost),
            "remaining_budget": float(record.remaining_budget),
            "metrics": _jsonable(dict(record.metrics)),
            "profiling": _jsonable(dict(record.profiling)),
            "diagnostics": _jsonable(dict(record.diagnostics)),
            "artifacts": artifact_paths,
        }
        if self.write_samples:
            payload["sampled_candidates"] = _jsonable(record.sampled_candidates)

        with self._rounds_path.open("a", encoding="utf-8") as rounds_handle:
            rounds_handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._append_experiment_row(
            round_index=record.round_index,
            observations=record.valid_observations,
            cumulative_cost=record.cumulative_cost,
        )

    def end_run(self, summary: dict[str, Any]) -> None:
        """Write the final summary."""
        self._write_json("run_summary.json", summary)
        self._write_experiment_log()

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        """Write one JSON artifact in a stable format."""
        path = self.output_dir / filename
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_figures(
        self,
        *,
        round_index: int,
        figures: Mapping[str, Figure],
    ) -> dict[str, str]:
        """Persist diagnostic figures under deterministic component-round paths."""
        artifact_paths: dict[str, str] = {}
        for key, figure in figures.items():
            relative_path = _figure_relative_path(key, round_index)
            path = self.output_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=150, bbox_inches="tight")
            artifact_paths[key] = relative_path.as_posix()
        return artifact_paths

    def _append_experiment_row(
        self,
        *,
        round_index: int,
        observations: Sequence[Observation],
        cumulative_cost: float,
    ) -> None:
        """Append the best-so-far objective trajectory for one round."""
        round_best_objective_value = _best_objective_value(observations)
        if round_best_objective_value is not None:
            if self._best_objective_value is None:
                self._best_objective_value = round_best_objective_value
            else:
                self._best_objective_value = max(
                    self._best_objective_value,
                    round_best_objective_value,
                )

        self._experiment_rows.append(
            {
                "index": self._next_experiment_row_index,
                "method": self._method,
                "seed": self._seed,
                "objective value": self._best_objective_value,
                "cost": float(cumulative_cost),
                "round": round_index - 1,
            }
        )
        self._next_experiment_row_index += 1
        self._write_experiment_log()

    def _write_experiment_log(self) -> None:
        """Write the per-round objective CSV used by downstream tooling."""
        path = self.output_dir / "experiment_log.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "method",
                    "seed",
                    "objective value",
                    "cost",
                    "round",
                ],
            )
            writer.writeheader()
            writer.writerows(self._experiment_rows)


class JSONLinesRunWriterConfig(BaseModel):
    """Configure durable JSON, JSONL, CSV, and artifact outputs for one run."""

    type: Literal["JSONLinesRunWriter"] = "JSONLinesRunWriter"
    output_dir: Path
    write_samples: bool = True
    write_config: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def build(self, metadata: dict[str, Any] | None = None) -> RunWriter:
        """Build the configured run writer.

        Parameters
        ----------
        metadata : dict[str, Any] | None, optional
            Additional metadata merged into the configured static metadata
            before the writer is constructed.

        Returns
        -------
        RunWriter
            Materialized JSON-lines run writer.
        """
        return JSONLinesRunWriter(
            output_dir=self.output_dir,
            write_samples=self.write_samples,
            write_config=self.write_config,
            metadata=_deep_merge(self.metadata, metadata or {}),
        )


RunWriterConfig = JSONLinesRunWriterConfig


def _figure_relative_path(key: str, round_index: int) -> Path:
    """Return the local artifact path for a stable diagnostic figure key."""
    segments = key.split("/")
    if len(segments) < 2 or any(
        not _is_safe_path_segment(segment) for segment in segments
    ):
        raise ValueError(f"Invalid diagnostic figure key: {key!r}")
    return Path("artifacts", *segments[:-1], f"round_{round_index:04d}") / (
        f"{segments[-1]}.png"
    )


def _is_safe_path_segment(segment: str) -> bool:
    """Return whether one figure-key path segment is safe for local storage."""
    return bool(segment) and all(
        character.isalnum() or character in "_-." for character in segment
    )


def _jsonable(value: Any) -> Any:
    """Convert common payloads into JSON-serializable structures."""
    if isinstance(value, (Candidate, Observation)):
        return _jsonable(asdict(value))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "detach") and callable(value.detach):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and callable(value.tolist):
        return _jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_seed(manifest: dict[str, Any]) -> int | str | None:
    """Extract the run seed from the manifest when available."""
    config = manifest.get("config", {})
    run_metadata = manifest.get("run", {})
    return run_metadata.get("seed", config.get("runtime", {}).get("seed"))


def _resolve_method(manifest: dict[str, Any], output_dir: Path) -> str | None:
    """Resolve the optional method label for CSV output."""
    run_metadata = manifest.get("run", {})
    raw_method = run_metadata.get("method")
    if raw_method is None and output_dir.parent != output_dir:
        raw_method = output_dir.parent.name
    if raw_method is None:
        return None
    return str(raw_method)


def _best_objective_value(observations: Sequence[Observation]) -> float | None:
    """Return the best finite scalar objective value observed in a round."""
    objective_values = [
        _coerce_scalar_float(observation.y) for observation in observations
    ]
    if not objective_values:
        return None
    finite_objective_values = [
        objective_value
        for objective_value in objective_values
        if math.isfinite(objective_value)
    ]
    if not finite_objective_values:
        return None
    return max(finite_objective_values)


def _coerce_scalar_float(value: Any) -> float:
    """Convert a scalar-like objective value into a float for CSV output."""
    jsonable_value = _jsonable(value)
    if isinstance(jsonable_value, bool):
        raise TypeError(
            "Boolean objective values cannot be written to experiment_log.csv."
        )
    if isinstance(jsonable_value, int | float):
        return float(jsonable_value)
    if isinstance(jsonable_value, list | tuple) and len(jsonable_value) == 1:
        return _coerce_scalar_float(jsonable_value[0])
    raise TypeError(
        "experiment_log.csv only supports scalar objective values, "
        f"received {type(jsonable_value).__name__}."
    )

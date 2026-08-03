"""Structured run writers for persisting active-learning artifacts."""

import csv
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from activelearning.utils.types import Candidate, Observation


class RunWriter(ABC):
    """Interface for structured active-learning run outputs."""

    @abstractmethod
    def start_run(self, metadata: dict[str, Any]) -> None:
        """Write run metadata before the first active-learning round."""

    @abstractmethod
    def record_round(
        self,
        *,
        round_index: int,
        sampled_candidates: Sequence[Candidate],
        sampled_scores: Sequence[float],
        selected_candidates: Sequence[Candidate],
        selected_scores: Sequence[float],
        selected_costs: Sequence[float],
        observations: Sequence[Observation],
        cumulative_cost: float,
        remaining_budget: float,
    ) -> None:
        """Write structured outputs for one completed round."""

    @abstractmethod
    def end_run(self, summary: dict[str, Any]) -> None:
        """Write the final run summary."""


class JSONLinesRunWriter(RunWriter):
    """Write run metadata and per-round artifacts as JSON and JSONL files."""

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
        self._rounds_handle = self._rounds_path.open("w", encoding="utf-8")

    def start_run(self, metadata: dict[str, Any]) -> None:
        """Write run metadata to disk."""
        self._manifest = _deep_merge(self.metadata, metadata)
        self._method = _resolve_method(self._manifest, self.output_dir)
        self._seed = _resolve_seed(self._manifest)
        if self.write_config:
            self._write_json("run_manifest.json", self._manifest)
        self._write_experiment_log()

    def record_round(
        self,
        *,
        round_index: int,
        sampled_candidates: Sequence[Candidate],
        sampled_scores: Sequence[float],
        selected_candidates: Sequence[Candidate],
        selected_scores: Sequence[float],
        selected_costs: Sequence[float],
        observations: Sequence[Observation],
        cumulative_cost: float,
        remaining_budget: float,
    ) -> None:
        """Append one round record to the JSONL history."""
        record: dict[str, Any] = {
            "round": round_index,
            "round_index": round_index,
            "selected_candidates": _jsonable(selected_candidates),
            "selected_scores": _jsonable(selected_scores),
            "selected_costs": _jsonable(selected_costs),
            "observations": _jsonable(observations),
            "new_observations": _jsonable(observations),
            "round_cost": float(sum(selected_costs)),
            "cumulative_cost": float(cumulative_cost),
            "remaining_budget": float(remaining_budget),
        }
        if self.write_samples:
            record["sampled_candidates"] = _jsonable(sampled_candidates)
            record["sampled_scores"] = _jsonable(sampled_scores)

        self._rounds_handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._rounds_handle.flush()
        self._append_experiment_row(
            round_index=round_index,
            observations=observations,
            cumulative_cost=cumulative_cost,
        )

    def end_run(self, summary: dict[str, Any]) -> None:
        """Write the final summary and close the round-history file."""
        self._write_json("run_summary.json", summary)
        self._write_experiment_log()
        self._rounds_handle.close()

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        """Write one JSON artifact in a stable format."""
        path = self.output_dir / filename
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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
    """Configuration for JSON and JSONL active-learning run outputs."""

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


def _jsonable(value: Any) -> Any:
    """Convert common payloads into JSON-serializable structures."""
    if isinstance(value, Candidate | Observation):
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
    """Extract the run method name for CSV output."""
    config = manifest.get("config", {})
    run_metadata = manifest.get("run", {})
    reproduce_paper = config.get("reproduce_paper", {})
    raw_method = run_metadata.get("method", reproduce_paper.get("method"))
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

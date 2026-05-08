import csv
import json
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
    """Write run metadata and per-round artifacts as JSON/JSONL files."""

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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rounds_path = self.output_dir / "round_history.jsonl"
        self._rounds_handle = self._rounds_path.open("w", encoding="utf-8")

    def start_run(self, metadata: dict[str, Any]) -> None:
        """Write run metadata to disk."""

        self._manifest = _deep_merge(self.metadata, metadata)
        if self.write_config:
            self._write_json("run_manifest.json", self._manifest)

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

    def end_run(self, summary: dict[str, Any]) -> None:
        """Write final summary and close the round-history file."""

        self._write_json("run_summary.json", summary)
        if self._manifest is not None:
            self._write_experiment_log(self._manifest, summary)
        self._rounds_handle.close()

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.output_dir / filename
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_experiment_log(
        self,
        manifest: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        """Write a flat single-row CSV summary for quick experiment inspection."""

        row = _build_experiment_log_row(
            manifest=_jsonable(manifest),
            summary=_jsonable(summary),
            output_dir=self.output_dir,
        )
        path = self.output_dir / "experiment_log.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)


class JSONLinesRunWriterConfig(BaseModel):
    """Configuration for JSON/JSONL active-learning run outputs."""

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
        """

        return JSONLinesRunWriter(
            output_dir=self.output_dir,
            write_samples=self.write_samples,
            write_config=self.write_config,
            metadata=_deep_merge(self.metadata, metadata or {}),
        )


RunWriterConfig = JSONLinesRunWriterConfig


def _jsonable(value: Any) -> Any:
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
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_experiment_log_row(
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Flatten the most useful run metadata into a CSV-friendly row."""

    config = manifest.get("config", {})
    cli_metadata = manifest.get("cli", {})
    provenance = manifest.get("provenance", {})
    run_metadata = manifest.get("run", {})
    dataset = config.get("dataset", {})
    initial_data = dataset.get("initial_data", {})
    budget = config.get("budget", {})
    schedule = budget.get("schedule", {})
    logger = config.get("logger", {})
    input_files = provenance.get("input_files", [])
    inferred_identity = _infer_identity_from_config_files(
        cli_metadata.get("config_files", [])
    )
    input_file_md5s = {
        str(item["path"]): str(item["md5"])
        for item in input_files
        if isinstance(item, dict) and "path" in item and "md5" in item
    }

    return {
        "run_directory": str(output_dir),
        "task_group": _string_or_empty(
            run_metadata.get("task_group", inferred_identity.get("task_group"))
        ),
        "task": _string_or_empty(
            run_metadata.get("task", inferred_identity.get("task"))
        ),
        "method": _string_or_empty(
            run_metadata.get("method", inferred_identity.get("method"))
        ),
        "seed": _string_or_empty(
            run_metadata.get("seed", config.get("runtime", {}).get("seed"))
        ),
        "logger_project_name": _string_or_empty(logger.get("project_name")),
        "logger_run_name": _string_or_empty(logger.get("run_name")),
        "dataset_type": _string_or_empty(dataset.get("type")),
        "dataset_initial_data_path": _string_or_empty(initial_data.get("path")),
        "surrogate_type": _string_or_empty(config.get("surrogate", {}).get("type")),
        "acquisition_type": _string_or_empty(config.get("acquisition", {}).get("type")),
        "sampler_type": _string_or_empty(config.get("sampler", {}).get("type")),
        "selector_type": _string_or_empty(config.get("selector", {}).get("type")),
        "oracle_type": _string_or_empty(config.get("oracle", {}).get("type")),
        "available_budget": _string_or_empty(budget.get("available_budget")),
        "budget_schedule_type": _string_or_empty(schedule.get("type")),
        "budget_schedule_value": _string_or_empty(schedule.get("value")),
        "git_commit_sha": _string_or_empty(provenance.get("git_commit_sha")),
        "resolved_config_md5": _string_or_empty(provenance.get("resolved_config_md5")),
        "config_files": ";".join(cli_metadata.get("config_files", [])),
        "config_overrides": ";".join(cli_metadata.get("config_overrides", [])),
        "input_files": ";".join(input_file_md5s),
        "input_file_md5s": json.dumps(input_file_md5s, sort_keys=True),
        "reproduce_command": _string_or_empty(cli_metadata.get("command")),
        "num_rounds": _string_or_empty(summary.get("num_rounds")),
        "total_cost": _string_or_empty(summary.get("total_cost")),
        "budget_remaining": _string_or_empty(summary.get("budget_remaining")),
    }


def _string_or_empty(value: Any) -> str:
    """Convert optional values into CSV-friendly strings."""

    return "" if value is None else str(value)


def _infer_identity_from_config_files(config_files: Sequence[str]) -> dict[str, str]:
    """Infer paper experiment identity from recorded config paths when possible."""

    for config_file in config_files:
        parts = Path(config_file).parts
        try:
            root_index = parts.index("reproduce_paper")
        except ValueError:
            continue
        if root_index + 2 >= len(parts):
            continue
        task = parts[root_index + 1]
        method = Path(parts[root_index + 2]).stem
        task_group = "molecules" if task.startswith("molecules_") else "synthetic"
        return {"task_group": task_group, "task": task, "method": method}
    return {}

"""Entry point for the active learning loop.

Usage
-----
    python -m activelearning <config.yaml> [<config2.yaml> ...] [key=value ...]

Positional arguments
--------------------
args
    One or more YAML configuration file paths, optionally followed by
    OmegaConf dotlist overrides. Arguments containing ``=`` are treated as
    overrides; all other arguments are treated as config file paths.
    Multiple config files are merged left to right (later files take
    precedence for shared keys). Overrides are applied last.

Examples
--------
    # Single config with overrides
    uv run activelearning config/base.yaml budget.available_budget=10

    # Merge two configs, then apply an override
    uv run activelearning config/base.yaml config/mf.yaml sampler.num_samples=500
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from omegaconf import OmegaConf

from activelearning.logger.config import bootstrap_logger_backend_imports
from activelearning.utils.config_loader import load_config


def _parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="activelearning",
        description=(
            "Run the active learning loop from one or more YAML config files. "
            "Arguments containing '=' are treated as OmegaConf dotlist overrides; "
            "all other arguments are treated as config file paths merged left to right."
        ),
    )
    parser.add_argument(
        "args",
        nargs="+",
        metavar="config_or_override",
        help=(
            "Config file path(s) and/or key=value overrides. "
            "Example: config/base.yaml config/mf.yaml budget.available_budget=5"
        ),
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load config, build components, run the active learning loop."""
    args, unknown = _parse_args(argv)

    configs = [a for a in args.args if "=" not in a]
    overrides = [a for a in args.args if "=" in a]

    if unknown:
        import warnings

        warnings.warn(
            f"Unrecognised arguments ignored: {unknown}. "
            "Pass OmegaConf overrides as positional key=value tokens, "
            "e.g. budget.available_budget=10.",
            stacklevel=2,
        )

    if not configs:
        raise ValueError(
            "At least one config file path must be provided. "
            "Arguments containing '=' are interpreted as overrides."
        )

    config_paths = [Path(config).resolve() for config in configs]

    raw_cfg = load_config(
        path=configs if len(configs) > 1 else configs[0],
        overrides=overrides or None,
    )
    bootstrap_logger_backend_imports(OmegaConf.to_container(raw_cfg, resolve=False))

    from activelearning.active_learning import active_learning
    from activelearning.config import ActiveLearningConfig
    from activelearning.utils.config_loader import parse_config
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed

    cfg = parse_config(raw_cfg, ActiveLearningConfig)
    set_global_seed(cfg.runtime.seed)

    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = cfg.acquisition.build()
    sampler = cfg.sampler.build()
    selector = cfg.selector.build()
    oracle = cfg.oracle.build()
    budget = cfg.budget.build()
    logger = cfg.logger.build() if cfg.logger is not None else None
    run_writer = (
        cfg.run_writer.build(
            metadata=_build_run_metadata(
                config_paths=config_paths,
                overrides=overrides,
                raw_cfg=raw_cfg,
            )
        )
        if cfg.run_writer is not None
        else None
    )
    runtime_context = cfg.runtime.build_context(logger=logger)

    bind_runtime_context(
        [dataset, surrogate, acquisition, sampler, selector, oracle],
        runtime_context,
    )

    if logger is not None:
        logger.log_config(OmegaConf.to_container(raw_cfg, resolve=True))

    dataset, total_cost, num_rounds = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
        runtime_context=runtime_context,
        run_writer=run_writer,
    )

    print(f"Done. Rounds: {num_rounds} | Total cost: {total_cost:.4f}")


def _build_run_metadata(
    *,
    config_paths: Sequence[Path],
    overrides: Sequence[str],
    raw_cfg: Any,
) -> dict[str, Any]:
    """Build compact, reproducible run metadata from the CLI inputs.

    Parameters
    ----------
    config_paths : Sequence[Path]
        Config files passed to the CLI, in merge order.
    overrides : Sequence[str]
        Dotlist overrides applied on top of the loaded config files.
    raw_cfg : Any
        Unparsed OmegaConf config used to derive the resolved run config.
    """

    resolved_config = OmegaConf.to_container(raw_cfg, resolve=True)
    return {
        "config": resolved_config,
        "cli": {
            "command": _format_activelearning_command(config_paths, overrides),
            "config_files": [_display_path(path) for path in config_paths],
            "config_overrides": list(overrides),
        },
        "provenance": {
            "git_commit_sha": _git_commit_sha(),
            "resolved_config_md5": _stable_md5(resolved_config),
            "input_files": _collect_input_files(config_paths, resolved_config),
        },
    }


def _format_activelearning_command(
    config_paths: Sequence[Path],
    overrides: Sequence[str],
) -> str:
    """Return a shell-friendly reproduction command for the current run."""

    args = [_display_path(path) for path in config_paths] + list(overrides)
    return "activelearning " + " ".join(args)


def _git_commit_sha() -> str | None:
    """Return the current Git commit SHA, if the working tree is in a repository."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _collect_input_files(
    config_paths: Sequence[Path],
    resolved_config: Any,
) -> list[dict[str, str]]:
    """Collect explicit input files and their hashes for reproducibility."""

    seen_paths: set[Path] = set()
    recorded_files: list[dict[str, str]] = []

    for path in config_paths:
        if path.is_file() and path not in seen_paths:
            recorded_files.append(_file_record(path))
            seen_paths.add(path)

    for candidate in _iter_config_file_candidates(resolved_config):
        resolved_candidate = _resolve_existing_file(candidate)
        if resolved_candidate is None or resolved_candidate in seen_paths:
            continue
        recorded_files.append(_file_record(resolved_candidate))
        seen_paths.add(resolved_candidate)

    return recorded_files


def _iter_config_file_candidates(value: Any, key: str | None = None) -> Iterator[str]:
    """Yield file-like config values while skipping known output paths."""

    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key in {"output_dir", "log_dir"}:
                continue
            yield from _iter_config_file_candidates(child_value, child_key)
        return

    if isinstance(value, list):
        for child_value in value:
            yield from _iter_config_file_candidates(child_value, key)
        return

    if isinstance(value, str) and key in {"path", "config_path", "file", "file_path"}:
        yield value


def _resolve_existing_file(path_like: str) -> Path | None:
    """Resolve a config path against the current working directory if it exists."""

    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if candidate.is_file():
        return candidate
    return None


def _file_record(path: Path) -> dict[str, str]:
    """Return a stable path/hash record for one file."""

    return {
        "path": _display_path(path),
        "md5": _file_md5(path),
    }


def _display_path(path: Path) -> str:
    """Prefer repository-relative paths when possible."""

    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def _file_md5(path: Path) -> str:
    """Compute the MD5 digest for a file on disk."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_md5(payload: Any) -> str:
    """Compute a deterministic MD5 for a JSON-serializable payload."""

    encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.md5(encoded_payload).hexdigest()


if __name__ == "__main__":
    main()

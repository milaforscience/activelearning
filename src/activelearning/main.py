"""Entry point for the active learning loop.

Usage
-----
    uv run activelearning <config.yaml> [<config2.yaml> ...] [key=value ...]

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

import argparse
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from activelearning.logger.config import bootstrap_logger_backend_imports
from activelearning.utils.config_loader import load_config

if TYPE_CHECKING:
    from activelearning.config import ActiveLearningConfig


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
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
    return parser.parse_known_args()


def process_arguments() -> tuple[DictConfig, "ActiveLearningConfig"]:
    """Parse CLI arguments, load configs, and validate the merged config."""
    args, unknown = _parse_args()

    configs = [arg for arg in args.args if "=" not in arg]
    overrides = [arg for arg in args.args if "=" in arg]

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

    raw_cfg = load_config(
        path=configs if len(configs) > 1 else configs[0],
        overrides=overrides or None,
    )
    bootstrap_logger_backend_imports(OmegaConf.to_container(raw_cfg, resolve=False))

    from activelearning.config import ActiveLearningConfig
    from activelearning.utils.config_loader import parse_config

    cfg = parse_config(raw_cfg, ActiveLearningConfig)
    return raw_cfg, cfg


def main() -> None:
    """Load config, build components, run the active learning loop."""
    raw_cfg, cfg = process_arguments()

    from activelearning.active_learning import active_learning
    from activelearning.runtime import bind_runtime_context

    # Build instances of active learning components from pydantic config models
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = cfg.acquisition.build()
    sampler = cfg.sampler.build()
    selector = cfg.selector.build()
    oracle = cfg.oracle.build()
    budget = cfg.budget.build()
    logger = cfg.logger.build() if cfg.logger is not None else None
    runtime_context = cfg.runtime.build(logger=logger)

    bind_runtime_context(
        [dataset, surrogate, acquisition, sampler, selector, oracle],
        runtime_context,
    )

    if logger is not None:
        logger.log_config(OmegaConf.to_container(raw_cfg, resolve=True))

    # Run active learning campaign
    dataset, total_cost, num_rounds = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
        runtime_context=runtime_context,
    )

    print(f"Done. Rounds: {num_rounds} | Total cost: {total_cost:.4f}")


if __name__ == "__main__":
    main()

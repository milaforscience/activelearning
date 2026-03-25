"""Entry point for the active learning loop.

Usage
-----
    python -m activelearning <config.yaml> [key=value ...]

Positional arguments
--------------------
config
    Path to a YAML configuration file (see ActiveLearningConfig).

Optional overrides
------------------
key=value
    OmegaConf dotlist overrides applied on top of the config file, e.g.:

        budget.available_budget=200 acquisition.beta=0.5
"""

import argparse
from omegaconf import OmegaConf

from activelearning.logger.config import bootstrap_logger_backend_imports
from activelearning.utils.config_loader import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="activelearning",
        description="Run the active learning loop from a YAML config file.",
    )
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="OmegaConf dotlist overrides (e.g. budget.available_budget=100).",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, build components, run the active learning loop."""
    args = _parse_args()

    raw_cfg = load_config(path=args.config, overrides=args.overrides or None)
    bootstrap_logger_backend_imports(OmegaConf.to_container(raw_cfg, resolve=False))

    from activelearning.active_learning import active_learning
    from activelearning.config import ActiveLearningConfig
    from activelearning.utils.config_loader import parse_config
    from activelearning.runtime import bind_runtime_context

    cfg = parse_config(raw_cfg, ActiveLearningConfig)

    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = cfg.acquisition.build()
    sampler = cfg.sampler.build(runtime=cfg.runtime)
    selector = cfg.selector.build()
    oracle = cfg.oracle.build()
    budget = cfg.budget.build()
    logger = cfg.logger.build() if cfg.logger is not None else None
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
    )

    print(f"Done. Rounds: {num_rounds} | Total cost: {total_cost:.4f}")


if __name__ == "__main__":
    main()

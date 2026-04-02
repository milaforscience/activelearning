# Running Experiments

This page covers everything you need to run, configure, and monitor experiments
with this framework. It assumes you have already [installed the package](../getting-started/installation.md).

## Setup

```sh
make setup
```

This installs the package and the `activelearning` CLI entrypoint.

## Run your first experiment

Pass a YAML config file to the CLI:

```sh
uv run activelearning config/branin_single_fidelity.yaml
```

You will see per-round metrics printed to the terminal:

```text
[Step 1] round=1 | num_new_samples=28 | round_cost=28.0 | total_cost=28.0 | budget_remaining=272.0
...
Done. Rounds: 10 | Total cost: 300.0
```

## Override any config value

Append OmegaConf dotlist overrides directly after the config path. Any
top-level or nested field can be overridden:

```sh
# Quick pilot with reduced budget
uv run activelearning config/branin_single_fidelity.yaml budget.available_budget=30.0

# Change the round budget
uv run activelearning config/branin_multi_fidelity.yaml budget.schedule.value=5.0

# Larger candidate pool
uv run activelearning config/branin_multi_fidelity.yaml sampler.num_samples=20000
```

## Compose multiple configs

Pass two or more YAML files. They are merged left to right — later files override
shared keys, everything else is inherited:

```sh
# Add Aim logging to any run without touching the base config
uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml
```

The bundled `config/aim_logging.yaml` replaces only the `logger` block.

## Add Aim logging

Install the optional Aim dependency:

```sh
uv sync --extra aim
```

Then compose with the Aim overlay:

```sh
uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml
```

Open the Aim UI:

```sh
uv run aim up
```

Other supported backends: `WandbLogger` (`uv sync --extra wandb`),
`CometLogger` (`uv sync --extra comet`). Override `logger.type` directly
or provide your own logger YAML overlay.

## Disable logging

```yaml
logger: null
```

Or as an override:

```sh
uv run activelearning config/branin_single_fidelity.yaml logger=null
```

## Validate a config without running

```python
from activelearning.utils.config_loader import load_and_parse
from activelearning.config import ActiveLearningConfig

load_and_parse("config/branin_single_fidelity.yaml", ActiveLearningConfig)
print("config ok")
```

## Available configs

See [Available Configs](../tutorials/configs.md) for the full list of bundled YAML files and when to use each.

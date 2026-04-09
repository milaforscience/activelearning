# Running Experiments

This page covers the CLI mechanics for configuring, composing, and monitoring experiments. If you haven't run your first experiment yet, start with the [Quickstart](../getting-started/quickstart.md).

## Override config values

Append `key=value` arguments after the config path to override any field without touching the YAML file. Keys use dot notation to address nested fields (`budget.schedule.value` maps to `schedule.value` inside the `budget` block). This follows standard [OmegaConf](https://omegaconf.readthedocs.io/) syntax:

```sh
# Change the round budget
uv run activelearning config/branin_multi_fidelity.yaml budget.schedule.value=5.0

# Larger candidate pool
uv run activelearning config/branin_multi_fidelity.yaml sampler.num_samples=20000
```

## Compose multiple configs

Pass two or more YAML files. They are merged left to right — later files override shared keys, everything else is inherited:

```sh
uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml
```

The bundled `config/aim_logging.yaml` replaces only the `logger` block, leaving the rest of the experiment unchanged. This pattern works with any base config.

## Logging

The framework supports several logging backends. Install the optional dependency for the one you want and compose with the corresponding overlay:

| Backend | Install | `logger.type` |
| --- | --- | --- |
| Console | — | [`ConsoleLogger`](../reference/activelearning/logger/logger.md#activelearning.logger.logger.ConsoleLogger) |
| [Aim](https://aimstack.io/) | `uv sync --extra aim` | [`AimLogger`](../reference/activelearning/logger/logger.md#activelearning.logger.logger.AimLogger) |
| [Weights & Biases](https://wandb.ai/) | `uv sync --extra wandb` | [`WandbLogger`](../reference/activelearning/logger/logger.md#activelearning.logger.logger.WandbLogger) |
| [Comet](https://www.comet.com/) | `uv sync --extra comet` | [`CometLogger`](../reference/activelearning/logger/logger.md#activelearning.logger.logger.CometLogger) |

You can override `logger.type` directly or provide your own logger YAML overlay.

For a step-by-step walkthrough of Aim logging, see the [Synthetic Function Examples](synthetic_function_experiment.md) tutorial.

### Disable logging

Set `logger` to `null` in the config or as a CLI override:

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

See the [Synthetic Function Examples](synthetic_function_experiment.md) tutorial for the full list of bundled configs and guided walkthroughs.

# Running Experiments

The recommended experimental workflow proceeds in four steps:

1. Set up the `uv` environment.
2. Validate a YAML experiment specification before launching a longer study.
3. Run a small-budget pilot.
4. Use overrides to scale the same study specification up or down.

## 1. Set up the repo

```sh
make setup
```

This installs the package into the project environment, so `uv run activelearning ...` works from the repository root.

## 2. Preflight a config

Validate a YAML file before launching a budget-constrained study:

```sh
uv run python - <<'PY'
from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_and_parse

load_and_parse("config/branin_botorch_toy.yaml", ActiveLearningConfig)
print("config ok")
PY
```

This step identifies schema errors—such as missing `fidelity_costs`, stale component tags, or invalid nested types—before budget is committed to a full run.

## 3. Run a short pilot experiment

```sh
uv run activelearning config/branin_botorch_toy.yaml \
  budget.available_budget=1.0 \
  sampler.num_samples=50 \
  selector.time_limit=5 \
  acquisition.num_fantasies=2 \
  acquisition.num_mv_samples=2 \
  acquisition.num_y_samples=4
```

This command runs the full active learning pipeline. It covers:

- YAML loading and override merging,
- component construction,
- candidate-fidelity proposal and scoring,
- the active-learning loop,
- the cost-aware knapsack selector, and
- console logging.

## 4. Expected Output

A normal run prints four kinds of information:

- `[Config] ...` — the fully resolved experiment specification after overrides.
- CBC solver output — expected when `KnapsackSelector` solves the round-level budget allocation.
- `[Step N] ...` — per-round metrics such as `num_new_samples`, `round_cost`, `total_cost` (accumulated oracle cost), and `budget_remaining`.
- `Done. Rounds: ...` — final summary from the CLI entrypoint.

If you use `ConsoleLogger`, Branin also acknowledges a landscape figure with a console message instead of rendering it inline.

## 5. Useful override patterns

Representative ablation patterns:

```sh
# Smaller candidate-fidelity pool
uv run activelearning config/branin_botorch_toy.yaml sampler.num_samples=20

# Higher or lower total oracle budget
uv run activelearning config/branin_botorch_toy.yaml budget.available_budget=5.0

# Faster or slower knapsack solve
uv run activelearning config/branin_botorch_toy.yaml selector.time_limit=10

# More or fewer samples inside the multi-fidelity acquisition
uv run activelearning config/branin_botorch_toy.yaml acquisition.num_y_samples=64
```

For GFlowNet experiments, the same dotlist syntax applies to `sampler.conf.*` overrides, as described in [Runtime and Configuration](../concepts/runtime_and_configuration.md).

## 6. Logging choices

`ConsoleLogger` requires no additional setup and is the minimal-overhead option for local inspection. The repo also includes `WandbLogger`, `CometLogger`, `AimLogger`, and
`MultiLogger`, but those depend on optional packages and external setup.

If you do not want any logging, set:

```yaml
logger: null
```

## 7. Validate your changes

Use the existing repository checks:

```sh
make check
```

In the current repo snapshot, the Branin BoTorch pilot run above is the
primary end-to-end validation path. `make check` is the recommended repository-wide check for documentation and configuration-related changes. The full test
suite still includes GFlowNet coverage that expects the bundled env config
discussed on the [GFlowNet Sampler Setup](../examples/gflownet_sampler.md) page.

`config/branin_botorch_toy.yaml` is the primary CLI reference configuration for
budget-constrained multi-fidelity discovery. The Branin, Hartmann, and GFlowNet
example pages document the other reference configs and their current caveats.

For the benchmark-level config map, use [Available Baselines and Configs](../examples/configs.md). For the remaining gap between those runnable paths and a fuller paper-style rerun workflow, use [Paper Replication](../paper_replication/index.md).

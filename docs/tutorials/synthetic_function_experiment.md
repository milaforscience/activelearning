# **Synthetic Function Examples**

This tutorial walks you through running complete active learning experiments on two synthetic benchmarks: the 2D **Branin** function and the 6-dimensional **Hartmann** function. Before starting, make sure you have completed [Installation](../getting-started/installation.md).

It is structured as three stages — each one runnable on its own:

1. **Single-fidelity** — get a working run and understand the terminal output.
2. **Multi-fidelity** — add fidelity levels and observe how the algorithm trades off cost vs. quality.
3. **Aim logging** — persist metrics and figures to a local Aim dashboard.

All four tutorial configs use [`ConsoleLogger`](../reference/activelearning/logger/logger.md#activelearning.logger.logger.ConsoleLogger) by default, so every stage gives you visible output immediately.

## **The benchmarks**

**Branin** is a 2-dimensional function commonly used in optimization research. Its small input space makes it quick to run and easy to visualize — the oracle logs a 2D contour plot after each round when Aim logging is enabled.

**Hartmann** is a 6-dimensional function with a more complex landscape, commonly used to evaluate multi-fidelity methods at higher dimensionality. Because there is no 2D representation, no landscape figure is logged.

Both use the same algorithmic stack: a Gaussian process surrogate, max-value entropy acquisition, uniform sampling, and cost-aware selection.

| Config | Fidelity costs | Total budget | Round budget |
| --- | --- | --- | --- |
| `branin_single_fidelity.yaml` | — | 300 | 30 |
| `branin_multi_fidelity.yaml` | 0.01 / 0.1 / 1.0 | 300 | 30 |
| `hartmann_single_fidelity.yaml` | — | 100 | 10 |
| `hartmann_multi_fidelity.yaml` | 0.125 / 0.25 / 1.0 | 100 | 10 |

!!! tip
    All configs are fully self-contained YAML files — open them before running to see the complete experiment specification.

## **1. Single-Fidelity Setting**

Start with the simplest runnable version of Branin — a short pilot with a reduced budget:

```sh
uv run activelearning config/branin_single_fidelity.yaml \
  budget.available_budget=30.0
```

You should see output like this in your terminal:

```text
[Step 1] round=1 | num_new_samples=30 | round_cost=30.0000 | total_cost=30.0000 | budget_remaining=0.0000
Done. Rounds: 1 | Total cost: 30.0000
```

!!! note "Cold start"
    The first round always runs with a random acquisition (the surrogate isn't fitted yet). From round 2 onwards, the acquisition function guides the search.

Each line represents one active learning round. The fields are:

| Field | Meaning |
| --- | --- |
| `round` | Round index |
| `num_new_samples` | Candidates queried this round |
| `round_cost` | Budget consumed this round |
| `total_cost` | Total budget consumed so far |
| `budget_remaining` | Budget still available |

Once the run looks right, drop the budget override to run the full experiment:

```sh
uv run activelearning config/branin_single_fidelity.yaml
```

The same workflow applies directly to Hartmann. With a round budget of `10.0`, a budget cap of `20.0` gives two rounds — enough to verify the output:

```sh
uv run activelearning config/hartmann_single_fidelity.yaml \
  budget.available_budget=20.0
```

Drop the override for the full run:

```sh
uv run activelearning config/hartmann_single_fidelity.yaml
```

## **2. Multi-Fidelity Setting**

Switch to the Branin multi-fidelity config:

```sh
uv run activelearning config/branin_multi_fidelity.yaml \
  budget.available_budget=30.0
```

The terminal output looks the same, but the numbers will differ:

```text
[Step 1] round=1 | num_new_samples=2999 | round_cost=29.9900 | total_cost=29.9900 | budget_remaining=0.0100
Done. Rounds: 1 | Total cost: 29.9900
```

!!! info "What changed with multi-fidelity?"
    With fidelity costs of `0.01`, `0.1`, and `1.0`, the selector can now pack many cheap low-fidelity queries into a single round budget. Notice that `num_new_samples` can be much larger when cheap queries dominate. The acquisition function learns to escalate to higher fidelities as evidence accumulates.

For Hartmann, the same applies with fidelity costs `0.125 / 0.25 / 1.0`:

```sh
uv run activelearning config/hartmann_multi_fidelity.yaml \
  budget.available_budget=20.0
```

Full run:

```sh
uv run activelearning config/hartmann_multi_fidelity.yaml
```

## **3. Add Aim Logging**

So far, results only exist in the terminal. [Aim](https://aimstack.io/) is an open-source experiment tracker that persists runs locally and lets you explore metrics, configs, and figures in an interactive UI.

Once the terminal workflow is clear, install the optional Aim dependency:

```sh
uv sync --extra aim
```

The repository ships `config/aim_logging.yaml` — a logger overlay that replaces the console-only logger with a `MultiLogger` (console + Aim). Compose it with any base config by passing both files to the CLI:

```sh
uv run activelearning config/branin_single_fidelity.yaml config/aim_logging.yaml
uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml
uv run activelearning config/hartmann_single_fidelity.yaml config/aim_logging.yaml
uv run activelearning config/hartmann_multi_fidelity.yaml config/aim_logging.yaml
```

All runs still print to the console and are now also persisted in Aim. Branin runs additionally log a 2D contour of the landscape after each oracle query; Hartmann runs do not.

## **4. Open Aim and inspect**

Aim stores its data in the `.aim/` directory at the repository root (already git-ignored).

Launch the Aim UI:

```sh
uv run aim up
```

Open the URL shown in the terminal. All runs appear under the project `activelearning_tutorials`.

The most useful views to start with:

- **Config tab** — the full resolved config that was used for the run
- **Metrics** — scalar time series for `round_cost`, `total_cost`, and `budget_remaining`
- **Images** — 2D contour of the Branin landscape logged after each oracle query (Branin runs only)

!!! tip "Comparing runs"
    Because all runs log to the same `activelearning_tutorials` project, you can compare all four side by side — overlaying `total_cost` curves to see how multi-fidelity covers more of the space per unit budget, and how Branin and Hartmann differ in their convergence behaviour.

## **5. What comes next**

This tutorial covers running and monitoring experiments from the checked-in configs. Natural follow-ups include:

- computing derived metrics on logged results (e.g. mean top-10 score, simple regret, inference regret),
- adapting the configs to your own oracle — see the [Extension Guide](../extension-guide/overview.md),
- or reading [Multi-Fidelity Setting](../concepts/multi_fidelity.md) for a deeper treatment of how fidelity costs and confidences propagate through the loop.

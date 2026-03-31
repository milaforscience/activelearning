# Branin Experiment Tutorial

This tutorial walks you through running a complete active learning experiment on the Branin benchmark, from a minimal first run to a fully monitored multi-fidelity study.

It is structured as three stages — each one runnable on its own:

1. **Single-fidelity Branin** — get a working run and understand the terminal output.
2. **Multi-fidelity Branin** — add fidelity levels and observe how the algorithm trades off cost vs. quality.
3. **Aim logging** — persist metrics and figures to a local Aim dashboard.

Both tutorial configs use `ConsoleLogger` by default, so every stage gives you visible output immediately.

## 1. Install the environment

From the repository root:

```sh
make setup
```

`make setup` installs the project and CLI entrypoint. You only need the optional Aim dependency in Stage 3.

## 2. The tutorial configs at a glance

The **single-fidelity** config (`config/branin_single_fidelity.yaml`) uses a Gaussian process surrogate with max-value entropy acquisition, uniform sampling over the Branin domain, cost-aware selection, and a total oracle budget of 300 with 30 per round.

The **multi-fidelity** config (`config/branin_multi_fidelity.yaml`) keeps the same stack but adds three fidelity levels with costs `0.01`, `0.1`, and `1.0`, so the algorithm can choose cheaper approximations to explore the space before committing to expensive high-fidelity queries.

!!! tip
    Both configs are fully self-contained YAML files — open them before running to see the complete experiment specification.

## 3. Single-Fidelity Setting

Start with the simplest runnable version — a short pilot with a reduced budget:

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

You're done with Stage 1. You have a fully working single-fidelity active learning run with nothing but a YAML file and one command.

## 4. Multi-Fidelity Setting

Switch to the multi-fidelity config:

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

## 5. Add Aim Logging

Once the terminal workflow is clear, install the optional Aim dependency:

```sh
uv sync --extra aim
```

The repository ships `config/aim_logging.yaml` — a logger overlay that replaces the console-only logger with a `MultiLogger` (console + Aim). Compose it with any base config by passing both files to the CLI:

**Multi-fidelity run with Aim:**

```sh
uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml
```

**Single-fidelity run with Aim:**

```sh
uv run activelearning config/branin_single_fidelity.yaml config/aim_logging.yaml
```

This stage still prints to the console, but now also persists the run in Aim, including a 2D contour of the Branin landscape logged after each oracle query.

## 6. Open Aim and inspect the run

Aim stores its data in the `.aim/` directory at the repository root (already git-ignored).

Launch the Aim UI:

```sh
uv run aim up
```

Open the URL shown in the terminal. Look for the project `activelearning_tutorials`.

The most useful views to start with:

- **Config tab** — the full resolved config that was used for the run
- **Metrics** — scalar time series for `round_cost`, `total_cost`, and `budget_remaining`
- **Images** — 2D contour of the Branin landscape logged after each oracle query

!!! tip "Comparing single-fidelity vs. multi-fidelity"
    Run both configs with Aim and open them side by side in the Aim UI. You can overlay the `total_cost` curves on the same plot to see how the multi-fidelity run covers more of the space per unit budget.

## 7. What comes next

This tutorial covers running and monitoring experiments from the checked-in configs. Natural follow-ups include:

- computing derived metrics on logged results (e.g. mean top-10 score, simple regret, inference regret),
- adapting the configs to your own oracle,
- or exploring the [Extension Guide](../extension-guide/index.md) to implement custom components.

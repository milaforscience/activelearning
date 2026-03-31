# Quickstart

After completing [Installation](installation.md), the recommended first run is a
reduced version of the single-fidelity Branin tutorial config. This keeps the
first experience simple and fully visible in the terminal.

## Run the minimal validated baseline

```bash
uv run activelearning config/branin_single_fidelity.yaml \
  budget.available_budget=30.0
```

You should see output like this in your terminal:

```text
[Step 1] round=1 | num_new_samples=30 | round_cost=30.0000 | total_cost=30.0000 | budget_remaining=0.0000
Done. Rounds: 1 | Total cost: 30.0000
```

!!! note "Cold start"
    The first round runs with a random acquisition — the surrogate isn't fitted yet. This is expected. From round 2 onwards the acquisition function guides the search.

The key output fields are:

| Field | Meaning |
| --- | --- |
| `round` | Active learning round index |
| `num_new_samples` | Candidates queried in that round |
| `round_cost` | Budget consumed in that round |
| `budget_remaining` | Total budget still available |

## Scale the same study specification

Remove the budget-reduction overrides incrementally as you build confidence.

**Use the bundled tutorial config as-is (budget 300):**

```bash
uv run activelearning config/branin_single_fidelity.yaml
```

**Keep the same study but double the total oracle budget:**

```bash
uv run activelearning config/branin_single_fidelity.yaml budget.available_budget=600
```

**Increase the candidate pool without changing anything else:**

```bash
uv run activelearning config/branin_single_fidelity.yaml sampler.num_samples=20000
```

!!! tip "Ready to go deeper?"
    The [Branin Experiment Tutorial](../tutorials/branin_experiment.md) continues from here — adding multi-fidelity, Aim logging, and a comparison between the two settings.

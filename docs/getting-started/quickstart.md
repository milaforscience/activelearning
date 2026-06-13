# **Quickstart**

After completing [Installation](installation.md), you are ready to run your first experiment. The framework is invoked with:

```bash
uv run activelearning <config.yaml> [key=value ...]
```

The YAML file defines the full experiment — surrogate, acquisition function, sampler, selector, oracle, and budget. Any arguments after it are [OmegaConf](https://omegaconf.readthedocs.io/) overrides that adjust the config on the fly without editing the file.

## **First run**

Start with a minimal version of the single-fidelity Branin config, capped to a small budget so the run completes in seconds:

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
    The first round uses a random acquisition — the surrogate has no data to fit yet. This is expected. From round 2 onwards the acquisition function guides the search.

The key output fields are:

| Field | Meaning |
| --- | --- |
| `round` | Active learning round index |
| `num_new_samples` | Candidates queried in that round |
| `round_cost` | Budget consumed in that round |
| `total_cost` | Cumulative budget consumed across all rounds so far |
| `budget_remaining` | Total budget still available |

## **Scale up**

Once you've confirmed the run completes successfully, remove the budget override to run the full experiment:

```bash
uv run activelearning config/branin_single_fidelity.yaml
```

From there, overrides let you explore without touching the config file:

```bash
# Double the total oracle budget
uv run activelearning config/branin_single_fidelity.yaml budget.available_budget=600

# Increase the candidate pool size
uv run activelearning config/branin_single_fidelity.yaml sampler.num_samples=20000
```

!!! tip "Ready to go deeper?"
    The [Synthetic Function Examples](../tutorials/synthetic_function_experiment.md) tutorial continues from here — adding multi-fidelity, Aim logging, and a comparison between the two settings.

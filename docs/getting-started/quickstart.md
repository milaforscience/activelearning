# **Quickstart**

After completing [Installation](installation.md), you are ready to run your first experiment. The framework is invoked with:

```bash
uv run activelearning <config.yaml> [key=value ...]
```

The YAML file defines the full experiment — surrogate, acquisition function, sampler, selector, oracle, and budget. Any arguments after it are [OmegaConf](https://omegaconf.readthedocs.io/) overrides that adjust the config on the fly without editing the file.

## **First run**

Start with a minimal, single-fidelity version of the Branin task. Branin is a widely used objective function with three global minima, defined on two dimensions. The oracle uses negated Branin to align with the framework's maximization setup. Here, the design space is a continuous 2D box sampled with a Latin hypercube. In this minimal example, the budget is capped so the run completes in seconds:

```bash
uv run activelearning config/branin/single_fidelity.yaml \
  budget.available_budget=3.0
```

You should see output like this in your terminal:

```text
[Step 1] active_learning/round=1 | active_learning/samples/proposed=10000 | active_learning/samples/selected=3 | active_learning/observations/new=3 | active_learning/cost/round=3.0000 | active_learning/cost/cumulative=3.0000 | active_learning/budget/remaining=0.0000 | profiling/...
Done. Rounds: 1 | Total cost: 3.0000
```

!!! note "Cold start"
    The first round uses a random acquisition — the surrogate has no data to fit yet. This is expected. From round 2 onwards the acquisition function guides the search.

The key output fields are:

| Field | Meaning |
| --- | --- |
| `active_learning/round` | Active learning round index |
| `active_learning/samples/proposed` | Candidates proposed by the sampler |
| `active_learning/samples/selected` | Candidates queried in that round |
| `active_learning/observations/new` | Finite observations added in that round |
| `active_learning/cost/round` | Budget consumed in that round |
| `active_learning/cost/cumulative` | Cumulative budget consumed across all rounds so far |
| `active_learning/budget/remaining` | Total budget still available |
| `profiling/<phase>` | Always-collected operational duration in seconds for one active-learning phase |

For live telemetry, durable run records, and optional diagnostic controls, see
[Monitoring and Diagnostics](../concepts/monitoring_and_diagnostics.md).

## **Scale up**

Once you've confirmed the run completes successfully, remove the budget override to run the full experiment:

```bash
uv run activelearning config/branin/single_fidelity.yaml
```

From there, overrides let you explore without touching the config file:

```bash
# Double the total oracle budget
uv run activelearning config/branin/single_fidelity.yaml budget.available_budget=600

# Increase the number of candidates the sampler proposes each round
# (the selector still only affords as many as fit within the round budget)
uv run activelearning config/branin/single_fidelity.yaml sampler.num_samples=200
```

!!! tip "Ready to go deeper?"
    The [Running Experiments](../tutorials/running_experiments.md) tutorial continues from here — introducing a second and third fidelity and Aim logging. From there, the [Branin Benchmark](../tutorials/branin_benchmark.md) tutorial puts those pieces to work in a full 5-method benchmark comparison.

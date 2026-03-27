# Quickstart

After completing [Installation](installation.md), the minimal validated baseline is a reduced version of the bundled BoTorch Branin study. This baseline uses the same study specification as the broader Branin benchmark path, reducing the candidate pool and oracle budget to minimize computation.

## Run the minimal validated baseline

```bash
uv run activelearning config/branin_botorch_toy.yaml \
  budget.available_budget=0.01 \
  sampler.num_samples=100 \
  selector.time_limit=5 \
  selector.verbose=false
```

## What this baseline does

- Loads `config/branin_botorch_toy.yaml`, the current validated Branin baseline.
- Constructs the surrogate, acquisition, sampler, selector, oracle, budget, and logger from the study specification.
- Generates 100 candidate-fidelity queries instead of the larger default pool.
- Caps the total budget at `0.01`, limiting the run to a single lower-cost query under this configuration.
- Logs the resolved config and round metrics to the console.

With an empty initial dataset $\mathcal{D}_0 = \emptyset$, the run operates in a cold-start regime: the algorithm proposes candidate-fidelity queries, evaluates one budget-feasible query $(x, m)$, updates $\mathcal{D}$, and terminates when the budget is exhausted.

## What the output means

A representative run ends with output like this:

```text
[Step 1] round=1 | num_new_samples=1 | round_cost=0.0100 | total_cost=0.0100 | budget_remaining=0.0000
Done. Rounds: 1 | Total cost: 0.0100
```

The key output fields are:

- `round=1`: one active learning round completed
- `num_new_samples=1`: one candidate-fidelity query was executed
- `round_cost=0.0100`: the selected query used the lowest-cost fidelity in this study
- `budget_remaining=0.0000`: the run terminates because the configured budget is exhausted

## Scale the same study specification

To scale toward a more informative study, remove the budget-reduction overrides incrementally.

### Use the bundled baseline config as-is

```bash
uv run activelearning config/branin_botorch_toy.yaml
```

### Keep the same study but increase the total oracle budget

```bash
uv run activelearning config/branin_botorch_toy.yaml budget.available_budget=1.0
```

### Increase the candidate pool without changing anything else

```bash
uv run activelearning config/branin_botorch_toy.yaml sampler.num_samples=1000
```

These variants preserve the validated baseline while scaling the budget or proposal set toward a larger study.

## Rationale

This command constitutes the current validated baseline without claiming a paper-complete reproduction:

- The system parses and validates the configuration.
- The runtime context is constructed.
- The sampler generates candidate-fidelity queries $(x, m)$.
- The acquisition function and selector enforce cost-aware budget allocation.
- The oracle returns a labeled observation.
- The logger reports the resolved config and per-round metrics.

To review the methodology before scaling the study, see:

- [Framework Overview](../concepts/overview.md)
- [Active Learning Loop](../concepts/active_learning_loop.md)
- [Runtime and Configuration](../concepts/runtime_and_configuration.md)

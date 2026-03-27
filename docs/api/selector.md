# Selector API

Selectors take a candidate pool from a sampler and return the final subset of candidates or candidate-fidelity queries to submit to the oracle.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.selector.selector` | `Selector` | Abstract selector contract. |
| `activelearning.selector.score_selector` | `TopKAcquisitionSelector` | Top-k by acquisition score. |
| `activelearning.selector.cost_aware_selector` | `CostAwareSelector` | Greedy budget-aware selection. |
| `activelearning.selector.knapsack_selector` | `KnapsackSelector`, `greedy_knapsack_indices` | Exact 0/1 knapsack selection with optional warm start. |
| `activelearning.selector.config` | selector config models | YAML-facing builders. |

## Core abstraction: `Selector`

Defined in `activelearning.selector.selector`.

| Parameter to `__call__` | Meaning |
| --- | --- |
| `candidates` | Candidate or candidate-fidelity query pool to choose from. |
| `acquisition` | Acquisition object used for scoring. |
| `cost_fn` | Callable that returns per-query costs, typically `oracle.get_costs`. |
| `round_budget` | Current round budget, typically `budget.get_round_budget(...)`. |

The current built-in selectors are all oriented around **singleton** acquisition scores. None of them call `acquisition.score_batches(...)` yet, so current round-level decision making is per-query rather than joint q-batch optimization.

## Implementations

| Class | Strategy | Requires cost function and budget | Config model |
| --- | --- | --- | --- |
| `TopKAcquisitionSelector` | Sort by acquisition score and keep the first `num_samples` | no | `TopKAcquisitionSelectorConfig` |
| `CostAwareSelector` | Greedy ratio `acquisition_value / cost` until budget is exhausted | yes | `CostAwareSelectorConfig` |
| `KnapsackSelector` | Exact 0/1 knapsack via PuLP and CBC | yes | `KnapsackSelectorConfig` |

### `TopKAcquisitionSelector`

- expects an acquisition with singleton scoring support
- ignores `cost_fn` and `round_budget`
- returns candidates in descending acquisition-score order

### `CostAwareSelector`

- expects `acquisition`, `cost_fn`, and `round_budget`
- computes per-candidate ratios `value / cost`
- treats zero-cost candidates as having infinite ratio
- returns the greedy feasible set in ratio order

### `KnapsackSelector`

- expects `acquisition`, `cost_fn`, and `round_budget`
- clamps negative acquisition values to zero
- if all acquisition scores are zero, substitutes a constant objective so the solver prefers feasible cheap items
- can optionally warm-start the CBC solver with `greedy_knapsack_indices(...)`

## Configuration

Defined in `activelearning.selector.config`.

| Config model | Builds | Key fields |
| --- | --- | --- |
| `TopKAcquisitionSelectorConfig` | `TopKAcquisitionSelector(...)` | `num_samples` |
| `CostAwareSelectorConfig` | `CostAwareSelector()` | no extra parameters |
| `KnapsackSelectorConfig` | `KnapsackSelector(...)` | `time_limit`, `verbose`, `warm_start` |
| `SelectorConfig` | discriminated union | one of the three rows above |

## Loop integration

The default loop wiring in `activelearning.active_learning.active_learning` is:

1. sampler returns `samples`
2. budget computes `round_budget`
3. selector is called as

   ```python
   selector(
       samples,
       acquisition=acquisition,
       cost_fn=oracle.get_costs,
       round_budget=round_budget,
   )
   ```

That is the main contract a custom selector needs to satisfy. In paper-aligned multi-fidelity runs, this is the point where the round-level budget-allocation problem over proposed candidate-fidelity queries is resolved.

## Class Reference

::: activelearning.selector.selector.Selector
    options:
      show_source: false
      heading_level: 3

::: activelearning.selector.score_selector.TopKAcquisitionSelector
    options:
      show_source: false
      heading_level: 3

::: activelearning.selector.cost_aware_selector.CostAwareSelector
    options:
      show_source: false
      heading_level: 3

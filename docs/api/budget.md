# Budget API

The budget family controls how much oracle work can be performed per active-learning round. In paper-aligned terms, it constrains how much accumulated oracle cost can be spent on candidate-fidelity queries during budget-constrained discovery.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.budget.budget` | `Budget` | Runtime budget accounting object. |
| `activelearning.budget.schedule_config` | `constant_schedule`, `sigmoid_iteration_schedule`, schedule config models | Schedule helpers and Pydantic schedule builders. |
| `activelearning.budget.config` | `BudgetConfig` | YAML-facing budget builder. |

## Core class: `Budget`

Defined in `activelearning.budget.budget`.

| Attribute or method | Purpose |
| --- | --- |
| `available_budget` | Remaining total budget. |
| `schedule` | Callable that maps round index to per-round allocation. |
| `get_round_budget(current_round)` | Returns the scheduled budget for the round, capped at the remaining budget. |
| `can_afford(cost)` | Side-effect-free affordability check. |
| `consume(cost)` | Deducts cost from `available_budget` or raises if it would overspend. |

`Budget` is purely about accounting; it does not know which candidates, fidelities, or selectors produced those costs.

## Schedule helpers

Defined in `activelearning.budget.schedule_config`.

| Helper or config | Behavior |
| --- | --- |
| `constant_schedule(value)` and `ConstantScheduleConfig` | Returns the same allocation every round. |
| `sigmoid_iteration_schedule(total_budget, num_iterations, midpoint_fraction, steepness)` and `SigmoidIterationScheduleConfig` | Converts a sigmoid-shaped cumulative spending curve into per-round allocations. |

### `SigmoidIterationScheduleConfig`

Fields:

- `num_iterations`
- `midpoint_fraction`
- `steepness`

The runtime schedule returns `0.0` for rounds outside `0 <= current_round < num_iterations`.

## Configuration

Defined in `activelearning.budget.config`.

| Config model | Fields | Build result |
| --- | --- | --- |
| `BudgetConfig` | `available_budget`, `schedule` | `Budget(available_budget=..., schedule=...)` |
| `ScheduleConfig` | discriminated union of `ConstantScheduleConfig` and `SigmoidIterationScheduleConfig` | nested inside `BudgetConfig` |

`BudgetConfig.build()` chooses the correct runtime schedule function based on the concrete `schedule` model type and returns a ready-to-use `Budget`.

## How the loop uses it

In `activelearning.active_learning.active_learning` the budget flow is:

1. `round_budget = budget.get_round_budget(num_rounds)`
2. selector receives `round_budget`
3. the loop sums `oracle.get_costs(selected_samples)` for the selected candidate-fidelity queries
4. `budget.can_afford(round_cost)` guards the query
5. `budget.consume(round_cost)` records the spend

The current loop passes `num_rounds` starting at `0`, so built-in schedules should be read as zero-based. This keeps budget accounting separate from acquisition scoring and from the oracle's fidelity-specific cost model.

## Class Reference

::: activelearning.budget.budget.Budget
    options:
      show_source: false
      heading_level: 3

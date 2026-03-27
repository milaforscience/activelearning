# Oracle API

Oracles define the evaluation side of the system: they validate fidelities, assign query costs, and turn candidate or candidate-fidelity queries into `Observation` objects. They define which fidelity levels exist, how much each query costs, and how those fidelities are encoded for the surrogate.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.oracle.oracle` | `Oracle` | Abstract oracle contract and fidelity validation helpers. |
| `activelearning.oracle.multi_fidelity_oracle` | `MultiFidelityOracle` | Reusable multi-fidelity implementation based on per-fidelity config dicts. |
| `activelearning.oracle.augmented_function_oracle` | `AugmentedFunctionOracle`, `BraninOracle`, `Hartmann6DOracle` | BoTorch synthetic test-function oracles. |
| `activelearning.oracle.composite_oracle` | `CompositeOracle` | Delegates to the cheapest supporting sub-oracle by fidelity. |
| `activelearning.oracle.plotting` | `build_augmented_2d_landscape_figure` | Helper used by `BraninOracle` figure logging. |
| `activelearning.oracle.config` | oracle config models | YAML-facing builders. |

All concrete oracles currently in the source tree are multi-fidelity or composed from multi-fidelity pieces.

## Core abstraction: `Oracle`

Defined in `activelearning.oracle.oracle`.

| Method or helper | Purpose | Notes |
| --- | --- | --- |
| `get_fidelity_confidences()` | Return `dict[int, float]` confidence values in `[0, 1]` | Surrogates consume this before fitting. |
| `get_supported_fidelities()` | Convenience wrapper around `get_fidelity_confidences().keys()` | Returns sorted fidelity ids. |
| `get_costs(candidates)` | Return per-candidate query costs | Used before budget consumption. |
| `query(candidates)` | Return `Observation` objects in input order | Budget consumption is the caller's responsibility. |
| `_validate_candidate_fidelity(...)` | Shared validation helper for single- and multi-fidelity candidates | Raises on missing or unsupported fidelities. |
| `_validate_fidelity_confidences(...)` | Shared validation for confidence mappings | Enforces numeric values in `[0, 1]`. |

## `MultiFidelityOracle`

Defined in `activelearning.oracle.multi_fidelity_oracle`.

`MultiFidelityOracle` is the reusable base for the concrete test-function oracles. Its constructor expects:

```python
fidelity_configs: dict[int, dict[str, Any]]
```

Each fidelity entry must contain:

- `cost_per_sample`
- `score_fn`
- `fidelity_confidence`

From that single mapping it implements `get_fidelity_confidences()`, `get_costs()`, and `query()` for candidate-fidelity queries.

## Concrete oracles

| Class | Module | What it does | Config-exposed |
| --- | --- | --- | --- |
| `AugmentedFunctionOracle` | `augmented_function_oracle.py` | Base class for BoTorch synthetic test functions whose last input dimension is fidelity | no |
| `BraninOracle` | `augmented_function_oracle.py` | Wraps `AugmentedBranin(negate=True)` and logs a 2-D landscape figure at query time when a logger is bound | yes |
| `Hartmann6DOracle` | `augmented_function_oracle.py` | Wraps `AugmentedHartmann(negate=True)` | yes |
| `CompositeOracle` | `composite_oracle.py` | Groups candidates by fidelity and delegates each group to the cheapest sub-oracle that supports that fidelity | yes |

### `AugmentedFunctionOracle`

`AugmentedFunctionOracle` is the bridge between integer fidelity ids used by the outer API and the continuous fidelity values expected by BoTorch's augmented test functions.

Key behaviors:

- `fidelity_costs` must be non-empty
- if `fidelity_confidences` is omitted, it is derived as `cost / max_cost`
- a per-fidelity `score_fn` appends the continuous confidence value as the last input dimension before evaluating the BoTorch function

That means the integer fidelity id never reaches the test function directly; the mapped confidence value does. When confidences are derived from costs, the highest-cost level stays aligned with the highest-fidelity available oracle.

### `BraninOracle` and plotting

`BraninOracle.query()` delegates to the base query path and then, when `self.logger` is available, logs `branin_landscape_query` using `build_augmented_2d_landscape_figure(...)`.

### `CompositeOracle`

`CompositeOracle` is useful when multiple sub-oracles can serve overlapping fidelity levels. Its behavior is:

1. group incoming candidates by fidelity
2. find the sub-oracle with the lowest total cost for that fidelity group
3. delegate `get_costs()` or `query()` to that oracle
4. restore the original candidate order in the result

It also merges fidelity-confidence mappings across child oracles and raises if two sub-oracles disagree about the confidence associated with the same fidelity id.

## Configuration

Defined in `activelearning.oracle.config`.

| Config model | Builds | Notes |
| --- | --- | --- |
| `BraninOracleConfig` | `BraninOracle(fidelity_costs, fidelity_confidences)` | `fidelity_confidences` is optional. |
| `Hartmann6DOracleConfig` | `Hartmann6DOracle(fidelity_costs, fidelity_confidences)` | Same pattern as Branin. |
| `CompositeOracleConfig` | `CompositeOracle(sub_oracles=[...])` | Recursively nests `OracleConfig` entries. |
| `OracleConfig` | discriminated union | Includes the three rows above. |

## Interaction with the rest of the package

- The loop calls `oracle.get_costs(selected_samples)` before `oracle.query(selected_samples)`, so oracle cost definitions directly shape budget-constrained discovery.
- The loop passes `oracle.get_fidelity_confidences()` into `surrogate.set_fidelity_confidences(...)`.
- Selectors typically receive `oracle.get_costs` as their `cost_fn`.

If you need a new oracle implementation, this is the family to extend; if you only need a new fidelity encoding or tensor representation, start with [Runtime and types](runtime_and_types.md).

## Class Reference

::: activelearning.oracle.oracle.Oracle
    options:
      show_source: false
      heading_level: 3

::: activelearning.oracle.multi_fidelity_oracle.MultiFidelityOracle
    options:
      show_source: false
      heading_level: 3

::: activelearning.oracle.augmented_function_oracle.AugmentedFunctionOracle
    options:
      show_source: false
      heading_level: 3

::: activelearning.oracle.augmented_function_oracle.BraninOracle
    options:
      show_source: false
      heading_level: 3

::: activelearning.oracle.augmented_function_oracle.Hartmann6DOracle
    options:
      show_source: false
      heading_level: 3

::: activelearning.oracle.composite_oracle.CompositeOracle
    options:
      show_source: false
      heading_level: 3

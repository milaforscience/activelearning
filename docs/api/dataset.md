# Dataset API

Datasets own observation storage and provide stable, reusable views of the observations produced by candidate or candidate-fidelity queries within each active-learning round.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.dataset.dataset` | `Dataset` | Abstract dataset contract. |
| `activelearning.dataset.list_dataset` | `ListDataset` | In-memory list-backed implementation. |
| `activelearning.dataset.config` | `ListDatasetConfig`, `DatasetConfig` | Config builders for datasets. |

## Core abstraction: `Dataset`

| Method | Responsibility | Important contract |
| --- | --- | --- |
| `add_observations(observations)` | Append newly labeled observations. | Must update the latest-batch view. |
| `get_observations_iterable()` | Return all stored observations for the current round. | The returned iterable must support consistent multiple iterations. |
| `get_latest_observations_iterable()` | Return only the most recent batch. | Must also be consistently re-iterable. |
| `get_best_candidates(k)` | Return the top-`k` observations by the dataset's notion of "best". | Ranking is implementation-specific. |

The multiple-pass guarantee is the key API rule here. The active-learning loop takes one round snapshot and shares it across surrogate fitting, acquisition-function updates, and sampling; dataset implementations are responsible for making repeated iteration over that snapshot safe.

## Concrete implementation: `ListDataset`

Defined in `activelearning.dataset.list_dataset`.

| Detail | Current behavior |
| --- | --- |
| storage | Keeps all observations in `_records: list[Observation]`. |
| latest-batch tracking | Uses `_latest_start_idx` and `_latest_end_idx` to slice the most recently appended block. |
| all-data view | `get_observations_iterable()` returns `list(self._records)`. |
| latest view | `get_latest_observations_iterable()` returns the latest slice without copying the whole history. |
| ranking logic | `get_best_candidates()` filters out `y is None`, keeps only the highest observed fidelity if fidelities are present, then uses `heapq.nlargest(..., key=observation.y)`. |

`ListDataset` therefore encodes a **maximization** convention and, in multi-fidelity runs, reports best observations from the highest observed fidelity only. If your problem is minimization or needs a different ranking rule, the extension point is a custom `Dataset` subclass with a different `get_best_candidates()` implementation.

## Configuration

Defined in `activelearning.dataset.config`.

| Config model | Builds | Notes |
| --- | --- | --- |
| `ListDatasetConfig` | `ListDataset()` | The only dataset builder currently exposed from YAML. |
| `DatasetConfig` | discriminated union | Currently a one-member union containing `ListDatasetConfig`. |

Minimal YAML:

```yaml
dataset:
  type: ListDataset
```

## Interaction with the rest of the package

- Oracles produce `Observation` objects from queried candidates.
- Datasets store those observations unchanged.
- Surrogates consume dataset iterables to fit or update predictive models.
- The loop logs `dataset.get_best_candidates(1)[0]` at each round when a logger is present, so this ranking also drives round summaries for budget-constrained discovery.

If you are deciding where to customize storage semantics, this family is the right place; if you are deciding how observations become tensors or predictions, move to the [Surrogate API](surrogate.md).

## Class Reference

::: activelearning.dataset.dataset.Dataset
    options:
      show_source: false
      heading_level: 3

::: activelearning.dataset.list_dataset.ListDataset
    options:
      show_source: false
      heading_level: 3

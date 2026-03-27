# Adding a New Selector

Selectors choose the final subset of candidates from the pool the sampler
produces. A selector implements the **round budget allocation policy**. Implement a new
selector to:

- Apply custom constraints (diversity, domain rules, batch coverage).
- Implement a new budget allocation policy (fractional fidelity budget, risk
  thresholds).
- Mix cost awareness with acquisition scoring in a custom way.

Before implementing a new selector, verify that `TopKAcquisitionSelector` or
`CostAwareSelector` does not already meet your requirements.

## What to implement

Subclass `activelearning.selector.selector.Selector` and implement one method:

| Method | Signature |
|---|---|
| `__call__` | `(candidates, acquisition=None, cost_fn=None, round_budget=None) -> list[Candidate]` |

Return a **subset** of the input candidates. Never query the oracle, compute
observations, or modify the budget inside the selector.

## Complete example: diversity-aware selector

This selector scores candidates with the acquisition function and then applies
a simple spread heuristic: candidates that are too close to an already-selected
point are penalized, promoting diversity.

```python
# src/activelearning/selector/diversity_selector.py

import math
from typing import Callable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate


def _euclidean(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two flat coordinate lists."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class DiversityAwareSelector(Selector):
    """Selects candidates by balancing acquisition score and spatial spread.

    Scores each candidate with the acquisition function, then greedily picks
    the next candidate whose distance to all already-selected candidates
    exceeds ``min_distance``. Falls back to the top-scoring remaining
    candidate when no diverse point exists.

    Parameters
    ----------
    num_samples : int
        Maximum number of candidates to select per round.
    min_distance : float
        Minimum Euclidean distance between any two selected candidates.
        Candidates closer than this threshold to an already-selected point
        are de-prioritized but not permanently excluded.
    """

    def __init__(self, num_samples: int, min_distance: float = 0.1) -> None:
        self._num_samples = num_samples
        self._min_distance = min_distance

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Acquisition] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ) -> list[Candidate]:
        """Select a diverse, high-scoring subset of candidates.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool to select from.
        acquisition : Acquisition, optional
            Used to score candidates. If ``None``, selection is random.
        cost_fn : callable, optional
            Unused by this selector. Accepted for interface compatibility.
        round_budget : float, optional
            Unused by this selector. Accepted for interface compatibility.

        Returns
        -------
        selected : list[Candidate]
            Up to ``num_samples`` candidates that balance score and spread.
            Returns an empty list when ``candidates`` is empty.
        """
        if not candidates:
            return []

        # Score every candidate (fall back to 0.0 when no acquisition)
        if acquisition is not None and acquisition.supports_singleton_scoring:
            scores = acquisition.score(candidates)
        else:
            scores = [0.0] * len(candidates)

        # Sort descending by score; work on index pairs for stable tracking
        indexed = sorted(
            enumerate(candidates), key=lambda ic: scores[ic[0]], reverse=True
        )

        selected: list[Candidate] = []
        selected_coords: list[list[float]] = []

        for _original_idx, candidate in indexed:
            if len(selected) >= self._num_samples:
                break

            x = candidate.x if isinstance(candidate.x, list) else list(candidate.x)

            # Check distance to all already-selected candidates
            too_close = any(
                _euclidean(x, sc) < self._min_distance
                for sc in selected_coords
            )

            if not too_close:
                selected.append(candidate)
                selected_coords.append(x)

        # If diversity filtering was too aggressive, top up with best remaining
        if len(selected) < self._num_samples:
            remaining = [
                c for _, c in indexed if c not in selected
            ]
            needed = self._num_samples - len(selected)
            selected.extend(remaining[:needed])

        return selected
```

## Config model and registration

Add a config model and extend `SelectorConfig` in
`src/activelearning/selector/config.py`:

```python
# src/activelearning/selector/config.py  (additions)
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from activelearning.selector.diversity_selector import DiversityAwareSelector
from activelearning.selector.selector import Selector


class DiversityAwareSelectorConfig(BaseModel):
    """Configuration for the diversity-aware selector."""

    type: Literal["DiversityAwareSelector"] = "DiversityAwareSelector"
    num_samples: int = Field(gt=0)
    min_distance: float = 0.1

    def build(self) -> Selector:
        """Instantiate and return the selector."""
        return DiversityAwareSelector(
            num_samples=self.num_samples,
            min_distance=self.min_distance,
        )


# Extend the union
SelectorConfig = Annotated[
    Union[
        TopKAcquisitionSelectorConfig,
        CostAwareSelectorConfig,
        DiversityAwareSelectorConfig,   # <-- new
    ],
    Field(discriminator="type"),
]
```

Then in your YAML:

```yaml
selector:
  type: DiversityAwareSelector
  num_samples: 10
  min_distance: 0.05
```

## Fidelity preservation

Selected candidates must carry their fidelity ids unchanged. Do not modify
`Candidate.fidelity` inside the selector — the oracle and dataset use it to
route and record observations correctly. Since `Candidate` is a frozen
dataclass, accidental mutation will raise an `AttributeError`.

## Returning an empty list

Return an empty list when no candidates are feasible (e.g. all candidates
violate a constraint or the budget is exhausted before any selection is made).
The loop treats an empty return as a valid termination condition for the current round.

```python
if round_budget is not None and min_cost > round_budget:
    return []   # nothing fits in budget — stop cleanly
```

## Using `acquisition.score()` inside the selector

Call `acquisition.score()` with the full candidate sequence. Always guard with
a `None` check and check `supports_singleton_scoring` before calling:

```python
if acquisition is not None and acquisition.supports_singleton_scoring:
    scores = acquisition.score(candidates)
else:
    scores = [0.0] * len(candidates)
```

For cost-aware policies, compute costs with `cost_fn` and divide scores by cost:

```python
if cost_fn is not None:
    costs = cost_fn(candidates)
    scores = [s / c for s, c in zip(scores, costs)]
```

## Common pitfalls

**Do not query the oracle** — the selector only allocates from a pre-built pool.
Any oracle call inside the selector corrupts budget accounting.

**Do not double-count budget** — the loop deducts costs after oracle query. The
selector's job is to check feasibility, not to deduct.

**`candidate.x` shape** — `Candidate.x` can be any type (list, numpy array,
tensor). Normalize to a plain list before arithmetic comparisons if you are not
sure of the upstream sampler's output format.

## Related pages

- [Sampler guide](sampler.md) — how the candidate pool is generated
- [Acquisition guide](acquisition.md) — how `score()` works
- [Extension guide overview](index.md)
- [Selector API](../api/selector.md)

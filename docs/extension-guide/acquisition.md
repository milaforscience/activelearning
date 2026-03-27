# Adding a New Acquisition Function

Acquisition functions map the surrogate's predictive posterior to a utility score $\alpha(x, m)$ over candidate-fidelity queries $(x, m)$. This score drives selector allocation decisions. Implement a new acquisition subclass to:

- Implement a custom information criterion (e.g. entropy search, knowledge
  gradient, bespoke cost-aware utility).
- Combine multiple signals (posterior mean + diversity bonus, multi-objective
  scalarization).
- Use a surrogate type that existing acquisitions do not expose.

## What to implement

Subclass `activelearning.acquisition.acquisition.Acquisition`. The core methods
are:

| Method | Required? | Notes |
|---|---|---|
| `update(surrogate, observations)` | Recommended | Cache the surrogate; validate compatibility |
| `score(candidates)` | For singleton scoring | Returns `list[float]` |
| `score_batches(candidate_batches)` | For batch scoring | Returns `list[float]` per batch |

Implement `score()` for independent per-candidate scoring, `score_batches()`
for joint batch utility, or both. The `supports_singleton_scoring` and
`supports_batch_scoring` properties are inferred automatically from which
methods you override.

## `DummyAcquisition` Reference

`DummyAcquisition` is the simplest acquisition in the library. It reads
`"mean"` and optionally `"std"` from `surrogate.predict()` and returns
`mean + beta * std` as the score. Review it before implementing a new acquisition —
most custom acquisitions require only minor additions to this pattern.

```python
# Simplified structure of DummyAcquisition
class DummyAcquisition(Acquisition):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self._beta = beta

    def score(self, candidates):
        pred = self.surrogate.predict(list(candidates))
        means = pred["mean"]
        stds = pred.get("std")
        if stds is None:
            return list(means)
        return [m + self._beta * s for m, s in zip(means, stds)]
```

## Complete Example: Posterior Mean with Diversity Bonus

This acquisition adds a diversity bonus to the posterior mean. Candidates far
from the current training set receive a higher score, balancing exploitation
with exploration without requiring uncertainty estimates.

```python
# src/activelearning/acquisition/mean_diversity_acquisition.py

import math
from typing import Callable, Iterable, Optional

from activelearning.acquisition.acquisition import Acquisition
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


def _min_distance_to_set(
    x: list[float], reference: list[list[float]]
) -> float:
    """Return the minimum Euclidean distance from x to any point in reference.

    Parameters
    ----------
    x : list[float]
        Query point coordinates.
    reference : list[list[float]]
        Reference set of coordinates.

    Returns
    -------
    float
        Minimum distance, or ``float("inf")`` when ``reference`` is empty.
    """
    if not reference:
        return float("inf")
    return min(
        math.sqrt(sum((a - b) ** 2 for a, b in zip(x, r))) for r in reference
    )


class MeanDiversityAcquisition(Acquisition):
    """Acquisition that combines posterior mean with a diversity bonus.

    Score = mean(x) + diversity_weight * min_distance(x, training_set)

    A higher ``diversity_weight`` encourages exploring undersampled regions;
    setting it to 0.0 recovers pure posterior mean (greedy exploitation).

    Parameters
    ----------
    diversity_weight : float
        Weight applied to the diversity bonus. Must be >= 0.
    maximize : bool
        If ``True``, higher posterior mean is better (maximization objective).
        If ``False``, lower is better; means are negated before scoring.

    Notes
    -----
    Requires ``surrogate.predict()`` to return a dict with a ``"mean"`` key.
    """

    def __init__(
        self,
        diversity_weight: float = 0.5,
        maximize: bool = True,
    ) -> None:
        super().__init__()
        self._diversity_weight = diversity_weight
        self._maximize = maximize
        self._training_coords: list[list[float]] = []

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Cache the surrogate and extract training set coordinates.

        Parameters
        ----------
        surrogate : Surrogate
            The fitted surrogate to use for posterior mean prediction.
            Must implement ``predict()`` with a ``"mean"`` key.
        observations : Iterable[Observation], optional
            All observations seen so far. Used to build the reference set
            for diversity scoring.
        """
        super().update(surrogate, observations)

        self._training_coords = []
        if observations is not None:
            for obs in observations:
                x = obs.x if isinstance(obs.x, list) else list(obs.x)
                self._training_coords.append(x)

    def score(
        self,
        candidates: Iterable[Candidate],
        cost_weighting: Optional[
            Callable[[list[float], list[Candidate]], list[float]]
        ] = None,
    ) -> list[float]:
        """Score candidates by posterior mean plus diversity bonus.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates to score independently.
        cost_weighting : callable, optional
            If provided, called as ``cost_weighting(raw_scores, candidates)``
            and its return value replaces the raw scores.

        Returns
        -------
        scores : list[float]
            Acquisition scores, one per candidate (higher is better).

        Raises
        ------
        ValueError
            If the surrogate has not been set via ``update()`` yet.
        ValueError
            If ``predict()`` does not return a ``"mean"`` key.
        """
        candidate_list = list(candidates)

        if self.surrogate is None:
            return [0.0] * len(candidate_list)

        pred = self.surrogate.predict(candidate_list)
        means = pred.get("mean")
        if means is None:
            raise ValueError(
                "MeanDiversityAcquisition requires surrogate.predict() "
                "to return a 'mean' key."
            )

        scores = []
        for candidate, mean in zip(candidate_list, means):
            x = candidate.x if isinstance(candidate.x, list) else list(candidate.x)
            diversity = _min_distance_to_set(x, self._training_coords)
            direction = 1.0 if self._maximize else -1.0
            score = direction * float(mean) + self._diversity_weight * diversity
            scores.append(score)

        if cost_weighting is not None:
            scores = cost_weighting(scores, candidate_list)

        return scores
```

## Config model and registration

Add a config model and extend `AcquisitionConfig` in
`src/activelearning/acquisition/config.py`:

```python
# src/activelearning/acquisition/config.py  (additions)
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from activelearning.acquisition.mean_diversity_acquisition import (
    MeanDiversityAcquisition,
)
from activelearning.acquisition.acquisition import Acquisition


class MeanDiversityAcquisitionConfig(BaseModel):
    """Configuration for the posterior-mean + diversity acquisition."""

    type: Literal["MeanDiversityAcquisition"] = "MeanDiversityAcquisition"
    diversity_weight: float = 0.5
    maximize: bool = True

    def build(self) -> Acquisition:
        """Instantiate and return the acquisition function."""
        return MeanDiversityAcquisition(
            diversity_weight=self.diversity_weight,
            maximize=self.maximize,
        )


# Extend the union (add alongside existing acquisition configs)
# AcquisitionConfig = Annotated[Union[..., MeanDiversityAcquisitionConfig], ...]
```

Then in your YAML:

```yaml
acquisition:
  type: MeanDiversityAcquisition
  diversity_weight: 0.3
  maximize: true
```

## `supports_singleton_scoring` property

This property is derived automatically by comparing the bound method to the
base class default. You do **not** need to override it:

- Override `score()` → `supports_singleton_scoring` returns `True`.
- Override `score_batches()` → `supports_batch_scoring` returns `True`.

You only need to be explicit if you want to disable a method that you have
partially inherited. This is rarely necessary.

## BoTorch acquisitions: subclass `BoTorchAcquisitionBase`

For acquisitions that wrap a BoTorch analytic or MC acquisition, subclass
`BoTorchAcquisitionBase` instead of `Acquisition` directly. It handles:

- Candidate encoding (including fidelity columns).
- Multi-fidelity and cost-aware infrastructure.
- BoTorch acquisition lifecycle (`_build_botorch_acquisition()`).

The only method you must implement is `_build_botorch_acquisition()`:

```python
from botorch.acquisition import qExpectedImprovement
from activelearning.acquisition.botorch.botorch_acquisition import BoTorchAcquisitionBase


class MyBoTorchAcquisition(BoTorchAcquisitionBase):
    """Custom BoTorch acquisition wrapping qEI."""

    def _build_botorch_acquisition(self):
        """Build and return the BoTorch acquisition object.

        Returns
        -------
        botorch acquisition
            An instantiated BoTorch acquisition ready for optimization.
        """
        model = self._botorch_surrogate.model
        best_f = max(o.y for o in self._observations_cache)
        return qExpectedImprovement(model=model, best_f=best_f)
```

The base class then uses this acquisition inside `score()`. For purely analytic
BoTorch acquisitions (see `UpperConfidenceBound`, `ExpectedImprovement` in the
library), the same pattern applies.

## Materializing `observations` in `update()`

The `observations` parameter passed to `update()` is an
`Iterable[Observation]` that may be a one-pass generator. If you iterate it
more than once — for example to extract coordinates and then compute statistics
— convert it to a list first:

```python
def update(self, surrogate, observations=None):
    super().update(surrogate, observations)
    if observations is not None:
        obs_list = list(observations)  # materialise immediately
        ...
```

## Common pitfalls

**Never call `surrogate.predict()` in `update()`** — `update()` is for caching
state, not scoring. Predict inside `score()` or `score_batches()` instead.

**Null surrogate guard** — `self.surrogate` is `None` before `update()` has
been called. Return neutral scores (e.g. `[0.0] * n`) when the surrogate is
not yet available rather than raising.

**The loop calls `update()` only after `is_fitted()` returns `True`** — your
acquisition will never see an unfitted surrogate in production, but defensive
`None` checks are still good practice for tests.

**Cost-weighting passthrough** — if you accept `cost_weighting` in `score()`,
call it as the last step after computing raw scores, and return its result. Do
not apply cost weighting twice.

## Related pages

- [Surrogate guide](surrogate.md) — what `predict()` returns
- [Selector guide](selector.md) — how `score()` is called during selection
- [Extension guide overview](index.md)
- [Acquisition API](../api/acquisition.md)

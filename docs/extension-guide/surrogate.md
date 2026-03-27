# Adding a New Surrogate

Surrogates approximate the objective function from collected observations,
enabling cheap candidate scoring without querying the oracle. Implement a new
surrogate subclass to:

- Use a different model family (neural network, random forest, ensemble).
- Wrap an external library (scikit-learn, GPyTorch custom kernel, JAX model).
- Implement incremental or online updates.
- Control fidelity-weighted training data.

Before implementing a new surrogate, verify that `DummyMeanSurrogate` (for
baselines) or `BoTorchGPSurrogate` (for GP-based work) does not already meet your
requirements.

## What to implement

Subclass `activelearning.surrogate.surrogate.Surrogate`. The methods you must
or should implement depend on your update strategy:

| Method | Required? | Notes |
|---|---|---|
| `updates_from_latest()` | **Yes** | Declares loop behavior: `False` = full refit, `True` = incremental |
| `fit(observations)` | If `updates_from_latest()` returns `False` | Full refit on all observations |
| `update(observations)` | If `updates_from_latest()` returns `True` | Incremental update from latest batch |
| `predict(candidates)` | If acquisition uses `predict()` | Returns `dict` with at least `"mean"` key |
| `is_fitted()` | If unsafe before training | Override to return `False` until first fit |
| `set_fidelity_confidences(confidences)` | For multi-fidelity surrogates | Called before `fit()` / `update()` |

## Complete example: scikit-learn GP surrogate

```python
# src/activelearning/surrogate/sklearn_gp_surrogate.py

from typing import Iterable, Mapping, Sequence, Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class SklearnGPSurrogate(Surrogate):
    """scikit-learn GP surrogate with Matérn 5/2 kernel.

    Performs a full refit on every loop iteration. Suitable for small to
    medium datasets (up to a few thousand observations).

    Parameters
    ----------
    noise_level : float
        Initial noise variance estimate passed to the GP kernel.
    n_restarts_optimizer : int
        Number of restarts for hyperparameter optimization.
    """

    def __init__(
        self,
        noise_level: float = 1e-3,
        n_restarts_optimizer: int = 5,
    ) -> None:
        self._noise_level = noise_level
        self._n_restarts_optimizer = n_restarts_optimizer
        self._gp: GaussianProcessRegressor | None = None

    # ------------------------------------------------------------------
    # Surrogate contract
    # ------------------------------------------------------------------

    def updates_from_latest(self) -> bool:
        """Declare full-refit semantics.

        Returns
        -------
        bool
            Always ``False``: this surrogate refits from scratch each round.
        """
        return False

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the GP to all current observations.

        Parameters
        ----------
        observations : Iterable[Observation]
            All observations collected so far.
        """
        obs_list = list(observations)
        if not obs_list:
            return

        X = np.array([o.x for o in obs_list], dtype=np.float64)
        y = np.array([o.y for o in obs_list], dtype=np.float64)

        kernel = Matern(nu=2.5) + Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3))
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self._noise_level,
            n_restarts_optimizer=self._n_restarts_optimizer,
            normalize_y=True,
        )
        self._gp.fit(X, y)

    def is_fitted(self) -> bool:
        """Return whether the surrogate has been fitted at least once.

        Returns
        -------
        bool
            ``False`` until ``fit()`` has been called with at least one
            observation. ``True`` thereafter.
        """
        return self._gp is not None

    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, Any]:
        """Predict mean and standard deviation for each candidate.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates to predict.

        Returns
        -------
        result : dict
            Dictionary with keys ``"mean"`` and ``"std"``, each a list of
            floats in the same order as ``candidates``.

        Raises
        ------
        RuntimeError
            If called before ``fit()`` (surrogate not yet fitted).
        """
        if self._gp is None:
            raise RuntimeError(
                "SklearnGPSurrogate.predict() called before fit(). "
                "Check is_fitted() before scoring candidates."
            )

        X = np.array([c.x for c in candidates], dtype=np.float64)
        mean, std = self._gp.predict(X, return_std=True)

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
```

## Config model and registration

Add a config model and extend `SurrogateConfig` in
`src/activelearning/surrogate/config.py`:

```python
# src/activelearning/surrogate/config.py  (additions)
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from activelearning.surrogate.sklearn_gp_surrogate import SklearnGPSurrogate
from activelearning.surrogate.surrogate import Surrogate


class SklearnGPSurrogateConfig(BaseModel):
    """Configuration for the scikit-learn GP surrogate."""

    type: Literal["SklearnGPSurrogate"] = "SklearnGPSurrogate"
    noise_level: float = 1e-3
    n_restarts_optimizer: int = 5

    def build(self) -> Surrogate:
        """Instantiate and return the surrogate."""
        return SklearnGPSurrogate(
            noise_level=self.noise_level,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )


# Extend the union
SurrogateConfig = Annotated[
    Union[
        DummyMeanSurrogateConfig,
        BoTorchGPSurrogateConfig,
        SklearnGPSurrogateConfig,   # <-- new
    ],
    Field(discriminator="type"),
]
```

Then in your YAML:

```yaml
surrogate:
  type: SklearnGPSurrogate
  noise_level: 0.001
  n_restarts_optimizer: 5
```

## The `is_fitted()` contract

The loop only calls `acquisition.update()` — which in turn calls
`surrogate.predict()` inside the acquisition — after `is_fitted()` returns
`True`. If your surrogate raises an error when `predict()` is called on an
empty model, **you must override `is_fitted()`** to return `False` until at
least one observation has been processed.

The default base-class implementation returns `True` (always ready). Override
it only when predictions are unsafe before fitting:

```python
def is_fitted(self) -> bool:
    return self._model is not None
```

## `updates_from_latest()` semantics

| Return value | Loop behavior | Method called |
|---|---|---|
| `False` (default) | Full refit every round | `fit(all_observations)` |
| `True` | Incremental update | `update(latest_observations_only)` |

Return `True` only when your model genuinely supports partial updates without
degrading accuracy. The `False` path is the safe default.

If `True` is returned without implementing `update()`, the base class raises
`NotImplementedError`.

## Multi-fidelity support

If your surrogate uses fidelity information, override
`set_fidelity_confidences()`. The loop calls this method once at startup with
the oracle's confidence mapping. Use the values to weight training data:

```python
def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
    """Store fidelity weights for use during fit().

    Parameters
    ----------
    confidences : dict[int, float]
        Mapping from fidelity id to confidence in [0, 1].
    """
    self._fidelity_confidences = confidences

def fit(self, observations: Iterable[Observation]) -> None:
    obs_list = list(observations)
    weights = [
        self._fidelity_confidences.get(o.fidelity, 1.0)
        for o in obs_list
    ]
    # use weights in model.fit(X, y, sample_weight=weights)
    ...
```

## `predict()` return format

Acquisition functions that use `surrogate.predict()` expect a `dict`. The
`"mean"` key is the minimum requirement. Add `"std"` to enable
uncertainty-aware acquisitions like `DummyAcquisition` (UCB-style):

```python
return {
    "mean": [...],   # list[float], length == len(candidates)
    "std": [...],    # list[float], optional but recommended
}
```

The keys your surrogate returns must match what the acquisition expects.
Document the requirements in both classes.

## Common Pitfalls

**`__init__()` must not construct tensors directly** — use `self.dtype` and `self.device`
inside `fit()` or `predict()` after the runtime context has been bound.

**Materialization** — `fit()` and `update()` receive an `Iterable[Observation]`
that may be a one-pass generator. Convert to a list immediately if you need
random access or multiple passes.

**Surrogate compatibility** — BoTorch acquisitions require a
`BoTorchGPSurrogate`, not a generic `Surrogate`. Custom surrogates that wrap
non-BoTorch models should pair with acquisitions that only use `predict()`.

## Related pages

- [Acquisition guide](acquisition.md) — how the acquisition consumes `predict()`
- [Extension guide overview](index.md)
- [Surrogate API](../api/surrogate.md)

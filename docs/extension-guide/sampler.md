# Adding a New Sampler

Samplers generate the pool of candidates that the selector then trims to the
final query set. Implement a new sampler to:

- Propose candidates from a new search space geometry (graph, sequence, grid).
- Use an acquisition-guided or model-based proposal (MCMC, GFlowNet-style).
- Integrate an external candidate generator.
- Apply domain constraints that `HypercubeSampler` cannot express.

Before implementing a new sampler, verify that `HypercubeSampler`,
`PoolUniformSampler`, or `PoolScoreSampler` does not already cover your use case.

## What to implement

Subclass `activelearning.sampler.sampler.Sampler` and implement one method:

| Method | Signature |
|---|---|
| `sample` | `(acquisition=None, observations=None) -> list[Candidate]` |

Both arguments are optional — your sampler may ignore either or both.

## Complete example: Gaussian random walk sampler

This sampler maintains a center point and proposes candidates by adding
Gaussian noise. It stamps fidelity ids when an explicit fidelity list is
provided.

```python
# src/activelearning/sampler/random_walk_sampler.py

from typing import Iterable, Optional, Sequence

import torch

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class GaussianRandomWalkSampler(Sampler):
    """Proposes candidates by Gaussian random walk from a fixed center.

    Each call to ``sample()`` draws ``num_samples`` points by adding
    isotropic Gaussian noise to ``center``. Points are clipped to ``bounds``.

    Parameters
    ----------
    center : list[float]
        Starting point for the random walk (one value per dimension).
    bounds : list[tuple[float, float]]
        Per-dimension ``(lower, upper)`` bounds used to clip proposals.
    num_samples : int
        Number of candidates to return per call.
    step_std : float
        Standard deviation of the Gaussian perturbation.
    fidelities : list[int] or None
        If provided, each candidate is stamped with a fidelity id chosen
        uniformly at random from this list. Pass ``None`` for single-fidelity
        setups.
    """

    def __init__(
        self,
        center: list[float],
        bounds: list[tuple[float, float]],
        num_samples: int,
        step_std: float = 0.1,
        fidelities: Optional[list[int]] = None,
    ) -> None:
        self._center = center
        self._bounds = bounds
        self._num_samples = num_samples
        self._step_std = step_std
        self._fidelities = fidelities

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        """Generate candidates by perturbing the center with Gaussian noise.

        The ``acquisition`` and ``observations`` arguments are accepted but
        ignored by this sampler; it is purely random-walk based.

        Parameters
        ----------
        acquisition : Acquisition, optional
            Unused. Accepted for interface compatibility.
        observations : Iterable[Observation], optional
            Unused. Accepted for interface compatibility.

        Returns
        -------
        candidates : list[Candidate]
            List of ``num_samples`` candidates.
        """
        center = torch.tensor(self._center, dtype=self.dtype, device=self.device)
        noise = torch.randn(
            self._num_samples, len(self._center), dtype=self.dtype, device=self.device
        ) * self._step_std

        proposals = center.unsqueeze(0) + noise

        # Clip to bounds
        for dim, (lower, upper) in enumerate(self._bounds):
            proposals[:, dim] = proposals[:, dim].clamp(lower, upper)

        candidates = []
        for i in range(self._num_samples):
            x = proposals[i].tolist()
            fidelity = None
            if self._fidelities:
                idx = torch.randint(len(self._fidelities), (1,)).item()
                fidelity = self._fidelities[int(idx)]  # type: ignore[arg-type]
            candidates.append(Candidate(x=x, fidelity=fidelity))

        return candidates
```

## Config model and registration

Add a config model and extend `SamplerConfig` in
`src/activelearning/sampler/config.py`:

```python
# src/activelearning/sampler/config.py  (additions)
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field
from activelearning.sampler.random_walk_sampler import GaussianRandomWalkSampler
from activelearning.sampler.sampler import Sampler


class GaussianRandomWalkSamplerConfig(BaseModel):
    """Configuration for the Gaussian random walk sampler."""

    type: Literal["GaussianRandomWalkSampler"] = "GaussianRandomWalkSampler"
    center: list[float]
    bounds: list[tuple[float, float]]
    num_samples: int = Field(gt=0)
    step_std: float = 0.1
    fidelities: Optional[list[int]] = None

    def build(self, runtime=None) -> Sampler:  # runtime param is required
        """Instantiate and return the sampler.

        Parameters
        ----------
        runtime : RuntimeConfig or None
            Passed by ``activelearning.main`` when wiring the loop. May be
            ``None`` during testing.
        """
        return GaussianRandomWalkSampler(
            center=self.center,
            bounds=self.bounds,
            num_samples=self.num_samples,
            step_std=self.step_std,
            fidelities=self.fidelities,
        )


# Extend the union
SamplerConfig = Annotated[
    Union[
        HypercubeSamplerConfig,
        GaussianRandomWalkSamplerConfig,  # <-- new
    ],
    Field(discriminator="type"),
]
```

!!! important "Always keep `build(runtime=None)` on sampler configs"
    The sampler is the **only** component whose config `build()` receives
    `runtime`. The framework calls `cfg.sampler.build(runtime=cfg.runtime)`.
    If your config omits the `runtime` parameter you will get a `TypeError`
    at startup.

Then reference it in your YAML:

```yaml
sampler:
  type: GaussianRandomWalkSampler
  center: [0.5, 0.5]
  bounds: [[0.0, 1.0], [0.0, 1.0]]
  num_samples: 200
  step_std: 0.15
  fidelities: [0, 1]
```

## Fidelity handling

The sampler is responsible for setting `Candidate.fidelity`. If your oracle
requires explicit fidelity ids (i.e. it is multi-fidelity), stamp each
`Candidate` before returning it:

```python
# uniform random assignment across two fidelity levels
import random
fidelity = random.choice([0, 1])
candidates.append(Candidate(x=x, fidelity=fidelity))
```

The fidelity ids you stamp must match exactly the ids declared in the oracle's
`get_fidelity_confidences()`. A mismatch causes a `ValueError` at query time.
See the [Oracle guide](oracle.md#fidelity-id-alignment-with-the-sampler) for
details.

For single-fidelity setups, leave `fidelity=None`.

## Materializing the `observations` iterable

The `observations` parameter is an `Iterable[Observation]` that may be a
one-pass generator. If your sampler needs to inspect observations more than
once (e.g., to filter out already-seen points), convert it to a list first:

```python
def sample(self, acquisition=None, observations=None):
    seen = set()
    if observations is not None:
        obs_list = list(observations)   # materialise before iterating
        seen = {tuple(o.x) for o in obs_list}
    ...
```

Do not iterate over `observations` twice without materializing it — the second
pass will silently yield nothing.

## Using acquisition scores in sampling

If your sampler uses the acquisition function to weight proposals, call
`acquisition.score()` inside `sample()`:

```python
def sample(self, acquisition=None, observations=None):
    candidates = self._generate_pool()
    if acquisition is not None and acquisition.supports_singleton_scoring:
        scores = acquisition.score(candidates)
        # use scores to weight selection from pool
        ...
    return candidates
```

Always guard with `acquisition is not None` — the sampler may be called before
the surrogate has been fitted.

## Common pitfalls

**Return `Candidate` objects, not tensors.** The selector and oracle expect
`Candidate` instances. Wrapping tensors as `Candidate.x` values is fine, but
the outer type must be `Candidate`.

**Do not query the oracle** inside the sampler. Sampling is purely generative;
evaluation happens in the oracle step.

**Avoid side effects from `observations`.** The observations iterable may be
a lazy generator. Never hold a reference to the generator and iterate it later.

## Related pages

- [Oracle guide](oracle.md) — aligning fidelity ids
- [Selector guide](selector.md) — what happens to the candidate pool after sampling
- [Extension guide overview](index.md)
- [Sampler API](../api/sampler.md)

# Adding a New Oracle

An oracle implements the ground-truth evaluation function $f(x)$ at each fidelity level $m \in \mathcal{M}$. Implement a new oracle subclass to:

- Wrap a new simulator, benchmark function, or lab instrument.
- Define a different fidelity structure (different levels, costs, or confidences).
- Implement a custom routing rule for multi-fidelity queries.

If you only need to **combine** existing oracles across fidelity levels, see
[Using CompositeOracle](#using-compositeoracle-instead-of-building-from-scratch)
before writing a new class.

## What to implement

Subclass `activelearning.oracle.oracle.Oracle` and implement three methods:

| Method | Purpose |
|---|---|
| `get_fidelity_confidences()` | Returns a dict mapping each fidelity id to a confidence in `[0, 1]` |
| `get_costs(candidates)` | Returns one cost per input candidate (same order) |
| `query(candidates)` | Returns one `Observation` per input candidate (same order) |

## Complete example: 2-D Gaussian oracle

The following oracle evaluates a 2-D Gaussian peak at two fidelity levels.
Fidelity 0 is a cheap, noisy approximation; fidelity 1 is the exact value.

```python
# src/activelearning/oracle/gaussian_oracle.py

import math
from typing import Sequence

import torch

from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate, Observation


class GaussianOracle(Oracle):
    """2-D Gaussian peak oracle with two fidelity levels.

    Fidelity 0 is a cheap, noisy low-fidelity evaluation.
    Fidelity 1 is the exact (noise-free) high-fidelity evaluation.

    Parameters
    ----------
    center : tuple[float, float]
        Peak location in (x1, x2).
    noise_std : float
        Standard deviation of Gaussian noise added at fidelity 0.
    low_cost : float
        Query cost for fidelity 0.
    high_cost : float
        Query cost for fidelity 1.
    """

    _SUPPORTED_FIDELITIES = {0, 1}

    def __init__(
        self,
        center: tuple[float, float] = (0.5, 0.5),
        noise_std: float = 0.1,
        low_cost: float = 1.0,
        high_cost: float = 5.0,
    ) -> None:
        self._center = center
        self._noise_std = noise_std
        self._costs = {0: low_cost, 1: high_cost}

    # ------------------------------------------------------------------
    # Oracle contract
    # ------------------------------------------------------------------

    def get_fidelity_confidences(self) -> dict[int, float]:
        """Return confidence values for each fidelity level.

        Returns
        -------
        confidences : dict[int, float]
            Fidelity 0 has low confidence (noisy); fidelity 1 is exact.
        """
        return {0: 0.2, 1: 1.0}

    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Return per-candidate query cost based on fidelity.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates to cost. Each must carry a fidelity id in {0, 1}.

        Returns
        -------
        costs : list[float]
            Cost for each candidate, in the same order as input.
        """
        costs = []
        for c in candidates:
            fidelity = self._validate_candidate_fidelity(
                c, self._SUPPORTED_FIDELITIES
            )
            costs.append(self._costs[fidelity])
        return costs

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        """Evaluate each candidate and return observations.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates to evaluate. Each must carry a fidelity id in {0, 1}.

        Returns
        -------
        observations : list[Observation]
            One observation per candidate in the same order as input.
        """
        observations = []
        for c in candidates:
            fidelity = self._validate_candidate_fidelity(
                c, self._SUPPORTED_FIDELITIES
            )
            x1, x2 = c.x[0], c.x[1]
            cx1, cx2 = self._center

            # Gaussian peak
            value = math.exp(
                -((x1 - cx1) ** 2 + (x2 - cx2) ** 2) / (2 * 0.1**2)
            )

            # Add noise at low fidelity
            if fidelity == 0:
                noise = torch.randn(1, dtype=self.dtype, device=self.device).item()
                value = value + self._noise_std * noise

            observations.append(
                Observation(x=c.x, y=float(value), fidelity=c.fidelity)
            )
        return observations
```

## Config model and registration

Add a Pydantic config model in `src/activelearning/oracle/config.py`:

```python
# src/activelearning/oracle/config.py  (additions)
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from activelearning.oracle.gaussian_oracle import GaussianOracle
from activelearning.oracle.oracle import Oracle


class GaussianOracleConfig(BaseModel):
    """Configuration for the 2-D Gaussian oracle."""

    type: Literal["GaussianOracle"] = "GaussianOracle"
    center: tuple[float, float] = (0.5, 0.5)
    noise_std: float = 0.1
    low_cost: float = 1.0
    high_cost: float = 5.0

    def build(self) -> Oracle:
        """Instantiate and return the oracle."""
        return GaussianOracle(
            center=self.center,
            noise_std=self.noise_std,
            low_cost=self.low_cost,
            high_cost=self.high_cost,
        )


# Extend the union — add GaussianOracleConfig
OracleConfig = Annotated[
    Union[
        BraninOracleConfig,
        Hartmann6DOracleConfig,
        CompositeOracleConfig,
        GaussianOracleConfig,   # <-- new
    ],
    Field(discriminator="type"),
]
```

Then specify it in your YAML:

```yaml
oracle:
  type: GaussianOracle
  center: [0.5, 0.5]
  noise_std: 0.1
  low_cost: 1.0
  high_cost: 5.0
```

## Fidelity id alignment with the sampler

The fidelity ids used in `get_fidelity_confidences()` must match the ids your
sampler stamps onto `Candidate.fidelity`. If the sampler emits fidelity `0` or
`1` but your oracle only declares fidelity `2`, every query will raise a
`ValueError`. The `HypercubeSampler` `fidelities` parameter controls which ids
it assigns — keep them consistent.

## Common pitfalls

**Order alignment** — `get_costs()` and `query()` must return one item per
input candidate, in the exact same order. Never sort, group, or filter the
input list before building your return list.

**Validation helper** — call `self._validate_candidate_fidelity(candidate, supported_fidelities)`
to get the validated fidelity int and raise a clear error on unexpected ids.
Pass `None` as the second argument for single-fidelity oracles.

**Confidence values** — `get_fidelity_confidences()` must return values in
`[0, 1]`. Use the `_validate_fidelity_confidences()` helper from the base class
to enforce this.

**Budget** — `query()` must not check or modify the budget. Budget tracking
happens in the loop; your oracle only observes.

**Runtime tensors** — build tensors inside `query()`, not in `__init__()`.
Use `self.dtype` and `self.device` so the runtime binding takes effect.

## Testing the new oracle

```python
from activelearning.utils.types import Candidate
from activelearning.oracle.gaussian_oracle import GaussianOracle

oracle = GaussianOracle()

candidates = [
    Candidate(x=[0.5, 0.5], fidelity=1),
    Candidate(x=[0.0, 0.0], fidelity=0),
]

costs = oracle.get_costs(candidates)
assert len(costs) == 2

observations = oracle.query(candidates)
assert len(observations) == 2
assert observations[0].fidelity == 1
assert observations[1].fidelity == 0

# High-fidelity peak should be close to 1.0
assert abs(observations[0].y - 1.0) < 1e-6
```

## Using CompositeOracle instead of building from scratch

When different oracle implementations each handle a subset of fidelity levels, use `CompositeOracle`. It merges `get_fidelity_confidences()` from all
sub-oracles, routes candidates to the cheapest sub-oracle that handles each
fidelity, and assembles results in the original candidate order.

```yaml
oracle:
  type: CompositeOracle
  sub_oracles:
    - type: BraninOracle
      fidelity_costs: {0: 1.0}
    - type: BraninOracle
      fidelity_costs: {1: 5.0}
```

Use `CompositeOracle` when fidelity levels are cleanly separated across
implementations. Write a new oracle class when the routing logic cannot be
expressed as independent sub-oracles.

## Related pages

- [Sampler guide](sampler.md) — aligning fidelity ids with the sampler
- [Extension guide overview](index.md)
- [Oracle API](../api/oracle.md)

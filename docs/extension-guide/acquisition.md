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

[`DummyAcquisition`](../api/acquisition.md#activelearning.acquisition.dummy_acquisition.DummyAcquisition) is the simplest acquisition in the library. It reads
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

## Reference implementations

Review the built-in acquisitions as concrete examples before writing your own:

- [`DummyAcquisition`](../api/acquisition.md#activelearning.acquisition.dummy_acquisition.DummyAcquisition) — scores candidates as `mean + beta * std`; the simplest possible acquisition and a good starting point for custom implementations.
- Analytic BoTorch acquisitions ([`ExpectedImprovement`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_analytic.ExpectedImprovement), [`UpperConfidenceBound`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_analytic.UpperConfidenceBound), etc.) — subclass [`AnalyticBoTorchAcquisition`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition).
- Q-batch BoTorch acquisitions ([`QMultiFidelityKnowledgeGradient`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityKnowledgeGradient), [`QMultiFidelityMaxValueEntropy`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityMaxValueEntropy), etc.) — subclass [`QBatchBoTorchAcquisition`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_acquisition.QBatchBoTorchAcquisition).

Source: `src/activelearning/acquisition/`.

## Config model and registration

Add a Pydantic config model in `src/activelearning/acquisition/config.py` and extend the `AcquisitionConfig` union. See the existing models in that file as reference.

```python
class MyAcquisitionConfig(BaseModel):
    type: Literal["MyAcquisition"] = "MyAcquisition"
    # your parameters here

    def build(self) -> Acquisition:
        return MyAcquisition(...)

AcquisitionConfig = Annotated[
    Union[..., MyAcquisitionConfig],
    Field(discriminator="type"),
]
```

Then in your YAML:

```yaml
acquisition:
  type: MyAcquisition
```

## BoTorch acquisitions

For acquisitions that wrap a BoTorch acquisition function, use the intermediate
base classes instead of [`Acquisition`](../api/acquisition.md#activelearning.acquisition.acquisition.Acquisition) directly. Both handle candidate encoding,
multi-fidelity infrastructure, and the BoTorch acquisition lifecycle:

- **[`AnalyticBoTorchAcquisition`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition)** — for analytic BoTorch acquisitions that
  score candidates independently (UCB, EI, PI, and similar).
- **[`QBatchBoTorchAcquisition`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_acquisition.QBatchBoTorchAcquisition)** — for Monte Carlo / q-batch BoTorch
  acquisitions.

The only method you must implement is `_build_botorch_acquisition()`, which
constructs and returns the internal BoTorch acquisition object. The base class
calls it during `update()` and then delegates `score()` to it.

```python
from botorch.acquisition import qExpectedImprovement
from activelearning.acquisition.botorch.botorch_acquisition import QBatchBoTorchAcquisition


class MyBoTorchAcquisition(QBatchBoTorchAcquisition):
    """Custom BoTorch acquisition wrapping qEI."""

    def _build_botorch_acquisition(self):
        model = self._botorch_surrogate.model
        best_f = max(o.y for o in self._observations_cache)
        return qExpectedImprovement(model=model, best_f=best_f)
```

For analytic acquisitions, subclass [`AnalyticBoTorchAcquisition`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition) and follow
the same pattern — see [`ExpectedImprovement`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_analytic.ExpectedImprovement) or [`UpperConfidenceBound`](../api/acquisition.md#activelearning.acquisition.botorch.botorch_analytic.UpperConfidenceBound) in
the library as reference.

## Common pitfalls

**Materializing `observations`** — the `observations` parameter in `update()` may be a one-pass generator. Convert it to a list immediately if you need to iterate it more than once.

**Never call `surrogate.predict()` in `update()`** — `update()` is for caching
state, not scoring. Predict inside `score()` or `score_batches()` instead.

**Null surrogate guard** — `self.surrogate` is `None` before `update()` has
been called. Return neutral scores (e.g. `[0.0] * n`) when the surrogate is
not yet available rather than raising.

**Cost-weighting passthrough** — if you accept `cost_weighting` in `score()`,
call it as the last step after computing raw scores, and return its result. Do
not apply cost weighting twice.

## Related pages

- [Surrogate guide](surrogate.md) — what `predict()` returns
- [Selector guide](selector.md) — how `score()` is called during selection
- [Extension guide overview](index.md)
- [Acquisition API](../api/acquisition.md)

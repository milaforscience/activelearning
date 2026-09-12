# **Adding a New Acquisition Function**

Acquisition functions map the surrogate's predictive posterior to a utility score $\alpha(x, m)$ over candidate-fidelity queries $(x, m)$. This score drives selector allocation decisions. Implement a new acquisition subclass to:

- Implement a custom information criterion (e.g. entropy search, knowledge
  gradient, bespoke cost-aware utility).
- Combine multiple signals (posterior mean + diversity bonus, multi-objective
  scalarization).
- Use a surrogate type that existing acquisitions do not expose.

## **What to implement**

Subclass `activelearning.acquisition.acquisition.Acquisition`. The core methods
are:

| Method | Required? | Notes |
|---|---|---|
| `update(surrogate, observations)` | **Recommended** | Cache the surrogate; validate compatibility |
| `score(candidates)` | **For singleton scoring** | Returns `list[float]` — one utility score per candidate |
| `score_batches(candidate_batches)` | **For batch scoring** | Returns `list[float]` — one utility score per batch |

Implement `score()` for independent per-candidate scoring, `score_batches()`
for joint batch utility, or both. The `supports_singleton_scoring` and
`supports_batch_scoring` properties are inferred automatically from which
methods you override.

## **Reference implementations**

Review the built-in acquisitions as concrete examples before writing your own:

- [`DummyAcquisition`](../reference/activelearning/acquisition/dummy_acquisition/#activelearning.acquisition.dummy_acquisition.DummyAcquisition) — scores candidates as `mean + beta * std`; the simplest possible acquisition and a good starting point for custom implementations.
- Analytic BoTorch acquisitions ([`ExpectedImprovement`](../reference/activelearning/acquisition/botorch/botorch_analytic/#activelearning.acquisition.botorch.botorch_analytic.ExpectedImprovement), [`UpperConfidenceBound`](../reference/activelearning/acquisition/botorch/botorch_analytic/#activelearning.acquisition.botorch.botorch_analytic.UpperConfidenceBound), etc.) — subclass [`AnalyticBoTorchAcquisition`](../reference/activelearning/acquisition/botorch/botorch_acquisition/#activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition).
- Q-batch BoTorch acquisitions ([`QMultiFidelityKnowledgeGradient`](../reference/activelearning/acquisition/botorch/botorch_multifidelity/#activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityKnowledgeGradient), [`QMultiFidelityMaxValueEntropy`](../reference/activelearning/acquisition/botorch/botorch_multifidelity/#activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityMaxValueEntropy), etc.) — subclass [`QBatchBoTorchAcquisition`](../reference/activelearning/acquisition/botorch/botorch_acquisition/#activelearning.acquisition.botorch.botorch_acquisition.QBatchBoTorchAcquisition).

Source: `src/activelearning/acquisition/`.

## **Config model and registration**

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

## **BoTorch acquisitions**

For acquisitions that wrap a BoTorch acquisition function, use the intermediate
base classes instead of [`Acquisition`](../reference/activelearning/acquisition/acquisition/#activelearning.acquisition.acquisition.Acquisition) directly. Both handle candidate encoding,
multi-fidelity infrastructure, and the BoTorch acquisition lifecycle:

- **[`AnalyticBoTorchAcquisition`](../reference/activelearning/acquisition/botorch/botorch_acquisition/#activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition)** — for analytic BoTorch acquisitions that
  score candidates independently (UCB, EI, PI, and similar).
- **[`QBatchBoTorchAcquisition`](../reference/activelearning/acquisition/botorch/botorch_acquisition/#activelearning.acquisition.botorch.botorch_acquisition.QBatchBoTorchAcquisition)** — for Monte Carlo / q-batch BoTorch
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

For analytic acquisitions, subclass [`AnalyticBoTorchAcquisition`](../reference/activelearning/acquisition/botorch/botorch_acquisition/#activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition) and follow
the same pattern — see [`ExpectedImprovement`](../reference/activelearning/acquisition/botorch/botorch_analytic/#activelearning.acquisition.botorch.botorch_analytic.ExpectedImprovement) or [`UpperConfidenceBound`](../reference/activelearning/acquisition/botorch/botorch_analytic/#activelearning.acquisition.botorch.botorch_analytic.UpperConfidenceBound) in
the framework as reference.

## **Common pitfalls**

**Materializing `observations`** — the `observations` parameter in `update()` may be a one-pass generator. Convert it to a list immediately if you need to iterate it more than once.

**Avoid calling `surrogate.predict()` in `update()`** — `update()` receives the
surrogate but not the candidates to score, so there is nothing meaningful to
predict yet. Cache the surrogate in `update()` and call `predict()` inside
`score()` or `score_batches()` where candidates are available.

**Null surrogate guard** — `self.surrogate` is `None` before `update()` has
been called. Return neutral scores (e.g. `[0.0] * n`) when the surrogate is
not yet available rather than raising.

**Two independent cost-weighting mechanisms exist** — labeling cost can
influence acquisition scores through two separate paths, and it is important
to understand their scope:

1. **Acquisition-level weighting** (configured at `update()` time).
   Some acquisition classes — notably BoTorch multi-fidelity acquisitions —
   accept a `cost_aware_utility` parameter that is wired into the internal
   BoTorch acquisition object when `update()` is called. From that point on,
   every call to `score()` returns values that *already* account for labeling
   cost. Because the sampler and the selector receive the same acquisition
   instance, this weighting affects both stages of candidate selection.

2. **Caller-side weighting** (the `cost_weighting` argument of `score()`).
   `score()` accepts an optional callable that post-processes raw acquisition
   scores — for example dividing each score by the candidate's labeling cost.
   This hook lets individual callers (such as `CostAwareSelector`) adjust
   scores without modifying the acquisition object itself. It is applied
   *after* any acquisition-level weighting has already been applied.

These two mechanisms are **independent**. If you configure a BoTorch MF
acquisition with `cost_aware_utility` *and* pair it with a cost-aware
selector that divides by cost again, both penalties compound. To avoid
double-counting, decide which level should own the cost penalty:

- Use **acquisition-level** weighting when you want cost awareness to
  propagate everywhere the acquisition is used (sampler score-guided
  proposals *and* selector ranking).
- Use **selector-level** weighting (via `cost_weighting` or
  `CostAwareSelector`) when you want cost to influence only the final
  selection step and leave the sampler's view of acquisition scores
  agnostic to cost.

**BoTorch MF cost utilities are frozen at `update()` time** — because the
`cost_aware_utility` is wired in during `update()`, later `score()` calls
cannot toggle it on or off per consumer. If you need different cost behavior
for the sampler and the selector, you must create separate acquisition
instances.

## **Related pages**

- [Surrogate guide](surrogate.md) — what `predict()` returns
- [Selector guide](selector.md) — how `score()` is called during selection
- [Extension guide overview](overview.md)
- [Acquisition API](../reference/activelearning/acquisition/acquisition/)

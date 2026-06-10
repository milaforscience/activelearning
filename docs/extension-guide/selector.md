# **Adding a New Selector**

Selectors choose the final subset of candidates from the pool the sampler
produces. A selector implements the **within-round candidate selection policy**
under the round budget provided by `Budget`. Implement a new selector to:

- Apply custom constraints (diversity, domain rules, batch coverage).
- Implement a new feasibility/ranking policy under a round budget (fractional
  fidelity spending, risk thresholds).
- Mix cost awareness with acquisition scoring in a custom way.

Before implementing a new selector, verify that [`TopKAcquisitionSelector`](../reference/activelearning/selector/score_selector/#activelearning.selector.score_selector.TopKAcquisitionSelector) or
[`CostAwareSelector`](../reference/activelearning/selector/cost_aware_selector/#activelearning.selector.cost_aware_selector.CostAwareSelector) does not already meet your requirements.

## **What to implement**

Subclass `activelearning.selector.selector.Selector` and implement one method:

| Method | Signature |
|---|---|
| `__call__` | `(candidates, acquisition=None, cost_fn=None, round_budget=None) -> list[Candidate]` |

Return a **subset** of the input candidates — including an empty list if no candidates are feasible. Never query the oracle, compute
observations, or modify the budget inside the selector.

## **Reference implementations**

Review the built-in selectors as concrete examples before writing your own:

- [`TopKAcquisitionSelector`](../reference/activelearning/selector/score_selector/#activelearning.selector.score_selector.TopKAcquisitionSelector) — picks the top-K candidates by acquisition score; the simplest selector.
- [`CostAwareSelector`](../reference/activelearning/selector/cost_aware_selector/#activelearning.selector.cost_aware_selector.CostAwareSelector) — selects candidates by score-per-unit-cost within the round budget; the default for multi-fidelity experiments.

Source: `src/activelearning/selector/`.

## **Config model and registration**

Add a Pydantic config model in `src/activelearning/selector/config.py` and extend the `SelectorConfig` union. See the existing models in that file as reference.

```python
class MySelectorConfig(BaseModel):
    type: Literal["MySelector"] = "MySelector"
    # your parameters here

    def build(self) -> Selector:
        return MySelector(...)

SelectorConfig = Annotated[
    Union[..., MySelectorConfig],
    Field(discriminator="type"),
]
```

Then in your YAML:

```yaml
selector:
  type: MySelector
```

## **Fidelity preservation**

Selected candidates must carry their fidelity ids unchanged. Do not modify
[`Candidate.fidelity`](../reference/activelearning/utils/types/#activelearning.utils.types.Candidate) inside the selector — the oracle and dataset use it to
route and record observations correctly. Since [`Candidate`](../reference/activelearning/utils/types/#activelearning.utils.types.Candidate) is a frozen
dataclass, accidental mutation will raise an `AttributeError`.

## **Using `acquisition.score()` inside the selector**

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

## **Common pitfalls**

**Do not query the oracle** — the selector only allocates from a pre-built pool.
Any oracle call inside the selector corrupts budget accounting.

**Do not double-count budget** — the loop deducts costs after oracle query. The
selector's job is to check feasibility, not to deduct.

**`candidate.x` shape** — `Candidate.x` can be any type (list, numpy array,
tensor). Normalize to a plain list before arithmetic comparisons if you are not
sure of the upstream sampler's output format.

## **Related pages**

- [Sampler guide](sampler.md) — how the candidate pool is generated
- [Acquisition guide](acquisition.md) — how `score()` works
- [Extension guide overview](index.md)
- [Selector API](../reference/activelearning/selector/selector/)

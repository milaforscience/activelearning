# **Adding a New Surrogate**

Surrogates approximate the objective function from collected observations,
enabling cheap candidate scoring without querying the oracle. Implement a new
surrogate subclass to:

- Use a different model family (neural network, random forest, ensemble).
- Wrap an external library (scikit-learn, GPyTorch custom kernel, JAX model).
- Implement incremental or online updates.
- Control fidelity-weighted training data.

Before implementing a new surrogate, verify that [`DummyMeanSurrogate`](../reference/activelearning/surrogate/dummy_mean_surrogate.md#activelearning.surrogate.dummy_mean_surrogate.DummyMeanSurrogate) (for
baselines) or [`BoTorchGPSurrogate`](../reference/activelearning/surrogate/botorch_surrogate.md#activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate) (for GP-based work) does not already meet your
requirements.

## **What to implement**

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

## **Reference implementations**

Review the built-in surrogates as concrete examples before writing your own:

- [`DummyMeanSurrogate`](../reference/activelearning/surrogate/dummy_mean_surrogate.md#activelearning.surrogate.dummy_mean_surrogate.DummyMeanSurrogate) — a minimal surrogate that returns a fixed mean; useful as a baseline or starting point.
- [`BoTorchGPSurrogate`](../reference/activelearning/surrogate/botorch_surrogate.md#activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate) — a full multi-fidelity GP surrogate using BoTorch; the primary production surrogate.

Source: `src/activelearning/surrogate/`.

## **Config model and registration**

Add a Pydantic config model in `src/activelearning/surrogate/config.py` and extend the `SurrogateConfig` union. See the existing models in that file as reference.

```python
class MySurrogateConfig(BaseModel):
    type: Literal["MySurrogate"] = "MySurrogate"
    # your parameters here

    def build(self) -> Surrogate:
        return MySurrogate(...)

SurrogateConfig = Annotated[
    Union[..., MySurrogateConfig],
    Field(discriminator="type"),
]
```

Then in your YAML:

```yaml
surrogate:
  type: MySurrogate
```

## **The `is_fitted()` contract**

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

## **`updates_from_latest()` semantics**

| Return value | Loop behavior | Method called |
|---|---|---|
| `False` (default) | Full refit every round | `fit(all_observations)` |
| `True` | Incremental update | `update(latest_observations_only)` |

Return `True` only when your model genuinely supports partial updates without
degrading accuracy. The `False` path is the safe default.

If `True` is returned without implementing `update()`, the base class raises
`NotImplementedError`.

## **Multi-fidelity support**

If your surrogate uses fidelity information, override `set_fidelity_confidences()`. The loop calls this method once at startup with the oracle's confidence mapping. Store the values and use them to weight training data inside `fit()` or `update()`.

## **Common Pitfalls**

**`__init__()` must not construct tensors directly** — use `self.dtype` and `self.device`
inside `fit()` or `predict()` after the runtime context has been bound.

**Materialization** — `fit()` and `update()` receive an `Iterable[Observation]`
that may be a one-pass generator. Convert to a list immediately if you need
random access or multiple passes.

**Surrogate compatibility** — BoTorch acquisitions require a
[`BoTorchGPSurrogate`](../reference/activelearning/surrogate/botorch_surrogate.md#activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate), not a generic [`Surrogate`](../reference/activelearning/surrogate/surrogate.md#activelearning.surrogate.surrogate.Surrogate). Custom surrogates that wrap
non-BoTorch models should pair with acquisitions that only use `predict()`.

## **Related pages**

- [Acquisition guide](acquisition.md) — how the acquisition consumes `predict()`
- [Extension guide overview](index.md)
- [Surrogate API](../reference/activelearning/surrogate/index.md)

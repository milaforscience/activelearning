# **Adding a New Surrogate**

Surrogates approximate the objective function from collected observations,
enabling cheap candidate scoring without querying the oracle. Implement a new
surrogate subclass to:

- Use a different model family (neural network, random forest, ensemble).
- Wrap an external library (scikit-learn, GPyTorch custom kernel, JAX model).
- Implement incremental or online updates.
- Customize how observations from different fidelity levels are encoded and
  weighted during model fitting.

Before implementing a new surrogate, verify that [`DummyMeanSurrogate`](../reference/activelearning/surrogate/dummy_mean_surrogate/#activelearning.surrogate.dummy_mean_surrogate.DummyMeanSurrogate) (for
baselines) or [`BoTorchGPSurrogate`](../reference/activelearning/surrogate/botorch_surrogate/#activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate) (for GP-based work) does not already meet your
requirements.

## **What to implement**

Subclass `activelearning.surrogate.surrogate.Surrogate`. The methods you must
or should implement depend on your update strategy:

| Method | Required? | Notes |
|---|---|---|
| `updates_from_latest()` | **Yes** | Declares loop behavior: `False` = full refit, `True` = incremental |
| `fit(observations)` | If `updates_from_latest()` returns `False` | Full refit on all observations |
| `update(observations)` | If `updates_from_latest()` returns `True` | Incremental update from latest batch |
| `predict(candidates)` | If acquisition uses `predict()` | Returns `dict` — must include at least a `"mean"` key |
| `is_fitted()` | If unsafe before training | Override to return `False` until first fit |
| `set_fidelity_confidences(confidences)` | For `MultiFidelitySurrogate` implementations | Called before `fit()` / `update()` |

## **Reference implementations**

Review the built-in surrogates as concrete examples before writing your own:

- [`DummyMeanSurrogate`](../reference/activelearning/surrogate/dummy_mean_surrogate/#activelearning.surrogate.dummy_mean_surrogate.DummyMeanSurrogate) — a minimal surrogate that returns a fixed mean; useful as a baseline or starting point.
- [`BoTorchGPSurrogate`](../reference/activelearning/surrogate/botorch_surrogate/#activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate) — a full multi-fidelity GP surrogate using BoTorch; the primary production surrogate.

Source: `src/activelearning/surrogate/`.

## **Config model and registration**

Add a `BuildableConfig` model and list it in the surrogate catalog in the same
file. External packages do not edit core's `SurrogateConfig`.

```python
from typing import ClassVar, Literal

from activelearning.config_registry import BuildableConfig


class MySurrogateConfig(BuildableConfig):
    type: Literal["MySurrogate"] = "MySurrogate"
    input_representation: ClassVar[str | None] = "numeric"
    # your parameters here

    def build(self) -> Surrogate:
        return MySurrogate(...)

    is_botorch_compatible: ClassVar[bool] = False


SURROGATE_CONFIGS = (MySurrogateConfig,)
```

Add `SURROGATE_CONFIGS` to the package's `CONFIG_CATALOGS` mapping in
`my_package/config_catalogs.py` as shown in the
[extension overview](overview.md#the-common-extension-recipe).

Then compose the application command with this mapping:

```python
from activelearning.main import run
from my_package.config_catalogs import CONFIG_CATALOGS

run(catalogs={"my-package": CONFIG_CATALOGS}, program_name="my-package")
```

```yaml
surrogate:
  type: MySurrogate
```

## **Adding a sequence encoder**

A DKL surrogate works with any encoder that maps a raw input to a fixed-width
latent feature vector. That input can be a molecular string, a protein
sequence, a text prompt, or any other domain-specific sequence.

```text
raw values -> prepare_inputs() -> model-space tensor -> forward() -> latent features
```

The surrogate interacts with the encoder through this interface and does not
need to know how tokenization works. For string inputs, a sequence encoder
handles tokenization internally: `prepare_inputs()` converts strings to token
IDs, and `forward()` maps those IDs to latent features.

!!! note "Choose the smallest applicable abstraction"
    Use [`LatentEncoder`](../reference/activelearning/surrogate/encoder/#activelearning.surrogate.encoder.LatentEncoder)
    for numeric, already-prepared, or non-tokenizer feature inputs. Subclass
    [`SequenceEncoder`](../reference/activelearning/surrogate/sequence/base/#activelearning.surrogate.sequence.base.SequenceEncoder)
    when raw strings need tokenization.

### **Implement the encoder**

Every DKL encoder exposes a positive integer `latent_dim` and implements two
methods: `prepare_inputs(values, *, device)`, which batches raw values into a
tensor, and `forward(model_inputs)`, which returns features whose last
dimension is `latent_dim`.

For tokenized strings, subclass `SequenceEncoder` and give it a
[`SequenceTokenizer`](../reference/activelearning/surrogate/sequence/tokenizer/#activelearning.surrogate.sequence.tokenizer.SequenceTokenizer).
The tokenizer exposes the vocabulary and special-token IDs, converts strings
through `batch_from_strings(strings, max_tokens, device)`, and returns a
matching mask from `attention_mask_from_batch(token_batch)`. Note that
`max_tokens` counts special tokens and padding, not just content tokens.

Models that accept raw values directly, such as graph or fingerprint
extractors, can implement `LatentEncoder` without a tokenizer. In that case,
`prepare_inputs()` performs the model-specific extraction and `forward()`
maps the resulting tensor to the latent features used by DKL.

Encoders built on a pretrained backbone should load their model in the
application layer and reuse
[`HuggingFaceSequenceEncoder`](../reference/activelearning/surrogate/sequence/huggingface_encoder/#activelearning.surrogate.sequence.huggingface_encoder.HuggingFaceSequenceEncoder),
which already handles frozen-backbone extraction, pooling, projection, and
caching.

### **Add configuration in the surrogate layer**

Concrete encoder configs belong to the distribution that owns the encoder.
Core provides the representation-independent `EncoderConfig` contract and
registry; an application package adds its concrete encoders to its
`CONFIG_CATALOGS` mapping.

Hugging Face models can subclass
[`HuggingFaceEncoderConfig`](../reference/activelearning/surrogate/sequence/config/#activelearning.surrogate.sequence.config.HuggingFaceEncoderConfig),
which already carries the shared loading and feature settings:

```python
from typing import ClassVar, Literal

from activelearning.surrogate.sequence.config import HuggingFaceEncoderConfig


class MySequenceEncoderConfig(HuggingFaceEncoderConfig):
    type: Literal["MySequenceEncoder"] = "MySequenceEncoder"
    # Optional metadata for application-level representation checks.
    input_representation: ClassVar[str | None] = None

    def _encoder_class(self) -> type[MySequenceEncoder]:
        return MySequenceEncoder
```

Any other model type uses a `BuildableConfig` config that builds its tokenizer
and encoder inside `build()`. Keep implementation imports inside `build()` or `_encoder_class()` so
importing `activelearning.surrogate` does not import an application package or
optional dependencies. Add the schema to the encoder tuple beside these
config classes, then add that tuple to the package's `CONFIG_CATALOGS` mapping.

Add an `input_representation` `ClassVar` such as `"smiles"` if the run needs
representation checks. It is metadata for the composition layer rather than
part of the encoder contract.

### **Check the integration**

Both DKL variants rely on this same contract: exact DKL calls the encoder
inside `EncoderKernel`, while variational DKL calls it before the sparse GP
head.

Cover input preparation, both DKL variants, candidate prediction, and config
parsing in tests, using fake components instead of downloaded model weights.

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

A regular `Surrogate` is single-fidelity only. To support an oracle with more
than one fidelity level, subclass `MultiFidelitySurrogate` and implement
`set_fidelity_confidences()`. The CLI composition step calls it once before
the loop starts with the oracle's `{fidelity_level: confidence}` mapping.

```python
class MyMultiFidelitySurrogate(MultiFidelitySurrogate):
    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        self._fidelity_confidences = dict(confidences)
```

Its config must implement the structural `FidelityAwareSurrogateConfig`
contract by providing `resolve_fidelity_confidences(confidences)`. This method
returns a revalidated config with any mode or target settings derived from the
oracle. `ActiveLearningConfig` rejects a multi-fidelity run when the surrogate
config does not implement this method.

```python
class MyMultiFidelitySurrogateConfig(BaseModel):
    def resolve_fidelity_confidences(
        self, confidences: dict[int, float]
    ) -> "MyMultiFidelitySurrogateConfig":
        data = self.model_dump()
        data["is_multi_fidelity"] = len(confidences) > 1
        return type(self).model_validate(data)
```

If a multi-fidelity acquisition must project encoded model inputs to a target
level, also implement the optional `TargetFidelityProjector` protocol:
`is_multi_fidelity`, `get_fidelity_dimension()`, and
`get_target_fidelity_value()`. Multi-fidelity surrogates that do not require
target projection do not need this protocol.

## **Common Pitfalls**

**`__init__()` must not construct tensors directly** — use `self.dtype` and `self.device`
inside `fit()` or `predict()` after the runtime context has been bound.

**Materialization** — `fit()` and `update()` receive an `Iterable[Observation]`
that may be a one-pass generator. Convert to a list immediately if you need
random access or multiple passes.

**Surrogate compatibility** — BoTorch acquisition configs declare
`requires_botorch_surrogate = True`. A custom surrogate that provides the
required BoTorch interfaces can opt in with `is_botorch_compatible = True`
without inheriting a specific built-in config class. Other surrogates should
pair with acquisitions that only use the generic `predict()` contract.

## **Related pages**

- [Acquisition guide](acquisition.md) — how the acquisition consumes `predict()`
- [Extension guide overview](overview.md)
- [Surrogate API](../reference/activelearning/surrogate/surrogate/)

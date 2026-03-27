# Runtime and Types

Two utility modules carry the shared state and lightweight data containers that move observations and candidate-fidelity queries through the loop:

- `activelearning.utils.runtime` manages shared torch and logging state.
- `activelearning.utils.types` defines the lightweight carrier objects that move through the loop.

## Runtime layer

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `resolve_torch_dtype(precision)` | function | Maps `32` or `64` to the corresponding `torch.dtype`. |
| `RuntimeContext` | frozen dataclass | Shared logger / device / dtype / precision bundle used at runtime. |
| `RuntimeConfig` | Pydantic model | Config entry point for materializing a `RuntimeContext`. |
| `DEFAULT_RUNTIME_CONTEXT` | constant | Default CPU / float64 fallback used when no explicit context is bound. |
| `ALRuntimeMixin` | mixin | Gives components `.runtime_context`, `.logger`, `.device`, `.dtype`, and `.precision` properties. |
| `bind_runtime_context(components, runtime_context)` | function | Batch-binds a `RuntimeContext` to each runtime-aware component in an iterable. |

### `RuntimeContext`

`RuntimeContext` stores:

- `logger`
- `device`
- `dtype`
- `precision`

Its `__post_init__` check keeps `dtype` and `precision` consistent, and `with_logger()` returns a copy with a replaced logger reference.

### How runtime binding happens

- `RuntimeConfig.build_context(logger=logger)` creates the concrete `RuntimeContext`.
- `activelearning.main` binds that context before calling the loop.
- `activelearning.active_learning.active_learning` binds it again to the assembled dataset, surrogate, acquisition, sampler, selector, and oracle instances, ensuring a consistent view.

For API consumers, the important detail is that most component classes do **not** accept `device`, `dtype`, or `logger` on every method call; they read those values from the mixin-bound runtime context instead.

## Data carrier types

Defined in `activelearning.utils.types`.

| Type | Fields | Typical producers | Typical consumers |
| --- | --- | --- | --- |
| `Candidate` | `x`, optional `fidelity` | samplers, selectors, user code | acquisitions, oracles, surrogate encoders |
| `Observation` | `x`, `y`, optional `fidelity` | oracles, datasets, helper functions | datasets, surrogates, loggers |

Both are frozen dataclasses, so they are cheap to pass around and safe to reuse. When `Candidate.fidelity` is present, the object represents the candidate-fidelity query `(x, m)` used throughout the multi-fidelity loop.

## Conversion helpers

| Function | Input | Output | Notes |
| --- | --- | --- | --- |
| `label_candidates(candidates, labels)` | `Candidate` iterable plus labels | `list[Observation]` | Lengths must match. |
| `observations_to_tensors(...)` | `Observation` iterable | `(X, y, fidelities)` | Preserves natural shapes; raises if fidelity values are present without a mapping. |
| `candidates_to_tensor(...)` | `Candidate` iterable | `(X, fidelities)` | Same fidelity rules as `observations_to_tensors`. |
| `_to_tensor(...)` | internal helper | `torch.Tensor` | Tries fast batch conversion first, then falls back to element-wise stacking. |

The `fidelities` return values are lists of continuous confidence values used internally by the multi-fidelity surrogate, not the original integer fidelity ids.

## Runtime/type connection points

| Connection | Where it happens |
| --- | --- |
| integer fidelity ids become continuous model-space values | `observations_to_tensors()` and `candidates_to_tensor()` consume an oracle-provided `fidelity_confidences` mapping |
| model-space fidelity is appended as the last input column | `BoTorchGPSurrogate._parse_observations()` and `BoTorchGPSurrogate.encode_candidates()` |
| components gain shared torch / logger state | `ALRuntimeMixin` properties read from the bound `RuntimeContext` |
| plotting and synthetic-oracle evaluation honor runtime settings | oracle plotting helpers and augmented-function score functions create tensors with the bound `dtype` and `device` |

## Practical implications

- If you build a multi-fidelity surrogate, set fidelity confidences **before** fitting so candidate-fidelity queries are encoded consistently. The active-learning loop does this automatically by calling `surrogate.set_fidelity_confidences(oracle.get_fidelity_confidences())`.
- If you write a custom component, inheriting `ALRuntimeMixin` is the standard way to participate in shared runtime state.
- If you write a custom tensor-based model, `Candidate` and `Observation` remain the outer API, while `activelearning.utils.types` handles conversion to tensors.

## Class Reference

::: activelearning.runtime.RuntimeContext
    options:
      show_source: false
      heading_level: 3

::: activelearning.runtime.RuntimeConfig
    options:
      show_source: false
      heading_level: 3

::: activelearning.runtime.ALRuntimeMixin
    options:
      show_source: false
      heading_level: 3

::: activelearning.utils.types.Candidate
    options:
      show_source: false
      heading_level: 3

::: activelearning.utils.types.Observation
    options:
      show_source: false
      heading_level: 3

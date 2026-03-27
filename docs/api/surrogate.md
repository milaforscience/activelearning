# Surrogate API

Surrogates turn `Observation` objects into predictive state that acquisition functions use to score candidate or candidate-fidelity queries.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.surrogate.surrogate` | `Surrogate` | Abstract contract used by the loop and acquisitions. |
| `activelearning.surrogate.dummy_mean_surrogate` | `DummyMeanSurrogate` | Small in-memory baseline surrogate. |
| `activelearning.surrogate.botorch_surrogate` | `BoTorchGPSurrogate` | BoTorch GP implementation with single- and multi-fidelity support. |
| `activelearning.surrogate.config` | `DummyMeanSurrogateConfig`, `BoTorchGPSurrogateConfig`, `SurrogateConfig` | Config builders. |

## Core abstraction: `Surrogate`

| Method | Meaning in the loop | Notes |
| --- | --- | --- |
| `updates_from_latest()` | Declares whether the loop should call `update(latest)` or `fit(all)` | `False` is the safe default. |
| `fit(observations)` | Full retraining path | Called when `updates_from_latest()` is `False`. |
| `update(observations)` | Incremental-update path | Only used when `updates_from_latest()` is `True`. |
| `is_fitted()` | Signals whether acquisitions may safely consume this surrogate | The loop skips `acquisition.update()` until this becomes `True`. |
| `set_fidelity_confidences(confidences)` | Receives oracle-provided fidelity metadata before multi-fidelity fitting | No-op by default. |
| `predict(candidates)` | Optional generic prediction surface | Some acquisitions use this; others work through framework-specific helpers instead. |

## Implementations

| Class | Module | Update strategy | Prediction surface | Config-exposed |
| --- | --- | --- | --- | --- |
| `DummyMeanSurrogate` | `dummy_mean_surrogate.py` | Full refit only (`updates_from_latest() -> False`) | Returns `{"mean": ..., "std": ...}` | yes |
| `BoTorchGPSurrogate` | `botorch_surrogate.py` | Full refit, or incremental conditioning when `use_partial_updates=True` and a model is already fitted | Returns `{"mean": ..., "std": ..., "posterior": ...}` and exposes BoTorch-specific helpers | yes |

### `DummyMeanSurrogate`

`DummyMeanSurrogate` is the baseline concrete surrogate in the tree:

- observations are cached in a dict keyed by `(x, fidelity)`
- seen points return their cached value with `_KNOWN_STD = 0.1`
- unseen points return the global mean with `_UNKNOWN_STD = 1.0`

That makes it a good fit for smoke tests, baseline behavior, and `DummyAcquisition`.

### `BoTorchGPSurrogate`

`BoTorchGPSurrogate` is the main production-oriented implementation for single- and multi-fidelity active learning.

#### Model construction

| Situation | Model built |
| --- | --- |
| single-fidelity observations | `botorch.models.SingleTaskGP` |
| multi-fidelity observations and no custom kernel | `botorch.models.gp_regression_fidelity.SingleTaskMultiFidelityGP` |
| multi-fidelity observations with a custom `covar_module` | `botorch.models.SingleTaskGP` with the supplied kernel |

Input scaling and output standardization are optional and configured by `scale_inputs` and `standardize_outputs`.

#### BoTorch-facing helpers

| Helper | Why it matters |
| --- | --- |
| `get_model()` | Gives BoTorch acquisitions the fitted GP model. |
| `get_train_data()` | Exposes the current model-space training tensors. |
| `state_dict()` and `load_state_dict()` | Support model checkpointing or preloaded hyperparameters. |
| `get_fidelity_confidences()` | Returns the configured integer-to-confidence mapping. |
| `get_fidelity_dimension()` | Returns the appended fidelity column index in model space. |
| `get_target_fidelity_value()` | Returns the highest configured confidence value, used by multi-fidelity acquisitions as the target corresponding to the highest-fidelity available oracle. |
| `encode_candidates()` | Converts `Candidate` objects into model-space tensors. |
| `encode_candidate_batches()` | Converts batches of candidates into BoTorch q-batch tensors. |

#### Runtime and type interaction

- `fit()` and `update()` convert `Observation` objects through `observations_to_tensors(...)`.
- `predict()` and the acquisition helpers convert `Candidate` objects through `candidates_to_tensor(...)`.
- In multi-fidelity mode, the continuous confidence value for the queried fidelity is appended as the **last input column**.
- All tensor creation honors the bound runtime `dtype` and `device`.

#### Update semantics

- Before the first successful fit, `is_fitted()` is `False`.
- With `use_partial_updates=False`, `update()` still rebuilds the full model on accumulated data.
- With `use_partial_updates=True`, `update()` uses BoTorch's `condition_on_observations(...)` path once a model already exists.

## Configuration

Defined in `activelearning.surrogate.config`.

| Config model | Key fields | Build result |
| --- | --- | --- |
| `DummyMeanSurrogateConfig` | `type` | `DummyMeanSurrogate()` |
| `BoTorchGPSurrogateConfig` | `scale_inputs`, `standardize_outputs`, `optimize_hyperparameters`, `fit_kwargs`, `custom_fit_function`, `covar_module`, `covar_module_kwargs`, `use_partial_updates` | `BoTorchGPSurrogate(...)` |

Config-time validation details:

- `custom_fit_function` is an `ImportString` and must resolve to a callable.
- `covar_module` is an `ImportString` and must resolve to either a `gpytorch.module.Module` instance or a callable that constructs one.
- `covar_module_kwargs` are only valid when `covar_module` is configured.
- If `covar_module` resolves to an already-instantiated module, `covar_module_kwargs` are rejected.

## Choosing a surrogate module

- Need a baseline surrogate for tests or examples? Start with `DummyMeanSurrogate`.
- Need BoTorch-based acquisition support, model access, or multi-fidelity scoring of candidate-fidelity queries? Use `BoTorchGPSurrogate`.
- Need to expose a new surrogate in YAML? Add it to `activelearning.surrogate.config.SurrogateConfig`.

## Class Reference

::: activelearning.surrogate.surrogate.Surrogate
    options:
      show_source: false
      heading_level: 3

::: activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate
    options:
      show_source: false
      heading_level: 3

::: activelearning.surrogate.dummy_mean_surrogate.DummyMeanSurrogate
    options:
      show_source: false
      heading_level: 3

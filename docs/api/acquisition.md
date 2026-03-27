# Acquisition API

Acquisition functions score `Candidate` objects using the current surrogate state. In multi-fidelity runs, that means scoring candidate-fidelity queries under the current predictive and cost-aware model.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.acquisition.acquisition` | `Acquisition` | Framework-agnostic acquisition base class. |
| `activelearning.acquisition.dummy_acquisition` | `DummyAcquisition` | Generic UCB-like scorer using `surrogate.predict()`. |
| `activelearning.acquisition.botorch_acquisition` | `BoTorchAcquisitionBase`, `AnalyticBoTorchAcquisition`, `QBatchBoTorchAcquisition` | Shared BoTorch plumbing and scoring semantics. |
| `activelearning.acquisition.botorch_analytic` | `UpperConfidenceBound`, `ExpectedImprovement`, `LogExpectedImprovement`, `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`, `PosteriorMean` | Analytic BoTorch wrappers. |
| `activelearning.acquisition.botorch_qbatch` | `QExpectedImprovement`, `QLogExpectedImprovement`, `QNoisyExpectedImprovement`, `QLogNoisyExpectedImprovement`, `QUpperConfidenceBound`, `QProbabilityOfImprovement`, `QSimpleRegret`, `QKnowledgeGradient` | Monte Carlo q-batch wrappers. |
| `activelearning.acquisition.botorch_multifidelity` | `QMultiFidelityKnowledgeGradient`, `QMultiFidelityMaxValueEntropy`, `QMultiFidelityLowerBoundMaxValueEntropy` | Multi-fidelity q-batch wrappers. |
| `activelearning.acquisition.botorch_posterior_mean_acquisition` | `BoTorchPosteriorMeanAcquisition` | Config-friendly name for posterior mean. |
| `activelearning.acquisition.botorch_multifidelity_mes_acquisition` | `BoTorchMultiFidelityMaxValueEntropyAcquisition` | Config-friendly multi-fidelity MES wrapper. |
| `activelearning.acquisition.config` | acquisition config models | YAML-facing builders. |

## Core abstraction: `Acquisition`

Defined in `activelearning.acquisition.acquisition`.

| Method or property | Purpose | Notes |
| --- | --- | --- |
| `surrogate` | Returns the currently attached surrogate | Populated by `update()`. |
| `update(surrogate, observations=None)` | Refresh internal state after the surrogate changes | Base implementation only stores the surrogate reference. |
| `score(candidates)` | Score candidates or candidate-fidelity queries independently | Used by all built-in selectors. |
| `score_batches(candidate_batches)` | Score batches jointly | Needed for true q-batch selection logic. |
| `supports_singleton_scoring()` | Capability check for `score()` | Prefer this over duck-typing. |
| `supports_batch_scoring()` | Capability check for `score_batches()` | Useful for custom selectors or samplers. |

The active-learning loop calls `acquisition.update(...)` only after `surrogate.is_fitted()` becomes `True`. Until then, concrete acquisitions typically fall back to zero-valued scores so the first round can still propose candidates or candidate-fidelity queries.

## BoTorch acquisition base classes

Defined in `activelearning.acquisition.botorch_acquisition`.

| Class | What it adds |
| --- | --- |
| `BoTorchAcquisitionBase` | Validates that the surrogate is a `BoTorchGPSurrogate`, materializes one-pass observation iterables, resolves target-fidelity projection, cost models, cost-aware utilities, and builds the internal BoTorch acquisition object. |
| `AnalyticBoTorchAcquisition` | Implements singleton `score()` using encoded candidates and an analytic BoTorch acquisition object. |
| `QBatchBoTorchAcquisition` | Implements `score()` as q=1 and `score_batches()` for true q-batch tensors. |

### Common BoTorch options

The shared base constructor accepts several options that matter across analytic, q-batch, and multi-fidelity subclasses:

| Option | Meaning |
| --- | --- |
| `maximize` | Whether objective-related quantities should be interpreted as maximization. |
| `target_fidelity_value` | Optional override for the encoded target fidelity used by multi-fidelity projection. |
| `project_to_target_fidelity_fn` | Custom projector from encoded model inputs to target fidelity. |
| `fidelity_costs` | Integer fidelity-to-cost mapping used to build a default `AffineFidelityCostModel`. |
| `cost_model` | Custom BoTorch-compatible cost model. |
| `cost_aware_utility` | Custom BoTorch-compatible cost-aware utility. |

If `fidelity_costs` are provided and no custom cost model or utility is supplied, the base class derives an `AffineFidelityCostModel` and wraps it in `InverseCostWeightedUtility`, giving the acquisition a default cost-aware view of candidate-fidelity queries.

## Concrete acquisitions

### Config-exposed acquisitions

| Class | Module | Behavior | Config model |
| --- | --- | --- | --- |
| `DummyAcquisition` | `dummy_acquisition.py` | Uses `surrogate.predict()` and computes `mean + beta * std` when `std` is available, otherwise returns `mean` | `DummyAcquisitionConfig` |
| `BoTorchPosteriorMeanAcquisition` | `botorch_posterior_mean_acquisition.py` | Thin, configuration-stable wrapper around the analytic `PosteriorMean` acquisition | `BoTorchPosteriorMeanAcquisitionConfig` |
| `BoTorchMultiFidelityMaxValueEntropyAcquisition` | `botorch_multifidelity_mes_acquisition.py` | Thin, configuration-stable wrapper around `QMultiFidelityMaxValueEntropy` for cost-aware candidate-fidelity query scoring that derives its `candidate_set` from surrogate training inputs | `BoTorchMultiFidelityMaxValueEntropyAcquisitionConfig` |

### Programmatic BoTorch wrappers present in source

| Family | Classes |
| --- | --- |
| analytic wrappers | `UpperConfidenceBound`, `ExpectedImprovement`, `LogExpectedImprovement`, `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`, `PosteriorMean` |
| q-batch wrappers | `QExpectedImprovement`, `QLogExpectedImprovement`, `QNoisyExpectedImprovement`, `QLogNoisyExpectedImprovement`, `QUpperConfidenceBound`, `QProbabilityOfImprovement`, `QSimpleRegret`, `QKnowledgeGradient` |
| multi-fidelity wrappers | `QMultiFidelityKnowledgeGradient`, `QMultiFidelityMaxValueEntropy`, `QMultiFidelityLowerBoundMaxValueEntropy` |

These classes are part of the source tree and can be instantiated directly from Python, but only the three config-exposed rows above are currently selectable through `AcquisitionConfig`.

## Family-specific notes

### `DummyAcquisition`

- supports singleton scoring only
- requires `surrogate.predict()` to return at least a `"mean"` key
- if no surrogate has been attached yet, returns zeros in candidate order

### Analytic BoTorch wrappers

The analytic classes in `botorch_analytic.py` build the corresponding BoTorch analytic acquisition object inside `_build_botorch_acquisition()`. They all depend on a fitted `BoTorchGPSurrogate` and use `encode_candidates()` to obtain model-space tensors.

### Q-batch BoTorch wrappers

The q-batch classes in `botorch_qbatch.py` are native batch scorers, but `QBatchBoTorchAcquisition.score()` still works by treating each singleton candidate as a q-batch of size 1. That makes them usable with the current built-in selectors, even though the selectors do not yet exploit `score_batches()`.

### Multi-fidelity BoTorch wrappers

The multi-fidelity classes in `botorch_multifidelity.py` reuse the shared cost-aware and target-fidelity helpers from `BoTorchAcquisitionBase`. `BoTorchMultiFidelityMaxValueEntropyAcquisition` adds one extra convenience layer by deriving the BoTorch `candidate_set` from the surrogate's fitted training inputs and projecting it to target fidelity when needed, so candidate-fidelity queries are compared against a consistent highest-fidelity target.

## Configuration

Defined in `activelearning.acquisition.config`.

| Config model | Builds | Notes |
| --- | --- | --- |
| `DummyAcquisitionConfig` | `DummyAcquisition(beta=...)` | Generic acquisition that only needs `predict()`. |
| `BoTorchPosteriorMeanAcquisitionConfig` | `BoTorchPosteriorMeanAcquisition(maximize=...)` | Small config surface; stable explicit name for YAML. |
| `BoTorchMultiFidelityMaxValueEntropyAcquisitionConfig` | `BoTorchMultiFidelityMaxValueEntropyAcquisition(...)` | Exposes `maximize`, fantasy and sample counts, and optional `fidelity_costs`. |
| `AcquisitionConfig` | discriminated union | Currently limited to the three models above. |

## Runtime and type connections

- All acquisitions consume `Candidate` objects at the public API boundary; when `candidate.fidelity` is present, that boundary is the candidate-fidelity query `(x, m)`.
- `DummyAcquisition` stays fully framework-agnostic and only relies on the surrogate's `predict()` payload.
- BoTorch acquisitions require `BoTorchGPSurrogate` and work in model space via `encode_candidates()` and `encode_candidate_batches()`.
- Multi-fidelity acquisitions use the oracle-to-surrogate fidelity confidence mapping indirectly, because the surrogate exposes encoded fidelity values and target-fidelity helpers.

If you are adding a new acquisition that should be selectable from YAML, update both the runtime module and `activelearning.acquisition.config.AcquisitionConfig`.

## Class Reference

::: activelearning.acquisition.acquisition.Acquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.dummy_acquisition.DummyAcquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_acquisition.BoTorchAcquisitionBase
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_acquisition.QBatchBoTorchAcquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.UpperConfidenceBound
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.ExpectedImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.LogExpectedImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.ProbabilityOfImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.LogProbabilityOfImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.PosteriorMean
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityMaxValueEntropy
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityLowerBoundMaxValueEntropy
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityKnowledgeGradient
    options:
      show_source: false
      heading_level: 3

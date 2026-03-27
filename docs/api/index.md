# API Reference

This section documents the hand-written, module-level API surface under `src/activelearning/`. It stays reference-oriented: the pages map the component families that assemble a multi-fidelity active learning run, from candidate-fidelity query scoring to budget-constrained oracle queries.

## How the package is assembled

| Step | Entry point | What it contributes |
| --- | --- | --- |
| 1 | `activelearning.main` | Loads YAML, applies OmegaConf dotlist overrides, bootstraps logger backends, and parses the config tree for a run. |
| 2 | `activelearning.config.ActiveLearningConfig` | Validates the top-level configuration model. |
| 3 | Family `config.py` modules | Build concrete `Dataset`, `Surrogate`, `Acquisition`, `Sampler`, `Selector`, `Oracle`, `Budget`, and optional `Logger` instances. |
| 4 | `activelearning.utils.runtime` | Builds a shared `RuntimeContext` and binds it to runtime-aware components. |
| 5 | `activelearning.active_learning.active_learning` | Runs the active-learning loop with the assembled runtime objects and selected candidate-fidelity queries. |

## Component-family roles and boundaries

The table below lists the **API-specific caveats** for each family. For a description of each component's conceptual role in the loop, see [Framework Overview](../concepts/overview.md#component-architecture).

<div class="schema-table" markdown>

| Family | Important boundary or caveat |
| --- | --- |
| `Dataset` | Observation store only: it does not model the objective, score future queries, or enforce budget. |
| `Surrogate` | Belief-state layer only: it does not choose queries or spend budget. The multi-fidelity surrogate path does not imply that every sampler-acquisition combination is already a paper-equivalent end-to-end workflow. |
| `Acquisition` | Scoring layer only. Built-in selectors currently consume singleton `score()` results even though q-batch wrappers exist in the source tree. |
| `Sampler` | Proposal layer only: it does not choose final queries or allocate budget. Current GFlowNet samplers are candidate-generation machinery, not a fully packaged end-to-end multi-fidelity GFlowNet workflow. |
| `Selector` | Built-in selectors are singleton-score oriented; `TopKAcquisitionSelector` ignores oracle cost and round budget entirely. |
| `Oracle` | The oracle is the authority for the multi-fidelity structure. Budget is still consumed by the active-learning loop rather than inside the oracle itself. |
| `Budget` | Accounting layer only: it constrains the search but does not score candidates or call the oracle. |
| `Logger` | Observability only: logging changes traceability, not query selection, surrogate fitting, or budget accounting. |

</div>

## Reference map

| Page | Covers | Main modules |
| --- | --- | --- |
| [Configuration](config.md) | Top-level config model, discriminated unions, and CLI build flow | `activelearning.config`, `activelearning.main`, `activelearning.utils.config_loader` |
| [Runtime and types](runtime_and_types.md) | Shared runtime state, `Candidate`, `Observation`, and tensor conversion helpers | `activelearning.utils.runtime`, `activelearning.utils.types` |
| [Dataset](dataset.md) | Observation storage and round-stable iterables | `activelearning.dataset.*` |
| [Surrogate](surrogate.md) | Predictive models and BoTorch integration | `activelearning.surrogate.*` |
| [Acquisition](acquisition.md) | Candidate-fidelity query scoring interfaces and BoTorch acquisition-function families | `activelearning.acquisition.*` |
| [Sampler](sampler.md) | Candidate or candidate-fidelity query proposal generation, including GFlowNet samplers | `activelearning.sampler.*` |
| [Selector](selector.md) | Final candidate-fidelity query selection under score or budget constraints | `activelearning.selector.*` |
| [Oracle](oracle.md) | Oracle query/cost interfaces and multi-fidelity objective oracles | `activelearning.oracle.*` |
| [Budget](budget.md) | Oracle-budget accounting and per-round schedules | `activelearning.budget.*` |
| [Logger](logger.md) | Optional logging backends and logger config helpers | `activelearning.logger.*` |

## Package structure notes

- The top-level package `activelearning` does not currently re-export submodules from `__init__.py`. In practice, imports target concrete modules such as `activelearning.surrogate.config` or `activelearning.oracle.augmented_function_oracle`.
- The config layer is intentionally explicit. Some runtime classes exist in source without a matching `...Config` wrapper yet; the family pages call out which implementations are configuration-friendly and which are programmatic only.
- Most core component families inherit `ALRuntimeMixin`, which is how the same logger, torch device, and dtype propagate across the loop.
- The detailed family pages in this API section are now the canonical reference;
  the legacy `docs/components/` pages are kept only as short redirects so they
  can be removed from navigation without losing module-family explanations.

## Fast lookup

| If you need to understand... | Start here |
| --- | --- |
| how YAML becomes runtime objects | [Configuration](config.md) |
| why components have `.device`, `.dtype`, and `.logger` | [Runtime and types](runtime_and_types.md) |
| where observations are stored between rounds | [Dataset](dataset.md) |
| which acquisition classes are actually config-exposed today | [Acquisition](acquisition.md) |
| how multi-fidelity confidence values flow from oracle to surrogate | [Oracle](oracle.md) and [Runtime and types](runtime_and_types.md) |

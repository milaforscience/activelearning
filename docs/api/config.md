# Configuration Model

The configuration layer is a thin Pydantic facade over the runtime modules in `src/activelearning/`. It validates the experiment contract for a multi-fidelity active learning run: which surrogate models the expensive objective, which acquisition function scores candidate-fidelity queries, which sampler proposes them, which selector spends the round budget, and which oracle defines costs and fidelities.

## Load, validate, build

The CLI path in `activelearning.main` is:

1. `load_config(path, overrides)` loads YAML and applies OmegaConf dotlist overrides.
2. `bootstrap_logger_backend_imports(raw_cfg)` eagerly imports logger SDKs that care about import order, notably Comet.
3. `parse_config(raw_cfg, ActiveLearningConfig)` validates the tree into Pydantic models.
4. `main()` builds each component explicitly and then starts `active_learning(...)`.

## Top-level model: `ActiveLearningConfig`

Defined in `activelearning.config`.

| Field | Config type | Built into | Notes |
| --- | --- | --- | --- |
| `runtime` | `RuntimeConfig` | `RuntimeContext` | Has a default factory, so a config can omit this section and still get CPU / float64 defaults. |
| `dataset` | `DatasetConfig` | `Dataset` | Discriminated union. |
| `surrogate` | `SurrogateConfig` | `Surrogate` | Discriminated union. |
| `acquisition` | `AcquisitionConfig` | `Acquisition` | Discriminated union. |
| `sampler` | `SamplerConfig` | `Sampler` | Discriminated union; the build step can inspect `runtime`. |
| `selector` | `SelectorConfig` | `Selector` | Discriminated union. |
| `oracle` | `OracleConfig` | `Oracle` | Discriminated union, including recursive composite oracles. |
| `budget` | `BudgetConfig` | `Budget` | Uses its own nested schedule union. |
| `logger` | optional `LoggerConfig` | optional `Logger` | The whole logging layer is optional. |

`ActiveLearningConfig` is only a validation container: it does **not** expose a single top-level `.build()` method. The orchestration lives in `activelearning.main.main()`.

## Build sequence in `activelearning.main`

The current CLI builds objects in this order:

- `dataset = cfg.dataset.build()`
- `surrogate = cfg.surrogate.build()`
- `acquisition = cfg.acquisition.build()`
- `sampler = cfg.sampler.build(runtime=cfg.runtime)`
- `selector = cfg.selector.build()`
- `oracle = cfg.oracle.build()`
- `budget = cfg.budget.build()`
- `logger = cfg.logger.build() if cfg.logger is not None else None`
- `runtime_context = cfg.runtime.build_context(logger=logger)`

That split is important for two families:

- `RuntimeConfig` builds a `RuntimeContext`, not a pipeline component.
- `SamplerConfig.build()` can reuse `RuntimeConfig` values when a sampler wants to inherit device / precision defaults.

## Family entry points

| Family | Config entry point | Concrete config models currently present | Build signature |
| --- | --- | --- | --- |
| runtime | `activelearning.utils.runtime.RuntimeConfig` | `RuntimeConfig` | `build_context(logger=...)` |
| dataset | `activelearning.dataset.config.DatasetConfig` | `ListDatasetConfig` | `build()` |
| surrogate | `activelearning.surrogate.config.SurrogateConfig` | `DummyMeanSurrogateConfig`, `BoTorchGPSurrogateConfig` | `build()` |
| acquisition | `activelearning.acquisition.config.AcquisitionConfig` | `DummyAcquisitionConfig`, `BoTorchPosteriorMeanAcquisitionConfig`, `BoTorchMultiFidelityMaxValueEntropyAcquisitionConfig` | `build()` |
| sampler | `activelearning.sampler.config.SamplerConfig` | `HypercubeSamplerConfig`, `GFlowNetSamplerConfig`, `GFlowNetGridSamplerConfig` | `build(runtime=...)` |
| selector | `activelearning.selector.config.SelectorConfig` | `TopKAcquisitionSelectorConfig`, `CostAwareSelectorConfig`, `KnapsackSelectorConfig` | `build()` |
| oracle | `activelearning.oracle.config.OracleConfig` | `BraninOracleConfig`, `Hartmann6DOracleConfig`, `CompositeOracleConfig` | `build()` |
| budget | `activelearning.budget.config.BudgetConfig` | `BudgetConfig` plus nested `ScheduleConfig` | `build()` |
| logger | `activelearning.logger.config.LoggerConfig` | `ConsoleLoggerConfig`, `WandbLoggerConfig`, `CometLoggerConfig`, `AimLoggerConfig`, `MultiLoggerConfig` | `build()` |

## Discriminator pattern

Every union-typed family uses a `type` field as the discriminator. For example:

```yaml
surrogate:
  type: BoTorchGPSurrogate
  use_partial_updates: true
```

The config layer does not scan modules dynamically. If a runtime class exists in source but is absent from the union, it is programmatically usable but not currently selectable from YAML.

## Helper functions worth knowing

| Symbol | Module | Purpose |
| --- | --- | --- |
| `load_config` | `activelearning.utils.config_loader` | Load YAML and apply OmegaConf dotlist overrides. |
| `parse_config` | `activelearning.utils.config_loader` | Validate a resolved OmegaConf tree into a Pydantic model. |
| `load_and_parse` | `activelearning.utils.config_loader` | Convenience wrapper around the two steps above. |
| `build_logger` | `activelearning.logger.config` | Build an optional logger from a logger config or return `None`. |
| `bootstrap_logger_backend_imports` | `activelearning.logger.config` | Import Comet early when the raw config references `CometLogger`. |

## What this layer does not do

- It does not expose a plugin registry beyond the explicit unions above.
- It does not hide runtime wiring; `main.py` still shows exactly how the pipeline is assembled.
- It does not replace the runtime APIs described in the rest of this section. Config models are builders, not the main execution surface.

## Class Reference

::: activelearning.config.ActiveLearningConfig
    options:
      show_source: false
      heading_level: 3

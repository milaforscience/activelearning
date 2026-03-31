# API Reference

This section documents the public API surface under `src/activelearning/`. Each page covers one component family — from surrogate fitting and acquisition scoring to budget accounting and logging — and shows the auto-generated class reference.

For a conceptual explanation of how these components interact at runtime, see [Active Learning Loop](../concepts/active_learning_loop.md) and [Framework Overview](../concepts/overview.md).

## Reference map

| Page | Covers |
| --- | --- |
| [Configuration](config.md) | Top-level config model, discriminated unions, and CLI build flow |
| [Runtime and Types](runtime_and_types.md) | Shared runtime state, `Candidate`, `Observation`, and tensor helpers |
| [Dataset](dataset.md) | Observation storage and round-stable iterables |
| [Surrogate](surrogate.md) | Predictive models and BoTorch GP integration |
| [Acquisition](acquisition.md) | Candidate-fidelity scoring and BoTorch acquisition families |
| [Sampler](sampler.md) | Candidate and candidate-fidelity proposal generation |
| [Selector](selector.md) | Budget-aware final query selection |
| [Oracle](oracle.md) | Query evaluation, cost assignment, and fidelity validation |
| [Budget](budget.md) | Oracle cost accounting and per-round spending schedules |
| [Logger](logger.md) | Optional logging backends: console, Aim, W&B, Comet |

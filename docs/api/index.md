# API Reference

This section documents the public API surface under `src/activelearning/`. Each page covers one component family — from surrogate fitting and acquisition scoring to budget accounting and logging — and shows the auto-generated class reference.

For a conceptual explanation of how these components interact at runtime, see [Active Learning Loop](../concepts/active_learning_loop.md) and [Framework Overview](../concepts/overview.md). For a complete auto-generated listing of every module, see the [Module Summary](../reference/summary.md).

## Reference map

| Module | Covers |
| --- | --- |
| [Configuration](../reference/activelearning/config.md) | Top-level config model, discriminated unions, and CLI build flow |
| [Runtime](../reference/activelearning/runtime.md) | Shared runtime state and execution context |
| [Types](../reference/activelearning/utils/types.md) | `Candidate`, `Observation`, and tensor helpers |
| [Dataset](../reference/activelearning/dataset/index.md) | Observation storage and round-stable iterables |
| [Surrogate](../reference/activelearning/surrogate/index.md) | Predictive models and BoTorch GP integration |
| [Acquisition](../reference/activelearning/acquisition/index.md) | Candidate-fidelity scoring and BoTorch acquisition families |
| [Sampler](../reference/activelearning/sampler/index.md) | Candidate and candidate-fidelity proposal generation |
| [Selector](../reference/activelearning/selector/index.md) | Budget-aware final query selection |
| [Oracle](../reference/activelearning/oracle/index.md) | Query evaluation, cost assignment, and fidelity validation |
| [Budget](../reference/activelearning/budget/index.md) | Oracle cost accounting and per-round spending schedules |
| [Logger](../reference/activelearning/logger/index.md) | Optional logging backends: console, Aim, W&B, Comet |

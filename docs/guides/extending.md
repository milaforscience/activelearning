# Extending the Framework

Detailed extension guides are in the **[Extension Guide](../extension-guide/index.md)** section, with one dedicated page per component type. The table below maps each extension target to its guide and required interface methods.

## Component guides

| What you want to change | Guide | Key method(s) |
| --- | --- | --- |
| Evaluation rule, simulator, fidelity structure | [Oracle](../extension-guide/oracle.md) | `get_fidelity_confidences()`, `get_costs()`, `query()` |
| Candidate proposal strategy, search space | [Sampler](../extension-guide/sampler.md) | `sample()` |
| Predictive model (GP, NN, ensemble…) | [Surrogate](../extension-guide/surrogate.md) | `updates_from_latest()`, `fit()` / `update()`, `predict()` |
| Information criterion, scoring function | [Acquisition](../extension-guide/acquisition.md) | `update()`, `score()` |
| Round budget allocation policy | [Selector](../extension-guide/selector.md) | `__call__()` |

## Related references

- [Extension Guide overview](../extension-guide/index.md) — common recipe, runtime context, quick-reference table
- [Runtime and configuration](../concepts/runtime_and_configuration.md)
- [API Reference](../api/index.md)

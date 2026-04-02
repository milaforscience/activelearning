# Available Configs

The repository ships several config files in `config/`. Use this page to choose the right starting point.

## Tutorial configs

These are the primary runnable baselines, validated against the current schema:

| File | Description |
| --- | --- |
| `config/branin_single_fidelity.yaml` | Single-fidelity Branin with GP surrogate, MES acquisition, and budget 300/30. |
| `config/branin_multi_fidelity.yaml` | Multi-fidelity Branin with fidelity costs 0.01 / 0.1 / 1.0 and budget 300/30. |
| `config/hartmann_single_fidelity.yaml` | Single-fidelity Hartmann6D with GP surrogate, MES acquisition, and budget 100/10. |
| `config/hartmann_multi_fidelity.yaml` | Multi-fidelity Hartmann6D with fidelity costs 0.125 / 0.25 / 1.0 and budget 100/10. |

See the [Branin tutorial](../tutorials/branin_experiment.md) and [Hartmann6D tutorial](../tutorials/hartmann_experiment.md) for guided walkthroughs.

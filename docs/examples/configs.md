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

## Other configs

| File | Description |
| --- | --- |
| `config/aim_logging.yaml` | Logger overlay — compose with any base config to add Aim alongside the console. |
| `config/branin_gflownet_toy.yaml` | Branin with GFlowNet sampler. Demonstrates the `sampler.conf.*` override structure. |
| `config/hartmann6d_toy.yaml` | Hartmann6D scaffold (legacy schema). Use as a reference for bounds and fidelity costs; requires schema updates before running. |

For GFlowNet sampler details, see [GFlowNet Sampler Setup](gflownet_sampler.md).

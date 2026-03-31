# Hartmann6D Benchmark

The Hartmann6D oracle is available in the repo. This page shows how to adapt the Branin tutorial config to run Hartmann6D experiments.

## Adapting the Branin config

Start from `config/branin_multi_fidelity.yaml` and change the following fields:

```yaml
surrogate:
  covar_module_kwargs:
    ard_num_dims: 7

acquisition:
  fidelity_costs: {1: 0.125, 2: 0.25, 3: 1.0}

sampler:
  bounds:
    - [0.0, 1.0]
    - [0.0, 1.0]
    - [0.0, 1.0]
    - [0.0, 1.0]
    - [0.0, 1.0]
    - [0.0, 1.0]

oracle:
  type: Hartmann6DOracle
  fidelity_costs: {1: 0.125, 2: 0.25, 3: 1.0}
```

!!! note "`ard_num_dims: 7`"
    The surrogate receives six design coordinates plus one appended fidelity-confidence feature, so ARD requires 7 dimensions.

## Scaffold config

`config/hartmann6d_toy.yaml` captures the benchmark shape (six-dimensional unit hypercube, fidelity costs `0.125 / 0.25 / 1.0`, sigmoid budget schedule) but uses legacy component tags. It is useful as a reference for copying Hartmann-specific fields into a current-schema config.

## Running

Once you have a current-schema Hartmann config, run it the same way as any other config:

```sh
uv run activelearning config/my_hartmann6d.yaml
```

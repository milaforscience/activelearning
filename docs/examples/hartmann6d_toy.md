# Hartmann6D Benchmark

This is the canonical Hartmann6D benchmark page. The current repo ships the
oracle and the benchmark scaffold, but not a committed current-schema Hartmann
run file, so the main task is to adapt the working Branin stack without
overstating paper replication.

## Status at a glance

| Layer | Asset | Status today | Use it for |
| --- | --- | --- | --- |
| Runnable path | local adaptation of `config/branin_botorch_toy.yaml` | current adaptation path | running Hartmann6D through the existing BoTorch baseline stack |
| Adaptable scaffold | `config/hartmann6d_toy.yaml` | legacy schema | preserving Hartmann-specific bounds, costs, and budget shaping |
| Future replication work | committed Hartmann YAML, paper manifests, and result scripts | not bundled | systematic paper-style reruns |

## Current runnable path

Until a dedicated Hartmann YAML is checked in, start from
`config/branin_botorch_toy.yaml` and change only the Hartmann-specific pieces:

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

`ard_num_dims: 7` is required because the surrogate receives six design coordinates plus
one appended fidelity-confidence feature. For an initial validation run, keep the
budget schedule constant; a staged schedule can be added later for longer allocation studies.

## Adaptable scaffold

`config/hartmann6d_toy.yaml` is still the right place to copy the benchmark
shape from:

- six input dimensions, each in `[0, 1]`,
- fidelity costs `0.125 / 0.25 / 1.0`, and
- a `sigmoid_iterations` schedule for spending less budget early and more later.

It is not a runnable baseline because it still uses legacy tags
`DummyDataset`, `DummySurrogate`, and `HypercubeUniformSampler`, and it omits
`oracle.fidelity_costs`.

## Future replication work

Hartmann6D still needs a packaged replication layer:

- a committed current-schema Hartmann YAML,
- exact paper-aligned seeds and hyperparameters,
- and result aggregation scripts for comparison or figure generation.

Use [Paper Replication](../paper_replication/index.md) for the remaining gap.
The current page is about benchmark wiring, not a claim that the full paper
pipeline is already bundled.

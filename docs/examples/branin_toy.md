# Branin Benchmark

This is the canonical Branin benchmark page. It separates the current runnable
baseline from the lighter scaffolds and from the paper-replication work that is
still outside the repo.

## Status at a glance

| Layer | Asset | Status today | Use it for |
| --- | --- | --- | --- |
| Runnable baseline | `config/branin_botorch_toy.yaml` | current and validated | end-to-end CLI runs, validation runs, and local ablations |
| Adaptable scaffold | `config/branin_toy.yaml` | legacy schema | the smallest multi-fidelity Branin benchmark skeleton |
| Adaptable GFlowNet scaffold | `config/branin_gflownet_toy.yaml` | partial integration | candidate-generation experiments, not a paper-equivalent MF-GFN baseline |
| Future replication work | paper-aligned manifests, seeds, and result scripts | not bundled | exact paper-style reruns |

## Current runnable baseline

`config/branin_botorch_toy.yaml` is the repo's benchmark-grade Branin entry
point. It combines `BoTorchGPSurrogate`,
`BoTorchMultiFidelityMaxValueEntropyAcquisition`, `HypercubeSampler`,
`KnapsackSelector`, and `BraninOracle` with the current three-level cost model
`0.01 / 0.1 / 1.0`.

A small validation run is:

```bash
uv run activelearning config/branin_botorch_toy.yaml \
  budget.available_budget=0.01 \
  sampler.num_samples=20 \
  selector.time_limit=5 \
  selector.verbose=false \
  acquisition.num_fantasies=2 \
  acquisition.num_mv_samples=2 \
  acquisition.num_y_samples=4
```

This path serves as the current validated baseline for budget-constrained
multi-fidelity discovery on Branin.

## Adaptable scaffolds

### `config/branin_toy.yaml`

The legacy scaffold serves as a compact benchmark reference, not a current run file.
It still captures the Branin domain `[-5, 10] x [0, 15]`, the intended
three-level fidelity structure, and a small constant-budget loop. To make it
current-schema, replace the sampler tag with `HypercubeSampler`, add a
`point_strategy`, and provide `oracle.fidelity_costs`.

### `config/branin_gflownet_toy.yaml`

The GFlowNet file provides the current `sampler.conf.*` override shape
for acquisition-guided candidate generation. The following caveats apply:

- sampler construction still expects a bundled `config/gflownet/env/grid.yaml`
  file that is not present in this snapshot,
- the sampler currently returns candidate points without fidelity labels, and
- the example uses a single oracle fidelity (`{0: 1.0}`), so it is not a
  packaged multi-fidelity paper baseline.

For the low-level sampler surface, see [GFlowNet Sampler Setup](gflownet_sampler.md).

## Future replication work

The missing Branin replication layer is mostly experiment packaging rather than
basic oracle support:

- checked-in paper-specific manifests and seeds,
- a fully bundled end-to-end GFlowNet path,
- and result scripts for tables, plots, or figure regeneration.

Use [Paper Replication](../paper_replication/index.md) for that boundary. The current runnable baseline does not constitute a claim of one-to-one paper parity.

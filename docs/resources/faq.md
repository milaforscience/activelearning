# FAQ

## What is the recommended first run?

Start from `config/branin_single_fidelity.yaml` with the reduced-budget command in
[Quickstart](../getting-started/quickstart.md). It is the smallest validated
path for confirming that configuration loading, runtime setup, acquisition,
budget-aware selection, oracle queries, and logging all work locally.

## What does multi-fidelity mean in this framework?

The action space is a candidate-fidelity pair $(x, m)$, not only a candidate $x$. The oracle defines which fidelities are valid, their query costs, and, optionally, their fidelity confidences. Lower-fidelity approximations can guide the search when the highest-fidelity available oracle is too costly to query everywhere.

## Where do fidelity costs and confidences belong in the config?

Define per-fidelity costs in `oracle.fidelity_costs`. Cost-aware acquisition functions read fidelity costs directly from the oracle at runtime — there is no separate `acquisition.fidelity_costs` field. If you want to override the default confidence heuristic in the built-in benchmark oracles, add `oracle.fidelity_confidences`.

## What does this framework support?

The framework supports fidelity-aware data types, multi-fidelity benchmark oracles (Branin, Hartmann), cost-aware acquisition functions, cost-weighted candidate-fidelity proposals, and modular samplers ([`HypercubeSampler`](../api/sampler.md#activelearning.sampler.hypercube_sampler.HypercubeSampler), [`PoolUniformSampler`](../api/sampler.md#activelearning.sampler.pool_uniform_sampler.PoolUniformSampler), [`PoolScoreSampler`](../api/sampler.md#activelearning.sampler.pool_score_sampler.PoolScoreSampler)). A GFlowNet-based sampler is planned as a future extension.

<!-- See [GFlowNet Sampler Setup](../tutorials/gflownet_sampler.md) for details on the GFlowNet integration. -->

## How should I cite this project?

See [References and Citation](references.md) for the recommended BibTeX entry and related software links.

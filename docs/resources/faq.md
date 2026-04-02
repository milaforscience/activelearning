# FAQ

## What is the recommended first run?

Start from `config/branin_single_fidelity.yaml` with the reduced-budget command in
[Quickstart](../getting-started/quickstart.md). It is the smallest validated
path for confirming that configuration loading, runtime setup, acquisition,
budget-aware selection, oracle queries, and logging all work locally.

## What does multi-fidelity mean in this framework?

The action space is a candidate-fidelity pair $(x, m)$, not only a candidate $x$. The oracle defines which fidelities are valid, their query costs, and, optionally, their fidelity confidences. Lower-fidelity approximations can guide the search when the highest-fidelity available oracle is too costly to query everywhere.

## Where do fidelity costs and confidences belong in the config?

Define per-fidelity costs in `oracle.fidelity_costs`. For cost-aware acquisition functions, pass the same mapping to `acquisition.fidelity_costs`. If you want to override the default confidence heuristic in the built-in benchmark oracles, add `oracle.fidelity_confidences`.

## What does this framework support?

The framework supports fidelity-aware data types, multi-fidelity benchmark oracles (Branin, Hartmann6D), cost-aware acquisition functions, cost-weighted candidate-fidelity proposals, and GFlowNet samplers for generative candidate generation.

<!-- See [GFlowNet Sampler Setup](../tutorials/gflownet_sampler.md) for details on the GFlowNet integration. -->

## How should I cite this project?

See [References and Citation](references.md) for the recommended BibTeX entry and related software links.

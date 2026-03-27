# FAQ

## Is this repository a full reproduction of the paper?

No. The current snapshot is best read as a modular re-implementation and benchmark scaffold for [*Multi-Fidelity Active Learning with GFlowNets*](http://arxiv.org/abs/2306.11715). The primary validated baseline is the BoTorch Branin path described in [Quickstart](../getting-started/quickstart.md) and [Branin Benchmark](../examples/branin_toy.md).

## What is the difference between a runnable baseline and a scaffold?

A **runnable baseline** is a checked-in path that is already validated end to end in the current repository. A **scaffold** captures the intended structure for a benchmark, component, or paper-aligned workflow, but still requires additional packaging or implementation work before it should be treated as the default run path.

## What is the recommended first run?

Start from `config/branin_botorch_toy.yaml` with the reduced-budget command in [Quickstart](../getting-started/quickstart.md). It is the smallest validated baseline for confirming that configuration loading, runtime setup, multi-fidelity acquisition, budget-aware selection, oracle queries, and logging all work locally.

## What does multi-fidelity mean in this framework?

The action space is a candidate-fidelity pair $(x, m)$, not only a candidate $x$. The oracle defines which fidelities are valid, their query costs, and, optionally, their fidelity confidences. Lower-fidelity approximations can guide the search when the highest-fidelity available oracle is too costly to query everywhere.

## Where do fidelity costs and confidences belong in the config?

Define per-fidelity costs in `oracle.fidelity_costs`. For cost-aware acquisition functions, pass the same mapping to `acquisition.fidelity_costs`. If you want to override the default confidence heuristic in the built-in benchmark oracles, add `oracle.fidelity_confidences`.

## How should I interpret the current GFlowNet support?

The repository includes GFlowNet samplers and configuration scaffolding, but the end-to-end paper-equivalent multi-fidelity GFlowNet path is not fully packaged in the current snapshot. Use [Paper Replication](../paper_replication/index.md) and [GFlowNet Sampler Setup](../examples/gflownet_sampler.md) for the current implementation boundary.

## How should I cite this project?

Cite the original paper that motivates the repository. The recommended citation and the most relevant software references are collected on [References and Citation](references.md).

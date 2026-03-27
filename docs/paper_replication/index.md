# Paper Replication

This section is now intentionally narrow: it tracks the gap between the current
benchmark docs and a fuller reproduction of
[*Multi-Fidelity Active Learning with GFlowNets*](http://arxiv.org/abs/2306.11715).
The benchmark instructions themselves live under **Examples**.

## Canonical benchmark pages

- [Branin Benchmark](../examples/branin_toy.md) — current runnable baseline,
  adaptable scaffolds, and Branin-specific caveats.
- [Hartmann6D Benchmark](../examples/hartmann6d_toy.md) — current runnable
  adaptation path and the Hartmann scaffold.
- [GFlowNet Sampler Setup](../examples/gflownet_sampler.md) — current partial
  sampler integration and its limits.

For setup and run hygiene before a paper-aligned study, see
[Running Experiments](../guides/experiments.md).

## Replication boundary today

| Area | Available now | Still missing for paper-style reruns |
| --- | --- | --- |
| Benchmarks | runnable Branin baseline, Hartmann oracle, and documented adaptation paths | paper-specific manifests, seeds, and locked comparison configs |
| GFlowNet | sampler config surface and code-level integration hooks | bundled env config, fidelity-aware end-to-end path, and packaged benchmark comparisons |
| Outputs | local logging and manual experimentation | automated tables, plots, and figure regeneration |

This section serves as a status record for future replication work, not a
duplicate of the benchmark walkthroughs.

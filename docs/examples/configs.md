# Benchmark Configs

Use this page to choose the right benchmark layer before editing YAML. The
canonical benchmark notes now live under **Examples**; the **Paper
Replication** section is only for what still needs to be added for paper-aligned
reruns.

## Canonical benchmark pages

| Benchmark | Canonical page | Runnable baseline today | Adaptable scaffolds |
| --- | --- | --- | --- |
| Branin | [Branin Benchmark](branin_toy.md) | `config/branin_botorch_toy.yaml` | `config/branin_toy.yaml`, `config/branin_gflownet_toy.yaml` |
| Hartmann6D | [Hartmann6D Benchmark](hartmann6d_toy.md) | local adaptation of `config/branin_botorch_toy.yaml` | `config/hartmann6d_toy.yaml` |

## Config roles at a glance

| File | Category | Current status | Best use |
| --- | --- | --- | --- |
| `config/branin_botorch_toy.yaml` | current runnable baseline | validates against the current top-level schema and is the repo's documented end-to-end benchmark path | validation runs, short pilots, and controlled ablations |
| `config/branin_toy.yaml` | adaptable scaffold | legacy sampler tag and missing `oracle.fidelity_costs`; not runnable as-is | preserving the small Branin search space, fidelity costs, and budget shape |
| `config/branin_gflownet_toy.yaml` | adaptable GFlowNet scaffold | useful sampler config surface, but still depends on a missing bundled env config and an external fidelity-assignment step | sampler-side experiments and `sampler.conf.*` overrides |
| `config/hartmann6d_toy.yaml` | adaptable scaffold | legacy dataset/surrogate/sampler tags plus missing `oracle.fidelity_costs`; not a current CLI baseline | carrying Hartmann bounds, fidelity costs, and the staged budget schedule into a local run file |

## Usage guidelines

- Start from a runnable baseline when one exists.
- Reuse scaffolds for benchmark-specific structure; they are not directly executable.
- Treat [Paper Replication](../paper_replication/index.md) as a status ledger
  for future work, not as a second copy of the benchmark instructions.
- Use [GFlowNet Sampler Setup](gflownet_sampler.md) for the current partial
  GFlowNet path.

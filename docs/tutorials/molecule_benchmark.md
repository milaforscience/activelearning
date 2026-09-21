# xTB IP/EA molecule benchmark

This how-to runs the modernized molecule benchmark based on the experiments in
[MF-GFN](https://arxiv.org/abs/2306.11715). It uses xTB IP/EA as the oracle,
frozen pooled GP-MoLFormer features with an exact GP surrogate, a GFlowNet over
SELFIES for molecule generation, and a cost-aware selector.

!!! warning
    These are modernized experiments, not bit-for-bit reproductions of the
    paper. They replace the paper's surrogate with an exact GP on frozen
    GP-MoLFormer features and its top-k selection with a cost-aware greedy
    selector, use the train-only initialization assets from
    `origin/reproduce_experiments`, and retain the historical fidelity-2 cost
    of `3.5` rather than the paper's printed `3`. S3-GFN overlays remain under
    `methods/` but are not part of this matrix.

## Prerequisites

- Install the repository and the molecule application:

  ```bash
  uv sync --all-packages --extra comet
  ```

- Install the `xtb` executable and make sure it is on `PATH`.
- Ensure the configured Hugging Face GP-MoLFormer checkpoints are available.
- Run commands from the repository root.

The benchmark uses canonical SMILES at runtime. The bundled initialization
files retain their original SELFIES values and include the converted SMILES
column for provenance.

## Experiment matrix

Each task uses seeds `42`, `43`, and `44`.

| Method | Task data | Molecule sampler | Fidelity queries |
| --- | --- | --- | --- |
| SF-GFN | EA-SF or IP-SF | SELFIES GFlowNet | Fidelity 3 only |
| MF-GFN | EA-MF or IP-MF | Learned-fidelity SELFIES GFlowNet | 1, 2, or 3 |
| Random-fidelity GFN | EA-MF or IP-MF | SELFIES GFlowNet with a uniform fidelity action | Uniformly sampled 1, 2, or 3 |
| Random | EA-MF or IP-MF | Uniform SELFIES-to-SMILES sampler | Uniformly assigned 1, 2, or 3 |

The xTB costs are `1`, `3.5`, and `7` for fidelities 1, 2, and 3. Each run
proposes 640 molecules, selects queries by acquisition value per unit cost up
to a per-round budget of 896 (128 fidelity-3 queries), and has a
post-initialization acquisition budget of 1260.

!!! note
    Fidelity-3 rescoring for the reported metrics is tracked separately from
    the active-learning acquisition budget. It is not added to the plotted
    cumulative acquisition cost.

## Run the benchmark

Preview the complete 24-run matrix without starting xTB or downloading model
weights:

```bash
uv run python scripts/run_molecule_benchmark.py --dry-run
```

Run one smoke configuration after reducing its budget and proposal counts:

```bash
uv run python scripts/run_molecule_benchmark.py \
  --task ea \
  --method mf_gfn \
  --seed 42 \
  --override budget.available_budget=7 \
  --override budget.schedule.value=7 \
  --override sampler.n_samples=4 \
  --override sampler.conf.gflownet.optimizer.n_train_steps=5
```

Run the full matrix:

```bash
uv run python scripts/run_molecule_benchmark.py
```

On Mila, run the launcher with `bash` from a login node. It submits one
independent Slurm job per EA/IP, method, and seed combination (24 jobs) on the
`main` partition, so the jobs run in parallel as resources allow:

```bash
uv sync --frozen --all-packages --extra comet
bash scripts/run_molecule_benchmark_slurm.sh
```

Runs log to the console and to Comet. Paste your Comet API key into the
`COMET_API_KEY` placeholder at the top of the launcher, or export
`COMET_API_KEY` before running it. Extra arguments are forwarded to every
`sbatch` call, for example
`bash scripts/run_molecule_benchmark_slurm.sh --partition=long`. Each job
requests one GPU and writes its run artifacts under
`outputs/xtb_ipea_benchmark/<task>/<method>/seed_<seed>`. Slurm logs are
written as `slurm-xtb-ipea-<task>-<method>-s<seed>-<job>.out` and `.err` in the
repository root. Set `MOLECULE_BENCHMARK_DRY_RUN=1` before submission to
validate the expanded commands without launching experiments.

Filter by task or method with repeatable options:

```bash
uv run python scripts/run_molecule_benchmark.py \
  --task ip \
  --method sf_gfn \
  --method random \
  --seed 42 \
  --seed 43 \
  --seed 44
```

## Evaluate and plot

After runs finish, evaluate every distinct acquired molecule at fidelity 3.
The evaluator resumes from its JSONL cache and records failed evaluations
explicitly:

```bash
uv run python scripts/evaluate_molecule_benchmark.py \
  outputs/xtb_ipea_benchmark \
  --output outputs/xtb_ipea_benchmark/metrics
```

Create the IP/EA score and diversity figure:

```bash
uv run python scripts/plot_molecule_benchmark.py \
  outputs/xtb_ipea_benchmark/metrics/molecule_metrics.json \
  --output-dir outputs/xtb_ipea_benchmark/figures
```

The plot contains mean top-100 fidelity-3 score and mean pairwise Tanimoto
distance, aggregated over the three seeds for all four methods. Diversity uses
radius-2, 2048-bit Morgan fingerprints. The evaluator writes the table used
for plotting alongside the metrics artifact.

## Add an ablation

Configurations are merged left to right. To test another frozen encoder, replace
the encoder overlay (for example, with a future `encoders/minimol.yaml`) while
keeping the task and method overlays unchanged:

```bash
uv run activelearning-molecules \
  applications/molecules/config/xtb_ipea_benchmark/base.yaml \
  applications/molecules/config/xtb_ipea_benchmark/encoders/gp_molformer.yaml \
  applications/molecules/config/xtb_ipea_benchmark/tasks/ea_mf.yaml \
  applications/molecules/config/xtb_ipea_benchmark/methods/gfn.yaml
```

S3-GFN comparisons follow the same pattern: use the `methods/s3gfn.yaml` or
`methods/random_fidelity_s3gfn.yaml` overlay with the existing task and encoder
overlays.

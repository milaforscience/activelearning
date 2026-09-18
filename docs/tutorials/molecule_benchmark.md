# xTB IP/EA molecule benchmark

This how-to runs the modernized molecule benchmark based on the experiments in
[MF-GFN](https://arxiv.org/abs/2306.11715). It uses xTB IP/EA as the oracle,
frozen pooled GP-MoLFormer features with a variational GP surrogate, and S3-GFN for
molecule generation.

!!! warning
    These are modernized experiments, not bit-for-bit reproductions of the
    paper. They use S3-GFN instead of classic GFlowNet and the train-only
    initialization assets from `origin/reproduce_experiments`, and retain the
    historical fidelity-2 cost of `3.5` rather than the paper's printed `3`.

## Prerequisites

- Install the repository and the molecule application:

  ```bash
  uv sync --package activelearning-molecules
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
| SF-S3-GFN | EA-SF or IP-SF | S3-GFN | Fidelity 3 only |
| MF-S3-GFN | EA-MF or IP-MF | Learned-fidelity S3-GFN | 1, 2, or 3 |
| Random-fidelity S3-GFN | EA-MF or IP-MF | Target-fidelity S3-GFN | Uniformly assigned 1, 2, or 3 |
| Random | EA-MF or IP-MF | Uniform SELFIES-to-SMILES sampler | Uniformly assigned 1, 2, or 3 |

The xTB costs are `1`, `3.5`, and `7` for fidelities 1, 2, and 3. Each run
proposes 640 molecules, selects up to 128 queries per round, and has a
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
  --method mf_s3gfn \
  --seed 42 \
  --override budget.available_budget=7 \
  --override sampler.n_samples=4 \
  --override selector.num_samples=1
```

Run the full matrix:

```bash
uv run python scripts/run_molecule_benchmark.py
```

Filter by task or method with repeatable options:

```bash
uv run python scripts/run_molecule_benchmark.py \
  --task ip \
  --method sf_s3gfn \
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
  applications/molecules/config/xtb_ipea_benchmark/methods/s3gfn.yaml
```

Classic GFlowNet comparisons follow the same pattern: add a method overlay
under `methods/` and reuse the existing task and encoder overlays.

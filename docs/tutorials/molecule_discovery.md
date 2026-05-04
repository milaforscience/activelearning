# **Molecule Discovery with SELFIES and xTB**

This tutorial assumes you are already comfortable with the previous tutorials: running experiments from YAML, composing logger overlays, interpreting multi-fidelity budgets, and configuring the GFlowNet sampler. We will focus on what changes when the search space is **molecules** rather than real-valued vectors.

The molecule examples follow the same budget-constrained active-learning loop introduced earlier, but the candidate \(x\) is now a molecular string. The objective is a molecular property such as **electron affinity** (EA) or **ionisation potential** (IP) computed by xTB, and the sampler proposes SELFIES strings instead of points in a box.

!!! note "Reference"
    The molecule workflow mirrors the molecular discovery setting in [Hernandez-Garcia et al., 2023](https://arxiv.org/abs/2306.11715): multi-fidelity active learning over a structured molecular search space, with GFlowNets used to discover diverse high-scoring candidates under a limited oracle budget.

## **The molecule pipeline**

At a high level, the framework models molecules as:

```text
SELFIES string
  -> tokenize into SELFIES tokens
  -> embed with a Transformer encoder
  -> fit a deep-kernel GP surrogate
  -> score candidate molecules with an acquisition function
  -> sample candidates from a pool or a SELFIES GFlowNet environment
  -> query xTB for IP/EA at the requested fidelity
```

The important distinction from Branin or Hartmann is that the input space is no longer a fixed-dimensional vector space. Instead, each molecule is a discrete structured object represented as a SELFIES string. The surrogate model operates on learned latent vectors derived from the SELFIES strings via the [`SelfiesTransformerEncoder`](../reference/activelearning/applications/molecules/selfies_transformer_encoder/#activelearning.applications.molecules.selfies_transformer_encoder.SelfiesTransformerEncoder), and the sampler generates new SELFIES strings either from a finite pool or by constructing them token by token in a GFlowNet environment. The oracle evaluates the molecular properties using xTB, which can be computationally expensive, hence the need for careful active learning and multi-fidelity strategies.

The repository includes five example molecule configs arranged as an incremental progression:

| Config | Sampler | Surrogate | Acquisition | Fidelity setting | Purpose |
|--------|---------|-----------|-------------|------------------|---------|
| `config/molecules/exact.yaml` | Pool file | Exact SELFIES DKL | UCB | pool fidelity `1` only | Stage 1: smallest pool-based baseline |
| `config/molecules/exact_multi_fidelity.yaml` | Pool file | Exact SELFIES DKL | MF-MES with cost utility | pool fidelities `1 / 2 / 3` | Stage 2: same pool setup with multi-fidelity scoring |
| `config/molecules/gflownet_exact.yaml` | SELFIES GFlowNet | Exact SELFIES DKL | UCB | fixed fidelity `1` | Stage 3: swap the pool sampler for a GFlowNet |
| `config/molecules/gflownet_exact_multi_fidelity.yaml` | SELFIES GFlowNet | Exact SELFIES DKL | MF-MES with cost utility | learned fidelity `1 / 2 / 3` | Stage 4: let the GFlowNet learn molecule-fidelity pairs |
| `config/molecules/gflownet_variational_multi_fidelity.yaml` | SELFIES GFlowNet | Variational SELFIES DKL | MF-MES with cost utility | learned fidelity `1 / 2 / 3` | Stage 5: keep the MF GFlowNet and swap in the scalable variational surrogate |

!!! tip
    The easiest way to understand the molecule examples is to work through them in order: start with `exact.yaml`, then add multi-fidelity structure with `exact_multi_fidelity.yaml`, then switch samplers with `gflownet_exact.yaml`, and only introduce the variational surrogate in the last step with `gflownet_variational_multi_fidelity.yaml`.

!!! note "Small defaults for fast checks"
    These examples are tuned to be runnable tutorial setups, not fully optimized molecule-discovery runs. The short command overrides below keep the active-learning budget small enough for a quick functional check, and the checked-in GFlowNet examples also use relatively short training schedules in the exact-surrogate stages so you can verify the full loop quickly. For better learning, increase both the oracle budget so the surrogate sees more observations and the GFlowNet optimization steps so the policy can better approximate reward-proportional sampling.

## **What are SELFIES?**

[SELFIES](https://arxiv.org/abs/1905.13741) (**Self-Referencing Embedded Strings**) are a string representation for molecules. Like SMILES, they encode molecular graphs as text. Unlike SMILES, SELFIES are designed so that generated token sequences decode into chemically valid molecular graphs under the SELFIES grammar.

That property is useful for active learning: a sampler can operate over discrete tokens without constantly producing invalid molecules. In this framework:

1. A candidate molecule is stored as a SELFIES string such as `[C][=C][O]`.
2. The tokenizer maps each SELFIES token to an integer ID, adds special tokens such as `[CLS]` and `[EOS]`, and pads to a fixed length.
3. The Transformer encoder embeds the token sequence and pools it into one latent molecule vector.
4. The DKL surrogate fits a GP on those latent vectors and provides posterior predictions to the acquisition function.
5. The oracle decodes SELFIES to a molecular graph before constructing a 3-D geometry for xTB.

!!! info "Why not optimise SMILES directly?"
    SMILES strings are compact and widely used, but arbitrary token sequences can be invalid. SELFIES make the generative part of the problem easier: the GFlowNet can learn over a constrained token language, and the oracle only has to reject molecules that fail downstream geometry construction or xTB evaluation.

## **xTB, IP, EA, and fidelities**

[xTB](https://xtb-docs.readthedocs.io/) is a semi-empirical quantum chemistry program. In these examples the oracle uses the [GFN2-xTB](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01176) method. It is much cheaper than high-level quantum chemistry, but still expensive enough that the active-learning loop should spend queries carefully.

The built-in [`XTBIPEAOracle`](../reference/activelearning/applications/molecules/xtb_oracle/#activelearning.applications.molecules.xtb_oracle.XTBIPEAOracle) computes one of two objectives:

| Task | Meaning | Intuition |
|------|---------|-----------|
| `ea` | [Electron affinity](https://en.wikipedia.org/wiki/Electron_affinity) | Energy change when the molecule accepts an electron. |
| `ip` | [Ionisation potential](https://en.wikipedia.org/wiki/Ionization_energy) | Energy required to remove an electron. |

The three xTB fidelities trade cost for accuracy:

| Fidelity | What happens | Cost in the provided configs |
|----------|--------------|------------------------------|
| `1` | RDKit/MMFF geometry, then vertical IP/EA with xTB. | `1.0` |
| `2` | xTB-optimise the neutral geometry, then compute vertical IP/EA. | `3.5` |
| `3` | Optimise neutral and ionic geometries, then compute adiabatic IP/EA. | `7.0` |

!!! warning "External dependency"
    The Python `molecules` extra installs SELFIES and RDKit, but the `xtb` executable must be installed separately and available on your `PATH`. Check this before running a molecule experiment:

    ```sh
    xtb --version
    ```

## **1. Prepare the molecule environment**

Install the optional molecule dependencies. If you also want the molecule grids in Aim, install the Aim extra at the same time:

```sh
uv sync --extra molecules --extra aim
```

The pool-based configs read one SELFIES string per line from:

```text
config/data/molecules.txt
```

You can replace that file with your own pool later. For now, keep the checked-in pool so the config paths work unchanged.

## **2. Run the pool-based DKL examples**

Start with the single-fidelity exact-DKL pool config and reduce it to a short run that still queries several molecules:

```sh
uv run activelearning config/molecules/exact.yaml \
  budget.available_budget=5.0 \
  budget.schedule.value=5.0 \
  selector.num_samples=5 \
  sampler.num_samples=50
```

You should see the same round-level fields as in the synthetic tutorials, plus a figure acknowledgement from the console logger:

```text
[Figure] 'xtb_ea_query_molecules' (not rendered in console)
[Step 1] round=1 | num_new_samples=5 | round_cost=5.0000 | total_cost=5.0000 | budget_remaining=0.0000
Done. Rounds: 1 | Total cost: 5.0000
```

The exact numbers depend on which candidate was selected and whether xTB succeeds for that molecule. Invalid molecules or failed xTB calculations are recorded as `NaN` and filtered before surrogate fitting.

Once the plumbing is working, run the full single-fidelity exact-DKL config:

```sh
uv run activelearning config/molecules/exact.yaml
```

Then move to the multi-fidelity pool version, which keeps the exact SELFIES DKL surrogate but exposes the full xTB fidelity ladder and uses MF-MES with cost-aware selection:

```sh
uv run activelearning config/molecules/exact_multi_fidelity.yaml
```

This second stage is the first place where the active-learning loop has to decide how much accuracy is worth paying for. The acquisition scores candidates with respect to the target high-fidelity objective, while the selector divides by query cost so cheap fidelities are favored when they carry similar information.

## **3. Log molecule visualizations**

The molecule configs enable xTB query visualizations in the oracle block:

```yaml
oracle:
    type: XTBIPEAOracle
    task: ea
    log_molecule_visualizations: true
    molecule_visualization_limit: 25
```

`log_molecule_visualizations` asks the oracle to render the queried molecules as a 2-D RDKit grid after each oracle call. `molecule_visualization_limit` caps the grid at 25 molecules; if a batch contains more than 25, the displayed molecules are the highest-scoring queried molecules in that batch.

The console logger only acknowledges the figure. To inspect a more informative grid, compose the run with the Aim overlay and keep the short budget large enough to query several molecules:

```sh
uv run activelearning config/molecules/exact.yaml config/aim_logging.yaml \
  budget.available_budget=5.0 \
  budget.schedule.value=5.0 \
  selector.num_samples=5 \
  sampler.num_samples=50
```

Open Aim:

```sh
uv run aim up
```

In the Aim UI, open the run and inspect **Images**. For EA runs, the image key is `xtb_ea_query_molecules`; for IP runs, it is `xtb_ip_query_molecules`. Each panel shows the query index, fidelity, observed score in eV, and the decoded molecule identifier.

The example below was generated from a short fidelity-1 EA run over the bundled SELFIES pool, then ranking the successful xTB evaluations by observed EA and rendering the top four molecules as a 2x2 grid:

![Top four molecules from a short xTB EA example](../assets/molecule_top4_xtb_ea.png)

!!! tip "Switching the objective"
    To optimise ionisation potential instead of electron affinity, override the task:

    ```sh
    uv run activelearning config/molecules/exact.yaml \
      oracle.task=ip \
      logger.run_name=molecules-dkl-exact-ip
    ```

## **4. Run the single-fidelity SELFIES GFlowNet sampler**

The single-fidelity GFlowNet molecule configs replace the finite pool sampler with a SELFIES sequence environment and stamp every sampled molecule with `fixed_fidelity: 1`:

```yaml
sampler:
    type: GFlowNetSampler
    n_samples: 16
    fixed_fidelity: 1
    conf:
        env:
            _target_: gflownet.envs.sequences.selfies.Selfies
            env_id: selfies
            id: selfies
            max_length: 8
```

Here, the GFlowNet constructs a molecule token by token in the SELFIES environment. The reward is still derived from the active-learning acquisition function, as in the previous GFlowNet tutorial, but the terminal object is now a SELFIES molecule rather than a grid coordinate.

Run a short exact-DKL GFlowNet check:

```sh
uv run activelearning config/molecules/gflownet_exact.yaml \
  sampler.n_samples=8 \
  selector.num_samples=1 \
  sampler.conf.gflownet.optimizer.n_train_steps=10 \
  budget.schedule.value=1.0 \
  budget.available_budget=1.0
```

For the full small tutorial run, drop the overrides:

```sh
uv run activelearning config/molecules/gflownet_exact.yaml
```

!!! note "Why keep the single-fidelity GFlowNet config?"
    It is the cheapest molecule GFlowNet example in the repository and is convenient for checking that the SELFIES environment, xTB oracle, and GFlowNet training loop all run end-to-end before spending budget on multi-fidelity experiments.

## **5. Run the multi-fidelity SELFIES GFlowNet sampler**

The exact multi-fidelity GFlowNet config lets the policy sample both a SELFIES string and a fidelity level:

```yaml
sampler:
    type: GFlowNetSampler
    n_samples: 64
    n_fidelities: 3
    fidelity_action: any
    conf:
        env:
            _target_: gflownet.envs.sequences.selfies.Selfies
            env_id: selfies
            id: selfies
            max_length: 8
```

In this setup, the GFlowNet is not limited to a fixed oracle level. It learns a policy over molecule-fidelity pairs, and the MF-MES acquisition rewards candidates that are informative about the target high-fidelity objective while the cost-aware utility discounts unnecessarily expensive queries. That is one of the main reasons GFlowNets are attractive in the multi-fidelity setting: the same policy can discover promising molecule structures and learn when a cheap xTB evaluation is enough versus when it is worth paying for a higher-fidelity query.

The smaller exact-GP multi-fidelity example is:

```sh
uv run activelearning config/molecules/gflownet_exact_multi_fidelity.yaml
```

For a shorter sanity check, reduce the GFlowNet training steps and selected batch size:

```sh
uv run activelearning config/molecules/gflownet_exact_multi_fidelity.yaml \
  sampler.conf.gflownet.optimizer.n_train_steps=25 \
  selector.num_samples=4 \
  budget.available_budget=28.0 \
  budget.schedule.value=28.0
```

The last step keeps the same multi-fidelity GFlowNet structure but swaps the exact GP for a variational surrogate:

```sh
uv run activelearning config/molecules/gflownet_variational_multi_fidelity.yaml
```

Because the variational surrogate scales better, it is the version to use once the exact multi-fidelity GFlowNet setup is working and you want a larger run.

## **Configuration reference**

The main molecule-specific fields are:

| Field | Meaning |
|-------|---------|
| `surrogate.encoder.max_length` | Maximum SELFIES token length before special tokens. |
| `surrogate.encoder.latent_dim` | Size of the learned molecule embedding passed to the GP. |
| `surrogate.multi_fidelity` | Whether to append fidelity to the surrogate features. |
| `surrogate.target_fidelity` | Fidelity level used when MF acquisitions project candidates to the target objective. |
| `sampler.candidate_pool_file` | SELFIES pool used by `PoolFileSampler`. |
| `sampler.conf.env._target_` | GFlowNet environment class for generated SELFIES. |
| `sampler.fixed_fidelity` | Fixed fidelity assigned to GFlowNet-generated molecules in the single-fidelity examples. |
| `sampler.n_fidelities` | Number of fidelity levels exposed to the GFlowNet policy. Values greater than `1` enable joint molecule-fidelity sampling. |
| `sampler.fidelity_action` | Where the fidelity choice appears in the trajectory; `"any"` lets the policy interleave fidelity selection with token actions. |
| `acquisition.type` | `UpperConfidenceBound` in stages 1 and 3, or `QMultiFidelityLowerBoundMaxValueEntropy` in stages 2, 4, and 5. |
| `acquisition.cost_aware_utility` | Cost model used to prefer cheap fidelities when the expected information gain is comparable. |
| `oracle.task` | `ea` for electron affinity or `ip` for ionisation potential. |
| `oracle.fidelity_costs` | Query costs used by the budget and the cost-aware utility. |
| `oracle.log_molecule_visualizations` | Whether to log RDKit grids of queried molecules. |
| `oracle.molecule_visualization_limit` | Maximum number of molecules shown per logged grid. |

!!! warning "Sequence length and oracle cost"
    Increasing `sampler.conf.env.max_length` or `surrogate.encoder.max_length` expands the molecular search space quickly. Start with the checked-in values, verify that xTB runs successfully, and only then increase sequence length or per-round budget.

## **What comes next**

You now have the same active-learning loop running on structured molecular strings. Natural follow-ups are:

- swap `oracle.task` between `ea` and `ip` and compare the selected molecules in Aim,
- replace `config/data/molecules.txt` with a domain-specific SELFIES pool,
- increase GFlowNet training steps once the short run works,
- compare the five staged examples under the same total budget,
- or adapt the xTB oracle for a different computational chemistry target using the [Oracle extension guide](../extension-guide/oracle.md).

# Multi-Fidelity Active Learning with GFlowNets

A modular, config-driven framework for **budget-constrained multi-fidelity active learning**. It enables cost-effective discovery of high-scoring candidates by intelligently choosing what to query — and at which fidelity level — under a finite oracle budget.

> **📖 Full documentation:** [milaforscience.github.io/activelearning](https://milaforscience.github.io/activelearning)

---

## Background

This repository re-implements and extends the codebase accompanying the paper
**[Multi-Fidelity Active Learning with GFlowNets](http://arxiv.org/abs/2306.11715)**,
with a focus on modularity and extensibility. It builds on the original implementation at
[nikita-0209/mf-al-gfn](https://github.com/nikita-0209/mf-al-gfn) and is also informed by
an intermediate implementation at [alexhernandezgarcia/activelearning](https://github.com/alexhernandezgarcia/activelearning),
with significant contributions by [ginihumer](https://github.com/ginihumer).

---

## Framework Architecture

The active learning loop connects eight independently configurable components:

```
Dataset → Surrogate → Acquisition → Sampler → Selector → Oracle → (back to Dataset)
```

| Component | Role |
|-----------|------|
| **Dataset** | Records observed queries and their outcomes |
| **Surrogate** | Probabilistic model fitted on the dataset (e.g. BoTorch GP) |
| **Acquisition** | Scores candidate-fidelity pairs by expected utility |
| **Sampler** | Generates candidate proposals (e.g. Latin Hypercube, GFlowNet) |
| **Selector** | Filters proposals to fit within the round budget |
| **Oracle** | Evaluates the true objective at the requested fidelity |
| **Budget** | Enforces per-round and total cost constraints |
| **Logger** | Records metrics and artifacts (console, W&B, Comet, Aim) |

Every component is selected and parameterised by a YAML config file. Swapping any single component requires only a config change — no code modifications needed.

For a deeper conceptual introduction, see the [Framework Overview](https://milaforscience.github.io/activelearning/concepts/overview/) in the docs.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**1. Install `uv`**

macOS with Homebrew:
```sh
brew install uv
```

macOS / Linux via the official installer:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Via pip:
```sh
pip install uv
```

**2. Set up the environment**

```sh
make setup
```

This installs all dependencies from the lockfile and sets up pre-commit hooks. A `.venv/` is created in the project root. To use an existing virtual environment instead:

```sh
UV_PROJECT_ENVIRONMENT=/path/to/venv make setup
```

> **Note:** `uv sync` strictly enforces the lockfile — packages not defined in the project will be removed from the environment.

For full details, see the [Installation guide](https://milaforscience.github.io/activelearning/getting-started/installation/).

---

## Quickstart

Run the minimal single-fidelity Branin baseline:

```sh
uv run activelearning config/branin_single_fidelity.yaml \
  budget.available_budget=30.0
```

Expected output:

```
[Step 1] round=1 | num_new_samples=30 | round_cost=30.0000 | total_cost=30.0000 | budget_remaining=0.0000
Done. Rounds: 1 | Total cost: 30.0000
```

From there, scale up by removing the budget override, or switch to the multi-fidelity config:

```sh
# Full single-fidelity run
uv run activelearning config/branin_single_fidelity.yaml

# Multi-fidelity run
uv run activelearning config/branin_multi_fidelity.yaml
```

Config values can be overridden inline using [OmegaConf dotlist](https://omegaconf.readthedocs.io/en/latest/usage.html#from-a-dot-list) syntax, and multiple YAML files can be composed (merged left-to-right):

```sh
# Override a config value
uv run activelearning config/branin_multi_fidelity.yaml budget.available_budget=50.0

# Compose configs (adds Aim logging on top of the base experiment)
uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml
```

The [Quickstart](https://milaforscience.github.io/activelearning/getting-started/quickstart/) and [Synthetic Function Examples](https://milaforscience.github.io/activelearning/tutorials/synthetic_function_experiment/) tutorial walk through the full workflow.

---

## Project Layout

```
src/activelearning/       # Framework source code
  acquisition/            # Acquisition functions
  dataset/                # Dataset backends
  oracle/                 # Oracle implementations
  sampler/                # Candidate samplers (Hypercube, pool-uniform, pool-score)
  selector/               # Budget-aware selectors
  surrogate/              # Probabilistic surrogate models
  logger/                 # Logging backends
  budget/                 # Budget schedulers
  config.py               # Top-level Pydantic config model
  main.py                 # CLI entry point and loop orchestration
config/                   # Bundled YAML experiment configs
tests/                    # Test suite
docs/                     # Documentation source (MkDocs)
```

---

## Loggers

Set `logger.type` in your config to choose a logging backend:

| Type | Backend | Optional dependency |
|------|---------|---------------------|
| `ConsoleLogger` | stdout | *(none)* |
| `WandbLogger` | Weights & Biases | `uv sync --extra wandb` |
| `CometLogger` | Comet ML | `uv sync --extra comet` |
| `AimLogger` | Aim | `uv sync --extra aim` |

Multiple loggers can be combined with `MultiLogger`. See the [Logger API reference](https://milaforscience.github.io/activelearning/reference/activelearning/logger/) for configuration details.

---

## Development

```sh
make test     # Run the pytest test suite
make check    # Run all pre-commit validation hooks
make clean    # Remove generated artifacts and caches
make help     # List all available Makefile targets
```

Pre-commit hooks are installed automatically by `make setup`. If they stop running, reinstall with:

```sh
uv run pre-commit install
```

---

## Extending the Framework

Each component has a documented extension interface. The [Extension Guide](https://milaforscience.github.io/activelearning/extension-guide/) covers:

- Adding a custom **Oracle**
- Adding a custom **Sampler** (including GFlowNet integration)
- Adding a custom **Surrogate**
- Adding a custom **Selector**
- Adding a custom **Acquisition function**

---

## Citation

If you use this codebase in your work, please cite the original paper:

```bibtex
@article{hernandezgarcia2024multifidelity,
  title={Multi-Fidelity Active Learning with {GF}lowNets},
  author={Alex Hernandez-Garcia and Nikita Saxena and Moksh Jain and Cheng-Hao Liu and Yoshua Bengio},
  journal={Transactions on Machine Learning Research},
  year={2024},
  issn={2835-8856},
  url={https://openreview.net/forum?id=dLaazW9zuF},
  note={Expert Certification}
}
```

**Related resources:**
- 📄 [Paper (arXiv)](http://arxiv.org/abs/2306.11715) · [OpenReview](https://openreview.net/forum?id=dLaazW9zuF)
- 🎥 [Talk (video)](https://www.dailymotion.com/video/k1k8KKYS67DgFCB516w) · [Slides](https://alexhernandezgarcia.com/slides/mfgfn-tmlr)
- 🔗 Original code: [nikita-0209/mf-al-gfn](https://github.com/nikita-0209/mf-al-gfn) · [alexhernandezgarcia/activelearning](https://github.com/alexhernandezgarcia/activelearning)

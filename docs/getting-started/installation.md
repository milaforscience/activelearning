# Installation

This repository uses [`uv`](https://docs.astral.sh/uv/) for deterministic environment management. The standard local setup is managed by the provided `Makefile`.

## 1. Install Dependencies

Install the `uv` package manager using one of the supported methods below.

### macOS with Homebrew
```bash
brew install uv
```

### macOS or Linux via Official Installer
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*(Alternatively, the helper target `make install-uv` executes this official installer script.)*

### Python pip
```bash
pip install uv
```

## 2. Initialize the Project Environment

Once `uv` is installed, install the project and its development dependencies from the repository root:

```bash
make setup
```

This command executes `uv sync` to align the local `.venv/` with the exact state of the lockfile, and it installs the requisite pre-commit hooks.

!!! note
    `uv sync` strictly enforces the lockfile state. This guarantees experimental reproducibility but will remove any extraneous, undocumented packages from the environment.

To use an existing virtual environment instead of generating the default local `.venv/`, define the environment path prior to initialization:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/venv make setup
```

## 3. Running the CLI

Following initialization, run an active learning study via the managed environment:

```bash
uv run activelearning <config.yaml> [key=value ...]
```

The YAML configuration file serves as the executable study specification. It defines the experimental parameters by selecting the surrogate, acquisition function, sampler, selector, oracle, budget constraints, and logging mechanisms.

**Example First Run:**

```bash
uv run activelearning config/branin_single_fidelity.yaml \
  budget.available_budget=30.0
```

Arguments following the YAML path are OmegaConf dotlist overrides that modify the configuration without altering the base file.

Later in the tutorial, you can add Aim logging with:

```bash
uv sync --extra aim
uv run aim up
```

## Maintenance

The repository provides the following utility targets:

```
make help   # List available Makefile targets
make test   # Run the pytest test suite
make check  # Run pre-commit validation hooks
make clean  # Remove generated artifacts and cache directories
```

## Next Steps

- Proceed to the [Quickstart](quickstart.md) to run a minimal validated baseline.
- Follow the [Branin Experiment Tutorial](../tutorials/branin_experiment.md) for
  the full config-to-Aim workflow.
- Review the [Framework Overview](../concepts/overview.md) for formal definitions of the architectural components.

# Installation

This repository uses [`uv`](https://docs.astral.sh/uv/) for deterministic environment management.

## 1. Install uv

Install `uv` using one of the supported methods below.

### macOS with Homebrew
```bash
brew install uv
```

### macOS or Linux via Official Installer
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python pip
```bash
pip install uv
```

## 2. Set Up the Environment

Once `uv` is installed, sync the project dependencies from the repository root:

```bash
uv sync
```

This installs all dependencies into a local `.venv/`, pinned to the exact versions in the lockfile.

!!! note
    `uv sync` strictly enforces the lockfile state. This guarantees experimental reproducibility but will remove any extraneous packages from the environment.

To use an existing virtual environment instead of the default `.venv/`:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/venv uv sync
```

## Next Steps

Once set up, head to the [Quickstart](quickstart.md) to run your first experiment.

# **Installation**

This repository uses [`uv`](https://docs.astral.sh/uv/) for deterministic environment management.

## **1. Clone the Repository**

```bash
git clone https://github.com/milaforscience/activelearning.git
cd activelearning
```

## **2. Install uv**

Install `uv` using one of the supported methods below.

### **macOS with Homebrew**
```bash
brew install uv
```

### **macOS or Linux via Official Installer**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### **Python pip**
```bash
pip install uv
```

## **3. Set Up the Environment**

Once `uv` is installed, choose the installation that matches the code you want
to use. The core package has no molecular runtime dependencies:

```bash
uv sync --package activelearning --group dev
```

To work on every workspace package, including molecular encoders, xTB
integration, and S3-GFN:

```bash
uv sync --all-packages --group dev
```

Both commands install dependencies into a local `.venv/`, pinned to the exact
versions in the lockfile. For an installed environment outside this repository,
use `pip install activelearning` and add `pip install activelearning-molecules`
only when molecular components are needed.

!!! note
    `uv sync` strictly enforces the lockfile state. This guarantees experimental reproducibility but will remove any extraneous packages from the environment.

To use an existing virtual environment instead of the default `.venv/`:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/venv uv sync
```

## **Next Steps**

Once set up, head to the [Quickstart](quickstart.md) to run your first experiment.
For molecular experiments, also read the
[molecular application guide](https://github.com/milaforscience/activelearning/tree/main/applications/molecules).

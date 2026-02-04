# Multi-Fidelity Active Learning with GFlowNets

This repository extends the codebase accompanying the paper
**[Multi-Fidelity Active Learning with GFlowNets](http://arxiv.org/abs/2306.11715)**.

This is a re-implementation of the original code base at https://github.com/nikita-0209/mf-al-gfn, with a focus on modularity and extensibility, enabling flexible experimental configurations and easier integration of future methods. Note that this is also inspired by a third, intermediate implementation in https://github.com/alexhernandezgarcia/activelearning, with significant contributions by [ginihumer](https://github.com/ginihumer).

## Setup Instructions

### 1. Install uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**On macOS (Homebrew)**
```sh
brew install uv
```

**On macOS or Linux**
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Using pip**
```sh
pip install uv
```

For more details see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Set up the environment

Install all dependencies (including dev dependencies) and pre-commit hooks:

```sh
make setup
```

By default, `make setup` (via `uv`) creates and uses a local `.venv/`. If you want to install into an existing virtual environment, set `UV_PROJECT_ENVIRONMENT` to its path before running the command:

```sh
UV_PROJECT_ENVIRONMENT=/path/to/venv
make setup
```

#### Note: This will sync the environment to the lockfile,  uninstalling any packages not defined in the project.

See the [uv documentation](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path) for details.

### 3. Run tests

```sh
make test
```

## Project Layout

- Package code: `src/activelearning/`
- Tests: `tests/`

## Development & Tooling Notes

- Pre-commit hooks are installed by `make setup`.
- Run all pre-commit checks locally:
  ```sh
  make check
  ```
- See available Makefile commands and descriptions:
  ```sh
  make help
  ```
- Clean generated files and caches:
  ```sh
  make clean
  ```
- Install uv via the Makefile (optional):
  ```sh
  make install-uv
  ```
- CI runs `make setup` and `make test`. Do not rename or remove these targets, as this will break GitHub Actions.
- If hooks are not running, re-install manually:
  ```sh
  uv run pre-commit install
  ```

## Citation
If you use this code in your work, please cite the original paper:
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

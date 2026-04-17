# **Contributing**

This guide describes the contribution workflow, coding conventions, and pull request
standards for the Multi-Fidelity Active Learning framework. It covers environment setup,
coding conventions, and pull request preparation.

If you are looking to adapt or extend the framework for your own experiments rather than
contribute upstream, start with the [Extension Guide](extension-guide/overview.md) instead.

---

## **1. Getting Started**

### **Fork and clone**

Fork the repository on GitHub, then clone your fork locally:

```sh
git clone https://github.com/<your-username>/activelearning.git
cd activelearning
```

### **Local setup**

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
A single command installs `uv` (if missing), syncs all development dependencies,
and registers the pre-commit hooks:

```sh
make setup
```

If `uv` is not on your `PATH` yet, install it first:

```sh
make install-uv
```

### **Verify the setup**

Confirm that linting, tests, and docs all pass before making any changes:

```sh
make check && make test && make docs-build
```

---

## **2. Coding Conventions**

Consistent, readable code reduces maintenance burden and simplifies review.

### **Type hints**

All functions and class methods must carry full type annotations.

```python
def select_candidates(scores: list[float], top_k: int) -> list[int]:
    ...
```

### **Naming**

| Construct | Style | Example |
|---|---|---|
| Functions and methods | `snake_case` | `compute_acquisition` |
| Classes | `PascalCase` | `SurrogateModel` |
| Module-level constants | `UPPER_SNAKE_CASE` | `MAX_BUDGET` |
| Private helpers | leading underscore | `_normalize_inputs` |

### **Function design**

- Keep functions **single-purpose**: one function should do one thing well.
- Prefer explicit arguments over `**kwargs` for public APIs so that type checkers
  and documentation tools can inspect them.

### **Comments**

Add comments only for logic that is complex or non-obvious. Avoid restating what the
code already says clearly:

```python
# Bad: restate the obvious
i = i + 1  # increment i

# Good: explain the why
# Skip the first row — it is always the header produced by the benchmark harness.
rows = rows[1:]
```

---

## **3. Docstring Guidelines**

All public classes and methods require a **NumPy-style docstring**. This is the style
consumed by the Sphinx AutoAPI reference under `docs_api/`, and is therefore the
authoritative format for generated API documentation.

### **Full example**

```python
def select_queries(
    candidates: list[tuple[float, int]],
    budget: float,
    top_k: int = 5,
) -> list[tuple[float, int]]:
    """Select the highest-scoring candidate-fidelity queries within a budget.

    Scores each ``(x, m)`` pair using the current acquisition function and
    returns the ``top_k`` pairs whose accumulated oracle cost does not exceed
    ``budget``.

    Parameters
    ----------
    candidates : list[tuple[float, int]]
        Pool of ``(x, m)`` candidate-fidelity pairs to score.
    budget : float
        Maximum accumulated oracle cost allowed for the returned queries.
    top_k : int, optional
        Number of queries to return. Default is ``5``.

    Returns
    -------
    selected : list[tuple[float, int]]
        The selected ``(x, m)`` pairs, ordered by acquisition score descending.

    Raises
    ------
    ValueError
        If ``budget`` is non-positive or ``top_k`` is less than 1.

    Notes
    -----
    When the budget is exhausted before ``top_k`` pairs are collected, only
    the affordable pairs are returned.

    Examples
    --------
    >>> select_queries([(0.1, 1), (0.5, 2)], budget=10.0, top_k=1)
    [(0.5, 2)]
    """
```

### **Section checklist**

| Section | Required? | Purpose |
|---|---|---|
| One-line summary | Always | First sentence, ends with `.` |
| Extended description | When helpful | Extra context, multi-paragraph |
| `Parameters` | When function has args | One entry per parameter |
| `Returns` | When function returns a value | Name and type of the return value |
| `Raises` | When exceptions are raised | Each exception type and condition |
| `Notes` | Optional | Implementation details, caveats |
| `Examples` | Encouraged for public API | Runnable `>>>` snippets |

---

## **4. Testing**

### **Running the test suite**

```sh
make test
```

This runs `uv run pytest` against the full `tests/` directory.

### **Writing tests**

- Place all tests under `tests/`, mirroring the source layout where practical.
- Every new public function or class should have at least one test.
- Use **pytest fixtures** for shared setup (datasets, configs, surrogate stubs).
- Use **`@pytest.mark.parametrize`** to cover multiple input cases without duplicating
  test bodies.

```python
import pytest
from activelearning.selector import GreedySelector

@pytest.mark.parametrize("top_k,expected_len", [(1, 1), (3, 3)])
def test_greedy_selector_returns_top_k(top_k: int, expected_len: int) -> None:
    selector = GreedySelector(top_k=top_k)
    result = selector.select(scores=[0.1, 0.9, 0.5, 0.2])
    assert len(result) == expected_len
```

---

## **5. Pre-commit Checks**

The repository uses [pre-commit](https://pre-commit.com/) to enforce quality gates
automatically. The hooks run on every commit once `make setup` has been called
and can also be run manually:

```sh
make check
```

The current hook stack includes:

| Hook | What it enforces |
|---|---|
| `trailing-whitespace` | No trailing spaces |
| `end-of-file-fixer` | Files end with a single newline |
| `check-merge-conflict` | No unresolved merge conflict markers |
| `check-yaml` / `check-toml` / `check-json` | Valid config file syntax |
| `check-added-large-files` | Prevents accidentally committing large binaries |
| `check-case-conflict` | No case-insensitive filename collisions |
| `nbstripout` | Strips output from Jupyter notebooks before commit |
| `codespell` | Catches common spelling mistakes in source and docs |
| `gitleaks` | Detects accidentally committed secrets or credentials |
| `ruff` (linter) | Enforces style rules and fixes auto-fixable issues |
| `ruff-format` (formatter) | Consistent code formatting (replaces Black) |

Fix any failures reported by `make check` before opening a pull request.

---

## **6. Documentation**

### **Preview locally**

```sh
make docs-serve
```

This serves the built documentation at `http://127.0.0.1:8000`. Re-run
`make docs-build` after changing prose docs or Python docstrings so the Zensical site
and Sphinx API reference stay in sync.

### **Build locally**

```sh
make docs-build
```

Builds the Zensical site and the Sphinx API reference into `site/`. The `site/`
directory is Git-ignored; delete it manually or run `make clean` when you no longer
need a local build.

### **Writing docs**

- Prose documentation lives under `docs/` and is written in Markdown.
- The standalone API reference is generated by Sphinx AutoAPI from the Python source in
  `src/`, using the configuration under `docs_api/`.
- Keep docstrings accurate and complete rather than duplicating API details in the prose
  docs.
- If you add a new top-level prose page, register it in the `nav` section of
  `zensical.toml`.

### **Terminology**

Keep all user-facing text aligned with the [Methodology](concepts/overview.md)
section (see also [Research-Facing Contributions](#9-research-facing-contributions) below).

---

## **7. Opening a Pull Request**

### **Branch naming**

```sh
git checkout -b feature/your-feature-name
# or
git checkout -b fix/short-description
```

### **Before you push**

Run the full validation suite:

```sh
make check && make test && make docs-build
```

All three must pass with no errors.

### **PR scope**

- **One logical change per PR.** If you are fixing a bug and adding a feature, open
  two separate PRs.
- Include tests for any new functionality.
- Update or add documentation if you are adding a new component or changing a public API.

### **PR description**

A complete description addresses three questions:

1. **What** does this change do?
2. **Why** is it needed?
3. **Caveats** — any known limitations, follow-up work, or decisions worth explaining?

---

## **8. Reporting Issues and Requesting Features**

### **Bug reports**

Open a [GitHub issue](https://github.com/milaforscience/activelearning/issues) and include:

- What you **expected** to happen.
- What **actually** happened (error message, traceback, unexpected output).
- A **minimal reproduction** command or script.
- Your Python version and relevant dependency versions (`uv run pip list`).

### **Feature requests**

Describe:

- The **use case** — the specific objective the feature addresses.
- How the feature aligns with the framework's goals (multi-fidelity active learning,
  GFlowNet-based sampling, benchmark reproducibility).
- Any prior art or references that support the design.

---

## **9. Research-Facing Contributions**

For methodology, benchmark, and paper-replication changes, keep terminology aligned with
the [Methodology](concepts/overview.md) and paper-replication pages:

- Describe explicit multi-fidelity actions as candidate-fidelity queries `(x, m)`.
- Describe spend as **oracle cost** or **accumulated oracle cost**.
- Use terms such as **study**, **run**, **runnable baseline**, and **scaffold** when they
  match the current implementation status.
- When referencing acquisition functions or surrogate models, use the names defined in
  the API reference rather than informal shorthand.

If your change affects reproducibility of published results, document the impact clearly
in the PR description and in the relevant benchmark page under `docs/examples/`.

---
description: "Every Monday, review PRs merged to main in the last week and open a documentation update PR if needed."

on:
  schedule: weekly on monday around 9am

permissions:
  contents: read
  pull-requests: read

tools:
  github:
    toolsets: [pull_requests, repos]
  edit:
  bash: ["gh:*"]

safe-outputs:
  create-pull-request:
    title-prefix: "[ai] "
    labels: [documentation]
    draft: true
    fallback-as-issue: false
    auto-close-issue: false
---

# Weekly Documentation Updater

You are a documentation maintainer for the **Multi-Fidelity Active Learning with GFlowNets**
framework — a modular, config-driven Python ML framework for budget-constrained active
learning experiments. Your task is to keep the documentation accurate and in sync with
recent code changes.

## Step 1 — Find uncovered PRs merged to main

Find the newest pull request in this repository with the label `documentation` and a title
starting with `[ai] ` (check both open and merged PRs). Use its **creation time** as the
coverage checkpoint — this is the moment up to which code PRs have already been analyzed.
If no such PR exists, fall back to the current time minus 7 days.

List all pull requests merged into `main` after the coverage checkpoint. Then exclude any PRs
already listed as analyzed in the newest docs-update PR's description (if one exists).

If no uncovered merged PRs remain, output a **noop** and stop.

## Step 2 — If a docs-update PR is already open, prepare to update it

If the newest docs-update PR is still open, you will update its branch later in Step 6 rather
than opening a new PR. Do **not** stop — an open docs-update PR does not mean newer merged PRs
are already covered.

## Step 3 — Collect and filter the diffs

For each merged PR, retrieve its diff. **Ignore** changes to `docs/`, `tests/`, `site/`,
and `.github/` — focus only on changes to `src/activelearning/` and project config files
(`pyproject.toml`, `zensical.toml`, `Makefile`, `config/`).

If no such changes exist across all PRs, output a **noop** and stop.

## Step 4 — Read the current documentation

Read the Markdown files under `docs/` to understand what is currently documented. Pay
attention to component interfaces, configuration options, and any examples.

## Step 5 — Decide whether an update is needed

Update the documentation **only** if the merged changes affect one or more of:

- Public APIs or component interfaces (e.g. a new method, changed signature, new class)
- YAML configuration options (new keys, changed types, renamed fields, new defaults)
- User-facing behavior or expected CLI outputs
- Conceptual accuracy of existing explanations
- Installation or usage instructions

**Do not** update the docs for:

- Internal refactors with no public-facing impact
- Test additions or CI/tooling changes
- Minor bug fixes that don't affect documented behavior
- Style, formatting, or comment-only changes

If no update is needed, output a **noop** explaining why and stop.

## Step 6 — Make the changes and open a PR

Edit the relevant files under `docs/` to reflect the code changes. If an open docs-update PR
already exists, update that PR branch; otherwise create a new pull request with those changes.

**PR description must include:**

- A list of the merged PRs that were analyzed (number and title)
- A concise explanation of what was changed in the docs and why
- A note that the changes were AI-generated and should be reviewed carefully before merging

**Guidelines for editing:**

- Preserve the existing writing style, tone, and structure.
- Do not invent or speculate about features not present in the code.
- Only modify files under `docs/`. Never touch source code, tests, or config files.
- When in doubt about whether a change is needed, be conservative and skip it.

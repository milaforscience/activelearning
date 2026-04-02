# Hartmann6D Experiment Tutorial

The Hartmann6D benchmark is a 6-dimensional test function commonly used to
evaluate multi-fidelity optimization methods. This tutorial follows the same
structure as the [Branin tutorial](branin_experiment.md) — if you haven't
done that one yet, start there. This page focuses on what's different about
Hartmann6D and how to run it.

## What's different from Branin

- **6 dimensions** instead of 2, each in $[0, 1]$
- **Single fidelity costs:** `1.0`; **multi-fidelity costs:** `0.125`, `0.25`, `1.0`
- **Budget:** total 100, round budget 10 (smaller, since each query is cheaper per round)
- **No landscape figure** — Hartmann6D is not 2D, so there is no oracle-level plot

The configs are otherwise identical in structure to the Branin ones.

## 1. Single-Fidelity Setting

```sh
uv run activelearning config/hartmann_single_fidelity.yaml \
  budget.available_budget=20.0
```

Expected terminal output:

```text
[Step 1] round=1 | num_new_samples=9 | round_cost=9.0 | total_cost=9.0 | budget_remaining=11.0
[Step 2] round=2 | num_new_samples=... | round_cost=... | total_cost=... | budget_remaining=...
Done. Rounds: 2 | Total cost: ...
```

Once the run looks right, drop the budget override to run the full config:

```sh
uv run activelearning config/hartmann_single_fidelity.yaml
```

## 2. Multi-Fidelity Setting

```sh
uv run activelearning config/hartmann_multi_fidelity.yaml \
  budget.available_budget=20.0
```

With fidelity costs `0.125 / 0.25 / 1.0`, a round budget of `10.0` fits many
cheap queries. Full run:

```sh
uv run activelearning config/hartmann_multi_fidelity.yaml
```

## 3. Add Aim Logging

Compose with the Aim overlay as with any other config:

```sh
uv run activelearning config/hartmann_single_fidelity.yaml config/aim_logging.yaml
uv run activelearning config/hartmann_multi_fidelity.yaml config/aim_logging.yaml
```

Then open the Aim UI:

```sh
uv run aim up
```

Look for `hartmann-single-fidelity` and `hartmann-multi-fidelity` under the
`activelearning_tutorials` project. The same metric views apply — `round_cost`,
`total_cost`, `budget_remaining` — with no landscape image (6D objective).

!!! tip "Comparing Branin and Hartmann in Aim"
    Because both benchmark sets log to the same `activelearning_tutorials`
    project, you can compare all four runs side by side in the Aim UI.

## What comes next

- Adapt either config to a **custom oracle** — see the [Extension Guide](../extension-guide/index.md) to swap in your own evaluation function.
<!-- - Explore the [GFlowNet Sampler Setup](../tutorials/gflownet_sampler.md) to replace the uniform hypercube sampler with an acquisition-guided generative proposal. -->
- Read [Multi-Fidelity Active Learning](../concepts/multi_fidelity.md) for a deeper conceptual treatment of how fidelity costs and confidences propagate through the loop.

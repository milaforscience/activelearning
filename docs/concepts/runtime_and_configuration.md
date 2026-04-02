# Runtime and Configuration

The YAML configuration file serves as the executable experiment specification for a multi-fidelity active-learning run. It specifies the objective oracle, surrogate model, acquisition rule, sampler, selector, budget policy, and runtime context. Reproducibility depends on expressing all study-defining choices explicitly in configuration rather than embedding them in custom driver code.

## Execution Model

A run follows this sequence:

1. **OmegaConf loads the YAML file.**
2. **CLI dotlist overrides are merged on top.**
3. **Pydantic validates the result** as `ActiveLearningConfig`.
4. **Each top-level block instantiates one concrete component.**
5. **A shared runtime context is constructed** and bound to runtime-aware components.
6. **The budget-constrained active-learning loop begins.**

The configuration file therefore constitutes a complete, reproducible specification of the study.

## Top-Level Schema

The following shows the full set of top-level configuration blocks with example component types:

```yaml
runtime:
  device: cpu
  precision: 64

dataset:
  type: ListDataset

surrogate:
  type: BoTorchGPSurrogate

acquisition:
  type: QMultiFidelityLowerBoundMaxValueEntropy
  candidate_set_spec:
    type: HypercubeCandidateSetSpec
    bounds:
      - [-5.0, 10.0]
      - [0.0, 15.0]
    n_points: 1000
    strategy: lhs

sampler:
  type: HypercubeSampler
  bounds:
    - [-5.0, 10.0]
    - [0.0, 15.0]
  num_samples: 10000
  fidelities: [1, 2, 3]
  point_strategy: lhs

selector:
  type: CostAwareSelector

oracle:
  type: BraninOracle
  fidelity_costs:
    1: 0.01
    2: 0.1
    3: 1.0
  log_landscape: true

budget:
  available_budget: 300.0
  schedule:
    type: constant
    value: 30.0

logger:
  type: ConsoleLogger
  project_name: activelearning_tutorials
  run_name: branin-multi-fidelity
```

The top-level sections correspond to the loop's conceptual components:

<div class="schema-table" markdown>

| Section | Methodological Role | Operational Scope |
| --- | --- | --- |
| `runtime` | Shared execution context | Device selection and floating-point precision; the logger is inserted into the same context after construction. |
| `dataset` | Observation store | Records all candidate-fidelity queries and their outcomes. |
| `surrogate` | Probabilistic model | Fitted on the current dataset to approximate the expensive objective. |
| `acquisition` | Utility function | Scores candidate-fidelity queries $(x, m)$ by expected utility per unit cost $c(x, m)$. |
| `sampler` | Proposal mechanism | Generates the tractable candidate set $\mathcal{P}$ or candidate-fidelity pairs $(x, m)$. |
| `selector` | Budget-aware filter | Selects the executable subset $\mathcal{B} \subset \mathcal{P}$ subject to the round budget. |
| `oracle` | Black-box evaluator | Evaluates the objective at the requested fidelity, realizing cost $c(x, m)$. |
| `budget` | Constraint scheduler | Enforces total budget and per-round spending policy. |
| `logger` | Experiment telemetry | Records runtime metrics, artifacts, and configurations; set to `null` to disable. |

</div>

Every non-null component block uses a `type` discriminator. The matching config model lives in `src/activelearning/<component>/config.py`; its `build()` method is the boundary between declarative YAML and runtime objects.

## Runtime Context

`runtime.device` and `runtime.precision` define the shared torch execution context. The logger, if configured, is constructed once and inserted into that context. The resulting runtime context is bound into runtime-aware components—dataset, surrogate, acquisition, sampler, selector, and oracle—so all components in one study share a single device, dtype, and logger reference.

The included configurations use `cpu` and `precision: 64`. Local sampler-level overrides (`sampler.device`, `sampler.float_precision`) inherit from `runtime` when omitted and take precedence when set explicitly. Local overrides are appropriate only when the sampler requires a different execution environment from the rest of the study.

## Study-Defining Multi-Fidelity Fields

For multi-fidelity experiments, the fields that materially define the study are:

- `oracle.fidelity_costs`: valid fidelity levels $m \in \mathcal{M}$ and per-query costs $c(x, m)$.
- `oracle.fidelity_confidences`: optional confidence map $\kappa(m)$; the built-in augmented-function oracles derive it from relative cost when omitted.
- `sampler.fidelities`: candidate-fidelity support for the sampler; accepts a simple list for uniform fidelity sampling, or a cost map that biases sampling inversely proportional to fidelity cost.
- `budget.available_budget` and `budget.schedule`: total expenditure limit and per-round spending policy (`constant` or `sigmoid_iterations` in the current schema).

Not every sampler emits explicit fidelity labels. That distinction is material when working with cost-aware, multi-fidelity acquisition functions.

## Configuration Overrides

The CLI accepts OmegaConf dotlist overrides following the config path. Overrides apply temporary perturbations—budget reductions, candidate-pool changes, or schedule adjustments—without duplicating the entire YAML file.

```bash
# Quick pilot with a reduced budget
uv run activelearning config/branin_single_fidelity.yaml \
  budget.available_budget=30

# Adjust the round budget
uv run activelearning config/branin_multi_fidelity.yaml \
  budget.schedule.value=5

# Larger candidate pool
uv run activelearning config/branin_multi_fidelity.yaml \
  sampler.num_samples=20000
```

Disable logging entirely via an inline override:

```bash
uv run activelearning config/branin_single_fidelity.yaml logger=null
```

Compose multiple YAML files by passing them in sequence — later files override shared keys:

```bash
# Add Aim logging to any run without touching the base config
uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml
```

!!! tip "When to override vs. when to edit the YAML"
    Edit the YAML when a change belongs to the canonical experiment definition.
    Apply CLI overrides when the change is comparative, exploratory, or local to one run.

## Starting Points

!!! tip "Good configs to start from"
    - `config/branin_single_fidelity.yaml` — simplest runnable baseline, single fidelity, Branin 2D.
    - `config/branin_multi_fidelity.yaml` — multi-fidelity Branin with fidelity costs 0.01 / 0.1 / 1.0.
    - `config/hartmann_single_fidelity.yaml` — single-fidelity Hartmann6D, budget 100/10.
    - `config/hartmann_multi_fidelity.yaml` — multi-fidelity Hartmann6D with fidelity costs 0.125 / 0.25 / 1.0.
    - `config/aim_logging.yaml` — logger overlay; compose with any base config to add Aim: `uv run activelearning config/branin_multi_fidelity.yaml config/aim_logging.yaml`

    For guided walkthroughs, see the [Branin Experiment Tutorial](../tutorials/branin_experiment.md) and the [Hartmann6D Tutorial](../tutorials/hartmann_experiment.md).

## Recommended Procedure

!!! note
    1. Begin from an included YAML configuration baseline.
    2. Restrict `runtime` to device and precision settings; modify only when a numerical or hardware constraint requires it.
    3. Apply CLI overrides for short validation runs and comparative experiments.
    4. Promote durable changes back into YAML.
    5. Introduce new component classes only when the configuration schema is insufficient.

For loop mechanics, see [Framework Overview](overview.md) and [Active Learning Loop](active_learning_loop.md).

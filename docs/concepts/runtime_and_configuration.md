# **Runtime and Configuration**

The YAML configuration file serves as the executable experiment specification for a multi-fidelity active-learning run. It specifies the objective oracle, surrogate model, acquisition rule, sampler, selector, budget policy, and runtime context. Reproducibility depends on expressing all study-defining choices explicitly in configuration rather than embedding them in custom driver code.

## **Execution Model**

A run follows this sequence:

1. **OmegaConf loads the YAML file.**
2. **CLI dotlist overrides are merged on top.**
3. **Pydantic validates the result** as [`ActiveLearningConfig`](../reference/activelearning/config/#activelearning.config.ActiveLearningConfig).
4. **Each top-level block instantiates one concrete component.**
5. **A shared runtime context is constructed** and bound to runtime-aware components.
6. **The budget-constrained active-learning loop begins.**

The configuration file therefore constitutes a complete, reproducible specification of the study.

## **Top-Level Schema**

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
  max_rounds: 300
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

| Section | Methodological Role | Operational Objective |
| --- | --- | --- |
| `runtime` | Infrastructure state | Synchronizes device (`cuda`/`cpu`) and tensor dtypes across modules. |
| `dataset` | State ($\mathcal{D}$) | Records observed candidate-fidelity queries and their outcomes. |
| `surrogate` | Multi-fidelity probabilistic model of the oracle | Computes the predictive distribution over $(x, m)$ pairs. |
| `acquisition` | Utility function | Quantifies the cost-aware expected utility of proposed queries. |
| `sampler` | Proposal mechanism | Generates a tractable set of candidate-fidelity pairs for evaluation. |
| `selector` | Budget-aware filter | Subsets proposed queries to satisfy per-iteration budget constraints. |
| `oracle` | Black-box evaluator | Evaluates the objective $f(x)$ at fidelity $m$, realizing cost $c(x, m)$. |
| `budget` | Constraint scheduler | Enforces strict per-iteration and total computational cost limits. |
| `logger` | Experiment telemetry | Persists runtime metrics, model artifacts, and configurations. |

</div>

Every non-null component block uses a `type` discriminator. The matching config model lives in `src/activelearning/<component>/config.py`; its `build()` method is the boundary between declarative YAML and runtime objects.

## **Runtime Context**

`runtime.device` and `runtime.precision` define the shared torch execution context. The logger, if configured, is constructed once and inserted into that context. The resulting runtime context is bound into runtime-aware components—dataset, surrogate, acquisition, sampler, selector, and oracle—so all components in one study share a single device, dtype, and logger reference.

The included configurations use `cpu` and `precision: 64`. Local sampler-level overrides (`sampler.device`, `sampler.float_precision`) inherit from `runtime` when omitted and take precedence when set explicitly. Local overrides are appropriate only when the sampler requires a different execution environment from the rest of the study.

## **Study-Defining Fidelity Fields**

The fidelity structure is determined at config parse time from the oracle's declared levels.
A study is *single-fidelity* when `oracle.fidelity_costs` has exactly one entry, and
*multi-fidelity* when it has two or more.

The fields that materially define the study are:

- `oracle.fidelity_costs`: valid fidelity levels $m \in \mathcal{M}$ and per-query costs $c(x, m)$.  **This is the authoritative source of the fidelity set.** All other components are validated against it.
- `oracle.fidelity_confidences`: optional confidence map $\kappa(m)$; the built-in augmented-function oracles derive it from relative cost when omitted.
- `sampler.fidelities`: fidelity levels the sampler will stamp on candidates.  Accepts a simple list for uniform fidelity sampling, or a cost map that biases sampling inversely proportional to fidelity cost.  **When omitted, the validator auto-fills this from the oracle's fidelity set.**  An error is raised if the sampler declares levels that are not in the oracle's set.
- `budget.available_budget` and `budget.schedule`: total expenditure limit and per-round spending policy (`constant` or `sigmoid_iterations` in the current schema).
- `budget.max_rounds`: optional positive limit on the number of completed active-learning rounds.

## **Configuration Overrides**

The CLI accepts OmegaConf dotlist overrides following the config path. Overrides apply temporary perturbations—budget reductions, candidate-pool changes, or schedule adjustments—without duplicating the entire YAML file.

```bash
# Quick pilot with a reduced budget
uv run activelearning config/branin/single_fidelity.yaml \
  budget.available_budget=3.0

# Adjust the round budget
uv run activelearning config/branin/multi_fidelity.yaml \
  budget.schedule.value=0.5

# Larger candidate pool per round
uv run activelearning config/branin/multi_fidelity.yaml \
  sampler.num_samples=200
```

Disable logging entirely via an inline override:

```bash
uv run activelearning config/branin/single_fidelity.yaml logger=null
```

Compose multiple YAML files by passing them in sequence — later files override shared keys:

```bash
# Add Aim logging to any run without touching the base config
uv run activelearning config/branin/multi_fidelity.yaml config/aim_logging.yaml
```

!!! tip "When to override vs. when to edit the YAML"
    Edit the YAML when a change belongs to the canonical experiment definition.
    Apply CLI overrides when the change is comparative, exploratory, or local to one run.

## **Starting Points**

!!! tip "Good configs to start from"
    - `config/branin/single_fidelity.yaml` — simplest runnable baseline, single fidelity, Branin 2D.
    - `config/branin/multi_fidelity.yaml` — multi-fidelity Branin with fidelity costs 0.01 / 0.1 / 1.0.
    - `config/hartmann/single_fidelity.yaml` — single-fidelity Hartmann, budget 100/10.
    - `config/hartmann/multi_fidelity.yaml` — multi-fidelity Hartmann with fidelity costs 0.125 / 0.25 / 1.0.
    - `config/aim_logging.yaml` — logger overlay; compose with any base config to add Aim: `uv run activelearning config/branin/multi_fidelity.yaml config/aim_logging.yaml`

    For guided walkthroughs, see the [Running Experiments](../tutorials/running_experiments.md) tutorial.

## **Recommended Procedure**

!!! note
    1. Begin from an included YAML configuration baseline.
    2. Restrict `runtime` to device and precision settings; modify only when a numerical or hardware constraint requires it.
    3. Apply CLI overrides for short validation runs and comparative experiments.
    4. Promote durable changes back into YAML.
    5. Introduce new component classes only when the configuration schema is insufficient.

For loop mechanics, see [Framework Overview](overview.md) and [Active Learning Loop](active_learning_loop.md).

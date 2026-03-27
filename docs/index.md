# Multi-Fidelity Active Learning

This framework provides a modular environment for **Multi-Fidelity Active Learning** under a finite oracle budget.

The motivating setting is an expensive black-box objective: the highest-fidelity available oracle is informative but computationally costly, while lower-fidelity approximations are less expensive and provide useful signals to guide the search.

The multi-fidelity acquisition framework extends the decision space beyond the candidate $x$ to include the fidelity level $m$. The algorithm thus selects a pair $(x, m)$, identifying not only where to evaluate but also at what level of precision.

An effective allocation policy distributes the oracle budget across candidate-fidelity decisions to facilitate the cost-effective discovery of diverse, high-scoring candidates.

## Problem Setting

- **Expensive black-box objective:** Direct evaluation is budget-constrained; the algorithm must optimize the allocation of finite computational or physical resources.
- **Multiple fidelities:** Lower-fidelity approximations provide useful signals at a lower cost than the highest-fidelity available oracle.
- **Finite budget:** Iterations must satisfy budget constraints and enforce total allowable budget limits.
- **Discovery focus:** The goal is to identify a diverse population of high-performing candidates. This is achieved through modular acquisition functions, samplers and selectors that balance candidate quality with design-space coverage.

## Framework Architecture

The framework implements the following modular execution loop:

```mermaid
graph LR
    D([Dataset]) -- fit --> S([Surrogate])
    S -- update --> A([Acquisition])
    A -- guide --> Sa([Sampler])
    Sa -- propose --> Se([Selector])
    Se -- query --> O([Oracle])
    O -- append --> D
```

Refer to the [Framework Overview](concepts/overview.md#component-architecture) for a formal definition of each component, and the [Active Learning Loop](concepts/active_learning_loop.md) for a detailed technical walkthrough of their interactions.

## Core Features

* **De Novo Query Synthesis:** Search across the full object space $\mathcal{X}$ via dynamic candidate generation, enabling discovery in settings where the search space is non-enumerable or does not exist a priori.
* **Multi-Fidelity Action Space:** Optimize candidate-fidelity pairs $(x, m)$ to strategically balance variable evaluation costs $c(x, m)$ against fidelity-specific confidence levels $\kappa(m)$.
* **Strict Budgetary Control:** Enforce prescribed round-wise schedules and global aggregate cost limits through a dedicated budget module integrated directly into the selection logic.
* **Interface-Driven Modularity:** Plug-and-play architecture for surrogates, acquisition functions, samplers, and selectors, all governed by unified interfaces and external YAML configurations.
* **Generative Discovery:** The goal is to identify a diverse population of high-performing candidates by proposing points proportional to an acquisition signal $\alpha(x, m)$ using GFlowNet or uniform sampling.
* **Reproducible Workflows:** Execute entire experiments from declarative YAML specifications with native support for OmegaConf-based CLI overrides for controlled perturbations.
* **Unified Runtime Context:** Automatic propagation of compute device, floating-point precision, and structured logging (W&B, Comet, Aim) across all framework components.

To learn more about the research motivating this framework, see [References and Citation](resources/references.md).

## Getting Started

<div class="grid cards" markdown>

-   :material-download-circle:{ .lg .middle } **Install & Run**

    ---

    Install the library and run a first end-to-end experiment in minutes.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)
    · [Quickstart](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Understand the Framework**

    ---

    Learn the multi-fidelity active learning methodology and the execution loop in depth.

    [:octicons-arrow-right-24: Framework Overview](concepts/overview.md)
    · [Active Learning Loop](concepts/active_learning_loop.md)
    · [Multi-Fidelity AL](concepts/multi_fidelity.md)

-   :material-puzzle-edit:{ .lg .middle } **Extend to Your Use Case**

    ---

    Implement custom surrogates, acquisitions, samplers, selectors, or oracles.

    [:octicons-arrow-right-24: Extension Guide](extension-guide/index.md)
    · [API Reference](api/index.md)

-   :material-flask-outline:{ .lg .middle } **Run Experiments**

    ---

    Explore provided configurations, benchmarks, and replication scripts.

    [:octicons-arrow-right-24: Examples](examples/configs.md)
    · [References](resources/references.md)

</div>

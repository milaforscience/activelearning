# Multi-Fidelity Active Learning

This page details how fidelity is represented and propagated through each component of the framework. It assumes familiarity with the problem setting (see [Home](../index.md)) and the execution loop (see [Active Learning Loop](active_learning_loop.md)).

In the multi-fidelity setting, the action space is the set of candidate-fidelity pairs $(x, m)$, where $m \in \mathcal{M}$ determines both the cost $c(x, m)$ and the confidence $\kappa(m)$ of the query. An effective policy exploits lower-fidelity approximations to improve budget efficiency, reserving higher-fidelity evaluation for promising regions.

## Oracle-Defined Fidelity Structure

The [`Oracle`](../api/oracle.md) defines the multi-fidelity structure of the experiment. It specifies:

- The valid fidelity levels $m \in \mathcal{M}$.
- The cost $c(x, m)$ associated with each fidelity level.
- The confidence $\kappa(m)$ associated with each fidelity level.

Fidelity costs are required; fidelity confidences are optional. When confidences are omitted, they are derived from relative cost: the highest-cost fidelity is assigned $\kappa(m) = 1.0$, and lower-fidelity levels carry signal proportional to their relative cost.

## Fidelity Integration Across Components

### Sampler

The [`Sampler`](../api/sampler.md) is responsible for emitting explicit candidate-fidelity pairs $(x, m)$ rather than candidates alone. Two strategies are supported:

- **Uniform fidelity sampling**: fidelity levels are drawn uniformly over $\mathcal{M}$.
- **Cost-weighted fidelity sampling**: fidelity levels are sampled inversely proportional to $c(x, m)$, increasing the proportion of lower-fidelity proposals and preserving budget headroom for high-fidelity queries in later rounds.

### Surrogate

Prior to loop execution, the [`Oracle`](../api/oracle.md) passes fidelity confidences $\kappa(m)$ to the [`Surrogate`](../api/surrogate.md) via [`set_fidelity_confidences()`](../api/surrogate.md#activelearning.surrogate.surrogate.Surrogate.set_fidelity_confidences). The surrogate uses these confidences to condition its probabilistic model on fidelity level, producing a posterior that accounts for the reduced reliability of lower-fidelity observations.

### Acquisition

A multi-fidelity [`Acquisition`](../api/acquisition.md) function $\alpha(x, m)$ scores candidate-fidelity pairs by expected utility per unit cost $c(x, m)$, balancing information gain against the cost of obtaining it. This cost-normalisation ensures that lower-fidelity queries remain competitive when they provide sufficient information gain relative to their cost.

### Selector and Budget

The [`Selector`](../api/selector.md) makes the final spending decision under the round budget $B_k$, jointly considering:

- Acquisition values $\alpha(x, m)$ over the proposed set $\mathcal{P}$.
- Oracle query costs $c(x, m)$.
- The round budget $B_k$ and remaining total budget.

Lower-fidelity queries support cost-effective allocation: they can fit within the round budget while still improving the surrogate sufficiently to guide subsequent higher-fidelity queries. The accumulated oracle cost $\sum_{i} c(x_i, m_i)$ across all executed queries is tracked and logged by the framework.

## Fidelity in the Configuration

Multi-fidelity behaviour is activated by specifying a fidelity cost map (and optionally a confidence map) consistently across the oracle, sampler, and acquisition blocks of the YAML configuration. The sampler uses the cost map to weight its fidelity proposals; the acquisition uses it to normalise utility scores; and the oracle uses it to compute query costs and derive default confidences.

For concrete configuration examples, see [Runtime and Configuration](runtime_and_configuration.md) and the [Quickstart](../getting-started/quickstart.md).

## Single-Fidelity vs. Multi-Fidelity

| Dimension | Single-Fidelity | Multi-Fidelity |
| --- | --- | --- |
| Query action | Select $x \in \mathcal{X}$ | Select pair $(x, m)$ |
| Query cost | Uniform | $c(x, m)$ varies by fidelity $m \in \mathcal{M}$ |
| Surrogate scope | One observation regime | Objective across fidelity levels |
| Budget role | Limits query count | Constrains query count and fidelity allocation |
| Search strategy | Targets best-so-far improvement | Mixes fidelities for cost-effective discovery |

## Codebase Status

The current repository supports fidelity-aware data types, multi-fidelity benchmark oracles, cost-aware acquisition functions, and cost-weighted candidate-fidelity proposals. Generative samplers are available for candidate generation. A full paper-equivalent end-to-end multi-fidelity generative workflow is not yet packaged.

For the current mapping between implemented functionality and paper-facing goals, see [Paper Replication](../paper_replication/index.md).

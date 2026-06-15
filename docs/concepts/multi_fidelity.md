# **Multi-Fidelity Setting**

This page details how fidelity is represented and propagated through each component of the framework. It assumes familiarity with the problem setting (see [Home](../index.md)) and the execution loop (see [Active Learning Loop](active_learning_loop.md)).

In the multi-fidelity setting, the queries for the oracles are candidate-fidelity pairs $(x, m)$, where $m \in \mathcal{M}$ determines both the cost $c(x, m)$ and the confidence $\kappa(m)$ of the query. An effective policy might use, for example, lower-fidelity approximations for wide exploration of the design space, reserving higher-fidelity evaluation for promising regions.

## **Oracle-Defined Fidelity Structure**

The [`Oracle`](../reference/activelearning/oracle/oracle/#activelearning.oracle.oracle.Oracle) defines the multi-fidelity structure of the experiment. It specifies:

- The valid fidelity levels $m \in \mathcal{M}$ and confidence $\kappa(m)$ associated with each fidelity level.
- The cost $c(x, m)$ associated with querying each candidate $x$ at fidelity level $m$.

Fidelity costs are required. Fidelity confidences are also generally required; a simple default is to scale them by relative cost, assigning the highest-cost fidelity $\kappa(m) = 1.0$ and lower-fidelity levels confidence proportional to their cost. This heuristic is the default in [`AugmentedFunctionOracle`](../reference/activelearning/oracle/augmented_function_oracle/#activelearning.oracle.augmented_function_oracle.AugmentedFunctionOracle) when `fidelity_confidences` is omitted.

## **Fidelity Integration Across Components**

### **Sampler**

The [`Sampler`](../reference/activelearning/sampler/sampler/#activelearning.sampler.sampler.Sampler) interface is responsible for emitting explicit candidate-fidelity pairs $(x, m)$ rather than candidates alone. As an example, the built-in [`HypercubeSampler`](../reference/activelearning/sampler/hypercube_sampler/#activelearning.sampler.hypercube_sampler.HypercubeSampler), fidelity assignment supports two strategies:

- **Uniform fidelity sampling**: fidelity levels are drawn uniformly over $\mathcal{M}$.
- **Cost-weighted fidelity sampling**: fidelity levels are sampled inversely proportional to $c(x, m)$, increasing the proportion of lower-fidelity proposals and preserving budget headroom for high-fidelity queries in later rounds.

### **Surrogate**

Prior to loop execution, the [`Oracle`](../reference/activelearning/oracle/oracle/#activelearning.oracle.oracle.Oracle) passes fidelity confidences $\kappa(m)$ to the [`Surrogate`](../reference/activelearning/surrogate/surrogate/#activelearning.surrogate.surrogate.Surrogate) interface via [`set_fidelity_confidences()`](../reference/activelearning/surrogate/surrogate/#activelearning.surrogate.surrogate.Surrogate.set_fidelity_confidences). In the built-in [`BoTorchGPSurrogate`](../reference/activelearning/surrogate/botorch_surrogate/#activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate), these confidences condition the probabilistic model on fidelity level, producing a posterior that accounts for the reduced reliability of lower-fidelity observations.

### **Acquisition**

The [`Acquisition`](../reference/activelearning/acquisition/acquisition/#activelearning.acquisition.acquisition.Acquisition) interface is generic, but the built-in multi-fidelity behaviour is implemented by q-batch BoTorch classes such as [`QMultiFidelityKnowledgeGradient`](../reference/activelearning/acquisition/botorch/botorch_multifidelity/#activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityKnowledgeGradient) and [`QMultiFidelityLowerBoundMaxValueEntropy`](../reference/activelearning/acquisition/botorch/botorch_multifidelity/#activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityLowerBoundMaxValueEntropy). These acquisition functions score candidate-fidelity pairs by expected utility per unit cost $c(x, m)$, balancing information gain against the cost of obtaining it. This cost-normalisation ensures that lower-fidelity queries remain competitive when they provide sufficient information gain relative to their cost.

### **Selector and Budget**

The [`Selector`](../reference/activelearning/selector/selector/#activelearning.selector.selector.Selector) makes the final spending decision under the round budget $B_k$, jointly considering:

- Acquisition values $\alpha(x, m)$ over the proposed set $\mathcal{P}$.
- Oracle query costs $c(x, m)$.
- The round budget $B_k$.

The [`Budget`](../reference/activelearning/budget/budget/#activelearning.budget.budget.Budget) tracks the remaining global budget across rounds. In the active learning loop, it caps each round allocation at the currently available budget via `get_round_budget()`, then checks the total cost of the selected candidates with `can_afford()` before the oracle is queried, ensuring that only affordable selections are executed.

Lower-fidelity queries support cost-effective allocation: they can fit within the round budget while still improving the surrogate sufficiently to guide subsequent higher-fidelity queries. The accumulated oracle cost $\sum_{i} c(x_i, m_i)$ across all executed queries is tracked and logged by the framework.

## **Fidelity in the Configuration**

Multi-fidelity behaviour is activated by specifying a fidelity cost map (with an associated confidence map) consistently across the YAML blocks for the relevant concrete components, e.g., [`AugmentedFunctionOracle`](../reference/activelearning/oracle/augmented_function_oracle/#activelearning.oracle.augmented_function_oracle.AugmentedFunctionOracle), [`HypercubeSampler`](../reference/activelearning/sampler/hypercube_sampler/#activelearning.sampler.hypercube_sampler.HypercubeSampler), or [`CostAwareSelector`](../reference/activelearning/selector/cost_aware_selector/#activelearning.selector.cost_aware_selector.CostAwareSelector). A sampler may use the cost map to weight its fidelity proposals; an acquisition function may use it to normalize utility scores; and the oracle uses it to compute query costs and derive default confidences from costs.

For concrete configuration examples, see [Runtime and Configuration](runtime_and_configuration.md) and the [Quickstart](../getting-started/quickstart.md).

## **Single-Fidelity vs. Multi-Fidelity**

| Dimension | Single-Fidelity            | Multi-Fidelity                                                       |
| --- |----------------------------|----------------------------------------------------------------------|
| Query action | Select $x \in \mathcal{X}$ | Select pair $(x, m)$                                                 |
| Query cost | Uniform                    | $c(x, m)$ varies by fidelity $m \in \mathcal{M}$                     |
| Surrogate scope | One observation regime     | Objective across fidelity levels                                     |
| Budget role | Constrains query costs     | Constrains query costs considering allocation across fidelity levels |

For concrete configuration examples, see the [Synthetic Function Examples](../tutorials/synthetic_function_experiment.md) tutorial, which walks through both settings side by side.

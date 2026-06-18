# **Related Work and Positioning**

This Multi-Fidelity Active Learning framework addresses a combination of research objectives not covered by existing active learning or Bayesian optimization libraries. The sections below identify the nearest related systems and characterize the specific gaps this framework fills.

## **Existing Active Learning Frameworks**

Most general-purpose active learning libraries share a common design: they operate over a fixed pool of unlabelled data and focus on reducing model uncertainty or prediction error across the entire input space. None address multi-fidelity queries or de novo synthesis.

| Library | Primary Capabilities | Limitations |
| --- | --- | --- |
| **modAL** ([modAL-python/modAL](https://github.com/modAL-python/modAL)) | Pool-based AL, scikit-learn estimators, uncertainty sampling, query-by-committee | No multi-fidelity; pool-based only |
| **scikit-activeml** ([scikit-activeml/scikit-activeml](https://github.com/scikit-activeml/scikit-activeml)) | Pool-based AL; broad query strategy set; classification and regression | No multi-fidelity; pool-based only |
| **BMDAL_reg** ([dholzmueller/bmdal_reg](https://github.com/dholzmueller/bmdal_reg)) | Batch mode deep AL for regression | No multi-fidelity; pool-based only; neural networks only |
| **ALiPy** ([NUAA-AL/ALiPy](https://github.com/NUAA-AL/ALiPy)) | Comprehensive pool-based AL toolbox; broad query strategy set | No multi-fidelity; pool-based only |
| **libact** ([ntucllab/libact](https://github.com/ntucllab/libact)) | Pool-based AL with active learning by learning | No multi-fidelity; pool-based only; classification focus |
| **Baal** ([baal-org/baal](https://github.com/baal-org/baal)) | Bayesian deep AL via Monte Carlo Dropout and ensemble uncertainty; image and text classification loops | No multi-fidelity; pool-based only; PyTorch models only |

All libraries listed above share two key limitations: they do not support multi-fidelity queries (querying the same candidate at different cost-accuracy trade-offs), and they are pool or stream-based only (selecting from a pre-enumerated set of candidates rather than generating novel candidates over continuous or structured spaces). To our knowledge, no existing open-source AL framework supports multi-fidelity surrogate modelling over candidate-fidelity pairs $(x, m)$, cost-aware budget accounting that tracks heterogeneous oracle costs $c(x, m)$ rather than simple query counts, or de novo candidate synthesis over continuous and structured input spaces.

## **Bayesian Optimization Libraries**

BO libraries operate over continuous input spaces and minimize expensive oracle evaluations, placing them closer in spirit to this framework than pool-based AL systems.

**BoTorch / Ax** (Balandat et al., 2020) underlies the surrogate and acquisition components of this framework. BoTorch provides multi-fidelity acquisition functions and supports GP-based surrogates with fidelity inputs. BoTorch does not include generative samplers, an active search loop for diverse candidate discovery, or a budget accounting layer that tracks accumulated oracle cost $\sum_i c(x_i, m_i)$ across rounds.

**GPyOpt** provides BO with GP surrogates and several acquisition strategies, but is single-fidelity and targets a single global optimum rather than a diverse set of high-scoring candidates.

**Dragonfly** (Kandasamy et al., 2020) supports multi-fidelity BO and certain structured input spaces, but targets global optimization rather than active search for diverse candidates and does not include GFlowNet-based samplers.

## **Multi-Fidelity and Scientific Discovery**

The following research directions are directly relevant to the multi-fidelity setting addressed here.

**Multi-fidelity BO (MFBO)** methods (e.g., Klein et al., 2017; Kandasamy et al., 2017; Wu & Frazier, 2019) extend BO to multiple fidelity levels $m \in \mathcal{M}$, typically targeting the global optimum of the highest-fidelity objective at reduced total cost. These methods share the cost-aware query selection formulation but focus on optimization rather than diverse discovery.

**BO with function networks (BOFN)** (Astudillo & Frazier, 2021) handles settings where objective evaluations proceed through a network of intermediate computations, some of which may correspond to lower-fidelity stages. This is structurally distinct from the fidelity-as-approximation model used here.

**GFlowNets** (Bengio et al., 2021, 2023) are generative models trained to sample objects proportional to a reward signal. They are suited to generating diverse high-scoring candidates in combinatorial or structured spaces. The GFlowNet sampler integration in this framework introduces diverse candidate generation into the multi-fidelity active learning loop—a combination absent from existing BO and AL libraries.

## **Framework Positioning**

The libraries and research directions above address subsets of the capabilities required for multi-fidelity active learning. BO libraries such as BoTorch and Dragonfly support multi-fidelity acquisition and continuous input spaces, but lack an active learning loop targeting diverse candidates under cost-aware budget constraints. AL libraries provide the iterative query loop but are limited to pool-based, single-fidelity selection. This framework combines these capabilities with the following design choices:

1. **Multi-fidelity active learning for diverse discovery**: combines multi-fidelity surrogate modeling over candidate-fidelity pairs $(x, m)$ with cost-aware budget accounting that tracks heterogeneous oracle costs $c(x, m)$. Unlike standard AL (which targets model accuracy) or BO (which targets a single global optimum), the objective is to discover diverse high-scoring candidates under a finite oracle budget.

2. **De novo query synthesis**: operates over the full input space $\mathcal{X}$ rather than a fixed pool of candidates. This is the appropriate setting for scientific discovery problems such as drug discovery and materials design, where the candidate space is too large to enumerate or does not exist a priori.

3. **GFlowNet integration** *(planned)*: the framework is designed to support GFlowNets as samplers for diverse candidate generation, where candidates are generated proportional to an acquisition signal $\alpha(x, m)$. This integration is a planned extension and is not yet implemented in the current codebase.

4. **Modular, config-driven design**: every component—surrogate, acquisition, sampler, selector, oracle, budget—is replaceable via YAML configuration. The orchestration logic is fixed; any component can be substituted or extended without modifying the loop.

## **References**

- Settles, B. (2009). *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin–Madison.
- King, R. D., et al. (2004). Functional genomic hypothesis generation and experimentation by a robot scientist. *Nature*, 427(6971), 247–252.
- Xue, D., et al. (2016). Accelerated search for materials with targeted properties by adaptive design. *Nature Communications*, 7, 11241.
- Yuan, R., et al. (2018). Accelerated discovery of large electrostrains in BaTiO3-based piezoelectrics using active learning. *Advanced Materials*, 30(7), 1702884.
- Kusne, A. G., et al. (2020). On-the-fly closed-loop materials discovery via Bayesian active learning. *Nature Communications*, 11, 5966.
- Garnett, R., Krishnamurthy, Y., Xiong, X., Schneider, J., & Mann, R. (2012). Bayesian optimal active search and surveying. *ICML 2012*.
- Jiang, S., Malkomes, G., Converse, G., Shofner, A., Moseley, B., & Garnett, R. (2017). Efficient nonmyopic active search. *ICML 2017*.
- Hernandez-Garcia, A., et al. (2024). Multi-Fidelity Active Learning with GFlowNets. *Transactions on Machine Learning Research (TMLR)*.

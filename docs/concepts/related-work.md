# **Related Work and Positioning**

The Multi-Fidelity Active Learning framework, described in Hernandez-Garcia et al. (2024), addresses a combination of research objectives not covered by existing active learning or Bayesian optimization libraries. The sections below identify the nearest related systems and characterize the specific gaps this framework fills.

## **Existing Active Learning Frameworks**

Most general-purpose active learning libraries share a common design: they operate over a fixed pool of unlabelled data and focus on reducing model uncertainty or prediction error across the entire input space. None address multi-fidelity queries or de novo synthesis.

| Library | Primary Capabilities | Limitations |
| --- | --- | --- |
| **modAL** (Danka & Horvath, 2018) | Pool-based AL, scikit-learn estimators, uncertainty sampling, query-by-committee | Single fidelity; pool-based only; no budget constraints |
| **scikit-activeml** | Pool-based AL; broad query strategy set; classification and regression | Single fidelity; pool-based only; no scientific discovery focus |
| **BMDAL_reg** | Batch mode deep AL for regression | Single fidelity; pool-based only; neural networks only |
| **ALiPy** | Comprehensive pool-based AL toolbox; broad query strategy set | Single fidelity; pool-based only; no multi-fidelity or continuous input support |
| **libact** | Pool-based AL with active learning by learning | Single fidelity; pool-based only; classification focus |
| **Baal** ([baal-org/baal](https://github.com/baal-org/baal)) | Bayesian deep AL via Monte Carlo Dropout and ensemble uncertainty; image and text classification loops | Single fidelity; pool-based only; deep learning classifiers/regressors only; no multi-fidelity or de novo synthesis |

 To our knowledge, no existing open-source framework addresses multi-fidelity experimentation, de novo synthesis over continuous or structured spaces, or budget-constrained discovery of diverse high-scoring candidates.

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

## **Framework Contributions**

This framework makes the following contributions relative to the libraries and research directions above:

1. **Multi-fidelity active search**: combines multiple fidelity levels $m \in \mathcal{M}$ with budget-constrained active search. The objective is to discover as many diverse high-scoring candidates as possible under a finite oracle budget, not to converge on a single global optimum.

2. **De novo query synthesis**: operates over the full input space $\mathcal{X}$ rather than a fixed pool of candidates. This is the appropriate setting for scientific discovery problems such as drug discovery and materials design, where the candidate pool is too large to enumerate or does not exist a priori.

3. **GFlowNet integration** *(planned)*: the framework is designed to support GFlowNets as samplers for diverse candidate generation, where candidates are generated proportional to an acquisition signal $\alpha(x, m)$. This integration is a planned extension and is not yet implemented in the current codebase.

4. **Modular, config-driven design**: every component—surrogate, acquisition, sampler, selector, oracle, budget—is replaceable via YAML configuration. The orchestration logic is fixed; any component can be substituted or extended without modifying the loop.

5. **Scientific discovery focus**: the framework targets settings where the goal is to identify a diverse set of high-scoring candidates for downstream experimental validation. This reflects the standard experimental workflow in computational biology, chemistry, and materials science.

## **References**

- Settles, B. (2009). *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin–Madison.
- King, R. D., et al. (2004). Functional genomic hypothesis generation and experimentation by a robot scientist. *Nature*, 427(6971), 247–252.
- Xue, D., et al. (2016). Accelerated search for materials with targeted properties by adaptive design. *Nature Communications*, 7, 11241.
- Yuan, R., et al. (2018). Accelerated discovery of large electrostrains in BaTiO3-based piezoelectrics using active learning. *Advanced Materials*, 30(7), 1702884.
- Kusne, A. G., et al. (2020). On-the-fly closed-loop materials discovery via Bayesian active learning. *Nature Communications*, 11, 5966.
- Garnett, R., Krishnamurthy, Y., Xiong, X., Schneider, J., & Mann, R. (2012). Bayesian optimal active search and surveying. *ICML 2012*.
- Jiang, S., Malkomes, G., Converse, G., Shofner, A., Moseley, B., & Garnett, R. (2017). Efficient nonmyopic active search. *ICML 2017*.
- Hernandez-Garcia, A., et al. (2024). Multi-Fidelity Active Learning with GFlowNets. *Transactions on Machine Learning Research (TMLR)*.

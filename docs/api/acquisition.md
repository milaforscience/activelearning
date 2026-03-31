# Acquisition API

Acquisition functions score `Candidate` objects using the current surrogate state; in multi-fidelity runs, that means scoring `(x, m)` pairs under a cost-aware predictive model.

## Class Reference

::: activelearning.acquisition.acquisition.Acquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.dummy_acquisition.DummyAcquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_acquisition.BoTorchAcquisitionBase
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_acquisition.AnalyticBoTorchAcquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_acquisition.QBatchBoTorchAcquisition
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.UpperConfidenceBound
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.ExpectedImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.LogExpectedImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.ProbabilityOfImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.LogProbabilityOfImprovement
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_analytic.PosteriorMean
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityMaxValueEntropy
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityLowerBoundMaxValueEntropy
    options:
      show_source: false
      heading_level: 3

::: activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityKnowledgeGradient
    options:
      show_source: false
      heading_level: 3

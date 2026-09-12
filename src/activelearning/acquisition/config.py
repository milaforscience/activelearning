"""Pydantic models of acquisition functions.

Changes in the interface of existing acquisition functions should be reflected in this
configuration. New acquisition functions should define a ``BuildableConfig`` schema
and list it in their distribution's configuration catalog.
"""

from typing import Annotated, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, Field

from activelearning.config_registry import BuildableConfig, registered_config
from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.botorch.candidate_set import (
    CandidateSetSpec,
    HypercubeCandidateSetSpec,
    TrainDataCandidateSetSpec,
)
from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.acquisition.botorch.botorch_analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from activelearning.acquisition.botorch.botorch_multifidelity import (
    QMultiFidelityKnowledgeGradient,
    QMultiFidelityLowerBoundMaxValueEntropy,
    QMultiFidelityMaxValueEntropy,
)


class HypercubeCandidateSetSpecConfig(BaseModel):
    type: Literal["HypercubeCandidateSetSpec"] = "HypercubeCandidateSetSpec"
    bounds: list[tuple[float, float]]
    n_points: int = Field(gt=0)
    strategy: Literal["uniform", "lhs"] = "uniform"

    def build(self) -> CandidateSetSpec:
        return HypercubeCandidateSetSpec(
            bounds=self.bounds,
            n_points=self.n_points,
            strategy=self.strategy,
        )


class TrainDataCandidateSetSpecConfig(BaseModel):
    type: Literal["TrainDataCandidateSetSpec"] = "TrainDataCandidateSetSpec"

    def build(self) -> CandidateSetSpec:
        return TrainDataCandidateSetSpec()


CandidateSetSpecConfig = Annotated[
    Union[HypercubeCandidateSetSpecConfig, TrainDataCandidateSetSpecConfig],
    Field(discriminator="type"),
]


class DummyAcquisitionConfig(BuildableConfig):
    type: Literal["DummyAcquisition"] = "DummyAcquisition"
    beta: float = 1.0

    def build(self) -> Acquisition:
        return DummyAcquisition(beta=self.beta)


class _BoTorchAcquisitionConfig(BuildableConfig):
    """Base contract for acquisitions that require a BoTorch surrogate."""

    requires_botorch_surrogate: ClassVar[bool] = True


class UpperConfidenceBoundConfig(_BoTorchAcquisitionConfig):
    type: Literal["UpperConfidenceBound"] = "UpperConfidenceBound"
    beta: float = 2.0
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return UpperConfidenceBound(
            beta=self.beta,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class ExpectedImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["ExpectedImprovement"] = "ExpectedImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return ExpectedImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class LogExpectedImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["LogExpectedImprovement"] = "LogExpectedImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return LogExpectedImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class ProbabilityOfImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["ProbabilityOfImprovement"] = "ProbabilityOfImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return ProbabilityOfImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class LogProbabilityOfImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["LogProbabilityOfImprovement"] = "LogProbabilityOfImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return LogProbabilityOfImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class PosteriorMeanConfig(_BoTorchAcquisitionConfig):
    type: Literal["PosteriorMean"] = "PosteriorMean"
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return PosteriorMean(
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class QMultiFidelityMaxValueEntropyConfig(_BoTorchAcquisitionConfig):
    type: Literal["QMultiFidelityMaxValueEntropy"] = "QMultiFidelityMaxValueEntropy"
    candidate_set_spec: CandidateSetSpecConfig
    num_fantasies: int = Field(default=16, gt=0)
    num_mv_samples: int = Field(default=10, gt=0)
    num_y_samples: int = Field(default=128, gt=0)
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return QMultiFidelityMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_fantasies=self.num_fantasies,
            num_mv_samples=self.num_mv_samples,
            num_y_samples=self.num_y_samples,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class QMultiFidelityLowerBoundMaxValueEntropyConfig(_BoTorchAcquisitionConfig):
    type: Literal["QMultiFidelityLowerBoundMaxValueEntropy"] = (
        "QMultiFidelityLowerBoundMaxValueEntropy"
    )
    candidate_set_spec: CandidateSetSpecConfig
    num_fantasies: int = Field(default=16, gt=0)
    num_mv_samples: int = Field(default=10, gt=0)
    num_y_samples: int = Field(default=128, gt=0)
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return QMultiFidelityLowerBoundMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_fantasies=self.num_fantasies,
            num_mv_samples=self.num_mv_samples,
            num_y_samples=self.num_y_samples,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


class QMultiFidelityKnowledgeGradientConfig(_BoTorchAcquisitionConfig):
    type: Literal["QMultiFidelityKnowledgeGradient"] = "QMultiFidelityKnowledgeGradient"
    num_fantasies: int = Field(default=64, gt=0)
    current_value: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return QMultiFidelityKnowledgeGradient(
            num_fantasies=self.num_fantasies,
            current_value=self.current_value,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
        )


ACQUISITION_CONFIGS = (
    DummyAcquisitionConfig,
    UpperConfidenceBoundConfig,
    ExpectedImprovementConfig,
    LogExpectedImprovementConfig,
    ProbabilityOfImprovementConfig,
    LogProbabilityOfImprovementConfig,
    PosteriorMeanConfig,
    QMultiFidelityMaxValueEntropyConfig,
    QMultiFidelityLowerBoundMaxValueEntropyConfig,
    QMultiFidelityKnowledgeGradientConfig,
)
AcquisitionConfig = registered_config("acquisition")

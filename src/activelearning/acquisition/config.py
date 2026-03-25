from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.candidate_set import (
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


class DummyAcquisitionConfig(BaseModel):
    type: Literal["DummyAcquisition"] = "DummyAcquisition"
    beta: float = 1.0

    def build(self) -> Acquisition:
        return DummyAcquisition(beta=self.beta)


class UpperConfidenceBoundConfig(BaseModel):
    type: Literal["UpperConfidenceBound"] = "UpperConfidenceBound"
    beta: float = 2.0
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return UpperConfidenceBound(
            beta=self.beta,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class ExpectedImprovementConfig(BaseModel):
    type: Literal["ExpectedImprovement"] = "ExpectedImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return ExpectedImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class LogExpectedImprovementConfig(BaseModel):
    type: Literal["LogExpectedImprovement"] = "LogExpectedImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return LogExpectedImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class ProbabilityOfImprovementConfig(BaseModel):
    type: Literal["ProbabilityOfImprovement"] = "ProbabilityOfImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return ProbabilityOfImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class LogProbabilityOfImprovementConfig(BaseModel):
    type: Literal["LogProbabilityOfImprovement"] = "LogProbabilityOfImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return LogProbabilityOfImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class PosteriorMeanConfig(BaseModel):
    type: Literal["PosteriorMean"] = "PosteriorMean"
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return PosteriorMean(
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class QMultiFidelityMaxValueEntropyConfig(BaseModel):
    type: Literal["QMultiFidelityMaxValueEntropy"] = "QMultiFidelityMaxValueEntropy"
    candidate_set_spec: CandidateSetSpecConfig
    num_fantasies: int = Field(default=16, gt=0)
    num_mv_samples: int = Field(default=10, gt=0)
    num_y_samples: int = Field(default=128, gt=0)
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return QMultiFidelityMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_fantasies=self.num_fantasies,
            num_mv_samples=self.num_mv_samples,
            num_y_samples=self.num_y_samples,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class QMultiFidelityLowerBoundMaxValueEntropyConfig(BaseModel):
    type: Literal["QMultiFidelityLowerBoundMaxValueEntropy"] = (
        "QMultiFidelityLowerBoundMaxValueEntropy"
    )
    candidate_set_spec: CandidateSetSpecConfig
    num_fantasies: int = Field(default=16, gt=0)
    num_mv_samples: int = Field(default=10, gt=0)
    num_y_samples: int = Field(default=128, gt=0)
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return QMultiFidelityLowerBoundMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_fantasies=self.num_fantasies,
            num_mv_samples=self.num_mv_samples,
            num_y_samples=self.num_y_samples,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


class QMultiFidelityKnowledgeGradientConfig(BaseModel):
    type: Literal["QMultiFidelityKnowledgeGradient"] = "QMultiFidelityKnowledgeGradient"
    num_fantasies: int = Field(default=64, gt=0)
    current_value: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    fidelity_costs: Optional[dict[int, float]] = None

    def build(self) -> Acquisition:
        return QMultiFidelityKnowledgeGradient(
            num_fantasies=self.num_fantasies,
            current_value=self.current_value,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            fidelity_costs=self.fidelity_costs,
        )


AcquisitionConfig = Annotated[
    Union[
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
    ],
    Field(discriminator="type"),
]

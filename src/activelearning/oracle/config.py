from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from activelearning.oracle.augmented_function_oracle import (
    BraninOracle,
    Hartmann6DOracle,
)
from activelearning.oracle.composite_oracle import CompositeOracle
from activelearning.oracle.oracle import Oracle


class BraninOracleConfig(BaseModel):
    type: Literal["BraninOracle"] = "BraninOracle"
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None
    log_landscape: bool = False

    def build(self) -> Oracle:
        return BraninOracle(
            self.fidelity_costs, self.fidelity_confidences, self.log_landscape
        )


class Hartmann6DOracleConfig(BaseModel):
    type: Literal["Hartmann6DOracle"] = "Hartmann6DOracle"
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None

    def build(self) -> Oracle:
        return Hartmann6DOracle(self.fidelity_costs, self.fidelity_confidences)


class CompositeOracleConfig(BaseModel):
    type: Literal["CompositeOracle"] = "CompositeOracle"
    sub_oracles: list["OracleConfig"]

    def build(self) -> Oracle:
        return CompositeOracle(sub_oracles=[cfg.build() for cfg in self.sub_oracles])


OracleConfig = Annotated[
    Union[BraninOracleConfig, Hartmann6DOracleConfig, CompositeOracleConfig],
    Field(discriminator="type"),
]

CompositeOracleConfig.model_rebuild()

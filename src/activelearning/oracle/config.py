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


class XTBIPEAOracleConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.xtb_oracle.XTBIPEAOracle`.

    Parameters
    ----------
    task : str
        ``"ea"`` (electron affinity) or ``"ip"`` (ionisation potential).
    fidelity_costs : dict[int, float]
        Computational cost per sample for each fidelity level.
    fidelity_confidences : dict[int, float], optional
        Confidence in ``[0, 1]`` per fidelity.  Defaults to costs normalised by max.
    gfn_version : int
        GFN-xTB parametrisation passed to ``--gfn`` (default: 2).
    ff : str
        RDKit force field for initial 3-D geometry: ``"mmff"`` or ``"uff"``.
    correction_factor : float
        Empirical correction subtracted from adiabatic IP/EA (eV).
    mol_repr : str
        Input molecules representation: ``"selfies"`` or ``"smiles"``.
    """

    type: Literal["XTBIPEAOracle"] = "XTBIPEAOracle"
    task: str
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None
    gfn_version: int = 2
    ff: str = "mmff"
    correction_factor: float = 4.8455
    mol_repr: str = "selfies"

    def build(self) -> Oracle:
        from activelearning.applications.molecules.xtb_oracle import XTBIPEAOracle

        return XTBIPEAOracle(
            task=self.task,
            fidelity_costs=self.fidelity_costs,
            fidelity_confidences=self.fidelity_confidences,
            gfn_version=self.gfn_version,
            ff=self.ff,
            correction_factor=self.correction_factor,
            mol_repr=self.mol_repr,
        )


OracleConfig = Annotated[
    Union[
        BraninOracleConfig,
        Hartmann6DOracleConfig,
        CompositeOracleConfig,
        XTBIPEAOracleConfig,
    ],
    Field(discriminator="type"),
]

CompositeOracleConfig.model_rebuild()

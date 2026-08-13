"""Pydantic models of oracles.

Changes in the interface of existing oracles should be reflected in this configuration.
New oracles should define their corresponding pydantic model here and be added to
``OracleConfig``.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    num_conformers : int
        Global/default number of RDKit conformers generated per molecule before
        selecting the lowest-energy starting geometry.
    per_fidelity_num_conformers : dict[int, int], optional
        Optional per-fidelity override for ``num_conformers``. Unlisted
        fidelities fall back to the global/default setting.
    correction_factor : float
        Empirical correction subtracted from adiabatic IP/EA (eV).
    mol_repr : str
        Input molecules representation: ``"selfies"`` or ``"smiles"``.
    negate_score : bool, optional
        Whether to negate the raw EA/IP value before it enters the active-
        learning loop. Defaults to ``True`` for IP and ``False`` for EA when
        omitted.
    log_molecule_visualizations : bool
        Whether to log queried molecule visualizations when a logger is bound.
    molecule_visualization_limit : int
        Maximum number of queried molecules to display in each logged grid.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["XTBIPEAOracle"] = "XTBIPEAOracle"
    task: str
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None
    gfn_version: int = 2
    ff: str = "mmff"
    num_conformers: int = Field(default=2, ge=1)
    per_fidelity_num_conformers: dict[int, int] | None = None
    correction_factor: float = 4.8455
    mol_repr: Literal["selfies", "smiles"] = "selfies"
    negate_score: bool | None = None
    log_molecule_visualizations: bool = False
    molecule_visualization_limit: int = Field(default=25, ge=1)

    @model_validator(mode="after")
    def validate_per_fidelity_num_conformers(self) -> "XTBIPEAOracleConfig":
        if self.per_fidelity_num_conformers is None:
            return self
        invalid_counts = {
            fidelity: count
            for fidelity, count in self.per_fidelity_num_conformers.items()
            if count < 1
        }
        if invalid_counts:
            raise ValueError(
                "per_fidelity_num_conformers must contain positive integers. "
                f"Got: {invalid_counts}."
            )
        unknown_fidelities = sorted(
            set(self.per_fidelity_num_conformers) - set(self.fidelity_costs)
        )
        if unknown_fidelities:
            raise ValueError(
                "per_fidelity_num_conformers contains unsupported fidelities: "
                f"{unknown_fidelities}."
            )
        return self

    def build(self) -> Oracle:
        from activelearning.applications.molecules.xtb_oracle import XTBIPEAOracle
        from activelearning.applications.molecules.xtb_oracle import ConformerConfig

        return XTBIPEAOracle(
            task=self.task,
            fidelity_costs=self.fidelity_costs,
            fidelity_confidences=self.fidelity_confidences,
            gfn_version=self.gfn_version,
            ff=self.ff,
            conformer_cfg=ConformerConfig(num_conformers=self.num_conformers),
            per_fidelity_num_conformers=self.per_fidelity_num_conformers,
            correction_factor=self.correction_factor,
            mol_repr=self.mol_repr,
            negate_score=self.negate_score,
            log_molecule_visualizations=self.log_molecule_visualizations,
            molecule_visualization_limit=self.molecule_visualization_limit,
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

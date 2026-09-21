"""Configuration models for molecular oracles."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from activelearning.config_registry import BuildableConfig
from activelearning.oracle.oracle import Oracle


class XTBIPEAOracleConfig(BuildableConfig):
    """Configuration for the xTB ionisation-potential/electron-affinity oracle."""

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

    @property
    def input_representation(self) -> str:
        """Expose the configured candidate representation to core validation."""
        return self.mol_repr

    @model_validator(mode="after")
    def validate_per_fidelity_num_conformers(self) -> "XTBIPEAOracleConfig":
        """Validate conformer overrides against declared fidelity levels."""
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
        """Build the xTB-backed oracle lazily."""
        from activelearning_molecules.oracles.xtb import (
            ConformerConfig,
            XTBIPEAOracle,
        )

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


MOLECULE_ORACLE_CONFIGS = (XTBIPEAOracleConfig,)

__all__ = ["MOLECULE_ORACLE_CONFIGS", "XTBIPEAOracleConfig"]

"""Pydantic models of oracles.

Changes in the interface of existing oracles should be reflected in this
configuration. Core oracles are registered by the core registry bootstrap;
application packages expose their oracle catalogs through their package
configuration mapping.
"""

from typing import ClassVar, Literal

from activelearning.config_registry import (
    BuildableConfig,
    registered_config,
)
from activelearning.oracle.augmented_function_oracle import (
    BraninOracle,
    Hartmann6DOracle,
)
from activelearning.oracle.composite_oracle import CompositeOracle
from activelearning.oracle.oracle import Oracle


class BraninOracleConfig(BuildableConfig):
    """Configuration for the analytic Branin multi-fidelity oracle."""

    type: Literal["BraninOracle"] = "BraninOracle"
    input_representation: ClassVar[str] = "numeric"
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None
    log_landscape: bool = False

    def build(self) -> Oracle:
        """Build the configured Branin oracle.

        Returns
        -------
        Oracle
            Configured :class:`~activelearning.oracle.augmented_function_oracle.BraninOracle`.
        """
        return BraninOracle(
            self.fidelity_costs, self.fidelity_confidences, self.log_landscape
        )


class Hartmann6DOracleConfig(BuildableConfig):
    """Configuration for the analytic six-dimensional Hartmann oracle."""

    type: Literal["Hartmann6DOracle"] = "Hartmann6DOracle"
    input_representation: ClassVar[str] = "numeric"
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None

    def build(self) -> Oracle:
        """Build the configured Hartmann six-dimensional oracle.

        Returns
        -------
        Oracle
            Configured :class:`~activelearning.oracle.augmented_function_oracle.Hartmann6DOracle`.
        """
        return Hartmann6DOracle(self.fidelity_costs, self.fidelity_confidences)


class CompositeOracleConfig(BuildableConfig):
    """Configuration for an oracle composed from multiple sub-oracles."""

    type: Literal["CompositeOracle"] = "CompositeOracle"
    sub_oracles: list["OracleConfig"]

    @property
    def input_representation(self) -> str | None:
        """Return the candidate representation shared by all sub-oracles."""
        representations = [
            getattr(sub_oracle, "input_representation", None)
            for sub_oracle in self.sub_oracles
        ]
        if not representations or any(
            not isinstance(representation, str) for representation in representations
        ):
            return None
        unique_representations = set(representations)
        if len(unique_representations) != 1:
            return None
        return unique_representations.pop()

    def build(self) -> Oracle:
        """Build each configured sub-oracle and combine their outputs.

        Returns
        -------
        Oracle
            Configured :class:`~activelearning.oracle.composite_oracle.CompositeOracle`.
        """
        return CompositeOracle(sub_oracles=[cfg.build() for cfg in self.sub_oracles])


ORACLE_CONFIGS = (BraninOracleConfig, Hartmann6DOracleConfig, CompositeOracleConfig)
OracleConfig = registered_config("oracle")

CompositeOracleConfig.model_rebuild(
    _types_namespace={"OracleConfig": OracleConfig},
)

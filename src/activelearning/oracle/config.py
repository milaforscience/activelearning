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
    """Configuration for the analytic Branin multi-fidelity oracle."""

    type: Literal["BraninOracle"] = "BraninOracle"
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


class Hartmann6DOracleConfig(BaseModel):
    """Configuration for the analytic six-dimensional Hartmann oracle."""

    type: Literal["Hartmann6DOracle"] = "Hartmann6DOracle"
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


class CompositeOracleConfig(BaseModel):
    """Configuration for an oracle composed from multiple sub-oracles."""

    type: Literal["CompositeOracle"] = "CompositeOracle"
    sub_oracles: list["OracleConfig"]

    def build(self) -> Oracle:
        """Build each configured sub-oracle and combine their outputs.

        Returns
        -------
        Oracle
            Configured :class:`~activelearning.oracle.composite_oracle.CompositeOracle`.
        """
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
        """Validate conformer overrides against declared fidelity levels.

        Returns
        -------
        XTBIPEAOracleConfig
            This validated configuration instance.

        Raises
        ------
        ValueError
            If an override has a non-positive count or references an
            undeclared fidelity.
        """
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
        """Build the xTB-backed oracle lazily.

        Returns
        -------
        Oracle
            Configured :class:`~activelearning.applications.molecules.xtb_oracle.XTBIPEAOracle`.

        Raises
        ------
        ImportError
            If the optional molecules dependencies are not installed.
        """
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


class Dock3OracleConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.dock3_oracle.Dock3Oracle`.

    Parameters
    ----------
    indock_template : str
        INDOCK template shipped with the receptor dockfiles.
    dockfiles_dir : str
        Directory of prepared receptor grid files.
    fidelity_costs : dict[int, float]
        Cost per sample. DOCK3 has no fidelity ladder, so exactly one fidelity
        level must be declared; compose with other oracles through
        ``CompositeOracle`` for multi-fidelity runs.
    fidelity_confidences : dict[int, float], optional
        Confidence in ``[0, 1]`` per fidelity.  Defaults to costs normalised by max.
    mol_repr : str
        Input molecules representation.  DOCK3 only accepts ``"smiles"``.
    dockenv_sh : str, optional
        Environment bootstrap script sourced before ligbuild.  Defaults to the
        shared cluster install.
    dock64_exe : str, optional
        Path to the dock64 binary.  Defaults to the shared cluster install.
    ligbuild_exe : str
        Name or path of the ligbuild executable, resolved from ``$PATH`` after
        sourcing ``dockenv_sh``.
    tmp_dir : str, optional
        Directory for per-call workdirs, preserved for debugging.  Keep it short
        (under ~25 characters); AMSOL silently corrupts builds when total paths
        exceed its fixed-width Fortran buffer.
    timeout : int
        Wall-clock seconds allowed for *each* of the ligbuild and dock64
        subprocesses.  Note that this, and not ``ligbuild_timeout``, is what
        actually bounds a ligbuild run.
    ligbuild_timeout : int
        Internal per-protomer timeout hint passed to ligbuild through
        ``custom_parms.json``.  Only meaningful when below ``timeout``.
    num_workers : int, optional
        Threads used per query batch.  ``null`` resolves to
        ``$SLURM_CPUS_PER_TASK``, else the CPU count, else 1.
    negate_score : bool
        Whether to return ``-score`` so the maximizing loop chases the
        strongest binders.  DOCK3 scores are negative-is-better.
    warmup : bool
        Whether to probe the docking environment during construction.  Leave
        enabled in production: the probe must run single-threaded before any
        parallel ligbuild call.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["Dock3Oracle"] = "Dock3Oracle"
    indock_template: str
    dockfiles_dir: str
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None
    mol_repr: Literal["smiles"] = "smiles"
    dockenv_sh: str | None = None
    dock64_exe: str | None = None
    ligbuild_exe: str = "ligbuild"
    tmp_dir: str | None = None
    timeout: int = Field(default=300, gt=0)
    ligbuild_timeout: int = Field(default=150, gt=0)
    num_workers: int | None = Field(default=1, ge=1)
    negate_score: bool = True
    warmup: bool = True

    @model_validator(mode="after")
    def validate_single_fidelity(self) -> "Dock3OracleConfig":
        """Validate that exactly one fidelity level is declared.

        Returns
        -------
        Dock3OracleConfig
            This validated configuration instance.

        Raises
        ------
        ValueError
            If zero or more than one fidelity level is declared.
        """
        if len(self.fidelity_costs) != 1:
            raise ValueError(
                "Dock3Oracle is single-fidelity and must declare exactly one "
                f"fidelity level in fidelity_costs; got {sorted(self.fidelity_costs)}."
            )
        return self

    def build(self) -> Oracle:
        """Build the DOCK3-backed oracle lazily.

        Returns
        -------
        Oracle
            Configured :class:`~activelearning.applications.molecules.dock3_oracle.Dock3Oracle`.
        """
        from activelearning.applications.molecules.dock3_oracle import Dock3Oracle

        return Dock3Oracle(
            indock_template=self.indock_template,
            dockfiles_dir=self.dockfiles_dir,
            fidelity_costs=self.fidelity_costs,
            fidelity_confidences=self.fidelity_confidences,
            dockenv_sh=self.dockenv_sh,
            dock64_exe=self.dock64_exe,
            ligbuild_exe=self.ligbuild_exe,
            tmp_dir=self.tmp_dir,
            timeout=self.timeout,
            ligbuild_timeout=self.ligbuild_timeout,
            num_workers=self.num_workers,
            negate_score=self.negate_score,
            warmup=self.warmup,
        )


OracleConfig = Annotated[
    Union[
        BraninOracleConfig,
        Hartmann6DOracleConfig,
        CompositeOracleConfig,
        XTBIPEAOracleConfig,
        Dock3OracleConfig,
    ],
    Field(discriminator="type"),
]

CompositeOracleConfig.model_rebuild()

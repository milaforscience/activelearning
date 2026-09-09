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

    The oracle observes an estimated probability of binding, converted from the
    raw DOCK3 score by the fitted hit-rate model. The raw score is retained in
    observation metadata.

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
        Threads used per query batch.  Defaults to ``null``, which sizes the
        pool from the CPUs this job actually holds -- the process CPU affinity
        mask and ``$SLURM_CPUS_PER_TASK``, whichever is smaller.  The achieved
        concurrency is still capped by the batch the selector hands the oracle.
    hitrate_params : str
        JSON file of fitted hit-rate parameters, keyed by target name.
    score_pprop_table : str
        Score/pProp lookup table for the reference library screen.
    pki_threshold : float
        Experimental pKi at or above which a molecule counts as a hit.
    hitrate_target : str
        Key to read from ``hitrate_params``.
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
    hitrate_params: str
    score_pprop_table: str
    pki_threshold: float
    hitrate_target: str = "ampc"
    dockenv_sh: str | None = None
    dock64_exe: str | None = None
    ligbuild_exe: str = "ligbuild"
    tmp_dir: str | None = None
    timeout: int = Field(default=300, gt=0)
    ligbuild_timeout: int = Field(default=150, gt=0)
    num_workers: int | None = Field(default=None, ge=1)
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
            hitrate_params=self.hitrate_params,
            score_pprop_table=self.score_pprop_table,
            pki_threshold=self.pki_threshold,
            hitrate_target=self.hitrate_target,
            warmup=self.warmup,
        )


class CxcalcOracleConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.cxcalc_oracle.CxcalcOracle`.

    The oracle observes an estimated probability of binding, converted from the
    raw anionic percentage by a configurable two-valued step. The percentage is
    retained in observation metadata. Reporting a probability is what lets this
    oracle share a multi-fidelity target with ``Dock3Oracle``.

    Parameters
    ----------
    fidelity_costs : dict[int, float]
        Cost per sample. cxcalc computes one property at one pH, so exactly one
        fidelity level must be declared; compose with other oracles through
        ``CompositeOracle`` for multi-fidelity runs.
    fidelity_confidences : dict[int, float], optional
        Confidence in ``[0, 1]`` per fidelity.  Defaults to costs normalised by max,
        which for a single-fidelity oracle is always ``1.0`` - set it explicitly
        when composing with another oracle, or both will claim full confidence.
    mol_repr : str
        Input molecules representation.  cxcalc only accepts ``"smiles"``.
    cxcalc_exe : str, optional
        Path to the cxcalc launcher.  Defaults to the shared cluster install.
    env_setup : str, optional
        Shell command run before cxcalc to put a JVM on ``PATH``.  Defaults to
        ``"module load java"``; set to ``null`` when java is already loaded.
    ph : float
        pH at which the microspecies distribution is computed.
    anionic_charges : list[int]
        Net formal charges counted as anionic.
    anion_percent_threshold : float
        Percentage strictly above which a molecule counts as anionic.  The
        default of ``0.0`` treats any trace of charge as anionic; raising it to
        ``0.5`` or ``1.0`` excludes a large band of barely-charged molecules
        that dock much more like the uncharged ones.
    zero_probability : float
        Probability reported at or below the threshold.
    nonzero_probability : float
        Probability reported above the threshold.
    timeout : int
        Wall-clock seconds allowed for each cxcalc subprocess.
    chunk_size : int
        Maximum molecules per cxcalc invocation.  A typical acquisition batch is
        a single chunk; cxcalc is a JVM application, so batching is what makes
        this oracle cheap.
    num_workers : int, optional
        Threads used to evaluate chunks.  Defaults to ``null``, which sizes
        the pool from the CPUs this job actually holds -- the process CPU
        affinity mask and ``$SLURM_CPUS_PER_TASK``, whichever is smaller.  Only
        relevant for batches larger than ``chunk_size``.
    warmup : bool
        Whether to prime the cxcalc environment during construction.  Leave
        enabled in production: the probe must run single-threaded before any
        parallel call.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["CxcalcOracle"] = "CxcalcOracle"
    fidelity_costs: dict[int, float]
    fidelity_confidences: dict[int, float] | None = None
    mol_repr: Literal["smiles"] = "smiles"
    cxcalc_exe: str | None = None
    env_setup: str | None = "module load java"
    ph: float = 7.4
    anionic_charges: list[int] = Field(default=[-1, -2], min_length=1)
    anion_percent_threshold: float = Field(default=0.0, ge=0.0)
    zero_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    nonzero_probability: float = Field(default=0.01, ge=0.0, le=1.0)
    timeout: int = Field(default=600, gt=0)
    chunk_size: int = Field(default=12500, ge=1)
    num_workers: int | None = Field(default=None, ge=1)
    warmup: bool = True

    @model_validator(mode="after")
    def validate_single_fidelity(self) -> "CxcalcOracleConfig":
        """Validate that exactly one fidelity level is declared.

        Returns
        -------
        CxcalcOracleConfig
            This validated configuration instance.

        Raises
        ------
        ValueError
            If zero or more than one fidelity level is declared.
        """
        if len(self.fidelity_costs) != 1:
            raise ValueError(
                "CxcalcOracle is single-fidelity and must declare exactly one "
                f"fidelity level in fidelity_costs; got {sorted(self.fidelity_costs)}."
            )
        return self

    def build(self) -> Oracle:
        """Build the cxcalc-backed oracle lazily.

        Returns
        -------
        Oracle
            Configured :class:`~activelearning.applications.molecules.cxcalc_oracle.CxcalcOracle`.
        """
        from activelearning.applications.molecules.cxcalc_oracle import CxcalcOracle

        return CxcalcOracle(
            fidelity_costs=self.fidelity_costs,
            fidelity_confidences=self.fidelity_confidences,
            cxcalc_exe=self.cxcalc_exe,
            env_setup=self.env_setup,
            ph=self.ph,
            anionic_charges=self.anionic_charges,
            anion_percent_threshold=self.anion_percent_threshold,
            zero_probability=self.zero_probability,
            nonzero_probability=self.nonzero_probability,
            timeout=self.timeout,
            chunk_size=self.chunk_size,
            num_workers=self.num_workers,
            warmup=self.warmup,
        )


OracleConfig = Annotated[
    Union[
        BraninOracleConfig,
        Hartmann6DOracleConfig,
        CompositeOracleConfig,
        XTBIPEAOracleConfig,
        Dock3OracleConfig,
        CxcalcOracleConfig,
    ],
    Field(discriminator="type"),
]

CompositeOracleConfig.model_rebuild()

"""Pydantic model for the configuration of an active learning run.

The configuration is based on Pydantic. :class:`ActiveLearningConfig` defines
the components of an active learning run and validates relationships between
them.

Changes in the active learning interface should be reflected in this configuration to
ensure consistency.

See the `Pydantic Docs <https://pydantic.dev/docs/validation/latest/get-started/>`_ for
further reference.
"""

from pydantic import BaseModel, Field, model_validator

from activelearning.acquisition.config import (
    AcquisitionConfig,
    ExpectedImprovementConfig,
    LogExpectedImprovementConfig,
    LogProbabilityOfImprovementConfig,
    PosteriorMeanConfig,
    ProbabilityOfImprovementConfig,
    QMultiFidelityKnowledgeGradientConfig,
    QMultiFidelityLowerBoundMaxValueEntropyConfig,
    QMultiFidelityMaxValueEntropyConfig,
    UpperConfidenceBoundConfig,
)
from activelearning.budget.config import BudgetConfig
from activelearning.dataset.config import DatasetConfig
from activelearning.logger.config import LoggerConfig
from activelearning.run_writer import RunWriterConfig
from activelearning.oracle.config import OracleConfig
from activelearning.runtime import RuntimeContextConfig
from activelearning.sampler.config import (
    SamplerConfig,
)
from activelearning.selector.config import SelectorConfig
from activelearning.surrogate.config import (
    DummyMeanSurrogateConfig,
    FidelityAwareSurrogateConfig,
    SurrogateConfig,
)


_BOTORCH_ACQUISITION_CONFIG_TYPES = (
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


class ActiveLearningConfig(BaseModel):
    """Validated composition of one active-learning experiment.

    Parameters
    ----------
    runtime : RuntimeContextConfig
        Device and floating-point runtime settings.
    dataset : DatasetConfig
        Initial observations and dataset persistence settings.
    surrogate : SurrogateConfig
        Model used to approximate the oracle objective.
    acquisition : AcquisitionConfig
        Rule used to score candidate queries.
    sampler : SamplerConfig
        Candidate-generation strategy.
    selector : SelectorConfig
        Rule used to select candidates for evaluation.
    oracle : OracleConfig
        Objective evaluator and fidelity metadata source.
    budget : BudgetConfig
        Query and round budget.
    logger : LoggerConfig, optional
        Logging configuration.
    run_writer : RunWriterConfig, optional
        Persistent run-output configuration.
    """

    runtime: RuntimeContextConfig = Field(default_factory=RuntimeContextConfig)
    dataset: DatasetConfig
    surrogate: SurrogateConfig
    acquisition: AcquisitionConfig
    sampler: SamplerConfig
    selector: SelectorConfig
    oracle: OracleConfig
    budget: BudgetConfig
    logger: LoggerConfig | None = None
    run_writer: RunWriterConfig | None = None

    @model_validator(mode="after")
    def _resolve_fidelities(self) -> "ActiveLearningConfig":
        """Resolve fidelity metadata across the configured components.

        The oracle defines the authoritative fidelity levels and their
        confidences. Missing sampler levels are filled from that set, explicit
        sampler levels are validated against it, and fidelity-aware surrogate
        configs resolve their settings from the resulting metadata. Known
        incompatible sampler, surrogate, acquisition, and oracle combinations
        are rejected after that resolution.

        Returns
        -------
        ActiveLearningConfig
            This validated config instance with resolved sampler and surrogate
            fidelity settings.

        Raises
        ------
        ValueError
            If the oracle declares no fidelity levels, or if the sampler
            or surrogate references a fidelity not declared by the oracle, or
            if built-in component contracts are incompatible.
        """
        fidelity_confidences = _extract_oracle_fidelity_confidences(self.oracle)
        if not fidelity_confidences:
            raise ValueError("Oracle config declares no fidelity levels.")

        self.sampler = _resolve_sampler_fidelities(
            self.sampler,
            set(fidelity_confidences),
        )
        self.surrogate = _resolve_surrogate_config_fidelities(
            self.surrogate,
            fidelity_confidences,
        )
        _validate_component_compatibility(
            sampler=self.sampler,
            surrogate=self.surrogate,
            acquisition=self.acquisition,
            oracle=self.oracle,
        )
        return self


def _direct_oracle_fidelity_confidences(oracle_config: object) -> dict[int, float]:
    """Resolve fidelity confidences for a non-composite oracle config.

    Explicit confidences are returned as floats. If they are omitted, each
    confidence is derived by dividing its fidelity cost by the maximum cost.

    Parameters
    ----------
    oracle_config : object
        Oracle config exposing ``fidelity_costs`` and
        ``fidelity_confidences`` attributes.

    Returns
    -------
    dict[int, float]
        Confidence associated with each declared fidelity. Returns an empty
        mapping when both costs and confidences are empty.

    Raises
    ------
    ValueError
        If explicit confidence keys do not exactly match the fidelity cost
        keys.
    """
    fidelity_costs = dict(oracle_config.fidelity_costs)
    fidelity_confidences = oracle_config.fidelity_confidences

    if fidelity_confidences is None:
        if not fidelity_costs:
            return {}
        max_cost = max(fidelity_costs.values())
        return {
            fidelity: float(cost / max_cost)
            for fidelity, cost in fidelity_costs.items()
        }

    if fidelity_confidences.keys() != fidelity_costs.keys():
        raise ValueError(
            "Oracle fidelity_confidences keys must match fidelity_costs keys. "
            f"Got {sorted(fidelity_confidences)}, "
            f"expected {sorted(fidelity_costs)}."
        )
    return {
        fidelity: float(confidence)
        for fidelity, confidence in fidelity_confidences.items()
    }


def _merge_fidelity_confidences(
    target: dict[int, float],
    source: dict[int, float],
) -> None:
    """Merge fidelity confidences into an existing mapping.

    Parameters
    ----------
    target : dict[int, float]
        Destination mapping, mutated in place.
    source : dict[int, float]
        Fidelity confidences to add to ``target``.

    Raises
    ------
    ValueError
        If both mappings contain the same fidelity with different confidence
        values.
    """
    for fidelity, confidence in source.items():
        existing = target.get(fidelity)
        if existing is not None and existing != confidence:
            raise ValueError(
                "Composite oracle declares inconsistent confidence for "
                f"fidelity {fidelity}: {existing} vs {confidence}."
            )
        target[fidelity] = confidence


def _extract_oracle_fidelity_confidences(
    oracle_config: object,
) -> dict[int, float]:
    """Recursively resolve fidelity confidences from an oracle config.

    Direct oracle configs derive their metadata from ``fidelity_costs`` and
    ``fidelity_confidences``. Composite configs recursively merge metadata
    from their ``sub_oracles``.

    Parameters
    ----------
    oracle_config : object
        Direct oracle config or composite config containing ``sub_oracles``.

    Returns
    -------
    dict[int, float]
        Combined confidence mapping for all declared fidelity levels.

    Raises
    ------
    ValueError
        If confidence and cost keys differ, composite sub-oracles disagree on
        a shared fidelity, or the config exposes no supported fidelity
        metadata.
    """
    if hasattr(oracle_config, "fidelity_costs"):
        return _direct_oracle_fidelity_confidences(oracle_config)

    if hasattr(oracle_config, "sub_oracles"):
        confidences: dict[int, float] = {}
        for sub_oracle in oracle_config.sub_oracles:
            _merge_fidelity_confidences(
                confidences,
                _extract_oracle_fidelity_confidences(sub_oracle),
            )
        return confidences

    raise ValueError(
        "Cannot extract fidelity metadata from oracle config "
        f"{type(oracle_config).__name__}. "
        "Oracle configs must expose 'fidelity_costs' or 'sub_oracles'."
    )


def _sampler_fidelity_keys(fidelities: object) -> set[int] | None:
    """Extract the unique fidelity levels configured for a sampler.

    Parameters
    ----------
    fidelities : object
        Sampler fidelity configuration: a list of levels, a mapping keyed by
        level, or ``None`` when levels should be inferred from the oracle.

    Returns
    -------
    set[int] or None
        Unique configured levels, or ``None`` when no levels were specified.

    Raises
    ------
    ValueError
        If a fidelity sequence contains duplicate levels.
    """
    if fidelities is None:
        return None
    if isinstance(fidelities, dict):
        return set(fidelities)

    fidelity_keys = set(fidelities)
    if len(fidelity_keys) != len(fidelities):
        raise ValueError("Sampler fidelities must not contain duplicate levels.")
    return fidelity_keys


def _resolve_sampler_fidelities(
    sampler: BaseModel,
    oracle_fidelities: set[int],
) -> BaseModel:
    """Resolve sampler fidelity levels against the authoritative oracle set.

    Parameters
    ----------
    sampler : BaseModel
        Sampler config exposing a ``fidelities`` field.
    oracle_fidelities : set[int]
        Fidelity levels declared by the oracle.

    Returns
    -------
    BaseModel
        A copied sampler config populated with all oracle levels when its
        fidelities were unset; otherwise, the original validated config.

    Raises
    ------
    ValueError
        If configured levels contain duplicates or include levels not declared
        by the oracle.
    """
    configured = _sampler_fidelity_keys(sampler.fidelities)
    if configured is None:
        return sampler.model_copy(update={"fidelities": sorted(oracle_fidelities)})

    unknown = configured - oracle_fidelities
    if unknown:
        raise ValueError(
            f"Sampler fidelities {sorted(unknown)} are not declared in the "
            "oracle's fidelity_costs. Oracle declares: "
            f"{sorted(oracle_fidelities)}."
        )
    return sampler


def _resolve_surrogate_config_fidelities(
    surrogate: BaseModel,
    fidelity_confidences: dict[int, float],
) -> BaseModel:
    """Resolve surrogate fidelity settings through its declared contract.

    Parameters
    ----------
    surrogate : BaseModel
        Parsed surrogate config.
    fidelity_confidences : dict[int, float]
        Non-empty oracle confidence mapping.

    Returns
    -------
    BaseModel
        Revalidated config with resolved fidelity settings.

    Raises
    ------
    ValueError
        If the surrogate does not support a multi-fidelity oracle or rejects
        the provided fidelity metadata.
    """
    if isinstance(surrogate, FidelityAwareSurrogateConfig):
        return surrogate.resolve_fidelity_confidences(fidelity_confidences)

    if len(fidelity_confidences) > 1:
        raise ValueError(
            f"{type(surrogate).__name__} does not support multi-fidelity "
            "oracles. Implement FidelityAwareSurrogateConfig to opt in."
        )
    return surrogate


def _extract_molecule_representation(oracle: BaseModel) -> str | None:
    """Return one molecular representation shared by an oracle configuration.

    Direct molecular oracles expose ``mol_repr``. Composite oracles are
    considered molecular only when every sub-oracle exposes the same
    representation. Returning ``None`` means that the oracle is not known to
    accept one consistent molecular string representation.

    Parameters
    ----------
    oracle : BaseModel
        Parsed oracle configuration.

    Returns
    -------
    str or None
        The shared molecular representation, or ``None`` when it cannot be
        established from the configuration.
    """
    representation = getattr(oracle, "mol_repr", None)
    if representation is not None:
        return str(representation)

    sub_oracles = getattr(oracle, "sub_oracles", None)
    if sub_oracles is None:
        return None

    representations = [
        _extract_molecule_representation(sub_oracle) for sub_oracle in sub_oracles
    ]
    if not representations or any(item is None for item in representations):
        return None
    if len(set(representations)) != 1:
        return None
    return representations[0]


def _encoder_molecule_representation(component: object) -> str | None:
    """Return the representation declared by an encoder config.

    Parameters
    ----------
    component : object
        Parsed encoder configuration.

    Returns
    -------
    str or None
        Declared representation, or ``None`` when no representation is known.
    """
    if not isinstance(component, BaseModel):
        return None
    representation = getattr(component, "input_representation", None)
    return representation if isinstance(representation, str) else None


def _validate_component_compatibility(
    *,
    sampler: BaseModel,
    surrogate: BaseModel,
    acquisition: BaseModel,
    oracle: BaseModel,
) -> None:
    """Reject parsed component combinations with incompatible contracts.

    This validator handles only incompatibilities that are knowable from the
    built-in configuration types. Candidate contents in pool files and opaque
    custom GFlowNet environments remain runtime concerns.

    Parameters
    ----------
    sampler : BaseModel
        Parsed sampler configuration.
    surrogate : BaseModel
        Parsed surrogate configuration.
    acquisition : BaseModel
        Parsed acquisition configuration.
    oracle : BaseModel
        Parsed oracle configuration.

    Raises
    ------
    ValueError
        If a known sampler, surrogate, acquisition, or oracle contract is
        incompatible with another configured component.
    """
    encoder_component = getattr(surrogate, "encoder", None)
    encoder_representation = _encoder_molecule_representation(encoder_component)
    oracle_representation = _extract_molecule_representation(oracle)

    sampler_representation = getattr(sampler, "output_representation", None)
    surrogate_representation = getattr(surrogate, "input_representation", None)

    if sampler_representation == "smiles":
        if encoder_representation == "selfies":
            raise ValueError(
                "S3GFNSampler generates canonical SMILES, but the configured "
                "SelfiesTransformerEncoder expects SELFIES strings. Use "
                "a SMILES encoder with S3-GFN or choose a "
                "SELFIES-producing sampler."
            )
        if oracle_representation != "smiles":
            configured = (
                repr(oracle_representation)
                if oracle_representation is not None
                else "no consistent molecular representation"
            )
            raise ValueError(
                "S3GFNSampler generates canonical SMILES and requires a "
                "molecular oracle with mol_repr='smiles'; got "
                f"{configured}."
            )

    if sampler_representation == "numeric":
        if encoder_representation is not None:
            raise ValueError(
                f"{type(sampler).__name__} generates numeric candidates, but "
                f"{type(encoder_component).__name__} expects "
                f"{encoder_representation.upper()} strings."
            )
        if oracle_representation is not None:
            raise ValueError(
                f"{type(sampler).__name__} generates numeric candidates, but "
                f"{type(oracle).__name__} expects molecular strings with "
                f"mol_repr={oracle_representation!r}."
            )

    if encoder_representation is not None:
        if oracle_representation is None:
            raise ValueError(
                f"{type(encoder_component).__name__} expects "
                f"{encoder_representation.upper()} strings, but the configured "
                f"{type(oracle).__name__} does not expose one consistent "
                "molecular representation."
            )
        if oracle_representation != encoder_representation:
            raise ValueError(
                f"{type(encoder_component).__name__} expects "
                f"{encoder_representation.upper()} strings, but the oracle is "
                f"configured with mol_repr={oracle_representation!r}."
            )

    if surrogate_representation == "numeric" and (
        sampler_representation == "smiles" or oracle_representation is not None
    ):
        raise ValueError(
            "BoTorchGPSurrogateConfig expects numeric tensor inputs, but the "
            "configured components use molecular strings. Use a molecular DKL "
            "surrogate with an appropriate encoder."
        )

    if isinstance(acquisition, _BOTORCH_ACQUISITION_CONFIG_TYPES) and isinstance(
        surrogate, DummyMeanSurrogateConfig
    ):
        raise ValueError(
            f"{type(acquisition).__name__} requires a BoTorch-compatible "
            "surrogate, but DummyMeanSurrogateConfig is not one."
        )

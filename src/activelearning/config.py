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

from activelearning.acquisition.config import AcquisitionConfig
from activelearning.budget.config import BudgetConfig
from activelearning.dataset.config import DatasetConfig
from activelearning.logger.config import LoggerConfig
from activelearning.monitoring.diagnostics_config import DiagnosticsConfig
from activelearning.monitoring.run_writer import RunWriterConfig
from activelearning.oracle.config import OracleConfig
from activelearning.runtime import RuntimeContextConfig
from activelearning.sampler.config import (
    SamplerConfig,
)
from activelearning.selector.config import SelectorConfig
from activelearning.surrogate.config import (
    FidelityAwareSurrogateConfig,
    SurrogateConfig,
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
        Optional live telemetry backend configuration.
    run_writer : RunWriterConfig, optional
        Optional durable structured-output configuration.
    diagnostics : DiagnosticsConfig
        Optional diagnostic enrichment controls for either configured sink.
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
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    @model_validator(mode="after")
    def _resolve_fidelities(
        self,
    ) -> "ActiveLearningConfig":
        """Resolve fidelity metadata across the configured components.

        The oracle defines the authoritative fidelity levels and their
        confidences. Missing sampler levels are filled from that set, explicit
        sampler levels are validated against it, and fidelity-aware surrogate
        configs resolve their settings from the resulting metadata. Declared
        representation and BoTorch compatibility are checked after that
        resolution.

        Returns
        -------
        ActiveLearningConfig
            This validated config instance with resolved sampler and surrogate
            fidelity settings.

        Raises
        ------
        ValueError
            If the oracle declares no fidelity levels, if the sampler or surrogate
            references a fidelity not declared by the oracle, or if declared
            component contracts are incompatible.
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


def _declared_representation(config: object, attribute: str) -> str | None:
    """Return a component representation declaration when one is available."""
    representation = getattr(config, attribute, None)
    return representation if isinstance(representation, str) else None


def _validate_representation_pair(
    *,
    producer: BaseModel,
    producer_attribute: str,
    consumer: BaseModel,
    consumer_attribute: str,
) -> None:
    """Reject a producer/consumer pair with incompatible declarations."""
    produced = _declared_representation(producer, producer_attribute)
    expected = _declared_representation(consumer, consumer_attribute)
    if produced is not None and expected is not None and produced != expected:
        raise ValueError(
            f"{type(producer).__name__} produces representation {produced!r}, "
            f"but {type(consumer).__name__} expects {expected!r}."
        )


def _validate_shared_input_representation(
    *,
    first: BaseModel,
    second: BaseModel,
) -> None:
    """Reject components that require different candidate representations."""
    first_expected = _declared_representation(first, "input_representation")
    second_expected = _declared_representation(second, "input_representation")
    if (
        first_expected is not None
        and second_expected is not None
        and first_expected != second_expected
    ):
        raise ValueError(
            f"{type(first).__name__} expects representation {first_expected!r}, "
            f"but {type(second).__name__} expects {second_expected!r}."
        )


def _validate_botorch_compatibility(
    *,
    acquisition: BaseModel,
    surrogate: BaseModel,
) -> None:
    """Reject BoTorch acquisitions paired with an incompatible surrogate."""
    requires_botorch = getattr(acquisition, "requires_botorch_surrogate", False)
    is_botorch_compatible = getattr(surrogate, "is_botorch_compatible", False)
    if requires_botorch and not is_botorch_compatible:
        raise ValueError(
            f"{type(acquisition).__name__} requires a BoTorch-compatible "
            f"surrogate, but {type(surrogate).__name__} is not compatible."
        )


def _validate_component_compatibility(
    *,
    sampler: BaseModel,
    surrogate: BaseModel,
    acquisition: BaseModel,
    oracle: BaseModel,
) -> None:
    """Reject parsed component combinations with incompatible contracts.

    The component schemas declare representation and BoTorch compatibility.
    Candidate contents in pool files and opaque custom GFlowNet environments
    remain runtime concerns.

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
        If a declared sampler, surrogate, acquisition, or oracle contract is
        incompatible with another configured component.
    """
    encoder = getattr(surrogate, "encoder", None)
    sampler_representation = _declared_representation(
        sampler,
        "output_representation",
    )
    oracle_representation = _declared_representation(
        oracle,
        "input_representation",
    )
    if sampler_representation is not None and oracle_representation is None:
        raise ValueError(
            f"{type(sampler).__name__} produces representation "
            f"{sampler_representation!r}, but {type(oracle).__name__} does not "
            "declare its input_representation."
        )

    if isinstance(encoder, BaseModel):
        _validate_representation_pair(
            producer=sampler,
            producer_attribute="output_representation",
            consumer=encoder,
            consumer_attribute="input_representation",
        )

    _validate_representation_pair(
        producer=sampler,
        producer_attribute="output_representation",
        consumer=oracle,
        consumer_attribute="input_representation",
    )

    if isinstance(encoder, BaseModel):
        _validate_representation_pair(
            producer=encoder,
            producer_attribute="input_representation",
            consumer=oracle,
            consumer_attribute="input_representation",
        )

    _validate_representation_pair(
        producer=sampler,
        producer_attribute="output_representation",
        consumer=surrogate,
        consumer_attribute="input_representation",
    )
    _validate_shared_input_representation(first=surrogate, second=oracle)
    _validate_botorch_compatibility(
        acquisition=acquisition,
        surrogate=surrogate,
    )

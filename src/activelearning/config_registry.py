"""Registry for extensible component configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, BeforeValidator, SerializeAsAny, ValidationInfo
from pydantic_core import PydanticCustomError

ConfigCategory = Literal[
    "dataset",
    "surrogate",
    "acquisition",
    "sampler",
    "selector",
    "oracle",
    "logger",
    "encoder",
    "fixed_encoder",
]


class BuildableConfig(BaseModel):
    """Base class for configuration models that construct runtime components."""

    type: str

    def build(self) -> object:
        """Build the configured runtime component."""
        raise NotImplementedError


ConfigCatalogs = Mapping[
    ConfigCategory,
    tuple[type[BuildableConfig], ...],
]


class ConfigRegistry:
    """Map YAML type names to configuration models."""

    def __init__(self) -> None:
        self._models: dict[ConfigCategory, dict[str, type[BuildableConfig]]] = {
            category: {} for category in get_args(ConfigCategory)
        }
        self._model_origins: dict[tuple[ConfigCategory, str], str] = {}

    def install(self, catalogs: ConfigCatalogs, *, origin: str) -> None:
        """Install the models in a package's category mapping."""
        if not isinstance(catalogs, Mapping):
            raise TypeError(
                f"Configuration catalogs from {origin} must be a mapping "
                "from categories to model tuples."
            )

        for category, models in catalogs.items():
            if category not in self._models:
                raise ValueError(f"Unsupported configuration category: {category!r}.")
            for model in models:
                self._register(category, model, origin)

    def _register(
        self,
        category: ConfigCategory,
        model: type[BuildableConfig],
        origin: str,
    ) -> None:
        if not isinstance(model, type) or not issubclass(model, BuildableConfig):
            raise TypeError(
                f"{model!r} must be a BuildableConfig subclass for {category!r}."
            )

        field = model.model_fields.get("type")
        model_name = field.default if field is not None else None
        if not isinstance(model_name, str) or not model_name:
            raise TypeError(
                f"{model.__module__}.{model.__qualname__} must define a "
                "non-empty literal type default."
            )

        existing = self._models[category].get(model_name)
        if existing is not None:
            existing_origin = self._model_origins[(category, model_name)]
            raise ValueError(
                f"Configuration type {model_name!r} is already registered "
                f"in category {category!r} by {existing_origin} "
                f"({existing.__module__}.{existing.__qualname__}); "
                f"cannot register {model.__module__}.{model.__qualname__} "
                f"from {origin}."
            )

        self._models[category][model_name] = model
        self._model_origins[(category, model_name)] = origin

    def resolve(
        self,
        category: ConfigCategory,
        name: str,
    ) -> type[BuildableConfig]:
        """Resolve a discriminator to a registered configuration model."""
        if category not in self._models:
            raise ValueError(f"Unsupported configuration category: {category!r}.")
        try:
            return self._models[category][name]
        except KeyError as error:
            available = sorted(self._models[category])
            raise PydanticCustomError(
                "registered_config_type",
                "Unknown {category} configuration type '{name}'. "
                "Available types: {available}.",
                {
                    "category": category,
                    "name": name,
                    "available": ", ".join(available) or "<none>",
                },
            ) from error


def registered_config(category: ConfigCategory) -> Any:
    """Create a Pydantic annotation that dispatches through the registry."""

    def dispatch(value: Any, info: ValidationInfo) -> BuildableConfig:
        context = dict(info.context) if isinstance(info.context, dict) else {}
        registry = context.get("config_registry") or _get_core_registry()
        # Nested configs must use the same registry as their parent.
        context["config_registry"] = registry

        if isinstance(value, BuildableConfig):
            # A concrete config object has already been validated. This also
            # keeps direct construction of composite config models convenient
            # for application packages, which cannot pass Pydantic context.
            if type(value) is not BuildableConfig:
                return value
            config_type = getattr(value, "type", None)
            if not isinstance(config_type, str):
                raise PydanticCustomError(
                    "registered_config_type",
                    "Registered {category} configuration objects must expose "
                    "a string type.",
                    {"category": category},
                )
            model = registry.resolve(category, config_type)
            return model.model_validate(value.model_dump(), context=context)

        if not isinstance(value, Mapping):
            raise PydanticCustomError(
                "registered_config_mapping",
                "{category} configuration must be a mapping with a type field.",
                {"category": category},
            )
        config_type = value.get("type")
        if not isinstance(config_type, str) or not config_type:
            raise PydanticCustomError(
                "registered_config_type",
                "{category} configuration must define a non-empty string type.",
                {"category": category},
            )
        model = registry.resolve(category, config_type)
        return model.model_validate(dict(value), context=context)

    # Serialize the concrete model, not only BuildableConfig's fields.
    return SerializeAsAny[Annotated[BuildableConfig, BeforeValidator(dispatch)]]


def create_config_registry(
    catalogs: Mapping[str, ConfigCatalogs] | None = None,
) -> ConfigRegistry:
    """Create a registry from core and application-provided catalogs."""
    registry = ConfigRegistry()
    from activelearning.core_config_catalogs import (
        CONFIG_CATALOGS as CORE_CONFIG_CATALOGS,
    )

    registry.install(CORE_CONFIG_CATALOGS, origin="core")
    for origin, package_catalogs in (catalogs or {}).items():
        registry.install(package_catalogs, origin=origin)
    return registry


@cache
def _get_core_registry() -> ConfigRegistry:
    """Return the core-only registry used by direct Pydantic validation."""
    return create_config_registry()

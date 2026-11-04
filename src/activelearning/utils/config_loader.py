from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from activelearning.config_registry import ConfigCatalogs, create_config_registry

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_config(
    path: str | Path | list[str | Path],
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load and merge one or more YAML config files, then apply CLI overrides.

    Parameters
    ----------
    path : str | Path | list[str | Path]
        A single YAML file path or an ordered list of paths. When a list is
        provided, files are merged left to right: later files override earlier
        ones for any shared keys.
    overrides : list[str] | None
        OmegaConf dotlist overrides applied on top of the merged config,
        e.g. ``["budget.available_budget=10", "sampler.num_samples=500"]``.

    Returns
    -------
    DictConfig
        The merged OmegaConf config object. Schema validation is performed
        separately via ``parse_config`` using a Pydantic model.
    """
    paths = [path] if not isinstance(path, list) else path
    cfg = OmegaConf.merge(*[OmegaConf.load(p) for p in paths])
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cast(DictConfig, cfg)


def parse_config(
    cfg: DictConfig,
    model: type[ModelT],
    *,
    catalogs: Mapping[str, ConfigCatalogs] | None = None,
) -> ModelT:
    """Validate an OmegaConf mapping with application-provided catalogs."""
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_config, Mapping):
        raise TypeError("The top-level configuration must be a mapping.")

    return model.model_validate(
        raw_config,
        context={"config_registry": create_config_registry(catalogs)},
    )


def load_and_parse(
    path: str | Path | list[str | Path],
    model: type[ModelT],
    overrides: list[str] | None = None,
    *,
    catalogs: Mapping[str, ConfigCatalogs] | None = None,
) -> ModelT:
    return parse_config(
        load_config(path=path, overrides=overrides),
        model=model,
        catalogs=catalogs,
    )

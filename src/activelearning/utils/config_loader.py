from pathlib import Path
from typing import TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cast(DictConfig, cfg)


def parse_config(cfg: DictConfig, model: type[ModelT]) -> ModelT:
    return model.model_validate(OmegaConf.to_container(cfg, resolve=True))


def load_and_parse(
    path: str | Path, model: type[ModelT], overrides: list[str] | None = None
) -> ModelT:
    return parse_config(load_config(path=path, overrides=overrides), model=model)

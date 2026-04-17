from typing import Any, Annotated, Literal, Union

from pydantic import BaseModel, Field

from activelearning.logger.logger import (
    AimLogger,
    CometLogger,
    MultiLogger,
    ConsoleLogger,
    Logger,
    WandbLogger,
)


class ConsoleLoggerConfig(BaseModel):
    type: Literal["ConsoleLogger"] = "ConsoleLogger"
    project_name: str
    run_name: str | None = None

    def build(self) -> Logger:
        return ConsoleLogger(project_name=self.project_name, run_name=self.run_name)


class WandbLoggerConfig(BaseModel):
    type: Literal["WandbLogger"] = "WandbLogger"
    project_name: str
    run_name: str | None = None

    def build(self) -> Logger:
        return WandbLogger(project_name=self.project_name, run_name=self.run_name)


class CometLoggerConfig(BaseModel):
    type: Literal["CometLogger"] = "CometLogger"
    project_name: str
    run_name: str | None = None
    workspace: str | None = None
    api_key: str | None = None

    def build(self) -> Logger:
        return CometLogger(
            project_name=self.project_name,
            run_name=self.run_name,
            workspace=self.workspace,
            api_key=self.api_key,
        )


class AimLoggerConfig(BaseModel):
    type: Literal["AimLogger"] = "AimLogger"
    project_name: str
    run_name: str | None = None
    repo: str | None = None

    def build(self) -> Logger:
        return AimLogger(
            project_name=self.project_name,
            run_name=self.run_name,
            repo=self.repo,
        )


# Forward reference needed for the recursive MultiLoggerConfig.
_ChildLoggerConfig = Annotated[
    Union[
        "ConsoleLoggerConfig",
        "WandbLoggerConfig",
        "CometLoggerConfig",
        "AimLoggerConfig",
    ],
    Field(discriminator="type"),
]


class MultiLoggerConfig(BaseModel):
    type: Literal["MultiLogger"] = "MultiLogger"
    loggers: list[_ChildLoggerConfig]

    def build(self) -> Logger:
        return MultiLogger(loggers=[child.build() for child in self.loggers])


LoggerConfig = Annotated[
    Union[
        ConsoleLoggerConfig,
        WandbLoggerConfig,
        CometLoggerConfig,
        AimLoggerConfig,
        MultiLoggerConfig,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of all supported logger configurations."""


def build_logger(config: LoggerConfig | None) -> Logger | None:
    """Build a logger from a config object, or return None if config is None.

    Parameters
    ----------
    config : LoggerConfig or None
        Logger configuration object or None.

    Returns
    -------
    logger : Logger or None
        Built logger instance, or None if config is None.
    """
    if config is None:
        return None
    return config.build()


def _config_uses_logger_type(node: Any, logger_type: str) -> bool:
    """Return whether a raw config tree references the given logger type."""
    if isinstance(node, dict):
        if node.get("type") == logger_type:
            return True
        return any(
            _config_uses_logger_type(value, logger_type) for value in node.values()
        )

    if isinstance(node, list):
        return any(_config_uses_logger_type(value, logger_type) for value in node)

    return False


def bootstrap_logger_backend_imports(raw_cfg: Any) -> None:
    """Import logger backends early when their SDKs require it.

    Comet ML emits late-import warnings when its SDK is initialized after
    framework modules such as torch. Bootstrapping it here keeps the CLI
    entrypoint thin while ensuring Comet is imported before runtime-heavy
    modules are loaded.
    """
    if _config_uses_logger_type(raw_cfg, "CometLogger"):
        import comet_ml  # type: ignore[import-not-found]  # noqa: F401

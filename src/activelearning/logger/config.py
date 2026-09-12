"""Pydantic configuration models for live telemetry backends."""

from typing import Any, Literal

from pydantic import model_validator

from activelearning.config_registry import BuildableConfig, registered_config
from activelearning.logger.logger import (
    AimLogger,
    CometLogger,
    MultiLogger,
    ConsoleLogger,
    Logger,
    WandbLogger,
)


class ConsoleLoggerConfig(BuildableConfig):
    type: Literal["ConsoleLogger"] = "ConsoleLogger"
    project_name: str
    run_name: str | None = None

    def build(self) -> Logger:
        return ConsoleLogger(project_name=self.project_name, run_name=self.run_name)


class WandbLoggerConfig(BuildableConfig):
    type: Literal["WandbLogger"] = "WandbLogger"
    project_name: str
    run_name: str | None = None

    def build(self) -> Logger:
        return WandbLogger(project_name=self.project_name, run_name=self.run_name)


class CometLoggerConfig(BuildableConfig):
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


class AimLoggerConfig(BuildableConfig):
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


class MultiLoggerConfig(BuildableConfig):
    type: Literal["MultiLogger"] = "MultiLogger"
    loggers: list["LoggerConfig"]

    @model_validator(mode="after")
    def reject_nested_multi_loggers(self) -> "MultiLoggerConfig":
        """Preserve the non-recursive MultiLogger configuration contract."""
        if any(isinstance(logger, MultiLoggerConfig) for logger in self.loggers):
            raise ValueError("MultiLogger cannot contain another MultiLogger.")
        return self

    def build(self) -> Logger:
        return MultiLogger(loggers=[child.build() for child in self.loggers])


LOGGER_CONFIGS = (
    ConsoleLoggerConfig,
    WandbLoggerConfig,
    CometLoggerConfig,
    AimLoggerConfig,
    MultiLoggerConfig,
)
LoggerConfig = registered_config("logger")

MultiLoggerConfig.model_rebuild(_types_namespace={"LoggerConfig": LoggerConfig})


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

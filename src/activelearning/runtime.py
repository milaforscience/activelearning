import torch
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Iterable, Literal

from activelearning.logger.logger import Logger


@dataclass
class RuntimeContext:
    """Shared runtime settings available to active-learning components.

    The optional logger is part of the runtime context so components can submit
    live telemetry during their own work. Durable ``RunWriter`` persistence is
    intentionally owned by the loop, so components cannot write partial round
    records.
    """

    logger: Logger | None = None
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    seed: int = 42


DEFAULT_RUNTIME_CONTEXT = RuntimeContext()


def resolve_torch_dtype(precision: int) -> torch.dtype:
    """Map a floating-point precision setting to a torch dtype."""
    if precision == 32:
        return torch.float32
    if precision == 64:
        return torch.float64
    raise ValueError(f"Unsupported precision {precision}. Expected 32 or 64.")


class RuntimeContextConfig(BaseModel):
    """Configuration for global torch runtime defaults and shared live telemetry."""

    device: str = "cpu"
    precision: Literal[32, 64] = 64
    seed: int = Field(default=42, ge=0)

    def build(self, logger: Logger | None = None) -> RuntimeContext:
        """Materialize runtime settings and attach the optional shared logger."""
        return RuntimeContext(
            logger=logger,
            device=torch.device(self.device),
            dtype=resolve_torch_dtype(self.precision),
            seed=self.seed,
        )


class ALRuntimeMixin:
    """Mixin exposing shared runtime state through convenience properties."""

    _runtime_context: RuntimeContext = DEFAULT_RUNTIME_CONTEXT

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind shared runtime state to this component instance."""
        self._runtime_context = runtime_context

    @property
    def runtime_context(self) -> RuntimeContext:
        """Return the currently bound runtime context."""
        return getattr(self, "_runtime_context", DEFAULT_RUNTIME_CONTEXT)

    @property
    def logger(self) -> Logger | None:
        """Return the bound experiment logger, if any."""
        return self.runtime_context.logger

    @property
    def device(self) -> torch.device:
        """Return the shared torch device."""
        return self.runtime_context.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the shared floating-point torch dtype."""
        return self.runtime_context.dtype


def bind_runtime_context(
    components: Iterable[object], runtime_context: RuntimeContext
) -> None:
    """Bind a runtime context to every runtime-aware component in an iterable."""
    for component in components:
        if isinstance(component, ALRuntimeMixin):
            component.bind_runtime_context(runtime_context)

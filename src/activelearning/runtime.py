from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from activelearning.logger.logger import Logger


@dataclass
class RuntimeContext:
    """Shared runtime settings available to active learning components."""

    logger: Logger | None = None
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    precision: Literal[32, 64] = 64


DEFAULT_RUNTIME_CONTEXT = RuntimeContext()


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

    @property
    def precision(self) -> Literal[32, 64]:
        """Return the shared floating-point precision in bits."""
        return self.runtime_context.precision


def bind_runtime_context(
    components: Iterable[object], runtime_context: RuntimeContext
) -> None:
    """Bind a runtime context to every runtime-aware component in an iterable."""
    for component in components:
        if isinstance(component, ALRuntimeMixin):
            component.bind_runtime_context(runtime_context)

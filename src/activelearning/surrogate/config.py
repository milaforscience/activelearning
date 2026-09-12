"""Pydantic models of surrogate models.

Changes in the interface of existing surrogates should be reflected in this
configuration. New surrogates should define their corresponding pydantic model here and
be added to ``SurrogateConfig``.
"""

from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Protocol,
    Union,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, Field, ImportString, field_validator, model_validator
from gpytorch.module import Module

from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.surrogate.surrogate import Surrogate
from activelearning.applications.molecules.config import (
    ExactSelfiesDKLSurrogateConfig,
    VariationalSelfiesDKLSurrogateConfig,
)


@runtime_checkable
class FidelityAwareSurrogateConfig(Protocol):
    """Configuration contract for surrogates that support multiple fidelities."""

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> BaseModel:
        """Return this config with oracle-derived fidelity settings resolved."""


class DummyMeanSurrogateConfig(BaseModel):
    type: Literal["DummyMeanSurrogate"] = "DummyMeanSurrogate"

    def build(self) -> Surrogate:
        return DummyMeanSurrogate()


class BoTorchGPSurrogateConfig(BaseModel):
    type: Literal["BoTorchGPSurrogate"] = "BoTorchGPSurrogate"
    scale_inputs: bool = True
    standardize_outputs: bool = True
    optimize_hyperparameters: bool = True
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)
    custom_fit_function: ImportString | None = None
    covar_module: ImportString | None = None
    covar_module_kwargs: dict[str, Any] = Field(default_factory=dict)
    use_partial_updates: bool = False
    # Derived from the oracle's fidelity set by the top-level ``ActiveLearningConfig``
    # validator when using the config system.  Can also be set directly when
    # constructing the surrogate outside of the config system.
    is_multi_fidelity: bool = False

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> "BoTorchGPSurrogateConfig":
        """Derive multi-fidelity mode from oracle confidence metadata.

        Parameters
        ----------
        confidences : dict[int, float]
            Oracle fidelity confidence mapping.

        Returns
        -------
        BoTorchGPSurrogateConfig
            Revalidated config with the derived ``is_multi_fidelity`` value.
        """
        data = self.model_dump()
        data["is_multi_fidelity"] = len(confidences) > 1
        return type(self).model_validate(data)

    @field_validator("custom_fit_function")
    @classmethod
    def validate_custom_fit_function(cls, value: Any) -> Any:
        """Ensure custom fit function imports resolve to a callable.

        Parameters
        ----------
        value : Any
            The imported object (or ``None``) to validate.

        Returns
        -------
        Any
            The original value, unchanged.

        Raises
        ------
        TypeError
            If ``value`` is not ``None`` and not callable.
        """
        if value is not None and not callable(value):
            raise TypeError("custom_fit_function must resolve to a callable.")
        return value

    @field_validator("covar_module")
    @classmethod
    def validate_covar_module(cls, value: Any) -> Any:
        """Ensure covar module imports resolve to a module or a constructor.

        Parameters
        ----------
        value : Any
            The imported object (or ``None``) to validate.

        Returns
        -------
        Any
            The original value, unchanged.

        Raises
        ------
        TypeError
            If ``value`` is not ``None``, not a
            :class:`gpytorch.module.Module`, and not callable.
        """
        if value is None:
            return value
        if isinstance(value, Module) or callable(value):
            return value
        raise TypeError(
            "covar_module must resolve to a gpytorch.module.Module or a callable "
            "that builds one."
        )

    @model_validator(mode="after")
    def validate_covar_module_kwargs(self) -> "BoTorchGPSurrogateConfig":
        """Reject constructor kwargs when no covar module target is configured.

        Returns
        -------
        BoTorchGPSurrogateConfig
            The validated config instance.

        Raises
        ------
        ValueError
            If ``covar_module_kwargs`` is non-empty but ``covar_module`` is
            ``None``.
        """
        if self.covar_module is None and self.covar_module_kwargs:
            raise ValueError("covar_module_kwargs requires covar_module.")
        return self

    def build(self) -> Surrogate:
        """Build and return a :class:`~activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate` instance.

        Returns
        -------
        Surrogate
            A configured :class:`~activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate`.

        Raises
        ------
        TypeError
            If ``covar_module_kwargs`` are provided for an already-instantiated
            :class:`gpytorch.module.Module`, or if the resolved ``covar_module``
            target does not produce a :class:`gpytorch.module.Module`.
        """
        covar_module = None

        if self.covar_module is not None:
            resolved_covar_module = self.covar_module
            if isinstance(resolved_covar_module, Module):
                if self.covar_module_kwargs:
                    raise TypeError(
                        "covar_module_kwargs cannot be provided when covar_module "
                        "resolves to an already-instantiated Module."
                    )
                covar_module = resolved_covar_module
            else:
                # noinspection PyUnnecessaryCast
                covar_module_candidate = cast(
                    Callable[..., Any], resolved_covar_module
                )(**self.covar_module_kwargs)
                if not isinstance(covar_module_candidate, Module):
                    raise TypeError(
                        "Resolved covar_module target produced "
                        f"{type(covar_module_candidate).__name__}, expected a "
                        "gpytorch.module.Module."
                    )
                covar_module = covar_module_candidate

        return BoTorchGPSurrogate(
            scale_inputs=self.scale_inputs,
            standardize_outputs=self.standardize_outputs,
            optimize_hyperparameters=self.optimize_hyperparameters,
            fit_kwargs=self.fit_kwargs,
            custom_fit_function=self.custom_fit_function,
            covar_module=covar_module,
            use_partial_updates=self.use_partial_updates,
            is_multi_fidelity=self.is_multi_fidelity,
        )


SurrogateConfig = Annotated[
    Union[
        DummyMeanSurrogateConfig,
        BoTorchGPSurrogateConfig,
        ExactSelfiesDKLSurrogateConfig,
        VariationalSelfiesDKLSurrogateConfig,
    ],
    Field(discriminator="type"),
]

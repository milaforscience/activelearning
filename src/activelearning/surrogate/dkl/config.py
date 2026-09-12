"""Configuration for representation-independent Deep Kernel Learning surrogates."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr

from activelearning.config_registry import BuildableConfig
from activelearning.surrogate.encoder_config import EncoderConfig
from activelearning.surrogate.surrogate import Surrogate


class DKLTrainingConfig(BaseModel):
    """Hyper-parameters for the DKL Adam training loop.

    Parameters
    ----------
    epochs : int
        Number of epochs for the GP training phase.
    lr : float
        Adam learning rate.
    mask_ratio : float
        Fraction of valid tokens masked per sequence when the encoder exposes
        an ``mlm_loss`` method.
    pretrain_epochs : int
        Number of auxiliary MLM warm-up epochs before GP training.
    """

    epochs: int = 50
    lr: float = 1e-3
    mask_ratio: float = 0.125
    pretrain_epochs: int = 0


class DKLSurrogateConfigBase(BuildableConfig):
    """Shared configuration fields for exact and variational DKL models."""

    is_botorch_compatible: ClassVar[bool] = True
    encoder: EncoderConfig
    training_params: DKLTrainingConfig = Field(default_factory=DKLTrainingConfig)
    target_fidelity: Optional[int] = None
    standardize_outputs: bool = True
    _is_multi_fidelity: bool = PrivateAttr(default=False)

    @property
    def is_multi_fidelity(self) -> bool:
        """Return the fidelity mode derived from the configured oracle."""
        return self._is_multi_fidelity

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> "DKLSurrogateConfigBase":
        """Resolve fidelity mode and target from oracle confidence metadata.

        Parameters
        ----------
        confidences : dict[int, float]
            Mapping from declared fidelity levels to their confidence values.

        Returns
        -------
        DKLSurrogateConfigBase
            Copy of this configuration with multi-fidelity mode and target
            fidelity resolved from ``confidences``.

        Raises
        ------
        ValueError
            If the configured target fidelity is not declared by the oracle.
        """
        is_multi_fidelity = len(confidences) > 1
        target_fidelity = self.target_fidelity

        if not is_multi_fidelity:
            target_fidelity = None
        elif target_fidelity is None:
            target_fidelity = max(confidences, key=confidences.__getitem__)
        elif target_fidelity not in confidences:
            raise ValueError(
                f"Surrogate target_fidelity {target_fidelity} is not declared by "
                f"the oracle. Oracle declares: {sorted(confidences)}."
            )

        resolved = self.model_copy(update={"target_fidelity": target_fidelity})
        resolved._is_multi_fidelity = is_multi_fidelity
        return resolved

    def build(self) -> Surrogate:
        """Build the configured DKL surrogate.

        Returns
        -------
        Surrogate
            Instantiated exact or variational DKL surrogate.
        """
        encoder = self.encoder.build()
        surrogate_class = self._surrogate_class()
        return surrogate_class(
            encoder=encoder,
            training_params=self.training_params,
            is_multi_fidelity=self.is_multi_fidelity,
            target_fidelity=self.target_fidelity,
            standardize_outputs=self.standardize_outputs,
            **self._additional_build_kwargs(),
        )

    def _surrogate_class(self) -> type[Surrogate]:
        """Return the concrete surrogate class constructed by this config."""
        raise NotImplementedError

    def _additional_build_kwargs(self) -> dict[str, Any]:
        """Return variant-specific constructor arguments."""
        return {}


class ExactDKLSurrogateConfig(DKLSurrogateConfigBase):
    """Configuration for an exact DKL surrogate.

    The inherited ``encoder`` and ``training_params`` settings define the
    feature representation and joint optimization schedule.
    """

    type: Literal["ExactDKLSurrogate"] = "ExactDKLSurrogate"

    def _surrogate_class(self) -> type[Surrogate]:
        """Return the exact DKL surrogate class."""
        from activelearning.surrogate.dkl.exact import ExactDKLSurrogate

        return ExactDKLSurrogate


class VariationalDKLSurrogateConfig(DKLSurrogateConfigBase):
    """Configuration for a sparse variational DKL surrogate.

    The inherited ``encoder`` and ``training_params`` settings define the
    feature representation and joint optimization schedule.

    Parameters
    ----------
    num_inducing : int, default=64
        Number of inducing points in the variational GP head.
    """

    type: Literal["VariationalDKLSurrogate"] = "VariationalDKLSurrogate"
    num_inducing: int = 64

    def _surrogate_class(self) -> type[Surrogate]:
        """Return the variational DKL surrogate class."""
        from activelearning.surrogate.dkl.variational import VariationalDKLSurrogate

        return VariationalDKLSurrogate

    def _additional_build_kwargs(self) -> dict[str, Any]:
        """Return the variational GP constructor arguments."""
        return {"num_inducing": self.num_inducing}

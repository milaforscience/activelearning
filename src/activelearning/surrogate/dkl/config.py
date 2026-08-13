"""Configuration shared by Deep Kernel Learning surrogates."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


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


class DKLSurrogateConfigBase(BaseModel):
    """Shared configuration fields for exact and variational DKL models."""

    training_params: DKLTrainingConfig = Field(default_factory=DKLTrainingConfig)
    is_multi_fidelity: bool = False
    target_fidelity: Optional[int] = None
    standardize_outputs: bool = True

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

        data = self.model_dump()
        data.update(
            {
                "is_multi_fidelity": is_multi_fidelity,
                "target_fidelity": target_fidelity,
            }
        )
        return type(self).model_validate(data)

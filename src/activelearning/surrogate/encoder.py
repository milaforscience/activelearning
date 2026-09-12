"""Shared encoder contracts for Deep Kernel Learning surrogates.

Deep Kernel Learning uses one encoder object for two distinct stages. First,
``prepare_inputs()`` converts raw domain values into fixed model-space tensors.
Second, the same encoder module is re-run inside the GP or kernel during each
forward pass so GPyTorch can backpropagate through the encoder parameters.
Keeping both stages on one module makes the preprocessing contract explicit
without breaking the gradient path required for joint encoder-GP training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn


class LatentEncoder(nn.Module, ABC):
    """Abstract base class for encoders used by DKL surrogates.

    Subclasses must expose a ``latent_dim`` attribute and implement
    :meth:`forward` to map prepared model inputs to latent feature vectors.
    They may override :meth:`prepare_inputs` when raw domain values require
    tokenizer- or modality-specific preprocessing.

    Attributes
    ----------
    latent_dim : int
        Width of the latent feature vectors returned by :meth:`forward`.
    """

    latent_dim: int

    def prepare_inputs(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Convert raw domain values into a model-space tensor.

        Parameters
        ----------
        values : Sequence[Any]
            Raw values taken from observations or candidates.
        device : torch.device
            Device on which the returned tensor should be allocated.

        Returns
        -------
        Tensor
            Batched model-space tensor accepted by :meth:`forward`.
        """
        return torch.as_tensor(values, device=device)

    @abstractmethod
    def forward(self, model_inputs: Tensor) -> Tensor:
        """Encode prepared model inputs into latent feature vectors.

        Parameters
        ----------
        model_inputs : Tensor
            Batched tensor produced by :meth:`prepare_inputs`.

        Returns
        -------
        Tensor
            Latent feature tensor with final dimension ``latent_dim``.
        """

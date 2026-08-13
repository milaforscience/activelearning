"""Convert molecular strings into tensors for sequence-based surrogates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer

__all__ = ["MoleculeStringInputAdapter"]


class MoleculeStringInputAdapter:
    """Adapt molecular string inputs to the tensor format used by an encoder.

    Parameters
    ----------
    tokenizer : SequenceTokenizer
        Tokenizer that converts molecular strings into padded token-ID tensors.
    max_tokens : int
        Maximum number of molecule tokens retained for each input. Special
        tokens added by the tokenizer are not included in this limit.
    """

    def __init__(self, tokenizer: SequenceTokenizer, max_tokens: int) -> None:
        """Initialize an adapter for a molecule sequence tokenizer.

        Parameters
        ----------
        tokenizer : SequenceTokenizer
            Tokenizer used to encode each molecular string.
        max_tokens : int
            Maximum number of molecule tokens retained for each input.
        """
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    def __call__(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Tokenize a batch of molecular strings on the requested device.

        Parameters
        ----------
        values : Sequence[Any]
            Molecular strings to tokenize. Every value must be a ``str``.
        device : torch.device
            Device on which the returned token-ID tensor should be allocated.

        Returns
        -------
        torch.Tensor
            Padded token IDs with one row per molecular string.

        Raises
        ------
        ValueError
            If any input is not a string.
        """
        string_values: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    "Molecule sequence surrogates require string inputs, got "
                    f"{type(value).__name__}."
                )
            string_values.append(value)
        return self._tokenizer.batch_from_strings(
            string_values,
            max_tokens=self._max_tokens,
            device=device,
        )

"""Base classes shared by token-sequence encoders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from activelearning.surrogate.encoder import LatentEncoder
from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer


class SequenceEncoder(LatentEncoder):
    """Base class for encoders backed by a sequence tokenizer.

    Parameters
    ----------
    tokenizer : SequenceTokenizer
        Tokenizer used to convert raw strings into token-ID tensors.
    max_tokens : int
        Total number of token positions consumed by the encoder, including
        special tokens and padding.
    """

    tokenizer: SequenceTokenizer
    max_tokens: int
    max_seq_len: int

    def __init__(self, tokenizer: SequenceTokenizer, max_tokens: int) -> None:
        """Initialize a tokenizer-backed sequence encoder.

        Parameters
        ----------
        tokenizer : SequenceTokenizer
            Tokenizer used to prepare raw string inputs.
        max_tokens : int
            Total number of token positions consumed by the encoder,
            including special tokens and padding.
        """
        super().__init__()
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least two.")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_seq_len = max_tokens

    def prepare_inputs(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Tokenize a batch of raw string inputs.

        Parameters
        ----------
        values : Sequence[Any]
            Raw input values. Every item must be a string.
        device : torch.device
            Device on which the token batch should be allocated.

        Returns
        -------
        Tensor
            Padded token-ID tensor accepted by sequence encoders.

        Raises
        ------
        ValueError
            If any input value is not a string.
        """
        string_values: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    "String sequence surrogates require string inputs, got "
                    f"{type(value).__name__}."
                )
            string_values.append(value)
        return self.tokenizer.batch_from_strings(
            string_values,
            max_tokens=self.max_tokens,
            device=device,
        )

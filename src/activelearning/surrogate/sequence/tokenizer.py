"""Shared tokenizer protocol for encoded sequence inputs."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class SequenceTokenizer(Protocol):
    """Common interface for tokenizers used by token-sequence encoders.

    A conforming tokenizer converts each input string into integer token IDs.
    It exposes the vocabulary size and the IDs of the special tokens required
    by the encoder. All token IDs, including special-token IDs, must be in
    ``[0, vocab_size)``.

    Attributes
    ----------
    vocab_size : int
        Total number of token IDs in the vocabulary, including special tokens.
    padding_idx : int
        ID of the padding token. This token fills unused positions when
        sequences in the same batch have different lengths, and the encoder
        ignores those positions.
    eos_idx : int
        ID of the end-of-sequence token, which marks where the sequence ends.
    cls_idx : int
        ID of the start-of-sequence token, which is placed before sequence
        tokens when required by the encoder.
    mask_idx : int
        ID of the mask token, which replaces selected tokens during masked
        token training.
    """

    vocab_size: int
    padding_idx: int
    eos_idx: int
    cls_idx: int
    mask_idx: int

    def batch_from_strings(
        self,
        strings: Sequence[str],
        max_tokens: int,
        device: torch.device | None = None,
    ) -> Tensor:
        """Convert strings into a padded batch of token IDs.

        The returned tensor is rectangular: row ``i`` encodes ``strings[i]``,
        rows retain the input order, and unused positions contain
        :attr:`padding_idx`. The tokenizer adds the start- and end-of-sequence
        tokens required by its encoder and truncates inputs that exceed the
        configured limit. Whether special-token positions count toward
        ``max_tokens`` is defined by the concrete tokenizer.

        Parameters
        ----------
        strings : Sequence[str]
            Strings to convert into token IDs.
        max_tokens : int
            Maximum number of token positions to encode. The concrete
            tokenizer determines whether positions for special tokens are
            included in this limit.
        device : torch.device, optional
            Device for the returned tensor. If ``None``, the tokenizer's
            default device is used.

        Returns
        -------
        Tensor
            Two-dimensional tensor of dtype ``torch.long`` with shape
            ``(len(strings), sequence_length)``. It contains token IDs,
            special tokens, and padding in the format expected by the
            sequence encoder. ``sequence_length`` is fixed for the batch and may
            include positions reserved for special tokens.
        """
        ...

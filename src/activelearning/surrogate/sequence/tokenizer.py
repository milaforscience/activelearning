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
    mask_idx : int or None
        ID of the mask token, which replaces selected tokens during masked
        token training. ``None`` means masked-token training is unavailable.
    """

    vocab_size: int
    padding_idx: int
    eos_idx: int
    cls_idx: int
    mask_idx: int | None

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
        configured limit. ``max_tokens`` is the total output length, including
        special-token and padding positions.

        Parameters
        ----------
        strings : Sequence[str]
            Strings to convert into token IDs.
        max_tokens : int
            Total number of token positions to encode, including special
            tokens and padding.
        device : torch.device, optional
            Device for the returned tensor. If ``None``, the tokenizer's
            default device is used.

        Returns
        -------
        Tensor
            Two-dimensional tensor of dtype ``torch.long`` with shape
            ``(len(strings), max_tokens)``. It contains token IDs, special
            tokens, and padding in the format expected by the sequence
            encoder.
        """
        ...

    def attention_mask_from_batch(self, token_batch: Tensor) -> Tensor:
        """Return a binary attention mask aligned with a token-ID batch.

        Parameters
        ----------
        token_batch : Tensor
            Two-dimensional token-ID tensor.

        Returns
        -------
        Tensor
            A ``torch.long`` tensor with the same first two dimensions as
            ``token_batch``. Non-padding positions contain one.
        """
        ...

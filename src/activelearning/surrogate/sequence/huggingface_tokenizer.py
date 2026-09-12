"""Hugging Face-backed tokenizer adapter for sequence inputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer
from activelearning.utils.optional import missing_optional_dependency_error

__all__ = ["HuggingFaceTokenizer"]


def _load_tokenizer() -> Any:
    """Load the Hugging Face tokenizer class on demand."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise missing_optional_dependency_error(
            component="Hugging Face tokenization",
            extra="molecules",
            error=error,
        ) from error
    return AutoTokenizer


class HuggingFaceTokenizer(SequenceTokenizer):
    """Adapt a Hugging Face tokenizer to the sequence tokenizer contract.

    This class does not define a vocabulary or implement tokenization rules.
    It delegates tokenization, special-token handling, truncation, and padding
    to a supplied or pretrained Hugging Face tokenizer, then exposes the
    resulting token IDs and special-token IDs required by sequence encoders.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str | None = None,
        *,
        tokenizer: Any | None = None,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        cache_size: int = 4096,
    ) -> None:
        """Load or wrap the Hugging Face tokenizer used for tokenization.

        If ``tokenizer`` is provided, it is used directly and
        ``tokenizer_name_or_path`` is ignored. Otherwise, a tokenizer is
        loaded with :func:`transformers.AutoTokenizer.from_pretrained`.
        This adapter does not tokenize input strings itself; all tokenization
        and special-token insertion are delegated to the wrapped tokenizer.
        The tokenizer must define a padding token, an end-of-sequence token,
        and either a CLS or BOS token. For the mask token, the adapter uses
        the tokenizer's mask token or unknown token when available. If neither
        exists, masked-token training is unavailable.

        Parameters
        ----------
        tokenizer_name_or_path : str, optional
            Hugging Face model identifier or local path from which to load a
            tokenizer when ``tokenizer`` is not supplied.
        tokenizer : object, optional
            Already-initialized Hugging Face-compatible tokenizer. It must
            expose the required special-token IDs and be callable with the
            keyword arguments used by :meth:`batch_from_strings`.
        trust_remote_code : bool, default=False
            Whether loading from Hugging Face may execute code supplied by the
            model repository.
        cache_dir : str, optional
            Directory in which Hugging Face stores or looks up cached files.
        cache_size : int, default=4096
            Maximum number of tokenizer attention masks retained in the LRU
            cache. Zero disables caching.

        Raises
        ------
        ValueError
            If neither ``tokenizer_name_or_path`` nor ``tokenizer`` is
            provided, or if the tokenizer is missing a required token ID.
        ImportError
            If the optional ``transformers`` dependency is not installed.
        """
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")
        if tokenizer is None and tokenizer_name_or_path is None:
            raise ValueError(
                "Provide tokenizer_name_or_path or an already-loaded tokenizer."
            )
        if tokenizer is None:
            auto_tokenizer = _load_tokenizer()
            tokenizer = auto_tokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
        self._tokenizer = tokenizer
        self.cache_size = cache_size
        self._attention_masks: OrderedDict[tuple[int, ...], tuple[int, ...]] = (
            OrderedDict()
        )
        self.padding_idx = self._require_token_id("pad_token_id", "padding")
        self.eos_idx = self._require_token_id("eos_token_id", "EOS")
        self.cls_idx = self._first_token_id(
            ("cls_token_id", "bos_token_id"),
            "CLS/BOS",
        )
        self.mask_idx = self._first_token_id(
            ("mask_token_id", "unk_token_id"),
            "MASK",
            required=False,
        )
        if self.mask_idx == self.padding_idx:
            self.mask_idx = None

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size reported by the wrapped tokenizer.

        The wrapped tokenizer's ``vocab_size`` attribute is preferred. If it
        does not define that attribute, its length is used instead.

        Returns
        -------
        int
            Number of token IDs in the tokenizer vocabulary.
        """
        size = getattr(self._tokenizer, "vocab_size", None)
        return int(len(self._tokenizer) if size is None else size)

    def _require_token_id(self, attribute: str, name: str) -> int:
        """Return a required special-token id."""
        token_id = getattr(self._tokenizer, attribute, None)
        if token_id is None:
            raise ValueError(f"Hugging Face tokenizer must define a {name} token.")
        return int(token_id)

    def _first_token_id(
        self,
        attributes: Sequence[str],
        name: str,
        *,
        required: bool = True,
    ) -> int | None:
        """Return the first available id from a list of tokenizer attributes."""
        for attribute in attributes:
            token_id = getattr(self._tokenizer, attribute, None)
            if token_id is not None:
                return int(token_id)
        if required:
            raise ValueError(f"Hugging Face tokenizer must define a {name} token.")
        return None

    def batch_from_strings(
        self,
        strings: Sequence[str],
        max_tokens: int,
        device: torch.device | None = None,
    ) -> Tensor:
        """Delegate tokenization and normalize the resulting ID batch.

        The wrapped Hugging Face tokenizer is called with
        ``add_special_tokens=True``, ``truncation=True``, and
        ``padding="max_length"``. Therefore, each non-empty input produces one
        row of exactly ``max_tokens`` IDs, including special tokens and
        padding. The adapter does not define how strings are split into tokens.
        An empty input sequence returns an empty tensor with shape
        ``(0, max_tokens)``.

        Parameters
        ----------
        strings : Sequence[str]
            Strings to tokenize. The output rows preserve this order.
        max_tokens : int
            Fixed number of token positions in each output row, including
            positions used for special tokens.
        device : torch.device, optional
            Device on which to place the returned tensor. If ``None``, the
            tensor remains on the tokenizer's default device.

        Returns
        -------
        Tensor
            Two-dimensional ``torch.long`` tensor of shape
            ``(len(strings), max_tokens)``. Padding positions contain
            :attr:`padding_idx`.

        Raises
        ------
        ValueError
            If ``max_tokens`` is smaller than two, if the wrapped tokenizer
            does not return two-dimensional ``input_ids``, or if it reuses the
            EOS ID for padding without returning an attention mask.
        """
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least two.")
        if not strings:
            return torch.empty(
                (0, max_tokens),
                dtype=torch.long,
                device=device,
            )

        encoded = self._tokenizer(
            list(strings),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2:
            raise ValueError(
                "The Hugging Face tokenizer must return two-dimensional input_ids."
            )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            if self.padding_idx == self.eos_idx:
                raise ValueError(
                    "The Hugging Face tokenizer must return attention_mask when "
                    "padding and EOS share an ID."
                )
            attention_mask = input_ids.ne(self.padding_idx).long()
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "The Hugging Face tokenizer must return an attention_mask with "
                "the same shape as input_ids."
            )
        if self.cache_size:
            for row, mask in zip(input_ids, attention_mask):
                key = tuple(row.tolist())
                self._attention_masks[key] = tuple(mask.tolist())
                self._attention_masks.move_to_end(key)
                while len(self._attention_masks) > self.cache_size:
                    self._attention_masks.popitem(last=False)
        return input_ids.to(device=device, dtype=torch.long)

    def attention_mask_from_batch(self, token_batch: Tensor) -> Tensor:
        """Return the tokenizer-produced attention mask for token IDs.

        Batches returned by :meth:`batch_from_strings` retain the exact mask
        produced by Hugging Face, including checkpoints whose padding and EOS
        IDs are equal. Unknown batches fall back to padding-ID detection.

        Parameters
        ----------
        token_batch : Tensor
            Two-dimensional token-ID tensor whose rows should be masked.

        Returns
        -------
        Tensor
            ``torch.long`` attention mask with the same shape and device as
            ``token_batch``.

        Raises
        ------
        ValueError
            If ``token_batch`` is not two-dimensional, or if a batch with
            shared padding/EOS IDs has no tokenizer-produced mask.
        """
        if token_batch.ndim != 2:
            raise ValueError(
                "token_batch must be 2-D (B, seq_len), got shape "
                f"{tuple(token_batch.shape)}"
            )
        if token_batch.shape[0] == 0:
            return torch.empty_like(token_batch, dtype=torch.long)
        rows: list[tuple[int, ...]] = []
        for row in token_batch.detach().to(device="cpu", dtype=torch.long):
            key = tuple(row.tolist())
            mask = self._attention_masks.get(key)
            if mask is not None:
                self._attention_masks.move_to_end(key)
            if mask is None:
                if self.padding_idx == self.eos_idx:
                    raise ValueError(
                        "An attention mask is required for token batches whose "
                        "padding and EOS IDs are equal."
                    )
                mask = tuple(row.ne(self.padding_idx).long().tolist())
            rows.append(mask)
        return torch.tensor(rows, dtype=torch.long, device=token_batch.device)

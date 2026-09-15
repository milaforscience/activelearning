from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from activelearning.applications.molecules._optional import (
    missing_molecules_dependency_error,
)
from activelearning.applications.molecules.constants import SELFIES_VOCAB_SMALL

try:
    import selfies as sf
except ImportError as error:  # pragma: no cover - exercised via subprocess test
    raise missing_molecules_dependency_error("SelfiesTokenizer", error) from error


class SelfiesTokenizer:
    """Minimal tokenizer that converts SELFIES strings to padded token-ID tensors.

    There are two layers of token handling:

    1. The base vocabulary encodes raw SELFIES alphabet tokens.
       ``encode_selfies`` converts a SELFIES string to token IDs using this
       base vocabulary.

    2. ``transform_batch`` augments the raw IDs by placing ``[EOS]`` at the
       first padding position and prepending ``[CLS]``, producing the final
       encoder input format.

    Parameters
    ----------
    selfies_vocab : Sequence[str]
        Ordered SELFIES alphabet. Defaults to :data:`SELFIES_VOCAB_SMALL`.
    """

    def __init__(self, selfies_vocab: Sequence[str] = SELFIES_VOCAB_SMALL) -> None:
        # Base vocabulary used by sf.selfies_to_encoding
        self.base_vocab = list(selfies_vocab) + ["[nop]", "[EOS]"]
        self.base_lookup: dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.base_vocab)
        }

        # Full vocabulary: base + any special tokens not already present
        special_tokens = ["[EOS]", "[CLS]", "[MASK]", "[nop]"]
        extra = [tok for tok in special_tokens if tok not in self.base_lookup]
        self.full_vocab: list[str] = self.base_vocab + extra
        self.lookup: dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.full_vocab)
        }
        self.inverse_lookup: dict[int, str] = {
            idx: tok for tok, idx in self.lookup.items()
        }

        self.padding_token = "[nop]"
        self.eos_token = "[EOS]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"

        self.padding_idx: int = self.lookup[self.padding_token]
        self.eos_idx: int = self.lookup[self.eos_token]
        self.cls_idx: int = self.lookup[self.cls_token]
        self.mask_idx: int = self.lookup[self.mask_token]

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.full_vocab)

    def encode_selfies(self, selfies_string: str, max_mol_tokens: int) -> Tensor:
        """Convert a SELFIES string into a fixed-length tensor of base token IDs.

        Parameters
        ----------
        selfies_string : str
            A valid SELFIES string, e.g. ``"[C][=C][C]"``.
        max_mol_tokens : int
            Pad or truncate to this many molecular tokens.  Does *not* include
            the ``[CLS]`` / ``[EOS]`` special tokens added later by
            :meth:`transform_batch`.

        Returns
        -------
        Tensor
            Shape ``(max_mol_tokens,)`` of dtype ``torch.long``.
        """
        raw_ids = sf.selfies_to_encoding(
            selfies=selfies_string,
            vocab_stoi=self.base_lookup,
            pad_to_len=max_mol_tokens,
            enc_type="label",
        )
        return torch.tensor(raw_ids[:max_mol_tokens], dtype=torch.long)

    def transform_batch(self, raw_batch: Tensor) -> Tensor:
        """Augment a batch of raw padded sequences to encoder input format.

        Steps:
        1. Insert ``[EOS]`` at the first padding position (or at the end if
           the sequence is fully occupied).
        2. Prepend ``[CLS]`` to each sequence.

        Parameters
        ----------
        raw_batch : Tensor
            Shape ``(B, max_mol_tokens)`` of dtype ``torch.long``.

        Returns
        -------
        Tensor
            Shape ``(B, max_mol_tokens + 2)`` of dtype ``torch.long``.

        Raises
        ------
        ValueError
            If ``raw_batch`` is not 2-D.
        """
        if raw_batch.ndim != 2:
            raise ValueError(
                f"raw_batch must be 2-D (B, seq_len), got shape {tuple(raw_batch.shape)}"
            )

        batch = raw_batch.clone().long()
        device = batch.device

        # Locate the first padding position per sequence to insert [EOS]
        eos_positions: list[int] = []
        for seq in batch:
            pad_positions = torch.where(seq == self.padding_idx)[0]
            eos_positions.append(
                int(pad_positions[0]) if len(pad_positions) > 0 else seq.size(0)
            )

        # Append one extra padding column so [EOS] always fits without overflow
        pad_col = torch.full(
            (batch.size(0), 1), self.padding_idx, dtype=torch.long, device=device
        )
        batch = torch.cat([batch, pad_col], dim=1)

        # Scatter [EOS] into the correct positions
        eos_col = torch.full(
            (batch.size(0), 1), self.eos_idx, dtype=torch.long, device=device
        )
        eos_idx_t = torch.tensor(eos_positions, dtype=torch.long, device=device)
        batch = batch.scatter(1, eos_idx_t.unsqueeze(1), eos_col)

        # Prepend [CLS]
        cls_col = torch.full(
            (batch.size(0), 1), self.cls_idx, dtype=torch.long, device=device
        )
        return torch.cat([cls_col, batch], dim=1)

    def batch_from_selfies(
        self,
        selfies_list: Sequence[str],
        max_mol_tokens: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Tokenize and transform a list of SELFIES strings into a batched tensor.

        Parameters
        ----------
        selfies_list : Sequence[str]
            SELFIES strings to tokenize.
        max_mol_tokens : int
            Maximum number of molecular tokens per sequence, not counting the
            ``[CLS]`` and ``[EOS]`` special tokens added by
            :meth:`transform_batch`.
        device : torch.device, optional
            Target device for the output tensor.

        Returns
        -------
        Tensor
            Shape ``(len(selfies_list), max_mol_tokens + 2)`` of dtype ``torch.long``.
        """
        raw = torch.stack(
            [self.encode_selfies(s, max_mol_tokens) for s in selfies_list], dim=0
        )
        batch = self.transform_batch(raw)
        if device is not None:
            batch = batch.to(device)
        return batch

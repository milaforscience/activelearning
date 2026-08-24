"""Transformer encoders for tokenized sequence representations.

Pipeline:
    token IDs → nn.Embedding → positional encoding → Transformer encoder
    → masked-mean pool → latent sequence vector

The encoder also exposes an MLM head so it can be trained jointly with:
- masked language modelling (MLM) loss on masked sequence tokens
- GP regression loss (exact MLL or variational ELBO)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from activelearning.surrogate.sequence.base import SequenceEncoder
from activelearning.surrogate.sequence.pooling import masked_mean
from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer

__all__ = [
    "MaskedMeanPool",
    "PositionalEncoding",
    "TransformerSequenceEncoder",
]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings.

    Parameters
    ----------
    embed_dim : int
        Embedding dimensionality.
    max_len : int
        Maximum sequence length the encoder will ever see.
    dropout : float
        Dropout rate applied after adding positional encodings.
    """

    def __init__(self, embed_dim: int, max_len: int, dropout: float = 0.0) -> None:
        """Initialize sinusoidal encodings for a fixed maximum sequence length.

        Parameters
        ----------
        embed_dim : int
            Embedding dimensionality.
        max_len : int
            Maximum sequence length represented by the encoding table.
        dropout : float, default=0.0
            Dropout rate applied after adding positional encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positional_dtype = torch.get_default_dtype()

        position = torch.arange(max_len, dtype=positional_dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=positional_dtype)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, 1, embed_dim, dtype=positional_dtype)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Odd embed_dim leaves one more sine slot than cosine slot.
        pe[:, 0, 1::2] = torch.cos(position * div_term[: pe[:, 0, 1::2].shape[1]])
        # Store as (1, max_len, embed_dim) for easy broadcasting
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to token embeddings.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, seq_len, embed_dim)``.

        Returns
        -------
        Tensor
            Same shape as ``x`` with positional information injected.
        """
        return self.dropout(x + self.pe[:, : x.size(1)])


class MaskedMeanPool(nn.Module):
    """Pool token features into one vector per sequence via masked mean + projection.

    Parameters
    ----------
    input_dim : int
        Dimensionality of per-token features.
    output_dim : int
        Dimensionality of the output (latent) vector.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the projection applied after masked mean pooling.

        Parameters
        ----------
        input_dim : int
            Dimensionality of each token feature vector.
        output_dim : int
            Dimensionality of the projected pooled vector.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, token_features: Tensor, mask: Tensor) -> Tensor:
        """Pool token features into one vector per sequence.

        Parameters
        ----------
        token_features : Tensor
            Shape ``(B, seq_len, input_dim)``.
        mask : Tensor
            Boolean tensor of shape ``(B, seq_len)``.  ``True`` where tokens
            should contribute to the pooled vector.

        Returns
        -------
        Tensor
            Shape ``(B, output_dim)``.
        """
        pooled = masked_mean(token_features, mask)
        return self.proj(pooled)


class TransformerSequenceEncoder(SequenceEncoder):
    """Sequence encoder used by the DKL surrogate.

    Architecture:
        token IDs → ``nn.Embedding`` → ``PositionalEncoding``
        → ``TransformerEncoder`` → ``embed_to_latent`` linear
        → ``MaskedMeanPool`` → latent vector

    The encoder also exposes an MLM head (``mlm_loss``) so it can be trained
    jointly with a GP loss inside a DKL surrogate.

    Parameters
    ----------
    tokenizer : SequenceTokenizer
        Tokenizer providing vocabulary and special-token indices.
    max_tokens : int
        Total number of sequence positions, including ``[CLS]`` and ``[EOS]``
        specials. Stored as ``max_seq_len = max_tokens``.
    embed_dim : int
        Embedding and Transformer hidden dimensionality.
    ff_dim : int
        Feedforward hidden size inside each Transformer layer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    latent_dim : int
        Output dimensionality of the pooled sequence vector.
    dropout : float
        Dropout rate applied throughout.
    """

    def __init__(
        self,
        tokenizer: SequenceTokenizer,
        max_tokens: int = 64,
        embed_dim: int = 64,
        ff_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        latent_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a tokenizer-backed Transformer sequence encoder.

        Parameters
        ----------
        tokenizer : SequenceTokenizer
            Tokenizer providing vocabulary and special-token indices.
        max_tokens : int, default=64
            Total number of sequence positions, including special tokens and
            padding.
        embed_dim : int, default=64
            Embedding and Transformer hidden dimensionality.
        ff_dim : int, default=256
            Feedforward hidden size inside each Transformer layer.
        num_heads : int, default=8
            Number of self-attention heads.
        num_layers : int, default=8
            Number of Transformer encoder layers.
        latent_dim : int, default=64
            Width of the pooled sequence representation.
        dropout : float, default=0.0
            Dropout rate used by the Transformer and positional encoding.
        """
        super().__init__(tokenizer=tokenizer, max_tokens=max_tokens)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(
            tokenizer.vocab_size,
            embed_dim,
            padding_idx=tokenizer.padding_idx,
        )
        self.positional = PositionalEncoding(embed_dim, self.max_seq_len, dropout)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.embed_to_latent = nn.Linear(embed_dim, latent_dim)
        self.pool = MaskedMeanPool(latent_dim, latent_dim)
        self.mlm_head = nn.Linear(embed_dim, tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def encode_tokens(self, token_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Run transformer forward pass and return per-token features + keep mask.

        Parameters
        ----------
        token_batch : Tensor
            Shape ``(B, seq_len)`` of dtype ``torch.long``.

        Returns
        -------
        token_features : Tensor
            Shape ``(B, seq_len, latent_dim)``.
        keep_mask : Tensor
            Boolean tensor ``(B, seq_len)``.  ``True`` for all non-padding
            tokens (including ``[CLS]`` and ``[EOS]``) that contribute to
            the pooled vector.
        """
        token_batch = token_batch.long()
        if token_batch.size(1) > self.max_seq_len:
            token_batch = token_batch[:, : self.max_seq_len]

        x = self.embedding(token_batch) * math.sqrt(self.embed_dim)
        x = self.positional(x)
        key_padding_mask = token_batch.eq(self.tokenizer.padding_idx)
        x = self.encoder_layers(x, src_key_padding_mask=key_padding_mask)
        x = self.embed_to_latent(x)

        # Pool over all non-padding tokens (CLS, sequence tokens, and EOS).
        keep_mask = ~key_padding_mask
        return x, keep_mask

    def forward(self, token_batch: Tensor) -> Tensor:
        """Encode a batch of token sequences to latent vectors.

        Parameters
        ----------
        token_batch : Tensor
            Shape ``(B, seq_len)`` of dtype ``torch.long``.

        Returns
        -------
        Tensor
            Shape ``(B, latent_dim)``.
        """
        token_features, keep_mask = self.encode_tokens(token_batch)
        return self.pool(token_features, keep_mask)

    # ------------------------------------------------------------------
    # MLM helpers
    # ------------------------------------------------------------------

    def logits_from_tokens(self, token_batch: Tensor) -> Tensor:
        """Compute per-token vocabulary logits for MLM training.

        Parameters
        ----------
        token_batch : Tensor
            Shape ``(B, seq_len)`` of dtype ``torch.long``.  Some positions
            should already have been replaced with ``[MASK]`` tokens.

        Returns
        -------
        Tensor
            Shape ``(B, seq_len, vocab_size)``.
        """
        token_batch = token_batch.long()
        if token_batch.size(1) > self.max_seq_len:
            token_batch = token_batch[:, : self.max_seq_len]

        x = self.embedding(token_batch) * math.sqrt(self.embed_dim)
        x = self.positional(x)
        key_padding_mask = token_batch.eq(self.tokenizer.padding_idx)
        x = self.encoder_layers(x, src_key_padding_mask=key_padding_mask)
        return self.mlm_head(x)

    def sample_mask_positions(self, token_batch: Tensor, mask_ratio: float) -> Tensor:
        """Sample token positions to mask for MLM training.

        Masking is applied only to regular sequence tokens, not to ``[CLS]``,
        ``[EOS]``, or ``[nop]`` padding positions.

        Parameters
        ----------
        token_batch : Tensor
            Shape ``(B, seq_len)`` of dtype ``torch.long``.
        mask_ratio : float
            Fraction of valid tokens to mask per sequence.  Sequences where
            ``int(n_valid * mask_ratio) == 0`` are left unmasked and will
            contribute zero MLM loss.

        Returns
        -------
        Tensor
            Boolean tensor of shape ``(B, seq_len)``.
        """
        tok = self.tokenizer
        valid = (
            token_batch.ne(tok.padding_idx)
            & token_batch.ne(tok.cls_idx)
            & token_batch.ne(tok.eos_idx)
        )
        mask = torch.zeros_like(token_batch, dtype=torch.bool)
        for row in range(token_batch.size(0)):
            valid_positions = torch.where(valid[row])[0]
            if len(valid_positions) == 0:
                continue
            n_mask = int(len(valid_positions) * mask_ratio)
            if n_mask == 0:
                continue
            perm = valid_positions[
                torch.randperm(len(valid_positions), device=token_batch.device)
            ]
            mask[row, perm[:n_mask]] = True
        return mask

    def mlm_loss(self, token_batch: Tensor, mask_ratio: float = 0.125) -> Tensor:
        """Compute masked language modelling cross-entropy loss.

        Parameters
        ----------
        token_batch : Tensor
            Shape ``(B, seq_len)`` of dtype ``torch.long``.
        mask_ratio : float
            Fraction of valid tokens to randomly mask per sequence.

        Returns
        -------
        Tensor
            Scalar cross-entropy loss. Returns zero when no tokens can be
            masked (e.g. a trivially short sequence).
        """
        mask = self.sample_mask_positions(token_batch, mask_ratio)
        masked = token_batch.clone()
        if self.tokenizer.mask_idx is None or (
            self.tokenizer.mask_idx == self.tokenizer.padding_idx
        ):
            raise ValueError(
                "MLM training requires a mask token distinct from the padding token."
            )
        masked[mask] = self.tokenizer.mask_idx

        logits = self.logits_from_tokens(masked)
        labels = token_batch[mask]
        pred = logits[mask]

        if labels.numel() == 0:
            return torch.zeros((), device=token_batch.device, dtype=logits.dtype)
        return F.cross_entropy(pred, labels)

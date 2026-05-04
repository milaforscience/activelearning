"""SELFIES Transformer encoder.

Pipeline:
    token IDs → nn.Embedding → positional encoding → Transformer encoder
    → masked-mean pool → latent molecules vector

The encoder also exposes an MLM head so it can be trained jointly with:
- masked language modelling (MLM) loss on masked SELFIES tokens
- GP regression loss (exact MLL or variational ELBO)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from activelearning.applications.molecules.selfies_tokenizer import SelfiesTokenizer


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
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, 1, embed_dim)
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
    """Pool token features into one vector per molecules via masked mean + projection.

    The mask excludes padding and ``[EOS]`` so the pooled vector represents the
    full encoded sequence content, including the leading ``[CLS]`` token.

    Parameters
    ----------
    input_dim : int
        Dimensionality of per-token features.
    output_dim : int
        Dimensionality of the output (latent) vector.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, token_features: Tensor, mask: Tensor) -> Tensor:
        """Pool token features into one vector per molecules.

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
        weights = mask.unsqueeze(-1).to(token_features.dtype)
        # Weighted mean; +1e-6 avoids division by zero for fully-masked inputs
        pooled = (weights * token_features).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        return self.proj(pooled)


class SelfiesTransformerEncoder(nn.Module):
    """Sequence encoder used by the DKL surrogate.

    Architecture:
        token IDs → ``nn.Embedding`` → ``PositionalEncoding``
        → ``TransformerEncoder`` → ``embed_to_latent`` linear
        → ``MaskedMeanPool`` → latent vector

    The encoder also exposes an MLM head (``mlm_loss``) so it can be trained
    jointly with a GP loss inside :class:`SelfiesDeepKernelSurrogate`.

    Parameters
    ----------
    tokenizer : SelfiesTokenizer
        Tokenizer providing vocabulary and special-token indices.
    max_length : int
        Base sequence length (not counting ``[CLS]`` / ``[EOS]``).
        Internally stored as ``max_length + 2`` to accommodate those tokens.
    embed_dim : int
        Embedding and Transformer hidden dimensionality.
    ff_dim : int
        Feedforward hidden size inside each Transformer layer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    latent_dim : int
        Output dimensionality of the pooled molecules vector.
    dropout : float
        Dropout rate applied throughout.
    """

    def __init__(
        self,
        tokenizer: SelfiesTokenizer,
        max_length: int = 64,
        embed_dim: int = 64,
        ff_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        latent_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length + 2  # +2 for [CLS] and [EOS]
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(
            tokenizer.vocab_size,
            embed_dim,
            padding_idx=tokenizer.padding_idx,
        )
        # +1 extra position buffer for safety (long molecules close to max_length)
        self.positional = PositionalEncoding(embed_dim, self.max_length + 1, dropout)
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
            Boolean tensor ``(B, seq_len)``.  ``True`` for non-padding,
            non-EOS tokens that contribute to pooling, including ``[CLS]``.
        """
        if token_batch.size(1) > self.max_length:
            token_batch = token_batch[:, : self.max_length]

        x = self.embedding(token_batch) * math.sqrt(self.embed_dim)
        x = self.positional(x)
        key_padding_mask = token_batch.eq(self.tokenizer.padding_idx)
        x = self.encoder_layers(x, src_key_padding_mask=key_padding_mask)
        x = self.embed_to_latent(x)

        # Pool over [CLS] and molecular tokens, excluding [EOS] and padding.
        keep_mask = (~key_padding_mask) & token_batch.ne(self.tokenizer.eos_idx)
        return x, keep_mask

    def forward(self, token_batch: Tensor) -> Tensor:
        """Encode a batch of token sequences to molecules latent vectors.

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
        if token_batch.size(1) > self.max_length:
            token_batch = token_batch[:, : self.max_length]

        x = self.embedding(token_batch) * math.sqrt(self.embed_dim)
        x = self.positional(x)
        key_padding_mask = token_batch.eq(self.tokenizer.padding_idx)
        x = self.encoder_layers(x, src_key_padding_mask=key_padding_mask)
        return self.mlm_head(x)

    def sample_mask_positions(self, token_batch: Tensor, mask_ratio: float) -> Tensor:
        """Sample token positions to mask for MLM training.

        Masking is applied only to regular SELFIES tokens, not to ``[CLS]``,
        ``[EOS]``, or ``[nop]`` padding positions.

        Parameters
        ----------
        token_batch : Tensor
            Shape ``(B, seq_len)`` of dtype ``torch.long``.
        mask_ratio : float
            Fraction of valid tokens to mask per sequence.

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
            n_mask = max(1, int(len(valid_positions) * mask_ratio))
            if len(valid_positions) == 0:
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
        masked[mask] = self.tokenizer.mask_idx

        logits = self.logits_from_tokens(masked)
        labels = token_batch[mask]
        pred = logits[mask]

        if labels.numel() == 0:
            return torch.zeros((), device=token_batch.device, dtype=logits.dtype)
        return F.cross_entropy(pred, labels)

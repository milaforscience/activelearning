"""Pooling helpers for token-sequence representations."""

from __future__ import annotations

from torch import Tensor


def masked_mean(token_features: Tensor, mask: Tensor) -> Tensor:
    """Compute a masked mean over the sequence dimension.

    Fully masked rows return zero rather than producing NaNs. The mask is
    broadcast over all dimensions after the sequence dimension.

    Parameters
    ----------
    token_features : Tensor
        Features with shape ``(B, seq_len, ...)``.
    mask : Tensor
        Boolean or numeric mask with shape ``(B, seq_len)``.

    Returns
    -------
    Tensor
        Features with shape ``(B, ...)``.
    """
    weights = mask.unsqueeze(-1).to(token_features.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (token_features * weights).sum(dim=1) / denominator

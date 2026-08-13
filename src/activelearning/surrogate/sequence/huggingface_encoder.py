"""Frozen Hugging Face transformer feature encoders for token sequences."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer

__all__ = ["HuggingFaceSequenceEncoder"]


class HuggingFaceSequenceEncoder(nn.Module):
    """Encode token sequences with a frozen Hugging Face backbone.

    The backbone produces per-token hidden states. A selected sequence
    representation is passed through a trainable linear projection. The
    backbone remains frozen and in evaluation mode, so only the projection is
    trained by this module.

    Parameters
    ----------
    backbone : torch.nn.Module
        Already-loaded Hugging Face-compatible model. Its forward method must
        return per-token hidden states through ``last_hidden_state`` or as the
        first positional output.
    tokenizer : SequenceTokenizer
        Tokenizer adapter that provides the padding token ID.
    max_tokens : int, default=140
        Maximum number of token positions consumed by the encoder.
    latent_dim : int, default=64
        Size of the projected latent representation.
    pooling : {"last", "mean"}, default="mean"
        Sequence pooling strategy. ``"last"`` selects each row's last
        non-padding hidden state. ``"mean"`` computes a masked mean over all
        non-padding positions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: SequenceTokenizer,
        *,
        max_tokens: int = 140,
        latent_dim: int = 64,
        pooling: Literal["last", "mean"] = "mean",
    ) -> None:
        super().__init__()
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least two.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if pooling not in ("last", "mean"):
            raise ValueError("pooling must be either 'last' or 'mean'.")

        hidden_dim = getattr(backbone.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(backbone.config, "d_model", None)
        if hidden_dim is None:
            raise ValueError(
                "Hugging Face backbone config must define hidden_size or d_model "
                "for projection."
            )
        self.backbone_hidden_dim = int(hidden_dim)
        self.max_tokens = max_tokens
        self.max_seq_len = max_tokens
        self.latent_dim = latent_dim
        self.pooling = pooling
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.projection = nn.Linear(self.backbone_hidden_dim, latent_dim)
        floating_parameters = [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.is_floating_point()
        ]
        self._backbone_dtype = (
            floating_parameters[0].dtype
            if floating_parameters
            else torch.get_default_dtype()
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "HuggingFaceSequenceEncoder":
        """Set train/eval mode while keeping the frozen backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "HuggingFaceSequenceEncoder":
        """Move the encoder while preserving the backbone checkpoint dtype."""
        result = super().to(*args, **kwargs)
        self.restore_backbone_dtype()
        return result

    def restore_backbone_dtype(self) -> None:
        """Restore the original dtype and evaluation mode of the backbone."""
        self.backbone.to(dtype=self._backbone_dtype)
        self.backbone.eval()

    def _forward_backbone(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Any:
        """Run the backbone and return its hidden-state output."""
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def forward(self, token_batch: Tensor) -> Tensor:
        """Convert a batch of padded token IDs into projected latent features.

        ``token_batch`` may have any floating dtype because DKL model-space
        tensors are floating-point; IDs are converted to ``torch.long`` at
        this sequence-encoder boundary.
        """
        if token_batch.ndim != 2:
            raise ValueError(
                f"token_batch must be 2-D (B, seq_len), got {tuple(token_batch.shape)}"
            )
        input_ids = token_batch[:, : self.max_seq_len].long()
        attention_mask = input_ids.ne(self.tokenizer.padding_idx).long()

        with torch.no_grad():
            outputs = self._forward_backbone(input_ids, attention_mask)
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs[0]

        if self.pooling == "last":
            valid_lengths = attention_mask.sum(dim=1)
            if torch.any(valid_lengths == 0):
                raise ValueError(
                    "token_batch must contain at least one non-padding token per row."
                )
            row_indices = torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
            )
            last_indices = (valid_lengths - 1).to(hidden_states.device)
            pooled = hidden_states[row_indices, last_indices]
        else:
            weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)
        pooled = pooled.to(
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )
        return self.projection(pooled)

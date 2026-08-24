"""Frozen Hugging Face transformer feature encoders for token sequences."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal

import torch
from torch import Tensor, nn

from activelearning.surrogate.sequence.base import SequenceEncoder
from activelearning.surrogate.sequence.pooling import masked_mean
from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer

__all__ = ["HuggingFaceSequenceEncoder"]


class HuggingFaceSequenceEncoder(SequenceEncoder):
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
        Total number of token positions consumed by the encoder, including
        special tokens and padding.
    latent_dim : int, default=64
        Size of the projected latent representation.
    pooling : {"last", "mean"}, default="mean"
        Sequence pooling strategy. ``"last"`` selects each row's last
        non-padding hidden state. ``"mean"`` computes a masked mean over all
        non-padding positions.
    cache_size : int, default=4096
        Maximum number of pooled backbone rows retained in the cache. Set to
        zero to disable caching.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: SequenceTokenizer,
        *,
        max_tokens: int = 140,
        latent_dim: int = 64,
        pooling: Literal["last", "mean"] = "mean",
        cache_size: int = 4096,
    ) -> None:
        """Initialize a frozen-backbone sequence encoder.

        Parameters
        ----------
        backbone : torch.nn.Module
            Hugging Face-compatible backbone returning per-token hidden states.
        tokenizer : SequenceTokenizer
            Tokenizer that supplies padding information and attention masks.
        max_tokens : int, default=140
            Total number of token positions consumed per sequence.
        latent_dim : int, default=64
            Width of the trainable projected representation.
        pooling : {"last", "mean"}, default="mean"
            Strategy used to reduce token-level hidden states to one vector.
        cache_size : int, default=4096
            Maximum number of pooled backbone rows to cache. Zero disables the
            cache.

        Raises
        ------
        ValueError
            If ``latent_dim`` is not positive, ``pooling`` is unsupported,
            ``cache_size`` is negative, or the backbone does not expose a
            hidden dimension through ``config.hidden_size`` or
            ``config.d_model``.
        """
        super().__init__(tokenizer=tokenizer, max_tokens=max_tokens)
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if pooling not in ("last", "mean"):
            raise ValueError("pooling must be either 'last' or 'mean'.")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")

        hidden_dim = getattr(backbone.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(backbone.config, "d_model", None)
        if hidden_dim is None:
            raise ValueError(
                "Hugging Face backbone config must define hidden_size or d_model "
                "for projection."
            )
        self.backbone_hidden_dim = int(hidden_dim)
        self.latent_dim = latent_dim
        self.pooling = pooling
        self.cache_size = cache_size
        self.backbone = backbone
        self._pooled_cache: OrderedDict[
            tuple[tuple[int, ...], tuple[int, ...]], Tensor
        ] = OrderedDict()
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
        """Set projection train/eval mode while keeping the backbone in eval mode.

        Parameters
        ----------
        mode : bool, default=True
            Whether the trainable projection should enter training mode.

        Returns
        -------
        HuggingFaceSequenceEncoder
            This encoder instance.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "HuggingFaceSequenceEncoder":
        """Move the encoder while preserving the backbone checkpoint dtype.

        Parameters
        ----------
        *args, **kwargs
            Arguments forwarded to :meth:`torch.nn.Module.to`.

        Returns
        -------
        HuggingFaceSequenceEncoder
            This encoder instance after moving its modules.
        """
        self._pooled_cache.clear()
        result = super().to(*args, **kwargs)
        self.restore_backbone_dtype()
        return result

    def restore_backbone_dtype(self) -> None:
        """Restore the original checkpoint dtype and evaluation mode.

        The trainable projection follows the active runtime dtype, while the
        frozen backbone returns to the dtype it had when it was loaded.

        Returns
        -------
        None
            The backbone is updated in place.
        """
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

    def _pool_hidden_states(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Pool per-token hidden states according to the configured strategy."""
        if self.pooling == "last":
            positions = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
            ).expand(hidden_states.shape[0], -1)
            last_indices = (
                (positions * attention_mask.to(dtype=positions.dtype)).max(dim=1).values
            )
            if torch.any(attention_mask.sum(dim=1) == 0):
                raise ValueError(
                    "token_batch must contain at least one non-padding token per row."
                )
            row_indices = torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
            )
            return hidden_states[row_indices, last_indices]
        return masked_mean(hidden_states, attention_mask)

    def _backbone_features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return pooled backbone features, using the bounded row cache."""
        if input_ids.shape[0] == 0:
            return torch.empty(
                (0, self.backbone_hidden_dim),
                device=input_ids.device,
                dtype=self._backbone_dtype,
            )

        cached_features: list[Tensor | None] = []
        missing_indices: list[int] = []
        missing_keys: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for index, (row, mask) in enumerate(zip(input_ids, attention_mask)):
            key = (tuple(row.tolist()), tuple(mask.tolist()))
            cached = None
            if self.cache_size:
                cached = self._pooled_cache.get(key)
                if cached is not None:
                    self._pooled_cache.move_to_end(key)
            cached_features.append(cached)
            if cached is None:
                missing_indices.append(index)
                missing_keys.append(key)

        if missing_indices:
            missing_ids = input_ids[missing_indices]
            missing_mask = attention_mask[missing_indices]
            with torch.no_grad():
                outputs = self._forward_backbone(missing_ids, missing_mask)
                hidden_states = getattr(outputs, "last_hidden_state", None)
                if hidden_states is None:
                    hidden_states = outputs[0]
            missing_features = self._pool_hidden_states(hidden_states, missing_mask)
            for index, key, feature in zip(
                missing_indices, missing_keys, missing_features
            ):
                feature = feature.detach().cpu()
                cached_features[index] = feature
                if self.cache_size:
                    self._pooled_cache[key] = feature
                    self._pooled_cache.move_to_end(key)
                    while len(self._pooled_cache) > self.cache_size:
                        self._pooled_cache.popitem(last=False)

        return torch.stack(
            [
                feature.to(device=input_ids.device, dtype=self._backbone_dtype)
                for feature in cached_features
                if feature is not None
            ],
            dim=0,
        )

    def forward(self, token_batch: Tensor) -> Tensor:
        """Convert a batch of padded token IDs into projected latent features.

        ``token_batch`` may have any floating dtype because DKL model-space
        tensors are floating-point; IDs are converted to ``torch.long`` at
        this sequence-encoder boundary.

        Parameters
        ----------
        token_batch : Tensor
            Two-dimensional tensor of shape ``(B, seq_len)`` containing padded
            token IDs. Inputs longer than ``max_seq_len`` are truncated.

        Returns
        -------
        Tensor
            Projected latent features of shape ``(B, latent_dim)``.

        Raises
        ------
        ValueError
            If ``token_batch`` is not two-dimensional, or if its tokenizer
            returns an attention mask with an incompatible shape.
        """
        if token_batch.ndim != 2:
            raise ValueError(
                f"token_batch must be 2-D (B, seq_len), got {tuple(token_batch.shape)}"
            )
        input_ids = token_batch[:, : self.max_seq_len].long()
        attention_mask_from_batch = getattr(
            self.tokenizer,
            "attention_mask_from_batch",
            None,
        )
        if attention_mask_from_batch is None:
            attention_mask = input_ids.ne(self.tokenizer.padding_idx).long()
        else:
            attention_mask = attention_mask_from_batch(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask_from_batch must return a mask matching token_batch."
            )
        attention_mask = attention_mask.to(device=input_ids.device)

        pooled = self._backbone_features(input_ids, attention_mask)
        pooled = pooled.to(
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )
        return self.projection(pooled)

"""Configuration helpers for reusable Hugging Face sequence encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from activelearning.config_registry import BuildableConfig

if TYPE_CHECKING:
    from activelearning.surrogate.sequence.huggingface_encoder import (
        HuggingFaceSequenceEncoder,
    )

__all__ = ["HuggingFaceEncoderConfig"]


class HuggingFaceEncoderConfig(BuildableConfig):
    """Shared configuration for frozen Hugging Face sequence encoders.

    Parameters
    ----------
    model_name_or_path : str
        Hugging Face model identifier or local model path.
    tokenizer_name_or_path : str
        Hugging Face tokenizer identifier or local tokenizer path.
    trust_remote_code : bool
        Whether loading may execute code supplied by the model repository.
    cache_dir : str, optional
        Directory used for Hugging Face model and tokenizer files.
    max_mol_tokens : int
        Total number of token positions consumed for each input sequence,
        including special tokens and padding. The existing field name is
        preserved for YAML compatibility.
    latent_dim : int
        Size of the projected latent representation.
    pooling : {"last", "mean"}
        Sequence pooling strategy. ``"last"`` selects the final non-padding
        state and ``"mean"`` computes a masked mean.
    cache_size : int
        Maximum number of pooled backbone rows retained in the cache. Set to
        zero to disable caching.
    """

    model_name_or_path: str
    tokenizer_name_or_path: str
    trust_remote_code: bool = True
    cache_dir: str | None = None
    max_mol_tokens: int = 140
    latent_dim: int = 64
    pooling: Literal["last", "mean"] = "mean"
    cache_size: int = Field(default=4096, ge=0)

    def _encoder_class(self) -> type[HuggingFaceSequenceEncoder]:
        """Return the concrete encoder class for this config."""
        raise NotImplementedError

    def build(self) -> HuggingFaceSequenceEncoder:
        """Instantiate the configured pretrained sequence feature encoder.

        Returns
        -------
        HuggingFaceSequenceEncoder
            A concrete encoder returned by :meth:`_encoder_class`.

        Raises
        ------
        NotImplementedError
            If a subclass does not provide an encoder class.
        """
        encoder_class = self._encoder_class()
        return encoder_class(
            model_name_or_path=self.model_name_or_path,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            max_mol_tokens=self.max_mol_tokens,
            latent_dim=self.latent_dim,
            pooling=self.pooling,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
            cache_size=self.cache_size,
        )

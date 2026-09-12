"""Concrete Hugging Face encoders for pretrained SMILES models."""

from __future__ import annotations

from typing import Any, Literal

from torch import Tensor, nn

from activelearning.applications.molecules._optional import (
    missing_molecules_dependency_error,
)
from activelearning.surrogate.sequence.huggingface_tokenizer import HuggingFaceTokenizer
from activelearning.surrogate.sequence.huggingface_encoder import (
    HuggingFaceSequenceEncoder,
)

__all__ = ["GPMoLFormerSmilesEncoder", "MoLFormerSmilesEncoder"]


def _load_transformers() -> tuple[Any, Any, Any]:
    """Load Hugging Face model classes on demand."""
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise missing_molecules_dependency_error(
            "SMILES Hugging Face encoders", error
        ) from error
    return AutoModel, AutoModelForCausalLM, AutoTokenizer


def _load_smiles_tokenizer(
    auto_tokenizer: Any,
    tokenizer_name_or_path: str,
    *,
    trust_remote_code: bool,
    cache_dir: str | None,
) -> HuggingFaceTokenizer:
    """Load and adapt a Hugging Face tokenizer for SMILES."""
    tokenizer = auto_tokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    return HuggingFaceTokenizer(tokenizer=tokenizer)


class _PretrainedSmilesEncoder(HuggingFaceSequenceEncoder):
    """Shared loader for frozen pretrained SMILES transformer encoders."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        *,
        max_mol_tokens: int,
        latent_dim: int,
        pooling: Literal["last", "mean"],
        trust_remote_code: bool,
        cache_dir: str | None,
        cache_size: int,
        use_causal_lm: bool,
    ) -> None:
        """Load a pretrained backbone and initialize the shared feature head.

        Parameters
        ----------
        model_name_or_path : str
            Hugging Face model identifier or local model path.
        tokenizer_name_or_path : str
            Hugging Face tokenizer identifier or local tokenizer path.
        max_mol_tokens : int
            Total number of token positions consumed per sequence.
        latent_dim : int
            Width of the projected latent representation.
        pooling : {"last", "mean"}
            Strategy used to pool the backbone hidden states.
        trust_remote_code : bool
            Whether Hugging Face may execute repository-provided model code.
        cache_dir : str, optional
            Directory used for downloaded model and tokenizer files.
        cache_size : int
            Maximum number of pooled backbone rows retained in the cache.
        use_causal_lm : bool
            Whether to load the causal-language-model wrapper.

        Raises
        ------
        ImportError
            If the optional Transformers dependency is not installed.
        ValueError
            If the tokenizer or model configuration is missing required
            special tokens or hidden-dimension metadata.
        """
        auto_model, auto_model_for_causal_lm, auto_tokenizer = _load_transformers()
        tokenizer = _load_smiles_tokenizer(
            auto_tokenizer,
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        model_loader = auto_model_for_causal_lm if use_causal_lm else auto_model
        backbone = model_loader.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            deterministic_eval=True,
            cache_dir=cache_dir,
        )
        super().__init__(
            backbone=backbone,
            tokenizer=tokenizer,
            max_tokens=max_mol_tokens,
            latent_dim=latent_dim,
            pooling=pooling,
            cache_size=cache_size,
        )


class GPMoLFormerSmilesEncoder(_PretrainedSmilesEncoder):
    """Encode SMILES with a frozen causal GP-MoLFormer backbone.

    GP-MoLFormer is loaded through ``AutoModelForCausalLM``. Its causal
    language-model wrapper is unwrapped for feature extraction, and causal
    key/value caching is disabled during each forward pass. Last-token pooling
    is the default because the final non-padding state has seen the complete
    preceding SMILES sequence.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        *,
        max_mol_tokens: int = 140,
        latent_dim: int = 64,
        pooling: Literal["last", "mean"] = "last",
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        cache_size: int = 4096,
    ) -> None:
        """Load GP-MoLFormer and initialize its frozen feature encoder.

        Parameters
        ----------
        model_name_or_path : str
            Hugging Face model identifier or local model path.
        tokenizer_name_or_path : str
            Hugging Face tokenizer identifier or local tokenizer path.
        max_mol_tokens : int, default=140
            Total number of token positions consumed per sequence.
        latent_dim : int, default=64
            Width of the projected latent representation.
        pooling : {"last", "mean"}, default="last"
            Strategy used to pool the backbone hidden states.
        trust_remote_code : bool, default=False
            Whether Hugging Face may execute repository-provided model code.
        cache_dir : str, optional
            Directory used for downloaded model and tokenizer files.
        cache_size : int, default=4096
            Maximum number of pooled backbone rows retained in the cache.

        Raises
        ------
        ImportError
            If the optional Transformers dependency is not installed.
        ValueError
            If the tokenizer or model configuration is missing required
            special tokens or hidden-dimension metadata.
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_mol_tokens=max_mol_tokens,
            latent_dim=latent_dim,
            pooling=pooling,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            cache_size=cache_size,
            use_causal_lm=True,
        )

    def _base_model(self) -> nn.Module:
        """Return the causal model component that emits hidden states."""
        base_model = getattr(self.backbone, "base_model", None)
        if base_model is None:
            prefix = getattr(self.backbone, "base_model_prefix", None)
            if prefix is not None:
                base_model = getattr(self.backbone, prefix, None)
        if base_model is None:
            raise TypeError(
                "GP-MoLFormer does not expose a base model for hidden states."
            )
        return base_model

    def _forward_backbone(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Any:
        """Run GP-MoLFormer's base model without causal key/value caching."""
        return self._base_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )


class MoLFormerSmilesEncoder(_PretrainedSmilesEncoder):
    """Encode SMILES with the frozen bidirectional MoLFormer backbone.

    The BERT-style MoLFormer is loaded through ``AutoModel`` rather than a
    language-model head. Its native sequence representation is a masked mean
    of the final hidden states, so masked-mean pooling is the default.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        *,
        max_mol_tokens: int = 140,
        latent_dim: int = 64,
        pooling: Literal["last", "mean"] = "mean",
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        cache_size: int = 4096,
    ) -> None:
        """Load MoLFormer and initialize its frozen feature encoder.

        Parameters
        ----------
        model_name_or_path : str
            Hugging Face model identifier or local model path.
        tokenizer_name_or_path : str
            Hugging Face tokenizer identifier or local tokenizer path.
        max_mol_tokens : int, default=140
            Total number of token positions consumed per sequence.
        latent_dim : int, default=64
            Width of the projected latent representation.
        pooling : {"last", "mean"}, default="mean"
            Strategy used to pool the backbone hidden states.
        trust_remote_code : bool, default=False
            Whether Hugging Face may execute repository-provided model code.
        cache_dir : str, optional
            Directory used for downloaded model and tokenizer files.
        cache_size : int, default=4096
            Maximum number of pooled backbone rows retained in the cache.

        Raises
        ------
        ImportError
            If the optional Transformers dependency is not installed.
        ValueError
            If the tokenizer or model configuration is missing required
            special tokens or hidden-dimension metadata.
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_mol_tokens=max_mol_tokens,
            latent_dim=latent_dim,
            pooling=pooling,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            cache_size=cache_size,
            use_causal_lm=False,
        )

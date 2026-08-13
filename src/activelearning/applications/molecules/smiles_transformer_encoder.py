"""Concrete Hugging Face encoders for pretrained SMILES models."""

from __future__ import annotations

from typing import Any, Literal

from torch import Tensor, nn

from activelearning.applications.molecules._optional import (
    missing_molecules_dependency_error,
)
from activelearning.surrogate.sequence.huggingface import HuggingFaceTokenizer
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


class GPMoLFormerSmilesEncoder(HuggingFaceSequenceEncoder):
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
    ) -> None:
        """Load GP-MoLFormer and initialize its frozen feature encoder.

        Parameters
        ----------
        model_name_or_path : str
            Hugging Face model identifier or local path for GP-MoLFormer.
        tokenizer_name_or_path : str
            Hugging Face model identifier or local path for its tokenizer.
        max_mol_tokens : int, default=140
            Maximum number of token positions consumed for each SMILES.
        latent_dim : int, default=64
            Size of the projected latent molecular representation.
        pooling : {"last", "mean"}, default="last"
            Sequence pooling strategy. ``"last"`` selects the last non-padding
            state, while ``"mean"`` computes a masked mean.
        trust_remote_code : bool, default=False
            Whether loading from Hugging Face may execute repository-supplied
            model code.
        cache_dir : str, optional
            Directory used for Hugging Face model and tokenizer files.
        """
        _, auto_model_for_causal_lm, auto_tokenizer = _load_transformers()
        tokenizer = _load_smiles_tokenizer(
            auto_tokenizer,
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        backbone = auto_model_for_causal_lm.from_pretrained(
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


class MoLFormerSmilesEncoder(HuggingFaceSequenceEncoder):
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
    ) -> None:
        """Load the encoder-style MoLFormer and initialize its feature head.

        Parameters
        ----------
        model_name_or_path : str
            Hugging Face model identifier or local path for MoLFormer.
        tokenizer_name_or_path : str
            Hugging Face model identifier or local path for its tokenizer.
        max_mol_tokens : int, default=140
            Maximum number of token positions consumed for each SMILES.
        latent_dim : int, default=64
            Size of the projected latent molecular representation.
        pooling : {"last", "mean"}, default="mean"
            Sequence pooling strategy. ``"mean"`` matches the model's native
            masked-mean pooler; ``"last"`` is available as an alternative.
        trust_remote_code : bool, default=False
            Whether loading from Hugging Face may execute repository-supplied
            model code.
        cache_dir : str, optional
            Directory used for Hugging Face model and tokenizer files.
        """
        auto_model, _, auto_tokenizer = _load_transformers()
        tokenizer = _load_smiles_tokenizer(
            auto_tokenizer,
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        backbone = auto_model.from_pretrained(
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
        )

"""Concrete Hugging Face encoders for pretrained SMILES models."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from activelearning.surrogate.encoder import FixedEncoder
from activelearning_molecules._optional import (
    missing_molecules_dependency_error,
)
from activelearning.surrogate.sequence.huggingface_tokenizer import HuggingFaceTokenizer
from activelearning.surrogate.sequence.huggingface_encoder import (
    HuggingFaceSequenceEncoder,
)

__all__ = [
    "GPMoLFormerSmilesEncoder",
    "GPMoLFormerSmilesFixedEncoder",
    "MoLFormerSmilesEncoder",
]


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


def _drop_fast_transformers_fallback_warning(record: logging.LogRecord) -> bool:
    """Drop GP-MoLFormer's per-layer notice that it uses PyTorch attention.

    GP-MoLFormer's causal attention tries the optional ``fast_transformers``
    CUDA kernel on every call and logs a warning before using an equivalent
    PyTorch implementation.
    """
    return "Falling back to (slow) pytorch implementation" not in record.getMessage()


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
        logging.getLogger(type(self.backbone).__module__).addFilter(
            _drop_fast_transformers_fallback_warning
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


class GPMoLFormerSmilesFixedEncoder(FixedEncoder):
    """Encode SMILES as frozen pooled GP-MoLFormer backbone features."""

    input_representation = "smiles"

    def __init__(
        self,
        model_name_or_path: str = "ibm-research/GP-MoLFormer-Uniq",
        tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct",
        *,
        max_mol_tokens: int = 140,
        pooling: Literal["last", "mean"] = "last",
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        cache_size: int = 4096,
        batch_size: int = 128,
        torch_compile: bool = False,
        torch_compile_mode: str = "default",
        torch_compile_dynamic: bool | None = None,
    ) -> None:
        """Load GP-MoLFormer and expose its pooled features without projection.

        Features are computed in chunks of ``batch_size`` molecules to bound
        activation memory. ``torch_compile`` compiles the backbone forward with
        the S3-GFN attention-mask adapter installed.
        """
        if max_mol_tokens < 2:
            raise ValueError("max_mol_tokens must be at least two.")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self._encoder = GPMoLFormerSmilesEncoder(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_mol_tokens=max_mol_tokens,
            latent_dim=1,
            pooling=pooling,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            cache_size=cache_size,
        )
        self.feature_dim = self._encoder.backbone_hidden_dim
        self.batch_size = batch_size
        if torch_compile:
            from activelearning_molecules.samplers.s3gfn.model import (
                _install_gp_molformer_attention_mask_adapter,
            )

            _install_gp_molformer_attention_mask_adapter(self._encoder.backbone)
            base_model = self._encoder._base_model()
            base_model.forward = torch.compile(
                base_model.forward,
                mode=torch_compile_mode,
                dynamic=torch_compile_dynamic,
            )

    def bind_runtime_context(self, runtime_context: Any) -> None:
        """Bind runtime settings and move the frozen backbone to its device."""
        super().bind_runtime_context(runtime_context)
        self._encoder.to(device=runtime_context.device)

    def encode(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Return pooled frozen backbone features for a batch of SMILES."""
        strings: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    "GP-MoLFormer fixed encoders require string inputs, got "
                    f"{type(value).__name__}."
                )
            strings.append(value)

        model_device = next(self._encoder.backbone.parameters()).device
        features: list[Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(strings), self.batch_size):
                token_batch = self._encoder.prepare_inputs(
                    strings[start : start + self.batch_size], device=model_device
                )
                attention_mask = self._encoder.tokenizer.attention_mask_from_batch(
                    token_batch
                )
                features.append(
                    self._encoder._backbone_features(token_batch, attention_mask)
                )
        return torch.cat(features).to(device=device)


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

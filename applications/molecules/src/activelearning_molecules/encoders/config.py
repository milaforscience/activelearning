"""Configuration models for molecular DKL encoders."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from activelearning.config_registry import BuildableConfig
from activelearning.surrogate.sequence.config import HuggingFaceEncoderConfig


def _default_selfies_vocab() -> list[str]:
    """Return the bundled default SELFIES vocabulary."""
    from activelearning_molecules.constants import SELFIES_VOCAB_SMALL

    return list(SELFIES_VOCAB_SMALL)


class SelfiesTransformerEncoderConfig(BuildableConfig):
    """Configuration for a Transformer encoder over SELFIES tokens."""

    type: Literal["SelfiesTransformerEncoder"] = "SelfiesTransformerEncoder"
    input_representation: ClassVar[str] = "selfies"
    vocab: list[str] = Field(default_factory=_default_selfies_vocab)
    max_mol_tokens: int = 66
    embed_dim: int = 64
    ff_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    latent_dim: int = 64
    dropout: float = 0.0

    def build(self) -> object:
        """Instantiate the tokenizer-backed SELFIES encoder lazily."""
        from activelearning_molecules.tokenization.selfies import SelfiesTokenizer
        from activelearning.surrogate.sequence.transformer_encoder import (
            TransformerSequenceEncoder,
        )

        return TransformerSequenceEncoder(
            tokenizer=SelfiesTokenizer(selfies_vocab=self.vocab),
            max_tokens=self.max_mol_tokens,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )


class GPMoLFormerSmilesEncoderConfig(HuggingFaceEncoderConfig):
    """Configuration for the frozen causal GP-MoLFormer SMILES encoder."""

    type: Literal["GPMoLFormerSmilesEncoder"] = "GPMoLFormerSmilesEncoder"
    input_representation: ClassVar[str] = "smiles"
    model_name_or_path: str = "ibm-research/GP-MoLFormer-Uniq"
    tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"
    pooling: Literal["last", "mean"] = "last"

    def _encoder_class(self) -> type[object]:
        """Return the pretrained GP-MoLFormer encoder class lazily."""
        from activelearning_molecules.encoders.molformer import (
            GPMoLFormerSmilesEncoder,
        )

        return GPMoLFormerSmilesEncoder


class MoLFormerSmilesEncoderConfig(HuggingFaceEncoderConfig):
    """Configuration for the frozen bidirectional MoLFormer SMILES encoder."""

    type: Literal["MoLFormerSmilesEncoder"] = "MoLFormerSmilesEncoder"
    input_representation: ClassVar[str] = "smiles"
    model_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"
    tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"

    def _encoder_class(self) -> type[object]:
        """Return the pretrained MoLFormer encoder class lazily."""
        from activelearning_molecules.encoders.molformer import (
            MoLFormerSmilesEncoder,
        )

        return MoLFormerSmilesEncoder


class MiniMolSmilesEncoderConfig(BuildableConfig):
    """Configuration for a frozen MiniMol SMILES fingerprint encoder."""

    type: Literal["MiniMolSmilesEncoder"] = "MiniMolSmilesEncoder"
    input_representation: ClassVar[str] = "smiles"
    batch_size: int = Field(default=100, ge=1)
    latent_dim: int = Field(default=32, ge=1)
    cache_size: int = Field(default=4096, ge=0)
    checkpoint_path: Path | None = None

    def build(self) -> object:
        """Instantiate the configured MiniMol encoder lazily."""
        from activelearning_molecules.encoders.minimol import MiniMolSmilesEncoder

        return MiniMolSmilesEncoder(
            batch_size=self.batch_size,
            latent_dim=self.latent_dim,
            cache_size=self.cache_size,
            checkpoint_path=self.checkpoint_path,
        )


class MiniMolSmilesFixedEncoderConfig(BuildableConfig):
    """Configuration for fixed stock MiniMol SMILES representations."""

    type: Literal["MiniMolSmilesFixedEncoder"] = "MiniMolSmilesFixedEncoder"
    input_representation: ClassVar[str] = "smiles"
    batch_size: int = Field(default=100, ge=1)
    cache_size: int = Field(default=4096, ge=0)
    checkpoint_path: Path | None = None

    def build(self) -> object:
        """Instantiate the fixed MiniMol encoder lazily."""
        from activelearning_molecules.encoders.minimol import (
            MiniMolSmilesFixedEncoder,
        )

        return MiniMolSmilesFixedEncoder(
            batch_size=self.batch_size,
            cache_size=self.cache_size,
            checkpoint_path=self.checkpoint_path,
        )


MOLECULE_ENCODER_CONFIGS = (
    SelfiesTransformerEncoderConfig,
    GPMoLFormerSmilesEncoderConfig,
    MoLFormerSmilesEncoderConfig,
    MiniMolSmilesEncoderConfig,
)

MOLECULE_FIXED_ENCODER_CONFIGS = (MiniMolSmilesFixedEncoderConfig,)

__all__ = [
    "GPMoLFormerSmilesEncoderConfig",
    "MOLECULE_ENCODER_CONFIGS",
    "MOLECULE_FIXED_ENCODER_CONFIGS",
    "MiniMolSmilesEncoderConfig",
    "MiniMolSmilesFixedEncoderConfig",
    "MoLFormerSmilesEncoderConfig",
    "SelfiesTransformerEncoderConfig",
]

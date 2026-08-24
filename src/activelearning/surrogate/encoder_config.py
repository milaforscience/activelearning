"""Configuration classes for built-in DKL encoder components.

The encoder configuration union lives with the core surrogate interfaces so
the surrogate package does not import application implementations eagerly.
Concrete ``build()`` methods keep application imports lazy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, Field

from activelearning.surrogate.sequence.config import HuggingFaceEncoderConfig

if TYPE_CHECKING:
    from activelearning.surrogate.sequence.huggingface_encoder import (
        HuggingFaceSequenceEncoder,
    )
    from activelearning.surrogate.sequence.transformer_encoder import (
        TransformerSequenceEncoder,
    )
    from activelearning.applications.molecules.minimol_encoder import (
        MiniMolSmilesEncoder,
    )


def _default_selfies_vocab() -> list[str]:
    """Load the default SELFIES vocabulary without an eager app import."""
    from activelearning.applications.molecules.constants import SELFIES_VOCAB_SMALL

    return list(SELFIES_VOCAB_SMALL)


class SelfiesTransformerEncoderConfig(BaseModel):
    """Configuration for a Transformer encoder over SELFIES tokens.

    All fields mirror the encoder constructor; changing them via YAML enables
    easy hyperparameter search (e.g. latent_dim sweep for benchmarking).

    Parameters
    ----------
    vocab : list[str]
        SELFIES alphabet. Defaults to
        :data:`~activelearning.applications.molecules.constants.SELFIES_VOCAB_SMALL`.
    max_mol_tokens : int
        Total number of token positions per sequence, including special
        tokens and padding.
    embed_dim : int
        Token embedding and Transformer hidden dimensionality.
    ff_dim : int
        Feedforward hidden size in each Transformer layer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    latent_dim : int
        Output dimensionality of the pooled molecular representation.
    dropout : float
        Dropout applied throughout the encoder.
    """

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

    def build(self) -> "TransformerSequenceEncoder":
        """Instantiate the Transformer encoder with its SELFIES tokenizer.

        Returns
        -------
        TransformerSequenceEncoder
            Configured tokenizer-backed Transformer encoder.

        Raises
        ------
        ImportError
            If the optional ``selfies`` dependency is not installed.
        """
        from activelearning.applications.molecules.selfies_tokenizer import (
            SelfiesTokenizer,
        )
        from activelearning.surrogate.sequence.transformer_encoder import (
            TransformerSequenceEncoder,
        )

        tokenizer = SelfiesTokenizer(selfies_vocab=self.vocab)
        return TransformerSequenceEncoder(
            tokenizer=tokenizer,
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

    def _encoder_class(self) -> type[HuggingFaceSequenceEncoder]:
        """Return the pretrained GP-MoLFormer encoder class."""
        from activelearning.applications.molecules.smiles_transformer_encoder import (
            GPMoLFormerSmilesEncoder,
        )

        return GPMoLFormerSmilesEncoder


class MoLFormerSmilesEncoderConfig(HuggingFaceEncoderConfig):
    """Configuration for the frozen bidirectional MoLFormer SMILES encoder."""

    type: Literal["MoLFormerSmilesEncoder"] = "MoLFormerSmilesEncoder"
    input_representation: ClassVar[str] = "smiles"
    model_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"
    tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"

    def _encoder_class(self) -> type[HuggingFaceSequenceEncoder]:
        """Return the pretrained MoLFormer encoder class."""
        from activelearning.applications.molecules.smiles_transformer_encoder import (
            MoLFormerSmilesEncoder,
        )

        return MoLFormerSmilesEncoder


class MiniMolSmilesEncoderConfig(BaseModel):
    """Configuration for a frozen MiniMol SMILES fingerprint encoder.

    MiniMol produces fixed 512-dimensional graph fingerprints. The encoder
    trains a projection from those fingerprints to ``latent_dim`` for DKL.

    Parameters
    ----------
    batch_size : int, default=100
        Maximum number of SMILES processed by MiniMol per extraction batch.
    latent_dim : int, default=32
        Width of the projected latent representation.
    cache_size : int, default=4096
        Maximum number of fingerprints retained in the encoder cache. Set to
        zero to disable caching.
    checkpoint_path : Path, optional
        Optional predictor state-dict checkpoint loaded over MiniMol's bundled
        pretrained weights.
    """

    type: Literal["MiniMolSmilesEncoder"] = "MiniMolSmilesEncoder"
    input_representation: ClassVar[str] = "smiles"
    batch_size: int = Field(default=100, ge=1)
    latent_dim: int = Field(default=32, ge=1)
    cache_size: int = Field(default=4096, ge=0)
    checkpoint_path: Path | None = None

    def build(self) -> "MiniMolSmilesEncoder":
        """Instantiate the configured MiniMol SMILES encoder.

        Returns
        -------
        MiniMolSmilesEncoder
            Frozen MiniMol fingerprint extraction with a trainable projection.

        Raises
        ------
        ImportError
            If the optional MiniMol dependency is not installed.
        """
        from activelearning.applications.molecules.minimol_encoder import (
            MiniMolSmilesEncoder,
        )

        return MiniMolSmilesEncoder(
            batch_size=self.batch_size,
            latent_dim=self.latent_dim,
            cache_size=self.cache_size,
            checkpoint_path=self.checkpoint_path,
        )


EncoderConfig = Annotated[
    Union[
        SelfiesTransformerEncoderConfig,
        GPMoLFormerSmilesEncoderConfig,
        MoLFormerSmilesEncoderConfig,
        MiniMolSmilesEncoderConfig,
    ],
    Field(discriminator="type"),
]

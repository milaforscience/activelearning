"""Configuration classes for molecule-specific components.

Encoder configs are declared here (not inside surrogate/config.py) so they can
be shared by both DKL surrogate variants without circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, Field

from activelearning.applications.molecules.constants import SELFIES_VOCAB_SMALL
from activelearning.surrogate.sequence.config import HuggingFaceEncoderConfig

if TYPE_CHECKING:
    from activelearning.surrogate.sequence.transformer_encoder import (
        TransformerSequenceEncoder,
    )
    from activelearning.surrogate.sequence.huggingface_encoder import (
        HuggingFaceSequenceEncoder,
    )


# ---------------------------------------------------------------------------
# Encoder configs
# ---------------------------------------------------------------------------


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
    vocab: list[str] = Field(default_factory=lambda: list(SELFIES_VOCAB_SMALL))
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


EncoderConfig = Annotated[
    Union[
        SelfiesTransformerEncoderConfig,
        GPMoLFormerSmilesEncoderConfig,
        MoLFormerSmilesEncoderConfig,
    ],
    Field(discriminator="type"),
]

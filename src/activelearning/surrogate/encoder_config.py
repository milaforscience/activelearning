"""Configuration classes for built-in DKL encoder components.

The encoder configuration union lives with the core surrogate interfaces so
the surrogate package does not import application implementations eagerly.
Concrete ``build()`` methods keep application imports lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from activelearning.surrogate.sequence.transformer_encoder import (
        TransformerSequenceEncoder,
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


EncoderConfig = Annotated[
    SelfiesTransformerEncoderConfig,
    Field(discriminator="type"),
]

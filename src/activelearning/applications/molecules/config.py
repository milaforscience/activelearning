"""Configuration classes for molecule-specific components.

Encoder configs are declared here (not inside surrogate/config.py) so they can
be shared by both DKL surrogate variants without circular imports.
The discriminated union ``EncoderConfig`` is the single source of truth for
encoder selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from activelearning.applications.molecules.constants import SELFIES_VOCAB_SMALL
from activelearning.surrogate.dkl.config import DKLSurrogateConfigBase

if TYPE_CHECKING:
    from activelearning.surrogate.sequence.transformer import (
        TransformerSequenceEncoder,
    )
    from activelearning.applications.molecules.smiles_transformer_encoder import (
        GPMoLFormerSmilesEncoder,
        MoLFormerSmilesEncoder,
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
        Maximum number of molecular tokens per sequence, not
        counting the ``[CLS]`` and ``[EOS]`` specials added internally.
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
    vocab: list[str] = Field(default_factory=lambda: list(SELFIES_VOCAB_SMALL))
    max_mol_tokens: int = 64
    embed_dim: int = 64
    ff_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    latent_dim: int = 64
    dropout: float = 0.0

    def build(self) -> "TransformerSequenceEncoder":
        """Instantiate the encoder with its tokenizer."""
        from activelearning.applications.molecules.selfies_tokenizer import (
            SelfiesTokenizer,
        )
        from activelearning.surrogate.sequence.transformer import (
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


class HuggingFaceSmilesEncoderConfig(BaseModel):
    """Shared configuration for frozen Hugging Face SMILES encoders.

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
        Maximum number of token positions consumed for each SMILES.
    latent_dim : int
        Size of the projected latent molecular representation.
    pooling : {"last", "mean"}
        Sequence pooling strategy. ``"last"`` selects the final non-padding
        state and ``"mean"`` computes a masked mean.
    """

    model_name_or_path: str
    tokenizer_name_or_path: str
    trust_remote_code: bool = True
    cache_dir: str | None = None
    max_mol_tokens: int = 140
    latent_dim: int = 64
    pooling: Literal["last", "mean"] = "mean"


class GPMoLFormerSmilesEncoderConfig(HuggingFaceSmilesEncoderConfig):
    """Configuration for the frozen causal GP-MoLFormer SMILES encoder."""

    type: Literal["GPMoLFormerSmilesEncoder"] = "GPMoLFormerSmilesEncoder"
    model_name_or_path: str = "ibm-research/GP-MoLFormer-Uniq"
    tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"
    pooling: Literal["last", "mean"] = "last"

    def build(self) -> "GPMoLFormerSmilesEncoder":
        """Instantiate the pretrained GP-MoLFormer feature encoder."""
        from activelearning.applications.molecules.smiles_transformer_encoder import (
            GPMoLFormerSmilesEncoder,
        )

        return GPMoLFormerSmilesEncoder(
            model_name_or_path=self.model_name_or_path,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            max_mol_tokens=self.max_mol_tokens,
            latent_dim=self.latent_dim,
            pooling=self.pooling,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )


class MoLFormerSmilesEncoderConfig(HuggingFaceSmilesEncoderConfig):
    """Configuration for the frozen bidirectional MoLFormer SMILES encoder."""

    type: Literal["MoLFormerSmilesEncoder"] = "MoLFormerSmilesEncoder"
    model_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"
    tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct"

    def build(self) -> "MoLFormerSmilesEncoder":
        """Instantiate the pretrained MoLFormer feature encoder."""
        from activelearning.applications.molecules.smiles_transformer_encoder import (
            MoLFormerSmilesEncoder,
        )

        return MoLFormerSmilesEncoder(
            model_name_or_path=self.model_name_or_path,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            max_mol_tokens=self.max_mol_tokens,
            latent_dim=self.latent_dim,
            pooling=self.pooling,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )


EncoderConfig = Annotated[
    Union[
        SelfiesTransformerEncoderConfig,
        GPMoLFormerSmilesEncoderConfig,
        MoLFormerSmilesEncoderConfig,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of encoder configuration types.

Add new encoder config classes to this union when new encoders are introduced).
The ``type`` discriminator must be unique.
"""

# ---------------------------------------------------------------------------
# Surrogate configs
# ---------------------------------------------------------------------------


class ExactDKLSurrogateConfig(DKLSurrogateConfigBase):
    """Configuration for the exact DKL surrogate with a molecule encoder.

    Parameters
    ----------
    encoder : EncoderConfig
        Encoder architecture.
    """

    type: Literal["ExactDKLSurrogate"] = "ExactDKLSurrogate"
    encoder: EncoderConfig

    def build(self) -> object:
        from activelearning.surrogate.dkl.dkl_surrogate import ExactDKLSurrogate

        encoder = self.encoder.build()
        return ExactDKLSurrogate(
            encoder=encoder,
            input_adapter=_molecule_string_input_adapter(encoder),
            training_params=self.training_params,
            is_multi_fidelity=self.is_multi_fidelity,
            target_fidelity=self.target_fidelity,
            standardize_outputs=self.standardize_outputs,
        )


class VariationalDKLSurrogateConfig(DKLSurrogateConfigBase):
    """Configuration for the variational DKL surrogate with a molecule encoder.

    Uses a sparse variational GP head (``ApproximateGP + VariationalELBO``),
    mirroring the reference sparse DKL implementation.
    Compatible with acquisitions that only require ``predict()`` (UCB, etc.).

    Parameters
    ----------
    encoder : EncoderConfig
        Encoder architecture.
    num_inducing : int
        Number of variational inducing points.
    """

    type: Literal["VariationalDKLSurrogate"] = "VariationalDKLSurrogate"
    encoder: EncoderConfig
    num_inducing: int = 64

    def build(self) -> object:
        from activelearning.surrogate.dkl.dkl_surrogate import VariationalDKLSurrogate

        encoder = self.encoder.build()
        return VariationalDKLSurrogate(
            encoder=encoder,
            input_adapter=_molecule_string_input_adapter(encoder),
            training_params=self.training_params,
            is_multi_fidelity=self.is_multi_fidelity,
            target_fidelity=self.target_fidelity,
            num_inducing=self.num_inducing,
            standardize_outputs=self.standardize_outputs,
        )


def _molecule_string_input_adapter(encoder: Any) -> Any:
    """Create a molecular-string adapter for a configured sequence encoder."""
    from activelearning.applications.molecules.input_adapter import (
        MoleculeStringInputAdapter,
    )

    return MoleculeStringInputAdapter(
        tokenizer=encoder.tokenizer,
        max_tokens=encoder.max_tokens,
    )

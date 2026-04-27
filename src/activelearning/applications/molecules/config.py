"""Configuration classes for molecules-specific components.

Encoder configs are declared here (not inside surrogate/config.py) so they can
be shared by both DKL surrogate variants without circular imports.
The discriminated union ``EncoderConfig`` is the single source of truth for
encoder selection.
"""

from __future__ import annotations
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from activelearning.applications.molecules.selfies_transformer_encoder import (
    SelfiesTransformerEncoder,
)
from activelearning.applications.molecules.selfies_tokenizer import (
    SELFIES_VOCAB_SMALL,
    SelfiesTokenizer,
)


# ---------------------------------------------------------------------------
# Encoder configs
# ---------------------------------------------------------------------------


class SelfiesTransformerEncoderConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.encoder.SelfiesTransformerEncoder`.

    All fields mirror the encoder constructor; changing them via YAML enables
    easy hyperparameter search (e.g. latent_dim sweep for benchmarking).

    Parameters
    ----------
    vocab : list[str]
        SELFIES alphabet.  Defaults to :data:`~activelearning.applications.molecules.tokenizer.SELFIES_VOCAB_SMALL`.
    max_length : int
        Base sequence length (special tokens added internally).
    embed_dim : int
        Token embedding and Transformer hidden dimensionality.
    ff_dim : int
        Feedforward hidden size in each Transformer layer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    latent_dim : int
        Output dimensionality of the pooled molecules vector.
    dropout : float
        Dropout applied throughout the encoder.
    """

    type: Literal["SelfiesTransformerEncoder"] = "SelfiesTransformerEncoder"
    vocab: list[str] = Field(default_factory=lambda: list(SELFIES_VOCAB_SMALL))
    max_length: int = 64
    embed_dim: int = 64
    ff_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    latent_dim: int = 64
    dropout: float = 0.0

    def build(self) -> SelfiesTransformerEncoder:
        """Instantiate the encoder with its tokenizer."""
        tokenizer = SelfiesTokenizer(selfies_vocab=self.vocab)
        return SelfiesTransformerEncoder(
            tokenizer=tokenizer,
            max_length=self.max_length,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )


EncoderConfig = Annotated[
    Union[SelfiesTransformerEncoderConfig],
    Field(discriminator="type"),
]
"""Discriminated union of encoder configuration types.

Add new encoder config classes to this union when new encoders are introduced).
The ``type`` discriminator must be unique.
"""


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


class SelfiesTrainingConfig(BaseModel):
    """Hyper-parameters for the joint MLM + GP Adam training loop.

    Parameters
    ----------
    epochs : int
        Number of epochs for the joint MLM + GP training phase.
    lr : float
        Adam learning rate.
    mask_ratio : float
        Fraction of valid tokens masked per sequence in MLM training.
    pretrain_epochs : int
        Number of MLM-only warm-up epochs run *before* joint training.
        Set to 0 to skip pre-training.
    """

    epochs: int = 50
    lr: float = 1e-3
    mask_ratio: float = 0.125
    pretrain_epochs: int = 0


# ---------------------------------------------------------------------------
# Surrogate configs
# ---------------------------------------------------------------------------


class ExactSelfiesDKLSurrogateConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.dkl_surrogate.ExactSelfiesDKLSurrogate`.

    Uses a BoTorch ``SingleTaskGP`` with a :class:`~activelearning.applications.molecules.kernel.SelfiesKernel`
    as the covariance module.  Compatible with all BoTorch acquisition functions.

    Parameters
    ----------
    encoder : EncoderConfig
        Encoder architecture.
    training_params : SelfiesTrainingConfig
        Training hyper-parameters for the joint MLM + GP Adam loop.
    multi_fidelity : bool
        Append fidelity scalar to feature tensors.
    target_fidelity : int, optional
        Required when ``multi_fidelity=True``; typically ``max(fidelity_costs)``.
    standardize_outputs : bool
        Normalise GP outputs to mean 0 / variance 1.
    """

    type: Literal["ExactSelfiesDKLSurrogate"] = "ExactSelfiesDKLSurrogate"
    encoder: EncoderConfig
    training_params: SelfiesTrainingConfig = Field(
        default_factory=SelfiesTrainingConfig
    )
    multi_fidelity: bool = False
    target_fidelity: Optional[int] = None
    standardize_outputs: bool = True

    @model_validator(mode="after")
    def _check_target_fidelity(self) -> "ExactSelfiesDKLSurrogateConfig":
        if self.multi_fidelity and self.target_fidelity is None:
            raise ValueError("target_fidelity is required when multi_fidelity=True")
        return self

    def build(self) -> object:
        from activelearning.applications.molecules.dkl_surrogate import (
            ExactSelfiesDKLSurrogate,
        )

        return ExactSelfiesDKLSurrogate(
            encoder=self.encoder.build(),
            training_params=self.training_params,
            multi_fidelity=self.multi_fidelity,
            target_fidelity=self.target_fidelity,
            standardize_outputs=self.standardize_outputs,
        )


class VariationalSelfiesDKLSurrogateConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.dkl_surrogate.VariationalSelfiesDKLSurrogate`.

    Uses a sparse variational GP head (``ApproximateGP + VariationalELBO``),
    mirroring the reference ``DeepKernelMoleculeRegressor`` implementation.
    Compatible with acquisitions that only require ``predict()`` (UCB, etc.).

    Parameters
    ----------
    encoder : EncoderConfig
        Encoder architecture.
    training_params : SelfiesTrainingConfig
        Training hyper-parameters.
    multi_fidelity : bool
        Append fidelity scalar to latent feature vectors.
    target_fidelity : int, optional
        Required when ``multi_fidelity=True``; typically ``max(fidelity_costs)``.
    num_inducing : int
        Number of variational inducing points.
    standardize_outputs : bool
        Normalise target observations before training.
    """

    type: Literal["VariationalSelfiesDKLSurrogate"] = "VariationalSelfiesDKLSurrogate"
    encoder: EncoderConfig
    training_params: SelfiesTrainingConfig = Field(
        default_factory=SelfiesTrainingConfig
    )
    multi_fidelity: bool = False
    target_fidelity: Optional[int] = None
    num_inducing: int = 64
    standardize_outputs: bool = True

    @model_validator(mode="after")
    def _check_target_fidelity(self) -> "VariationalSelfiesDKLSurrogateConfig":
        if self.multi_fidelity and self.target_fidelity is None:
            raise ValueError("target_fidelity is required when multi_fidelity=True")
        return self

    def build(self) -> object:
        from activelearning.applications.molecules.dkl_surrogate import (
            VariationalSelfiesDKLSurrogate,
        )

        return VariationalSelfiesDKLSurrogate(
            encoder=self.encoder.build(),
            training_params=self.training_params,
            multi_fidelity=self.multi_fidelity,
            target_fidelity=self.target_fidelity,
            num_inducing=self.num_inducing,
            standardize_outputs=self.standardize_outputs,
        )

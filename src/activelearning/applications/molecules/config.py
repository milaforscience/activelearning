"""Configuration classes for molecules-specific components.

Encoder configs are declared here (not inside surrogate/config.py) so they can
be shared by both DKL surrogate variants without circular imports.
The discriminated union ``EncoderConfig`` is the single source of truth for
encoder selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from activelearning.applications.molecules.constants import SELFIES_VOCAB_SMALL

if TYPE_CHECKING:
    from activelearning.applications.molecules.selfies_transformer_encoder import (
        SelfiesTransformerEncoder,
    )


# ---------------------------------------------------------------------------
# Encoder configs
# ---------------------------------------------------------------------------


class SelfiesTransformerEncoderConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.selfies_transformer_encoder.SelfiesTransformerEncoder`.

    All fields mirror the encoder constructor; changing them via YAML enables
    easy hyperparameter search (e.g. latent_dim sweep for benchmarking).

    Parameters
    ----------
    vocab : list[str]
        SELFIES alphabet.  Defaults to :data:`~activelearning.applications.molecules.constants.SELFIES_VOCAB_SMALL`.
    max_mol_tokens : int
        Maximum number of molecular (SELFIES) tokens per sequence, not
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
        Output dimensionality of the pooled molecules vector.
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

    def build(self) -> "SelfiesTransformerEncoder":
        """Instantiate the encoder with its tokenizer."""
        from activelearning.applications.molecules.selfies_tokenizer import (
            SelfiesTokenizer,
        )
        from activelearning.applications.molecules.selfies_transformer_encoder import (
            SelfiesTransformerEncoder,
        )

        tokenizer = SelfiesTokenizer(selfies_vocab=self.vocab)
        return SelfiesTransformerEncoder(
            tokenizer=tokenizer,
            max_mol_tokens=self.max_mol_tokens,
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


def _resolve_dkl_fidelity_confidences(
    config: BaseModel,
    confidences: dict[int, float],
) -> BaseModel:
    """Resolve shared DKL fidelity settings from oracle metadata."""
    is_multi_fidelity = len(confidences) > 1
    target_fidelity = config.target_fidelity

    if not is_multi_fidelity:
        target_fidelity = None
    elif target_fidelity is None:
        target_fidelity = max(confidences, key=confidences.__getitem__)
    elif target_fidelity not in confidences:
        raise ValueError(
            f"Surrogate target_fidelity {target_fidelity} is not declared by "
            f"the oracle. Oracle declares: {sorted(confidences)}."
        )

    data = config.model_dump()
    data.update(
        {
            "is_multi_fidelity": is_multi_fidelity,
            "target_fidelity": target_fidelity,
        }
    )
    return type(config).model_validate(data)


class ExactSelfiesDKLSurrogateConfig(BaseModel):
    """Configuration for :class:`~activelearning.applications.molecules.dkl_surrogate.ExactSelfiesDKLSurrogate`.

    Uses a BoTorch ``SingleTaskGP`` with a :class:`~activelearning.applications.molecules.selfies_kernel.SelfiesKernel`
    as the covariance module.  Compatible with all BoTorch acquisition functions.

    Parameters
    ----------
    encoder : EncoderConfig
        Encoder architecture.
    training_params : SelfiesTrainingConfig
        Training hyper-parameters for the joint MLM + GP Adam loop.
    is_multi_fidelity : bool
        Append the encoded fidelity confidence to feature tensors.
    target_fidelity : int, optional
        Target level for multi-fidelity acquisition. The top-level active
        learning config derives it from oracle confidence metadata when
        omitted. Standalone multi-fidelity builds must provide it explicitly.
    standardize_outputs : bool
        Normalise GP outputs to mean 0 / variance 1.
    """

    type: Literal["ExactSelfiesDKLSurrogate"] = "ExactSelfiesDKLSurrogate"
    encoder: EncoderConfig
    training_params: SelfiesTrainingConfig = Field(
        default_factory=SelfiesTrainingConfig
    )
    is_multi_fidelity: bool = False
    target_fidelity: Optional[int] = None
    standardize_outputs: bool = True

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> "ExactSelfiesDKLSurrogateConfig":
        """Resolve DKL fidelity mode and target from oracle confidences.

        Parameters
        ----------
        confidences : dict[int, float]
            Oracle fidelity confidence mapping.

        Returns
        -------
        ExactSelfiesDKLSurrogateConfig
            Revalidated config with resolved fidelity settings.

        Raises
        ------
        ValueError
            If the explicit target fidelity is not declared by the oracle.
        """
        return _resolve_dkl_fidelity_confidences(self, confidences)

    def build(self) -> object:
        from activelearning.applications.molecules.dkl_surrogate import (
            ExactSelfiesDKLSurrogate,
        )

        return ExactSelfiesDKLSurrogate(
            encoder=self.encoder.build(),
            training_params=self.training_params,
            is_multi_fidelity=self.is_multi_fidelity,
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
    is_multi_fidelity : bool
        Append the encoded fidelity confidence to latent feature vectors.
    target_fidelity : int, optional
        Target level for multi-fidelity acquisition. The top-level active
        learning config derives it from oracle confidence metadata when
        omitted. Standalone multi-fidelity builds must provide it explicitly.
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
    is_multi_fidelity: bool = False
    target_fidelity: Optional[int] = None
    num_inducing: int = 64
    standardize_outputs: bool = True

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> "VariationalSelfiesDKLSurrogateConfig":
        """Resolve DKL fidelity mode and target from oracle confidences.

        Parameters
        ----------
        confidences : dict[int, float]
            Oracle fidelity confidence mapping.

        Returns
        -------
        VariationalSelfiesDKLSurrogateConfig
            Revalidated config with resolved fidelity settings.

        Raises
        ------
        ValueError
            If the explicit target fidelity is not declared by the oracle.
        """
        return _resolve_dkl_fidelity_confidences(self, confidences)

    def build(self) -> object:
        from activelearning.applications.molecules.dkl_surrogate import (
            VariationalSelfiesDKLSurrogate,
        )

        return VariationalSelfiesDKLSurrogate(
            encoder=self.encoder.build(),
            training_params=self.training_params,
            is_multi_fidelity=self.is_multi_fidelity,
            target_fidelity=self.target_fidelity,
            num_inducing=self.num_inducing,
            standardize_outputs=self.standardize_outputs,
        )

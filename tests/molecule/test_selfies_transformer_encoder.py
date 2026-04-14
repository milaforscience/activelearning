"""Tests for SelfiesTransformerEncoder."""

import pytest
import torch

from activelearning.applications.molecule.selfies_transformer_encoder import (
    MaskedMeanPool,
    PositionalEncoding,
    SelfiesTransformerEncoder,
)
from activelearning.applications.molecule.selfies_tokenizer import (
    SELFIES_VOCAB_SMALL,
    SelfiesTokenizer,
)

BENZENE = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
ALANINE = "[C][Branch1][C][N][C][Branch1][C][C][C][=O][O]"


@pytest.fixture
def tokenizer() -> SelfiesTokenizer:
    return SelfiesTokenizer(SELFIES_VOCAB_SMALL)


@pytest.fixture
def encoder(tokenizer: SelfiesTokenizer) -> SelfiesTransformerEncoder:
    return SelfiesTransformerEncoder(
        tokenizer=tokenizer,
        max_length=32,
        embed_dim=16,
        ff_dim=32,
        num_heads=2,
        num_layers=1,
        latent_dim=8,
    )


@pytest.fixture
def token_batch(tokenizer: SelfiesTokenizer) -> torch.Tensor:
    return tokenizer.batch_from_selfies([BENZENE, ALANINE], max_length=32)


class TestPositionalEncoding:
    def test_shape_preserved(self):
        pe = PositionalEncoding(embed_dim=16, max_len=64)
        x = torch.randn(2, 20, 16)
        out = pe(x)
        assert out.shape == x.shape

    def test_values_differ_from_input(self):
        pe = PositionalEncoding(embed_dim=16, max_len=64)
        x = torch.zeros(1, 10, 16)
        out = pe(x)
        assert not torch.allclose(out, x)


class TestMaskedMeanPool:
    def test_output_shape(self):
        pool = MaskedMeanPool(input_dim=16, output_dim=8)
        features = torch.randn(3, 20, 16)
        mask = torch.ones(3, 20, dtype=torch.bool)
        out = pool(features, mask)
        assert out.shape == (3, 8)

    def test_all_masked_out_stays_finite(self):
        pool = MaskedMeanPool(input_dim=8, output_dim=4)
        features = torch.randn(2, 10, 8)
        mask = torch.zeros(2, 10, dtype=torch.bool)  # all masked
        out = pool(features, mask)
        assert torch.isfinite(out).all()


class TestSelfiesTransformerEncoder:
    def test_forward_shape(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        out = encoder(token_batch)
        assert out.shape == (2, 8)  # (batch, latent_dim)

    def test_forward_float_output(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        out = encoder(token_batch)
        assert out.dtype in {torch.float32, torch.float64}

    def test_double_precision(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        encoder_d = encoder.double()
        out = encoder_d(token_batch)
        assert out.dtype == torch.float64

    def test_encode_tokens_shapes(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        features, mask = encoder.encode_tokens(token_batch)
        assert features.shape[:2] == token_batch.shape  # (B, seq_len, latent_dim)
        assert mask.shape == token_batch.shape

    def test_mlm_loss_scalar(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        loss = encoder.mlm_loss(token_batch, mask_ratio=0.15)
        assert loss.ndim == 0
        assert float(loss) >= 0.0

    def test_mlm_loss_zero_mask_ratio_handled(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        # mask_ratio=0 → n_mask = max(1, 0) = 1; should still succeed
        loss = encoder.mlm_loss(token_batch, mask_ratio=0.0)
        assert torch.isfinite(loss)

    def test_sample_mask_positions_ratio(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        mask = encoder.sample_mask_positions(token_batch, mask_ratio=0.5)
        assert mask.dtype == torch.bool
        assert mask.shape == token_batch.shape

    def test_max_length_attribute(self, encoder: SelfiesTransformerEncoder):
        # max_length stored as base + 2
        assert encoder.max_length == 32 + 2

    def test_latent_dim_attribute(self, encoder: SelfiesTransformerEncoder):
        assert encoder.latent_dim == 8

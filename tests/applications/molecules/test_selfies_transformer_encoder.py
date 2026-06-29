"""Tests for SelfiesTransformerEncoder."""

import math

import pytest
import selfies as sf
import torch

from activelearning.applications.molecules.selfies_transformer_encoder import (
    MaskedMeanPool,
    PositionalEncoding,
    SelfiesTransformerEncoder,
)
from activelearning.applications.molecules.selfies_tokenizer import (
    SELFIES_VOCAB_SMALL,
    SelfiesTokenizer,
)

BENZENE = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
ALANINE = "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]"
ETHANOL = "[C][C][O]"
METHANE = "[C]"
EMPTY_SELFIES = ""


@pytest.fixture
def tokenizer() -> SelfiesTokenizer:
    return SelfiesTokenizer(SELFIES_VOCAB_SMALL)


@pytest.fixture
def encoder(tokenizer: SelfiesTokenizer) -> SelfiesTransformerEncoder:
    return SelfiesTransformerEncoder(
        tokenizer=tokenizer,
        max_mol_tokens=32,
        embed_dim=16,
        ff_dim=32,
        num_heads=2,
        num_layers=1,
        latent_dim=8,
    )


@pytest.fixture
def token_batch(tokenizer: SelfiesTokenizer) -> torch.Tensor:
    return tokenizer.batch_from_selfies([BENZENE, ALANINE], max_mol_tokens=32)


@pytest.fixture
def short_token_batch(tokenizer: SelfiesTokenizer) -> torch.Tensor:
    """A batch with very short sequences: ethanol (len 3) and methane (len 1)."""
    return tokenizer.batch_from_selfies([ETHANOL, METHANE], max_mol_tokens=16)


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

    def test_odd_embed_dim_supported(self):
        pe = PositionalEncoding(embed_dim=7, max_len=64)
        x = torch.randn(2, 20, 7)
        out = pe(x)
        assert out.shape == x.shape

    def test_sinusoidal_encoding_values(self) -> None:
        """PE should emit the canonical sine/cosine values at positions 0 and 1."""
        pe = PositionalEncoding(embed_dim=4, max_len=8)
        out = pe(torch.zeros(1, 2, 4))

        # Position 0: sin(0)=0 and cos(0)=1 alternate across channels
        assert torch.allclose(
            out[0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=out.dtype), atol=1e-6
        )
        # Position 1: first channel pair follows sin(1) / cos(1)
        assert torch.allclose(
            out[0, 1, 0], torch.tensor(math.sin(1.0), dtype=out.dtype), atol=1e-5
        )
        assert torch.allclose(
            out[0, 1, 1], torch.tensor(math.cos(1.0), dtype=out.dtype), atol=1e-5
        )


class TestMaskedMeanPool:
    def test_output_shape(self):
        pool = MaskedMeanPool(input_dim=16, output_dim=8)
        features = torch.randn(3, 20, 16)
        mask = torch.ones(3, 20, dtype=torch.bool)
        out = pool(features, mask)
        assert out.shape == (3, 8)

    def test_all_masked_out_stays_finite(self) -> None:
        """Fully masked inputs should remain finite."""
        pool = MaskedMeanPool(input_dim=8, output_dim=4)
        features = torch.randn(2, 10, 8)
        mask = torch.zeros(2, 10, dtype=torch.bool)  # all masked
        out = pool(features, mask)
        assert torch.isfinite(out).all()

    def test_all_masked_returns_zeros_with_zero_bias(self) -> None:
        """Fully masked inputs should project to zeros when the bias is zero."""
        pool = MaskedMeanPool(input_dim=3, output_dim=3)
        features = torch.randn(2, 4, 3)
        mask = torch.zeros(2, 4, dtype=torch.bool)

        with torch.no_grad():
            pool.proj.bias.zero_()

        out = pool(features, mask)

        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    @pytest.mark.parametrize(
        ("features", "mask", "expected"),
        [
            # Single token: output equals that token unchanged
            (
                torch.tensor([[[1.0, 2.0, 3.0]]]),
                torch.tensor([[True]]),
                torch.tensor([[1.0, 2.0, 3.0]]),
            ),
            # Multiple tokens, partial mask: mean of the two unmasked tokens
            (
                torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]),
                torch.tensor([[True, True, False]]),
                torch.tensor([[2.5, 3.5, 4.5]]),
            ),
        ],
    )
    def test_masked_mean_correct(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        """MaskedMeanPool computes the exact masked mean with identity projection."""
        pool = MaskedMeanPool(input_dim=3, output_dim=3)
        with torch.no_grad():
            pool.proj.weight.copy_(torch.eye(3))
            pool.proj.bias.zero_()

        out = pool(features, mask)

        assert torch.allclose(out, expected, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype_matches_input(self, dtype: torch.dtype) -> None:
        """Output dtype should match the input feature dtype."""
        pool = MaskedMeanPool(input_dim=8, output_dim=4).to(dtype)
        features = torch.randn(2, 10, 8, dtype=dtype)
        mask = torch.ones(2, 10, dtype=torch.bool)
        out = pool(features, mask)
        assert out.dtype == dtype


class TestSelfiesTransformerEncoder:
    def test_forward_shape(
        self, encoder: SelfiesTransformerEncoder, token_batch: torch.Tensor
    ):
        out = encoder(token_batch)
        assert out.shape == (2, 8)  # (batch, latent_dim)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_forward_output_dtype_matches(
        self,
        encoder: SelfiesTransformerEncoder,
        token_batch: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        """Output dtype should match the encoder weight dtype."""
        out = encoder.to(dtype)(token_batch)
        assert out.dtype == dtype

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
    ) -> None:
        """Zero mask ratio should still mask one token and yield positive loss."""
        # mask_ratio=0 → n_mask = max(1, 0) = 1; should still succeed
        loss = encoder.mlm_loss(token_batch, mask_ratio=0.0)
        assert float(loss) > 0.0

    def test_sample_mask_positions_ratio(
        self,
        encoder: SelfiesTransformerEncoder,
        token_batch: torch.Tensor,
        tokenizer: SelfiesTokenizer,
    ) -> None:
        """Mask sampling should pick the expected count and skip special tokens."""
        mask = encoder.sample_mask_positions(token_batch, mask_ratio=0.5)
        assert mask.dtype == torch.bool
        assert mask.shape == token_batch.shape

        expected_counts = [
            max(1, int(sf.len_selfies(BENZENE) * 0.5)),
            max(1, int(sf.len_selfies(ALANINE) * 0.5)),
        ]
        assert mask.sum(dim=1).tolist() == expected_counts
        assert not mask[:, 0].any()

        # Every masked position must be a regular content token: never CLS,
        # EOS, or padding. This positively asserts the sampled positions are
        # valid, rather than only checking individual special positions.
        is_special = (
            (token_batch == tokenizer.cls_idx)
            | (token_batch == tokenizer.eos_idx)
            | (token_batch == tokenizer.padding_idx)
        )
        assert not (mask & is_special).any()

        for row in range(token_batch.shape[0]):
            eos_positions = torch.where(token_batch[row] == tokenizer.eos_idx)[0]
            assert eos_positions.numel() == 1
            eos_position = int(eos_positions[0])
            assert not mask[row, eos_position]
            if eos_position + 1 < token_batch.shape[1]:
                assert not mask[row, eos_position + 1 :].any()
            padding_positions = token_batch[row] == tokenizer.padding_idx
            assert not mask[row, padding_positions].any()

    def test_max_seq_len_attribute(self, encoder: SelfiesTransformerEncoder):
        # max_seq_len = max_mol_tokens + 2
        assert encoder.max_seq_len == 32 + 2
        assert encoder.max_mol_tokens == 32

    def test_latent_dim_attribute(self, encoder: SelfiesTransformerEncoder):
        assert encoder.latent_dim == 8

    def test_odd_embed_dim_encoder_forward(self, tokenizer: SelfiesTokenizer):
        odd_encoder = SelfiesTransformerEncoder(
            tokenizer=tokenizer,
            max_mol_tokens=16,
            embed_dim=9,
            ff_dim=16,
            num_heads=1,
            num_layers=1,
            latent_dim=5,
        )
        batch = tokenizer.batch_from_selfies([BENZENE], max_mol_tokens=16)
        out = odd_encoder(batch)
        assert out.shape == (1, 5)

    def test_encode_tokens_mask_excludes_only_padding(
        self, encoder: SelfiesTransformerEncoder, tokenizer: SelfiesTokenizer
    ) -> None:
        token_batch = tokenizer.batch_from_selfies([BENZENE], max_mol_tokens=16)
        _, mask = encoder.encode_tokens(token_batch)
        # CLS at position 0 must be included.
        assert mask[0, 0].item() is True
        # EOS must also be included (only padding is excluded).
        eos_positions = torch.where(token_batch[0] == tokenizer.eos_idx)[0]
        assert eos_positions.numel() == 1
        eos_position = int(eos_positions[0])
        assert mask[0, eos_position].item() is True
        # Padding positions after EOS must be excluded.
        if eos_position + 1 < token_batch.shape[1]:
            assert mask[0, eos_position + 1 :].any().item() is False

    def test_empty_sequence_produces_finite_output(
        self, encoder: SelfiesTransformerEncoder, tokenizer: SelfiesTokenizer
    ) -> None:
        """Empty SELFIES should encode finitely and never receive MLM masks."""
        token_batch = tokenizer.batch_from_selfies(
            [EMPTY_SELFIES, BENZENE], max_mol_tokens=8
        )

        out = encoder(token_batch)
        mask = encoder.sample_mask_positions(token_batch, mask_ratio=0.5)

        assert out.shape == (2, encoder.latent_dim)
        assert torch.isfinite(out).all()
        assert mask[0].sum().item() == 0

    def test_short_sequences_forward_shape(
        self,
        encoder: SelfiesTransformerEncoder,
        short_token_batch: torch.Tensor,
    ) -> None:
        """Very short sequences (ethanol len 3, methane len 1) should produce finite embeddings."""
        out = encoder(short_token_batch)
        assert out.shape == (2, encoder.latent_dim)
        assert torch.isfinite(out).all()

    def test_short_sequences_encode_tokens_shapes(
        self,
        encoder: SelfiesTransformerEncoder,
        short_token_batch: torch.Tensor,
    ) -> None:
        """encode_tokens should return matching feature and mask shapes for short sequences."""
        features, mask = encoder.encode_tokens(short_token_batch)
        assert features.shape[:2] == short_token_batch.shape
        assert mask.shape == short_token_batch.shape

    @pytest.mark.parametrize(
        ("selfies_str", "expected_valid_tokens"),
        [
            # METHANE has 1 SELFIES token; CLS + token + EOS → only the 1 real token is valid
            (METHANE, 1),
            # ETHANOL has 3 SELFIES tokens
            (ETHANOL, 3),
        ],
    )
    def test_sample_mask_positions_short_sequence_floor(
        self,
        encoder: SelfiesTransformerEncoder,
        tokenizer: SelfiesTokenizer,
        selfies_str: str,
        expected_valid_tokens: int,
    ) -> None:
        """max(1, …) floor ensures at least 1 mask even on minimal sequences.

        Special tokens (CLS, EOS, padding) must never be masked.
        """
        batch = tokenizer.batch_from_selfies([selfies_str], max_mol_tokens=16)
        mask = encoder.sample_mask_positions(batch, mask_ratio=0.15)

        n_masked = int(mask.sum())
        expected_n = max(1, int(expected_valid_tokens * 0.15))
        assert n_masked == expected_n

        # CLS at position 0 must never be masked
        assert not mask[0, 0].item()

        # EOS and subsequent padding must never be masked
        eos_positions = torch.where(batch[0] == tokenizer.eos_idx)[0]
        assert eos_positions.numel() == 1
        eos_pos = int(eos_positions[0])
        assert not mask[0, eos_pos].item()
        if eos_pos + 1 < batch.shape[1]:
            assert not mask[0, eos_pos + 1 :].any().item()

    def test_mlm_loss_short_sequence_positive(
        self,
        encoder: SelfiesTransformerEncoder,
        short_token_batch: torch.Tensor,
    ) -> None:
        """MLM loss on very short sequences should be a finite positive scalar."""
        loss = encoder.mlm_loss(short_token_batch, mask_ratio=0.5)
        assert loss.ndim == 0
        assert float(loss) > 0.0

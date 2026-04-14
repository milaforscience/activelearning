"""Tests for SelfiesTokenizer."""

import pytest
import torch

from activelearning.applications.molecule.selfies_tokenizer import (
    SELFIES_VOCAB_SMALL,
    SelfiesTokenizer,
)

BENZENE = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
ALANINE = "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]"


@pytest.fixture
def tokenizer() -> SelfiesTokenizer:
    return SelfiesTokenizer(SELFIES_VOCAB_SMALL)


class TestSelfiesVocab:
    def test_vocab_not_empty(self):
        assert len(SELFIES_VOCAB_SMALL) > 0

    def test_vocab_contains_common_tokens(self):
        assert "[C]" in SELFIES_VOCAB_SMALL
        assert "[N]" in SELFIES_VOCAB_SMALL
        assert "[O]" in SELFIES_VOCAB_SMALL


class TestSelfiesTokenizer:
    def test_vocab_size(self, tokenizer: SelfiesTokenizer):
        # base_vocab = SELFIES_VOCAB_SMALL + [nop, EOS] + dedup specials
        assert tokenizer.vocab_size > len(SELFIES_VOCAB_SMALL)

    def test_special_token_indices(self, tokenizer: SelfiesTokenizer):
        assert tokenizer.padding_idx == tokenizer.lookup["[nop]"]
        assert tokenizer.eos_idx == tokenizer.lookup["[EOS]"]
        assert tokenizer.cls_idx == tokenizer.lookup["[CLS]"]
        assert tokenizer.mask_idx == tokenizer.lookup["[MASK]"]

    def test_encode_selfies_shape(self, tokenizer: SelfiesTokenizer):
        ids = tokenizer.encode_selfies(BENZENE, max_length=32)
        assert ids.shape == (32,)
        assert ids.dtype == torch.long

    def test_encode_selfies_padding(self, tokenizer: SelfiesTokenizer):
        ids = tokenizer.encode_selfies(BENZENE, max_length=64)
        # Benzene is short; remaining positions should be padding
        pad_count = (ids == tokenizer.padding_idx).sum().item()
        assert pad_count > 0

    def test_transform_batch_shape(self, tokenizer: SelfiesTokenizer):
        raw = torch.stack([tokenizer.encode_selfies(BENZENE, 32) for _ in range(4)])
        out = tokenizer.transform_batch(raw)
        # transform_batch adds 1 extra padding column + 1 CLS column = +2
        assert out.shape == (4, 34)
        assert out.dtype == torch.long

    def test_transform_batch_cls_prefix(self, tokenizer: SelfiesTokenizer):
        raw = torch.stack([tokenizer.encode_selfies(BENZENE, 32)])
        out = tokenizer.transform_batch(raw)
        assert int(out[0, 0]) == tokenizer.cls_idx

    def test_transform_batch_eos_inserted(self, tokenizer: SelfiesTokenizer):
        raw = torch.stack([tokenizer.encode_selfies(BENZENE, 32)])
        out = tokenizer.transform_batch(raw)
        # [EOS] should appear exactly once per sequence
        eos_count = (out[0] == tokenizer.eos_idx).sum().item()
        assert eos_count == 1

    def test_transform_batch_requires_2d(self, tokenizer: SelfiesTokenizer):
        with pytest.raises(ValueError, match="2-D"):
            tokenizer.transform_batch(torch.zeros(10, dtype=torch.long))

    def test_batch_from_selfies_shape(self, tokenizer: SelfiesTokenizer):
        batch = tokenizer.batch_from_selfies([BENZENE, ALANINE], max_length=64)
        assert batch.shape == (2, 66)  # 64 + 2
        assert batch.dtype == torch.long

    def test_batch_from_selfies_device(self, tokenizer: SelfiesTokenizer):
        device = torch.device("cpu")
        batch = tokenizer.batch_from_selfies([BENZENE], max_length=32, device=device)
        assert batch.device.type == "cpu"

    def test_inverse_lookup_roundtrip(self, tokenizer: SelfiesTokenizer):
        for idx, tok in tokenizer.inverse_lookup.items():
            assert tokenizer.lookup[tok] == idx

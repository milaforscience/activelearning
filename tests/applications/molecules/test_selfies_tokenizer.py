"""Tests for SelfiesTokenizer."""

import pytest
import selfies as sf
import torch

from activelearning.applications.molecules.selfies_tokenizer import (
    SELFIES_VOCAB_SMALL,
    SelfiesTokenizer,
)
from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer

BENZENE = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
ALANINE = "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]"
ETHANOL = "[C][C][O]"
LONG_SELFIES = "[C]" * 80


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
    def test_implements_molecule_tokenizer_protocol(
        self, tokenizer: SelfiesTokenizer
    ) -> None:
        assert isinstance(tokenizer, SequenceTokenizer)

    def test_vocab_size(self, tokenizer: SelfiesTokenizer):
        # base_vocab = SELFIES_VOCAB_SMALL + [nop, EOS] + dedup specials
        assert tokenizer.vocab_size > len(SELFIES_VOCAB_SMALL)

    def test_special_token_indices(self, tokenizer: SelfiesTokenizer):
        assert tokenizer.padding_idx == tokenizer.lookup["[nop]"]
        assert tokenizer.eos_idx == tokenizer.lookup["[EOS]"]
        assert tokenizer.cls_idx == tokenizer.lookup["[CLS]"]
        assert tokenizer.mask_idx == tokenizer.lookup["[MASK]"]

    def test_encode_selfies_shape(self, tokenizer: SelfiesTokenizer):
        ids = tokenizer.encode_selfies(BENZENE, max_mol_tokens=32)
        assert ids.shape == (32,)
        assert ids.dtype == torch.long

    @pytest.mark.parametrize("selfies_string", [BENZENE, ALANINE, ETHANOL])
    def test_encode_selfies_padding(
        self, tokenizer: SelfiesTokenizer, selfies_string: str
    ) -> None:
        """encode_selfies pads to exact length and encodes the right real-token count."""
        max_mol_tokens = 32
        real_count = sf.len_selfies(selfies_string)
        ids = tokenizer.encode_selfies(selfies_string, max_mol_tokens=max_mol_tokens)

        assert (ids != tokenizer.padding_idx).sum().item() == real_count
        assert (
            ids == tokenizer.padding_idx
        ).sum().item() == max_mol_tokens - real_count

    @pytest.mark.parametrize(
        ("selfies_string", "expected_distinct_count"),
        [
            (BENZENE, 4),
            (ALANINE, 6),
            (ETHANOL, 2),
        ],
    )
    def test_encode_selfies_distinct_token_count(
        self,
        tokenizer: SelfiesTokenizer,
        selfies_string: str,
        expected_distinct_count: int,
    ) -> None:
        """Encoded SELFIES retain the expected number of distinct token IDs."""
        ids = tokenizer.encode_selfies(selfies_string, max_mol_tokens=32)
        non_padding_ids = ids[ids != tokenizer.padding_idx].tolist()

        assert len(set(non_padding_ids)) == expected_distinct_count

    def test_encode_selfies_truncates_long_sequence(
        self, tokenizer: SelfiesTokenizer
    ) -> None:
        ids = tokenizer.encode_selfies(LONG_SELFIES, max_mol_tokens=16)
        assert ids.shape == (16,)
        assert (ids == tokenizer.padding_idx).sum().item() == 0

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

    @pytest.mark.parametrize(
        "selfies_list",
        [
            [BENZENE],
            [BENZENE, ALANINE, ETHANOL],
        ],
    )
    def test_transform_batch_eos_inserted(
        self, tokenizer: SelfiesTokenizer, selfies_list: list[str]
    ) -> None:
        """Each row in the transformed batch contains exactly one EOS token."""
        raw = torch.stack([tokenizer.encode_selfies(s, 32) for s in selfies_list])
        out = tokenizer.transform_batch(raw)

        for row in out:
            assert (row == tokenizer.eos_idx).sum().item() == 1

    def test_transform_batch_requires_2d(self, tokenizer: SelfiesTokenizer):
        with pytest.raises(ValueError, match="2-D"):
            tokenizer.transform_batch(torch.zeros(10, dtype=torch.long))

    def test_batch_from_selfies_shape(self, tokenizer: SelfiesTokenizer):
        batch = tokenizer.batch_from_selfies([BENZENE, ALANINE], max_mol_tokens=64)
        assert batch.shape == (2, 64)
        assert batch.dtype == torch.long

    def test_batch_from_selfies_mixed_short_and_long_sequences(
        self, tokenizer: SelfiesTokenizer
    ) -> None:
        batch = tokenizer.batch_from_selfies([BENZENE, LONG_SELFIES], max_mol_tokens=16)
        assert batch.shape == (2, 16)
        assert batch.dtype == torch.long

    @pytest.mark.parametrize(
        "selfies_string",
        [BENZENE, ALANINE, ETHANOL],
    )
    def test_batch_from_selfies_exact_layout(
        self, tokenizer: SelfiesTokenizer, selfies_string: str
    ) -> None:
        """Single-sequence batches place CLS, tokens, EOS, and padding exactly."""
        max_mol_tokens = 32
        real_token_count = sf.len_selfies(selfies_string)
        batch = tokenizer.batch_from_selfies(
            [selfies_string], max_mol_tokens=max_mol_tokens
        )
        row = batch[0]
        eos_index = real_token_count + 1

        assert int(row[0]) == tokenizer.cls_idx
        assert int(row[eos_index]) == tokenizer.eos_idx
        assert torch.all(row[eos_index + 1 :] == tokenizer.padding_idx)
        assert int((row == tokenizer.padding_idx).sum().item()) == (
            max_mol_tokens - real_token_count - 2
        )

    def test_batch_from_selfies_device(self, tokenizer: SelfiesTokenizer):
        device = torch.device("cpu")
        batch = tokenizer.batch_from_selfies(
            [BENZENE], max_mol_tokens=32, device=device
        )
        assert batch.device.type == "cpu"

    def test_transform_batch_cls_eos_for_all_rows(
        self, tokenizer: SelfiesTokenizer
    ) -> None:
        """Each transformed row starts with CLS and contains exactly one EOS."""
        raw = torch.stack(
            [
                tokenizer.encode_selfies(BENZENE, 32),
                tokenizer.encode_selfies(ALANINE, 32),
            ]
        )
        out = tokenizer.transform_batch(raw)

        assert torch.all(out[:, 0] == tokenizer.cls_idx)
        for row in out:
            assert int((row == tokenizer.eos_idx).sum().item()) == 1

    def test_inverse_lookup_roundtrip(self, tokenizer: SelfiesTokenizer):
        for idx, tok in tokenizer.inverse_lookup.items():
            assert tokenizer.lookup[tok] == idx

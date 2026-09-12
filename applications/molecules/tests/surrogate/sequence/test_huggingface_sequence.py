from __future__ import annotations

import pytest
import torch
from torch import nn

import activelearning_molecules.encoders.molformer as encoder_module
from activelearning_molecules.encoders.config import (
    GPMoLFormerSmilesEncoderConfig,
    MoLFormerSmilesEncoderConfig,
)
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.dkl import (
    ExactDKLSurrogate,
    VariationalDKLSurrogate,
)
from activelearning.surrogate.sequence.huggingface_tokenizer import HuggingFaceTokenizer
from activelearning.surrogate.sequence.huggingface_encoder import (
    HuggingFaceSequenceEncoder,
)
from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer
from activelearning.surrogate.sequence.transformer_encoder import (
    TransformerSequenceEncoder,
)
from activelearning_molecules.encoders.molformer import (
    GPMoLFormerSmilesEncoder,
    MoLFormerSmilesEncoder,
)
from activelearning.utils.types import Observation


class _FakeHuggingFaceTokenizer:
    vocab_size = 12
    pad_token_id = 2
    eos_token_id = 1
    bos_token_id = 0
    cls_token_id = 0
    mask_token_id = 3
    unk_token_id = 4

    def __call__(
        self,
        strings,
        *,
        add_special_tokens,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        assert add_special_tokens is True
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        rows = []
        for string in strings:
            content = [5 + (ord(character) % 3) for character in string]
            row = [self.bos_token_id, *content, self.eos_token_id]
            row = row[:max_length]
            row.extend([self.pad_token_id] * (max_length - len(row)))
            rows.append(row)
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}

    def __len__(self):
        return self.vocab_size


class _FakePadEqualsEosTokenizer(_FakeHuggingFaceTokenizer):
    pad_token_id = 2
    eos_token_id = 2

    def __call__(self, strings, **kwargs):
        encoded = super().__call__(strings, **kwargs)
        attention_masks = []
        for string in strings:
            valid_length = min(len(string) + 2, kwargs["max_length"])
            attention_masks.append(
                [1] * valid_length + [0] * (kwargs["max_length"] - valid_length)
            )
        encoded["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
        return encoded


class _FakeNoMaskTokenizer(_FakeHuggingFaceTokenizer):
    mask_token_id = None
    unk_token_id = None


class _FakeGPMoLFormerBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        self.forward_calls += 1
        assert use_cache is False
        assert return_dict is True
        hidden = input_ids.to(dtype=self.scale.dtype).unsqueeze(-1)
        hidden = hidden * self.scale
        hidden = hidden.expand(-1, -1, 4)
        return type("FakeOutput", (), {"last_hidden_state": hidden})()


class _FakeGPMoLFormerModel(nn.Module):
    base_model_prefix = "base"

    def __init__(self) -> None:
        super().__init__()
        self.config = type("FakeConfig", (), {"hidden_size": 4})()
        self.base = _FakeGPMoLFormerBase()

    @property
    def base_model(self):
        return self.base


class _FakeMoLFormerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("FakeConfig", (), {"hidden_size": 4})()
        self.scale = nn.Parameter(torch.ones(()))
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, return_dict):
        self.forward_calls += 1
        assert return_dict is True
        hidden = input_ids.to(dtype=self.scale.dtype).unsqueeze(-1)
        hidden = hidden * self.scale
        hidden = hidden.expand(-1, -1, 4)
        return type("FakeOutput", (), {"last_hidden_state": hidden})()


class _FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        assert name == "tokenizer"
        assert kwargs["trust_remote_code"] is True
        return _FakeHuggingFaceTokenizer()


class _FakeAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        assert name == "model"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["deterministic_eval"] is True
        return _FakeGPMoLFormerModel()


class _FakeAutoModel:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        assert name == "encoder-model"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["deterministic_eval"] is True
        return _FakeMoLFormerModel()


def _patch_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        encoder_module,
        "_load_transformers",
        lambda: (
            _FakeAutoModel,
            _FakeAutoModelForCausalLM,
            _FakeAutoTokenizer,
        ),
    )


def test_huggingface_tokenizer_uses_pretrained_ids() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer=_FakeHuggingFaceTokenizer())
    encoded = tokenizer.batch_from_strings(["C[Si]"], max_tokens=8)

    assert isinstance(tokenizer, SequenceTokenizer)
    assert encoded.shape == (1, 8)
    assert int(encoded[0, 0]) == tokenizer.cls_idx
    assert int(encoded[0, -1]) == tokenizer.padding_idx
    assert tokenizer.vocab_size == 12


def test_huggingface_tokenizer_bounds_attention_mask_cache() -> None:
    """Attention-mask entries are evicted after reaching the configured limit."""
    tokenizer = HuggingFaceTokenizer(
        tokenizer=_FakeHuggingFaceTokenizer(),
        cache_size=2,
    )

    for string in ["C", "CC", "CCC", "CCCC", "CCCCC"]:
        tokenizer.batch_from_strings([string], max_tokens=8)
        assert len(tokenizer._attention_masks) <= 2

    assert len(tokenizer._attention_masks) == 2


def test_huggingface_tokenizer_preserves_attention_mask_when_pad_equals_eos() -> None:
    """The EOS position must remain attended when it shares the pad ID."""
    tokenizer = HuggingFaceTokenizer(tokenizer=_FakePadEqualsEosTokenizer())
    encoded = tokenizer.batch_from_strings(["C"], max_tokens=8)

    attention_mask = tokenizer.attention_mask_from_batch(encoded)

    assert attention_mask.tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]


def test_mlm_rejects_tokenizer_without_distinct_mask_token() -> None:
    """MLM must not silently replace tokens with padding."""
    tokenizer = HuggingFaceTokenizer(tokenizer=_FakeNoMaskTokenizer())
    encoder = TransformerSequenceEncoder(
        tokenizer=tokenizer,
        max_tokens=4,
        embed_dim=4,
        ff_dim=8,
        num_heads=2,
        num_layers=1,
        latent_dim=2,
    )

    assert tokenizer.mask_idx is None
    with pytest.raises(ValueError, match="mask token"):
        encoder.mlm_loss(torch.tensor([[2, 5, 1, 2]]), mask_ratio=1.0)


def test_gpmolformer_encoder_freezes_backbone_and_trains_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=2,
        trust_remote_code=True,
    )

    encoder.train()
    token_batch = encoder.tokenizer.batch_from_strings(["C[Si]"], max_tokens=8)
    output = encoder(token_batch)
    output.sum().backward()

    assert output.shape == (1, 2)
    assert encoder.training is True
    assert encoder.backbone.training is False
    assert all(
        not parameter.requires_grad for parameter in encoder.backbone.parameters()
    )
    assert encoder.projection.weight.grad is not None
    assert encoder.backbone.base.scale.grad is None


def test_huggingface_encoder_caches_frozen_backbone_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated token batches should not rerun a frozen backbone."""
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=2,
        trust_remote_code=True,
    )
    token_batch = encoder.tokenizer.batch_from_strings(["C[Si]"], max_tokens=8)

    first = encoder(token_batch)
    second = encoder(token_batch)

    torch.testing.assert_close(first, second)
    assert encoder.backbone.base.forward_calls == 1


def test_huggingface_encoder_cache_opt_out_matches_cached_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling the cache must preserve the encoder's numerical result."""
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=2,
        trust_remote_code=True,
    )
    token_batch = encoder.tokenizer.batch_from_strings(["C[Si]"], max_tokens=8)

    cached = encoder(token_batch)
    encoder.cache_size = 0
    uncached = encoder(token_batch)

    torch.testing.assert_close(cached, uncached)
    assert encoder.backbone.base.forward_calls == 2


def test_huggingface_encoder_reprojects_cached_features_after_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Updating the trainable projection must affect cached-backbone outputs."""
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=2,
        trust_remote_code=True,
    )
    token_batch = encoder.tokenizer.batch_from_strings(["C[Si]"], max_tokens=8)
    optimizer = torch.optim.SGD(encoder.projection.parameters(), lr=0.1)

    first = encoder(token_batch)
    first.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    second = encoder(token_batch)

    assert not torch.equal(first, second)
    assert encoder.backbone.base.forward_calls == 1


def test_gpmolformer_encoder_defaults_to_last_non_padding_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=4,
        trust_remote_code=True,
    )
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(4))
        encoder.projection.bias.zero_()

    output = encoder(
        torch.tensor(
            [
                [5, 6, 2, 2],
                [7, 2, 2, 2],
            ],
            dtype=torch.long,
        )
    )

    expected = torch.tensor(
        [
            [6, 6, 6, 6],
            [7, 7, 7, 7],
        ],
        dtype=output.dtype,
    )
    torch.testing.assert_close(output, expected)


def test_molformer_encoder_uses_direct_backbone_and_mean_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    encoder = MoLFormerSmilesEncoder(
        model_name_or_path="encoder-model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=4,
        trust_remote_code=True,
    )
    assert isinstance(encoder, HuggingFaceSequenceEncoder)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(4))
        encoder.projection.bias.zero_()

    output = encoder(
        torch.tensor(
            [
                [5, 6, 2, 2],
                [7, 8, 9, 2],
            ],
            dtype=torch.long,
        )
    )

    expected = torch.tensor(
        [
            [5.5, 5.5, 5.5, 5.5],
            [8, 8, 8, 8],
        ],
        dtype=output.dtype,
    )
    torch.testing.assert_close(output, expected)
    assert encoder.backbone.training is False
    assert all(
        not parameter.requires_grad for parameter in encoder.backbone.parameters()
    )


def test_gpmolformer_encoder_supports_masked_mean_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=4,
        pooling="mean",
        trust_remote_code=True,
    )
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(4))
        encoder.projection.bias.zero_()

    output = encoder(
        torch.tensor(
            [
                [5, 6, 2, 2],
                [7, 8, 9, 2],
            ],
            dtype=torch.long,
        )
    )

    expected = torch.tensor(
        [
            [5.5, 5.5, 5.5, 5.5],
            [8, 8, 8, 8],
        ],
        dtype=output.dtype,
    )
    torch.testing.assert_close(output, expected)


def test_gpmolformer_encoder_rejects_invalid_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    with pytest.raises(ValueError, match="pooling"):
        GPMoLFormerSmilesEncoder(
            model_name_or_path="model",
            tokenizer_name_or_path="tokenizer",
            pooling="cls",  # type: ignore[arg-type]
            trust_remote_code=True,
        )
    with pytest.raises(ValueError):
        GPMoLFormerSmilesEncoderConfig(pooling="cls")
    with pytest.raises(ValueError):
        MoLFormerSmilesEncoderConfig(pooling="cls")


def test_gpmolformer_encoder_keeps_backbone_dtype_when_runtime_is_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=8,
        latent_dim=2,
        trust_remote_code=True,
    )

    encoder.to(dtype=torch.float64)

    assert encoder.projection.weight.dtype == torch.float64
    assert encoder.backbone.base.scale.dtype == torch.float32


def test_gpmolformer_encoder_works_with_exact_dkl_without_mlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=12,
        latent_dim=2,
        trust_remote_code=True,
    )
    projection_before = encoder.projection.weight.detach().clone()
    backbone_before = encoder.backbone.base.scale.detach().clone()
    surrogate = ExactDKLSurrogate(
        encoder=encoder,
        training_params=DKLTrainingConfig(
            epochs=1,
            pretrain_epochs=2,
            lr=1e-2,
        ),
    )

    surrogate.fit(
        [
            Observation(x="C[Si](C)C", y=1.0),
            Observation(x="[O-][N+](=O)O", y=2.0),
        ]
    )

    assert surrogate.is_fitted()
    assert not torch.equal(encoder.projection.weight, projection_before)
    assert torch.equal(encoder.backbone.base.scale, backbone_before)
    assert all(
        not parameter.requires_grad for parameter in encoder.backbone.parameters()
    )


def test_gpmolformer_encoder_works_with_variational_dkl_without_mlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transformers(monkeypatch)
    encoder = GPMoLFormerSmilesEncoder(
        model_name_or_path="model",
        tokenizer_name_or_path="tokenizer",
        max_mol_tokens=12,
        latent_dim=2,
        trust_remote_code=True,
    )
    projection_before = encoder.projection.weight.detach().clone()
    backbone_before = encoder.backbone.base.scale.detach().clone()
    surrogate = VariationalDKLSurrogate(
        encoder=encoder,
        training_params=DKLTrainingConfig(
            epochs=1,
            pretrain_epochs=2,
            lr=1e-2,
        ),
        num_inducing=2,
    )

    surrogate.fit(
        [
            Observation(x="C[Si](C)C", y=1.0),
            Observation(x="[O-][N+](=O)O", y=2.0),
        ]
    )

    assert surrogate.is_fitted()
    assert not torch.equal(encoder.projection.weight, projection_before)
    assert torch.equal(encoder.backbone.base.scale, backbone_before)
    assert all(
        not parameter.requires_grad for parameter in encoder.backbone.parameters()
    )

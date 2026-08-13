from __future__ import annotations

import pytest
import torch
from torch import nn

import activelearning.applications.molecules.smiles_transformer_encoder as encoder_module
from activelearning.applications.molecules.config import (
    GPMoLFormerSmilesEncoderConfig,
    MoLFormerSmilesEncoderConfig,
)
from activelearning.applications.molecules.input_adapter import (
    MoleculeStringInputAdapter,
)
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.dkl.dkl_surrogate import (
    ExactDKLSurrogate,
    VariationalDKLSurrogate,
)
from activelearning.surrogate.sequence.huggingface import HuggingFaceTokenizer
from activelearning.surrogate.sequence.huggingface_encoder import (
    HuggingFaceSequenceEncoder,
)
from activelearning.surrogate.sequence.tokenizer import SequenceTokenizer
from activelearning.applications.molecules.smiles_transformer_encoder import (
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


class _FakeGPMoLFormerBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
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

    def forward(self, input_ids, attention_mask, return_dict):
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
        input_adapter=MoleculeStringInputAdapter(encoder.tokenizer, encoder.max_tokens),
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
        input_adapter=MoleculeStringInputAdapter(encoder.tokenizer, encoder.max_tokens),
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

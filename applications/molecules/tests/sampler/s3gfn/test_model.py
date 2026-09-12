from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import activelearning_molecules.samplers.s3gfn.model as model_module
from activelearning_molecules.samplers.s3gfn.fidelity import FidelityActionHead
from conftest import FakeTokenizer

_UNSET = object()


class _FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask=None):
        logits = self.weight.expand(input_ids.shape[0], input_ids.shape[1], 4)
        return type("Output", (), {"logits": logits})()


class _FakeFeatureMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.eye(2))

    def orthogonal_random_weights(self, device=None) -> None:
        self.register_buffer(
            "weight",
            torch.eye(2, device=device, dtype=torch.float32),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        self.orthogonal_random_weights(query.device)
        return torch.matmul(query, self.weight)


class _FeatureMapCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(2))
        self.feature_map = _FakeFeatureMap()

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden_states = input_ids.to(self.scale.dtype).unsqueeze(-1).expand(-1, -1, 2)
        logits = self.feature_map(hidden_states) * self.scale
        return type("Output", (), {"logits": logits})()


class _HiddenLanguageModel(nn.Module):
    config = SimpleNamespace(hidden_size=2)

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.forward_calls = 0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
    ):
        self.forward_calls += 1
        logits = self.weight.expand(input_ids.shape[0], input_ids.shape[1], -1)
        hidden_states = (
            input_ids.to(dtype=self.weight.dtype)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                2,
            )
        )
        return type(
            "Output",
            (),
            {
                "logits": logits,
                "hidden_states": (hidden_states,) if output_hidden_states else None,
            },
        )()

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 0], [1, 3, 2]])


class _GenerationTokenizer(FakeTokenizer):
    @staticmethod
    def batch_decode(input_ids, skip_special_tokens=True):
        return ["CC", "CO"]


class MolformerSelfAttention(nn.Module):
    """Small source-inspectable stand-in for the pinned remote attention."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        del past_key_value
        if attention_mask is not None:
            per_query_attn = attention_mask[:, 0, -1]
            per_query_extended = per_query_attn[:, None, None, :]
            if not torch.equal(attention_mask, per_query_extended):
                raise ValueError(model_module._ATTENTION_MASK_ERROR)
            hidden_states = hidden_states * per_query_attn.unsqueeze(-1)
        return hidden_states * self.scale


class _AdapterPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = MolformerSelfAttention()

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        extended_mask = (
            None
            if attention_mask is None
            else attention_mask[:, None, None, :].to(hidden_states.dtype)
        )
        return self.attention(
            hidden_states,
            attention_mask=extended_mask,
            past_key_value=past_key_value,
        )


_ORIGINAL_ADAPTER_FORWARD_CODE = MolformerSelfAttention.forward.__code__
_ORIGINAL_ADAPTER_POLICY_FORWARD_CODE = _AdapterPolicy.forward.__code__


@pytest.fixture(autouse=True)
def restore_adapter_test_forward(monkeypatch):
    """Isolate class-level code replacement between adapter tests."""
    monkeypatch.setattr(
        MolformerSelfAttention.forward,
        "__code__",
        _ORIGINAL_ADAPTER_FORWARD_CODE,
    )
    monkeypatch.delattr(
        MolformerSelfAttention.forward,
        "_s3gfn_attention_mask_adapter",
        raising=False,
    )
    monkeypatch.setattr(
        _AdapterPolicy.forward,
        "__code__",
        _ORIGINAL_ADAPTER_POLICY_FORWARD_CODE,
    )
    monkeypatch.delattr(
        _AdapterPolicy.forward,
        "_s3gfn_attention_mask_validator",
        raising=False,
    )


@pytest.fixture
def make_model():
    """Return a factory building ``S3GFNModel`` from language-model doubles.

    Defaults to the hidden-state double and a two-level fidelity head, which
    is what most terminal-fidelity tests need.
    """

    def build(
        language_model=_HiddenLanguageModel,
        tokenizer=None,
        fidelity_head=_UNSET,
        n_fidelities: int = 2,
    ):
        if fidelity_head is _UNSET:
            fidelity_head = FidelityActionHead(
                hidden_size=2,
                n_fidelities=n_fidelities,
            )
        return model_module.S3GFNModel(
            policy=language_model(),
            prior=language_model(),
            tokenizer=tokenizer or FakeTokenizer(),
            fidelity_head=fidelity_head,
        )

    return build


def test_pretrained_loading_passes_deterministic_eval_to_both_models(monkeypatch):
    model_calls: list[dict] = []

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append({"name": name, **kwargs})
            return _FakeCausalLM()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            assert name == "tokenizer"
            assert kwargs["trust_remote_code"] is True
            return FakeTokenizer()

    monkeypatch.setattr(
        model_module,
        "require_transformers",
        lambda: (_AutoModel, _AutoTokenizer),
    )

    model = model_module.S3GFNModel.from_pretrained(
        policy_model_name_or_path="policy",
        tokenizer_name_or_path="tokenizer",
        trust_remote_code=True,
        deterministic_eval=True,
    )

    assert [call["name"] for call in model_calls] == ["policy", "policy"]
    assert all(call["deterministic_eval"] is True for call in model_calls)
    assert all(call["trust_remote_code"] is True for call in model_calls)
    assert model.prior.training is False
    assert all(not parameter.requires_grad for parameter in model.prior.parameters())


@pytest.mark.parametrize(
    "attention_mask",
    [torch.ones((2, 3)), torch.tensor([[1, 1, 0], [1, 0, 0]])],
)
def test_attention_mask_adapter_preserves_outputs_and_gradients(
    monkeypatch,
    attention_mask,
):
    revision_module = (
        "transformers_modules.ibm-research.GP-MoLFormer-Uniq."
        f"{model_module._SUPPORTED_GP_MOLFORMER_REVISION}.modeling_molformer"
    )
    monkeypatch.setattr(MolformerSelfAttention.forward, "__module__", revision_module)
    eager_policy = _AdapterPolicy()
    adapted_policy = copy.deepcopy(eager_policy)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

    eager_input = hidden_states.clone().requires_grad_()
    eager_output = eager_policy(eager_input, attention_mask=attention_mask)
    eager_output.sum().backward()
    adapted_count = model_module._install_gp_molformer_attention_mask_adapter(
        adapted_policy
    )
    adapted_input = hidden_states.clone().requires_grad_()
    adapted_output = adapted_policy(adapted_input, attention_mask=attention_mask)
    adapted_output.sum().backward()

    assert adapted_count == 1
    assert adapted_policy._s3gfn_attention_mask_adapter_count == 1
    assert adapted_policy.attention._s3gfn_attention_mask_adapter is True
    assert adapted_policy.attention.forward.__globals__["torch"] is torch
    torch.testing.assert_close(adapted_output, eager_output)
    torch.testing.assert_close(adapted_input.grad, eager_input.grad)
    torch.testing.assert_close(
        adapted_policy.attention.scale.grad,
        eager_policy.attention.scale.grad,
    )


def test_attention_mask_adapter_rejects_arbitrary_caller_mask(monkeypatch):
    revision_module = (
        "transformers_modules.ibm-research.GP-MoLFormer-Uniq."
        f"{model_module._SUPPORTED_GP_MOLFORMER_REVISION}.modeling_molformer"
    )
    monkeypatch.setattr(MolformerSelfAttention.forward, "__module__", revision_module)
    policy = _AdapterPolicy()
    model_module._install_gp_molformer_attention_mask_adapter(policy)

    with pytest.raises(ValueError, match="does not support arbitrary 3D attention"):
        policy(torch.ones((1, 2, 2)), attention_mask=torch.ones((1, 2, 2)))


def test_attention_mask_adapter_rejects_unknown_revision(monkeypatch):
    monkeypatch.setattr(
        MolformerSelfAttention.forward,
        "__module__",
        "transformers_modules.unknown.modeling_molformer",
    )

    with pytest.raises(RuntimeError, match="Unsupported GP-MoLFormer revision"):
        model_module._install_gp_molformer_attention_mask_adapter(_AdapterPolicy())


def test_pretrained_loading_requests_deterministic_eval_by_default(monkeypatch):
    """Both models must pin deterministic eval so likelihoods are reproducible.

    GP-MoLFormer's checkpoint config sets ``deterministic_eval`` to ``False``,
    which redraws the linear-attention random features on every forward pass.
    The frozen prior is always in eval mode, so omitting the flag would give
    the same molecule a different prior likelihood on each call.
    """
    model_calls: list[dict] = []

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append({"name": name, **kwargs})
            return _FakeCausalLM()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return FakeTokenizer()

    monkeypatch.setattr(
        model_module,
        "require_transformers",
        lambda: (_AutoModel, _AutoTokenizer),
    )

    model_module.S3GFNModel.from_pretrained(
        policy_model_name_or_path="policy",
        tokenizer_name_or_path="tokenizer",
    )

    assert model_calls
    assert all(call["deterministic_eval"] is True for call in model_calls)


def test_pretrained_loading_omits_deterministic_eval_when_explicitly_none(monkeypatch):
    """``None`` is the escape hatch for checkpoints without the custom flag."""
    model_calls: list[dict] = []

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append({"name": name, **kwargs})
            return _FakeCausalLM()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return FakeTokenizer()

    monkeypatch.setattr(
        model_module,
        "require_transformers",
        lambda: (_AutoModel, _AutoTokenizer),
    )

    model_module.S3GFNModel.from_pretrained(
        policy_model_name_or_path="policy",
        tokenizer_name_or_path="tokenizer",
        deterministic_eval=None,
    )

    assert model_calls
    assert all("deterministic_eval" not in call for call in model_calls)


def test_pretrained_loading_preserves_bfloat16_feature_map_redraw(monkeypatch):
    """BF16 loading must avoid CPU QR and preserve redraw tensor dtypes."""
    model_calls: list[dict] = []

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append({"name": name, **kwargs})
            return _FeatureMapCausalLM()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            del name, kwargs
            return FakeTokenizer()

    monkeypatch.setattr(
        model_module,
        "require_transformers",
        lambda: (_AutoModel, _AutoTokenizer),
    )

    model = model_module.S3GFNModel.from_pretrained(
        policy_model_name_or_path="policy",
        tokenizer_name_or_path="tokenizer",
        dtype=torch.bfloat16,
    )
    model.policy.train()

    output = model.policy(torch.ones((1, 3), dtype=torch.long))

    assert all("torch_dtype" not in call for call in model_calls)
    assert output.logits.dtype is torch.bfloat16
    assert next(model.policy.parameters()).dtype is torch.bfloat16
    assert next(model.prior.parameters()).dtype is torch.bfloat16
    assert model.policy.feature_map.weight.dtype is torch.bfloat16
    assert model.prior.feature_map.weight.dtype is torch.bfloat16
    assert model.log_z.dtype is torch.float32


def test_feature_map_redraw_dtype_adapter_is_idempotent():
    """Repeated model wrapping must not stack redraw adapters."""
    feature_map = _FakeFeatureMap()
    model = nn.Sequential(feature_map)

    model_module._preserve_feature_map_redraw_dtype(model)
    wrapped_redraw = feature_map.orthogonal_random_weights.__func__
    model_module._preserve_feature_map_redraw_dtype(model)

    assert feature_map.orthogonal_random_weights.__func__ is wrapped_redraw

    plain_model = _FakeCausalLM()
    model_module._preserve_feature_map_redraw_dtype(plain_model)
    assert not hasattr(plain_model, "_s3gfn_dtype_safe_redraw")


def test_feature_map_redraw_dtype_adapter_clones_redrawn_weight():
    """Redrawn weights must not alias storage owned by a compiled graph."""

    class _AliasingFeatureMap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", torch.eye(2))
            self.source = torch.ones((2, 2))

        def orthogonal_random_weights(self, device=None) -> None:
            self.weight = self.source.to(device=device)

    feature_map = _AliasingFeatureMap()
    model_module._preserve_feature_map_redraw_dtype(feature_map)

    feature_map.orthogonal_random_weights()

    assert feature_map.weight.data_ptr() != feature_map.source.data_ptr()
    assert torch.equal(feature_map.weight, feature_map.source)


def test_feature_map_redraw_dtype_adapter_survives_policy_deepcopy():
    """A fresh round policy must redraw its own BF16 feature-map weights."""
    model = _FeatureMapCausalLM().to(dtype=torch.bfloat16)
    model_module._preserve_feature_map_redraw_dtype(model)

    copied_model = copy.deepcopy(model)
    copied_model.feature_map.orthogonal_random_weights()

    assert copied_model.feature_map.weight.dtype is torch.bfloat16
    assert copied_model.feature_map.weight.data_ptr() != (
        model.feature_map.weight.data_ptr()
    )


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="TorchInductor C++ compilation requires a configured macOS C++ SDK",
)
def test_compiled_bfloat16_feature_map_redraw_keeps_matching_dtypes():
    """Compiled BF16 policy forwards must survive training-time redraws."""
    model = model_module.S3GFNModel(
        policy=_FeatureMapCausalLM(),
        prior=_FeatureMapCausalLM(),
        tokenizer=FakeTokenizer(),
    ).to(dtype=torch.bfloat16)
    model.policy.train()
    model.compile_policy()

    output = model.policy(torch.ones((1, 3), dtype=torch.long))

    assert output.logits.dtype is torch.bfloat16
    assert model.policy.feature_map.weight.dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("dtype", "expected_options"),
    [
        (
            torch.float32,
            {"max_autotune": True, "triton.cudagraphs": False},
        ),
        (
            torch.bfloat16,
            {
                "max_autotune": True,
                "triton.cudagraphs": False,
                "max_autotune_gemm_backends": "ATEN",
            },
        ),
    ],
)
def test_max_autotune_compilation_uses_safe_options(
    monkeypatch,
    dtype,
    expected_options,
):
    """Max-autotune must avoid unsafe CUDA graphs and BF16 Triton BMM."""
    compile_arguments = {}

    def fake_compile(forward, **kwargs):
        compile_arguments.update(kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = model_module.S3GFNModel(
        policy=_FakeCausalLM(),
        prior=_FakeCausalLM(),
        tokenizer=FakeTokenizer(),
    ).to(dtype=dtype)

    model.compile_policy(mode="max-autotune")

    assert compile_arguments == {"dynamic": True, "options": expected_options}


def test_compile_policy_forwards_dynamic_shape_setting(monkeypatch):
    """The model API must preserve an explicit dynamic-shape experiment."""
    compile_arguments = {}

    def fake_compile(forward, **kwargs):
        compile_arguments.update(kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = model_module.S3GFNModel(
        policy=_FakeCausalLM(),
        prior=_FakeCausalLM(),
        tokenizer=FakeTokenizer(),
    )

    model.compile_policy(dynamic=None)

    assert compile_arguments["dynamic"] is None


def test_training_only_compilation_keeps_generation_forward_eager(monkeypatch):
    """Training-only compilation must leave Hugging Face generation eager."""
    compiled_forward = object()

    def fake_compile(forward, **kwargs):
        return compiled_forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = model_module.S3GFNModel(
        policy=_FakeCausalLM(),
        prior=_FakeCausalLM(),
        tokenizer=FakeTokenizer(),
    )
    eager_forward = model.policy.forward

    model.compile_policy(training_only=True)

    assert model.policy.forward.__func__ is eager_forward.__func__
    assert model._compiled_policy_forward is compiled_forward


def test_training_only_compilation_covers_terminal_fidelity_likelihoods(
    monkeypatch,
    make_model,
):
    """Multi-fidelity training must use the compiled policy forward."""
    calls: list[dict[str, object]] = []

    def fake_compile(forward, **kwargs):
        del kwargs

        def compiled_forward(*args, **forward_kwargs):
            calls.append(forward_kwargs)
            return forward(*args, **forward_kwargs)

        return compiled_forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = make_model()
    model.compile_policy(training_only=True)

    model.policy_trajectory_log_probabilities(
        torch.tensor([[1, 2, 0], [1, 3, 2]]),
        fidelity_indices=torch.tensor([0, 1]),
    )

    assert len(calls) == 1
    assert torch.equal(calls[0]["input_ids"], torch.tensor([[1, 2, 0], [1, 3, 2]]))
    assert torch.equal(
        calls[0]["attention_mask"],
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )
    assert calls[0]["output_hidden_states"] is True


def test_training_only_compilation_preserves_fixed_trajectory_values_and_gradients(
    monkeypatch,
    make_model,
) -> None:
    """Compiled dispatch must preserve trajectory values and policy gradients."""
    eager_model = make_model(language_model=_CausalPrefixLanguageModel)
    compiled_model = make_model(language_model=_CausalPrefixLanguageModel)
    compiled_model.load_state_dict(eager_model.state_dict())
    monkeypatch.setattr(torch, "compile", lambda forward, **kwargs: forward)
    compiled_model.compile_policy(training_only=True)
    input_ids = torch.tensor([[1, 3, 2, 0], [1, 1, 3, 2]])
    fidelity_indices = torch.tensor([0, 1])

    eager_values = eager_model.policy_trajectory_log_probabilities(
        input_ids,
        fidelity_indices=fidelity_indices,
    )
    compiled_values = compiled_model.policy_trajectory_log_probabilities(
        input_ids,
        fidelity_indices=fidelity_indices,
    )
    eager_values.sum().backward()
    compiled_values.sum().backward()

    assert torch.allclose(compiled_values, eager_values)
    eager_gradients = {
        name: parameter.grad
        for name, parameter in eager_model.named_parameters()
        if parameter.grad is not None
    }
    compiled_gradients = {
        name: parameter.grad
        for name, parameter in compiled_model.named_parameters()
        if parameter.grad is not None
    }
    assert eager_gradients.keys() == compiled_gradients.keys()
    for name, eager_gradient in eager_gradients.items():
        assert torch.allclose(compiled_gradients[name], eager_gradient)


def test_compiled_prior_scorer_preserves_values_detachment_and_ownership(
    monkeypatch,
    make_model,
) -> None:
    """Compiled prior scoring must stay frozen and preserve state ownership."""
    eager_model = make_model(language_model=_CausalPrefixLanguageModel)
    compiled_model = make_model(language_model=_CausalPrefixLanguageModel)
    compiled_model.load_state_dict(eager_model.state_dict())
    original_state_keys = tuple(compiled_model.state_dict())
    monkeypatch.setattr(torch, "compile", lambda callable_, **kwargs: callable_)
    compiled_model.compile_prior_scorer()
    input_ids = torch.tensor([[1, 3, 2, 0], [1, 1, 3, 2]])

    eager_values = eager_model.prior_sequence_log_probabilities(input_ids)
    compiled_values = compiled_model.prior_sequence_log_probabilities(input_ids)

    assert torch.allclose(compiled_values, eager_values)
    assert compiled_values.requires_grad is False
    assert all(
        parameter.grad is None for parameter in compiled_model.prior.parameters()
    )
    assert all(
        parameter.requires_grad is False
        for parameter in compiled_model.prior.parameters()
    )
    assert tuple(compiled_model.state_dict()) == original_state_keys


def test_model_rejects_shared_pad_and_eos_ids() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {"pad_token_id": 2, "eos_token_id": 2, "padding_side": "right"},
    )()

    with pytest.raises(ValueError, match="distinct pad and EOS"):
        model_module.S3GFNModel(
            policy=_FakeCausalLM(),
            prior=_FakeCausalLM(),
            tokenizer=tokenizer,
        )


def test_model_rejects_left_padding() -> None:
    tokenizer = type(
        "Tokenizer",
        (),
        {"pad_token_id": 0, "eos_token_id": 2, "padding_side": "left"},
    )()

    with pytest.raises(ValueError, match="right-side padding"):
        model_module.S3GFNModel(
            policy=_FakeCausalLM(),
            prior=_FakeCausalLM(),
            tokenizer=tokenizer,
        )


def test_joint_trajectory_probabilities_include_terminal_fidelity_action(
    make_model,
) -> None:
    model = make_model()
    input_ids = torch.tensor([[1, 2, 0], [1, 3, 2]])
    fidelity_indices = torch.tensor([0, 1])

    policy_sequence = model.policy_sequence_log_probabilities(input_ids)
    prior_sequence = model.prior_sequence_log_probabilities(input_ids)
    model.policy.forward_calls = 0
    policy_trajectory = model.policy_trajectory_log_probabilities(
        input_ids,
        fidelity_indices,
    )
    prior_trajectory = model.prior_trajectory_log_probabilities(
        input_ids,
        fidelity_indices,
    )
    expected_action_log_probability = torch.full_like(
        policy_sequence,
        -torch.log(torch.tensor(2.0)),
    )

    assert torch.allclose(
        policy_trajectory,
        policy_sequence + expected_action_log_probability,
    )
    assert torch.allclose(
        prior_trajectory,
        prior_sequence + expected_action_log_probability,
    )
    assert model.policy.forward_calls == 1


def test_joint_trajectory_probabilities_select_the_terminal_hidden_state(
    make_model,
) -> None:
    """A trained head must read the last real token, not a padding column."""
    head = FidelityActionHead(hidden_size=2, n_fidelities=2)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[0.5, -0.25], [-0.75, 1.5]]))
        head.projection.bias.copy_(torch.tensor([0.3, -0.2]))
    model = make_model(fidelity_head=head)
    # ``_HiddenLanguageModel`` emits each token id broadcast across the hidden
    # dimension, so the terminal rows are ``[3, 3]`` and ``[2, 2]``. Both differ
    # from the padding column, making this sensitive to the terminal index.
    input_ids = torch.tensor([[1, 3, 0, 0], [1, 1, 2, 0]])
    fidelity_indices = torch.tensor([0, 1])

    policy_sequence = model.policy_sequence_log_probabilities(input_ids)
    policy_trajectory = model.policy_trajectory_log_probabilities(
        input_ids,
        fidelity_indices,
    )

    terminal_hidden_states = torch.tensor([[3.0, 3.0], [2.0, 2.0]])
    expected_action_log_probability = (
        torch.log_softmax(
            head.projection(terminal_hidden_states),
            dim=-1,
        )
        .gather(1, fidelity_indices.unsqueeze(-1))
        .squeeze(-1)
    )

    assert not torch.allclose(
        expected_action_log_probability,
        torch.full_like(expected_action_log_probability, -torch.log(torch.tensor(2.0))),
    )
    assert torch.allclose(
        policy_trajectory,
        policy_sequence + expected_action_log_probability,
    )


def test_generate_returns_terminal_fidelity_actions(make_model) -> None:
    model = make_model(tokenizer=_GenerationTokenizer())

    generated = model.generate(count=2, max_length=4)

    assert generated.fidelity_indices is not None
    assert generated.fidelity_indices.shape == (2,)
    assert torch.all(
        (generated.fidelity_indices >= 0) & (generated.fidelity_indices < 2)
    )


def test_joint_rtb_loss_backpropagates_through_fidelity_action_head(make_model) -> None:
    model = make_model()

    loss = model.on_policy_loss(
        positive_input_ids=torch.tensor([[1, 2, 0], [1, 3, 2]]),
        reward_scores=torch.tensor([0.1, 0.2]),
        beta=2.0,
        fidelity_indices=torch.tensor([0, 1]),
    )

    assert loss is not None
    loss.backward()
    assert model.fidelity_head.projection.weight.grad is not None


@pytest.mark.parametrize(
    ("negative_input_ids", "aux_coefficient"),
    [
        pytest.param(torch.tensor([[1, 1, 2]]), 0.0, id="coefficient-disabled"),
        pytest.param(None, 1.0, id="no-negative-batch"),
        pytest.param(
            torch.zeros((0, 3), dtype=torch.long),
            1.0,
            id="empty-negative-batch",
        ),
    ],
)
def test_replay_loss_reports_no_auxiliary_loss_when_it_is_not_computed(
    make_model,
    negative_input_ids,
    aux_coefficient,
) -> None:
    """An unused auxiliary term must read as absent, not as a measured zero."""
    model = make_model(language_model=_FakeCausalLM, fidelity_head=None)

    model.replay_loss(
        positive_input_ids=torch.tensor([[1, 3, 2], [1, 3, 2]]),
        reward_scores=torch.tensor([0.1, 0.2]),
        beta=2.0,
        negative_input_ids=negative_input_ids,
        aux_coefficient=aux_coefficient,
    )

    assert model.last_auxiliary_loss is None


def test_replay_loss_reports_auxiliary_loss_when_it_is_computed(make_model) -> None:
    """The auxiliary term is reported whenever the contrastive branch runs."""
    model = make_model(language_model=_FakeCausalLM, fidelity_head=None)

    model.replay_loss(
        positive_input_ids=torch.tensor([[1, 3, 2], [1, 3, 2]]),
        reward_scores=torch.tensor([0.1, 0.2]),
        beta=2.0,
        negative_input_ids=torch.tensor([[1, 1, 2]]),
        aux_coefficient=1.0,
    )

    assert model.last_auxiliary_loss is not None
    assert model.last_auxiliary_loss > 0.0


class _CausalPrefixLanguageModel(nn.Module):
    """Fake causal LM whose logits depend on the scored prefix.

    Position ``i`` is scored only from tokens ``0..i``, mirroring causal
    masking. Unlike the other fakes in this module the per-position
    distributions genuinely differ, so scoring the full sequence and slicing
    is only equivalent to scoring a truncated sequence if causality holds.
    """

    config = SimpleNamespace(hidden_size=2)

    def __init__(self, vocabulary_size: int = 4) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.vocabulary_size = vocabulary_size

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix = (input_ids * attention_mask).cumsum(dim=1).to(self.scale.dtype)
        vocabulary = torch.arange(self.vocabulary_size, dtype=self.scale.dtype)
        logits = self.scale * prefix.unsqueeze(-1) * vocabulary * 0.25
        hidden_states = input_ids.to(self.scale.dtype).unsqueeze(-1).expand(-1, -1, 2)
        return type(
            "Output",
            (),
            {
                "logits": logits,
                "hidden_states": (hidden_states,) if output_hidden_states else None,
            },
        )()


def test_fused_scoring_matches_the_standalone_sequence_scorer(make_model) -> None:
    """The single-pass path must agree with the truncated-forward scorer.

    The fused path feeds the full sequence to obtain a terminal hidden state
    and slices the logits, while ``policy_sequence_log_probabilities`` feeds a
    truncated sequence. They are equivalent only because causal masking makes
    the scored positions independent of the final token, so that invariant is
    asserted directly rather than left to the out-of-tree parity harness.
    """
    model = make_model(language_model=_CausalPrefixLanguageModel)
    input_ids = torch.tensor([[1, 3, 2, 0], [1, 1, 3, 2], [1, 2, 0, 0]])

    fused, _ = model._policy_sequence_log_probabilities_and_terminal_hidden_states(
        input_ids
    )
    standalone = model.policy_sequence_log_probabilities(input_ids)

    # Guard against a degenerate fake that would make the comparison vacuous.
    assert not torch.allclose(standalone, torch.zeros_like(standalone))
    assert torch.allclose(fused, standalone)


def test_terminal_hidden_states_require_the_policy_to_expose_them(make_model) -> None:
    """A policy without hidden states must fail loudly, not silently."""

    class _NoHiddenStateModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
            logits = self.weight.expand(input_ids.shape[0], input_ids.shape[1], -1)
            return type("Output", (), {"logits": logits, "hidden_states": None})()

    model = make_model(language_model=_NoHiddenStateModel, fidelity_head=None)

    with pytest.raises(ValueError, match="must return hidden states"):
        model._terminal_hidden_states(torch.tensor([[1, 3, 2]]))


def test_terminal_hidden_states_reject_rows_without_real_tokens(make_model) -> None:
    """An all-padding row has no terminal token to select."""
    model = make_model(fidelity_head=None)

    with pytest.raises(ValueError, match="at least one non-padding token"):
        model._terminal_hidden_states(torch.tensor([[1, 3, 2], [0, 0, 0]]))

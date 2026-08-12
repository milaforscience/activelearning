from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import activelearning.sampler.s3gfn.model as model_module
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask=None):
        logits = self.weight.expand(input_ids.shape[0], input_ids.shape[1], 4)
        return type("Output", (), {"logits": logits})()


class _HiddenLanguageModel(nn.Module):
    config = SimpleNamespace(hidden_size=2)

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
    ):
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


class _GenerationTokenizer(_FakeTokenizer):
    @staticmethod
    def batch_decode(input_ids, skip_special_tokens=True):
        return ["CC", "CO"]


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
            return _FakeTokenizer()

    monkeypatch.setattr(
        model_module,
        "require_transformers",
        lambda: (_AutoModel, _AutoTokenizer),
    )

    model = model_module.S3GFNModel.from_pretrained(
        policy_model_name_or_path="policy",
        tokenizer_name_or_path="tokenizer",
        trust_remote_code=True,
    )

    assert [call["name"] for call in model_calls] == ["policy", "policy"]
    assert all(call["deterministic_eval"] is True for call in model_calls)
    assert all(call["trust_remote_code"] is True for call in model_calls)
    assert model.prior.training is False
    assert all(not parameter.requires_grad for parameter in model.prior.parameters())


def test_joint_trajectory_probabilities_include_terminal_fidelity_action() -> None:
    model = model_module.S3GFNModel(
        policy=_HiddenLanguageModel(),
        prior=_HiddenLanguageModel(),
        tokenizer=_FakeTokenizer(),
        fidelity_head=FidelityActionHead(hidden_size=2, n_fidelities=2),
    )
    input_ids = torch.tensor([[1, 2, 0], [1, 3, 2]])
    fidelity_indices = torch.tensor([0, 1])

    policy_sequence = model.policy_sequence_log_probabilities(input_ids)
    prior_sequence = model.prior_sequence_log_probabilities(input_ids)
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


def test_generate_returns_terminal_fidelity_actions() -> None:
    model = model_module.S3GFNModel(
        policy=_HiddenLanguageModel(),
        prior=_HiddenLanguageModel(),
        tokenizer=_GenerationTokenizer(),
        fidelity_head=FidelityActionHead(hidden_size=2, n_fidelities=2),
    )

    generated = model.generate(count=2, max_length=4)

    assert generated.fidelity_indices is not None
    assert generated.fidelity_indices.shape == (2,)
    assert torch.all(
        (generated.fidelity_indices >= 0) & (generated.fidelity_indices < 2)
    )


def test_joint_rtb_loss_backpropagates_through_fidelity_action_head() -> None:
    model = model_module.S3GFNModel(
        policy=_HiddenLanguageModel(),
        prior=_HiddenLanguageModel(),
        tokenizer=_FakeTokenizer(),
        fidelity_head=FidelityActionHead(hidden_size=2, n_fidelities=2),
    )

    loss = model.on_policy_loss(
        positive_input_ids=torch.tensor([[1, 2, 0], [1, 3, 2]]),
        reward_scores=torch.tensor([0.1, 0.2]),
        beta=2.0,
        fidelity_indices=torch.tensor([0, 1]),
    )

    assert loss is not None
    loss.backward()
    assert model.fidelity_head.projection.weight.grad is not None

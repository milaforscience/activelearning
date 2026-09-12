from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import activelearning.sampler.s3gfn.model as model_module
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead
from tests.sampler.s3gfn.conftest import FakeTokenizer

_UNSET = object()


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

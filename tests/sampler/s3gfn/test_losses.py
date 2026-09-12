from __future__ import annotations

import torch
from torch import nn
import pytest

from activelearning.sampler.s3gfn.losses import (
    negative_replay_contrastive_loss,
    relative_trajectory_balance_loss,
    sequence_log_probabilities,
    sequence_log_probabilities_from_logits,
)
from activelearning.sampler.s3gfn.model import S3GFNModel
from tests.sampler.s3gfn.conftest import FakeTokenizer


class _FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        batch, sequence_length = input_ids.shape
        logits = self.logits[:sequence_length].unsqueeze(0).expand(batch, -1, -1)
        return type("Output", (), {"logits": logits})()


class _CountingLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int = 8) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(vocabulary_size))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        self.forward_calls += 1
        logits = self.bias.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return type("Output", (), {"logits": logits})()


def test_sequence_log_probabilities_include_eos_but_mask_padding() -> None:
    logits = torch.zeros(3, 4)
    logits[0, 2] = 2.0
    logits[1, 0] = 2.0
    logits[2, 0] = 2.0
    model = _FixedLogitModel(logits)
    input_ids = torch.tensor([[1, 2, 0, 0], [1, 2, 3, 0]])

    result = sequence_log_probabilities(model, input_ids, pad_token_id=0)

    expected_first = torch.log_softmax(logits[0], dim=-1)[2]
    expected_second = (
        torch.log_softmax(logits[0], dim=-1)[2]
        + torch.log_softmax(logits[1], dim=-1)[3]
    )
    assert torch.allclose(result, torch.stack([expected_first, expected_second]))


def test_rtb_uses_beta_scaled_reward_scores() -> None:
    policy = torch.tensor([1.0, 2.0], requires_grad=True)
    prior = torch.tensor([0.5, 1.0])
    reward_scores = torch.tensor([0.1, 0.2])
    log_z = torch.tensor(0.3, requires_grad=True)

    result = relative_trajectory_balance_loss(
        policy,
        prior,
        reward_scores,
        log_z,
        beta=2.0,
    )
    expected = (((0.3 + 1.0 - 0.5 - 0.2) ** 2) + ((0.3 + 2.0 - 1.0 - 0.4) ** 2)) / 2
    assert result.item() == pytest.approx(expected)


def test_negative_replay_loss_matches_upstream_normalized_formula() -> None:
    positive = torch.tensor([-0.3, -1.1])
    negatives = torch.tensor([-0.2, -0.7, -1.4])
    negative_log_mass = torch.logsumexp(negatives, dim=0) - torch.log(
        torch.tensor(float(negatives.numel()))
    )
    expected = (torch.logaddexp(positive, negative_log_mass) - positive).mean()

    assert negative_replay_contrastive_loss(positive, negatives) == pytest.approx(
        expected.item()
    )


def test_zero_auxiliary_coefficient_skips_negative_forward() -> None:
    policy = _CountingLanguageModel()
    prior = _CountingLanguageModel()
    model = S3GFNModel(policy=policy, prior=prior, tokenizer=FakeTokenizer())
    input_ids = torch.tensor([[1, 2, 0], [1, 3, 2]])

    output = model.replay_loss(
        positive_input_ids=input_ids[:1],
        reward_scores=torch.tensor([0.2]),
        beta=2.0,
        negative_input_ids=input_ids[1:],
        aux_coefficient=0.0,
    )

    assert output is not None
    assert policy.forward_calls == 1


def test_sequence_log_probabilities_from_logits_masks_padding_and_counts_eos() -> None:
    """The shared scoring core sums non-padding labels only."""
    shifted_logits = torch.zeros(1, 3, 4)
    shifted_logits[0, 0, 2] = 2.0
    shifted_logits[0, 1, 3] = 2.0
    shifted_logits[0, 2, 0] = 2.0
    labels = torch.tensor([[2, 3, 0]])

    result = sequence_log_probabilities_from_logits(
        shifted_logits,
        labels=labels,
        pad_token_id=0,
    )

    expected = (
        torch.log_softmax(shifted_logits[0, 0], dim=-1)[2]
        + torch.log_softmax(shifted_logits[0, 1], dim=-1)[3]
    )
    assert torch.allclose(result, expected.reshape(1))


def test_sequence_log_probabilities_from_logits_rejects_misaligned_logits() -> None:
    """Misaligned logits must raise rather than score the wrong positions."""
    with pytest.raises(ValueError, match="must align with the shifted sequence"):
        sequence_log_probabilities_from_logits(
            torch.zeros(1, 2, 4),
            labels=torch.tensor([[2, 3, 1]]),
            pad_token_id=0,
        )

from __future__ import annotations

import torch
from torch import nn
import pytest

from activelearning.sampler.s3gfn.losses import (
    relative_trajectory_balance_loss,
    sequence_log_probabilities,
    summed_negative_infonce_loss,
)
from activelearning.sampler.s3gfn.model import S3GFNModel


class _FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        batch, sequence_length = input_ids.shape
        logits = self.logits[:sequence_length].unsqueeze(0).expand(batch, -1, -1)
        return type("Output", (), {"logits": logits})()


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


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


def test_negative_infonce_matches_normalized_upstream_objective() -> None:
    positive = torch.tensor([0.0, 0.0])
    negatives = torch.tensor([0.0, 0.0])

    result = summed_negative_infonce_loss(positive, negatives)

    assert result.item() == torch.log(torch.tensor(2.0)).item()


def test_zero_auxiliary_coefficient_skips_negative_forward() -> None:
    policy = _CountingLanguageModel()
    prior = _CountingLanguageModel()
    model = S3GFNModel(policy=policy, prior=prior, tokenizer=_Tokenizer())
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

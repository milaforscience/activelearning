from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from benchmarks.optimized_s3gfn_benchmark import _build_models
from activelearning.sampler.optimized_s3gfn.model import OptimizedS3GFNModel
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead
from activelearning.sampler.s3gfn.model import S3GFNModel
from tests.sampler.s3gfn.conftest import FakeTokenizer


class _CountingLanguageModel(nn.Module):
    """Small causal model used to count frozen-prior forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        logits = self.weight.expand(input_ids.shape[0], input_ids.shape[1], 4)
        return SimpleNamespace(logits=logits)


class _CachedGenerationLanguageModel(nn.Module):
    """Causal model exposing tuple KV caches through a tiny HF-like generator."""

    config = SimpleNamespace(hidden_size=2)

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.forward_shapes: list[tuple[int, int]] = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        past_key_values=None,
        use_cache=None,
        return_dict=True,
    ):
        del attention_mask, use_cache, return_dict
        self.forward_shapes.append(tuple(input_ids.shape))
        batch_size, sequence_length = input_ids.shape
        hidden_states = input_ids.to(self.weight.dtype).unsqueeze(-1).expand(-1, -1, 2)
        logits = self.weight.expand(batch_size, sequence_length, 4)
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        key = torch.zeros(
            (batch_size, 1, past_length + sequence_length, 1),
            device=input_ids.device,
        )
        value = key.clone()
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            past_key_values=((key, value),),
        )

    def generate(
        self,
        *,
        max_length,
        num_return_sequences,
        pad_token_id,
        eos_token_id,
        **kwargs,
    ):
        del kwargs
        sequences = torch.ones((num_return_sequences, 1), dtype=torch.long)
        past_key_values = None
        finished = torch.zeros(num_return_sequences, dtype=torch.bool)
        step = 0
        while sequences.shape[1] < max_length:
            outputs = self(
                input_ids=sequences[:, -1:],
                attention_mask=torch.ones_like(sequences),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            if step == 0:
                next_tokens = torch.tensor([eos_token_id, 3])[:num_return_sequences]
            else:
                next_tokens = torch.full(
                    (num_return_sequences,),
                    eos_token_id,
                    dtype=torch.long,
                )
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )
            sequences = torch.cat((sequences, next_tokens[:, None]), dim=1)
            finished |= next_tokens.eq(eos_token_id)
            if bool(finished.all()):
                break
            step += 1
        return sequences


class _GenerationTokenizer(FakeTokenizer):
    @staticmethod
    def batch_decode(input_ids, skip_special_tokens=True):
        del skip_special_tokens
        return ["CC"] * input_ids.shape[0]


def _build_prior_model(
    *, cache_enabled: bool
) -> tuple[OptimizedS3GFNModel, _CountingLanguageModel]:
    prior = _CountingLanguageModel()
    model = OptimizedS3GFNModel(
        policy=_CountingLanguageModel(),
        prior=prior,
        tokenizer=FakeTokenizer(),
        fidelity_head=None,
        prior_cache_enabled=cache_enabled,
    )
    return model, prior


def test_prior_cache_deduplicates_and_ignores_padding_width() -> None:
    model, prior = _build_prior_model(cache_enabled=True)
    first_batch = torch.tensor([[1, 2, 0], [1, 2, 0], [1, 3, 2]])
    second_batch = torch.tensor([[1, 2, 0, 0], [1, 3, 2, 0]])

    first_values = model.prior_sequence_log_probabilities(first_batch)
    second_values = model.prior_sequence_log_probabilities(second_batch)

    assert prior.forward_calls == 1
    assert model.prior_cache_hits == 3
    assert model.prior_cache_misses == 2
    assert torch.allclose(first_values[[0, 2]], second_values)


def test_prior_cache_does_not_collapse_internal_padding() -> None:
    model, prior = _build_prior_model(cache_enabled=True)

    model.prior_sequence_log_probabilities(torch.tensor([[1, 0, 2, 0]]))
    model.prior_sequence_log_probabilities(torch.tensor([[1, 2, 0]]))

    assert prior.forward_calls == 2
    assert model.prior_cache_hits == 0
    assert model.prior_cache_misses == 2


def test_prior_cache_can_be_disabled_for_nondeterministic_models() -> None:
    model, prior = _build_prior_model(cache_enabled=False)
    input_ids = torch.tensor([[1, 2, 0]])

    model.prior_sequence_log_probabilities(input_ids)
    model.prior_sequence_log_probabilities(input_ids)

    assert prior.forward_calls == 2
    assert model.prior_cache_hits == 0
    assert model.prior_cache_misses == 0


def test_prior_cache_is_invalidated_when_model_dtype_changes() -> None:
    model, prior = _build_prior_model(cache_enabled=True)
    input_ids = torch.tensor([[1, 2, 0]])

    model.prior_sequence_log_probabilities(input_ids)
    model.to(dtype=torch.float64)
    values = model.prior_sequence_log_probabilities(input_ids)

    assert prior.forward_calls == 2
    assert values.dtype == torch.float64


def test_prior_cache_uses_least_recently_used_eviction() -> None:
    prior = _CountingLanguageModel()
    model = OptimizedS3GFNModel(
        policy=_CountingLanguageModel(),
        prior=prior,
        tokenizer=FakeTokenizer(),
        fidelity_head=None,
        prior_cache_capacity=2,
    )

    model.prior_sequence_log_probabilities(torch.tensor([[1, 2, 0], [1, 3, 0]]))
    model.prior_sequence_log_probabilities(torch.tensor([[1, 2, 0]]))
    model.prior_sequence_log_probabilities(torch.tensor([[2, 3, 0]]))
    model.prior_sequence_log_probabilities(torch.tensor([[1, 3, 0]]))

    assert prior.forward_calls == 3


def test_prior_cache_returns_a_batch_larger_than_its_capacity() -> None:
    model, prior = _build_prior_model(cache_enabled=True)
    model.prior_cache_capacity = 1

    values = model.prior_sequence_log_probabilities(
        torch.tensor([[1, 2, 0], [1, 3, 0]])
    )

    assert values.shape == (2,)
    assert prior.forward_calls == 1


def test_generation_capture_matches_reference_and_avoids_full_terminal_forward() -> (
    None
):
    tokenizer = _GenerationTokenizer()
    head = FidelityActionHead(hidden_size=2, n_fidelities=2)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    reference_policy = _CachedGenerationLanguageModel()
    reference = S3GFNModel(
        policy=reference_policy,
        prior=_CountingLanguageModel(),
        tokenizer=tokenizer,
        fidelity_head=deepcopy(head),
    )
    optimized_policy = _CachedGenerationLanguageModel()
    optimized = OptimizedS3GFNModel(
        policy=optimized_policy,
        prior=_CountingLanguageModel(),
        tokenizer=tokenizer,
        fidelity_head=deepcopy(head),
    )
    reference.policy.eval()
    optimized.policy.eval()

    torch.manual_seed(11)
    reference_result = reference.generate(count=2, max_length=4)
    torch.manual_seed(11)
    optimized_result = optimized.generate(count=2, max_length=4)

    assert torch.equal(reference_result.input_ids, optimized_result.input_ids)
    assert torch.equal(
        reference_result.fidelity_indices,
        optimized_result.fidelity_indices,
    )
    assert (2, 3) in reference_policy.forward_shapes
    assert (2, 3) not in optimized_policy.forward_shapes


def test_training_mode_keeps_reference_terminal_forward() -> None:
    tokenizer = _GenerationTokenizer()
    head = FidelityActionHead(hidden_size=2, n_fidelities=2)
    reference_policy = _CachedGenerationLanguageModel()
    optimized_policy = _CachedGenerationLanguageModel()
    reference = S3GFNModel(
        policy=reference_policy,
        prior=_CountingLanguageModel(),
        tokenizer=tokenizer,
        fidelity_head=deepcopy(head),
    )
    optimized = OptimizedS3GFNModel(
        policy=optimized_policy,
        prior=_CountingLanguageModel(),
        tokenizer=tokenizer,
        fidelity_head=deepcopy(head),
    )

    torch.manual_seed(11)
    reference_result = reference.generate(count=2, max_length=4)
    torch.manual_seed(11)
    optimized_result = optimized.generate(count=2, max_length=4)

    assert torch.equal(reference_result.input_ids, optimized_result.input_ids)
    assert torch.equal(
        reference_result.fidelity_indices,
        optimized_result.fidelity_indices,
    )
    assert (2, 3) in optimized_policy.forward_shapes


def test_generation_preserves_sampled_padding_ids() -> None:
    reference, optimized = _build_models(
        device=torch.device("cpu"),
        hidden_size=32,
        vocabulary_size=8,
    )
    reference.policy.eval()
    optimized.policy.eval()

    torch.manual_seed(0)
    reference_result = reference.generate(count=8, max_length=16)
    torch.manual_seed(0)
    optimized_result = optimized.generate(count=8, max_length=16)

    assert torch.equal(reference_result.input_ids, optimized_result.input_ids)
    assert torch.equal(
        reference_result.fidelity_indices,
        optimized_result.fidelity_indices,
    )
    terminal_positions = reference_result.input_ids.ne(0).sum(dim=1) - 1
    assert any(
        bool(
            reference_result.input_ids[index, :position].eq(0).any(),
        )
        for index, position in enumerate(terminal_positions.tolist())
        if position > 0
    )

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from benchmarks.optimized_s3gfn_benchmark import _build_models
from activelearning.sampler.optimized_s3gfn.model import OptimizedS3GFNModel
from activelearning.sampler.optimized_s3gfn.replay_buffer import OptimizedReplayBuffer
from activelearning.sampler.optimized_s3gfn.sampler import OptimizedS3GFNSampler
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead
from tests.sampler.s3gfn.conftest import FakeTokenizer


class MolformerFeatureMap(nn.Module):
    """Small feature-map double with the remote class name."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones((2, 2)))

    def forward(self, query, key):
        if self.training:
            self.register_buffer("weight", torch.zeros_like(self.weight))
        return query @ self.weight, key @ self.weight


class _FeatureMapPolicy(nn.Module):
    config = SimpleNamespace(hidden_size=2, bos_token_id=1)

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = MolformerFeatureMap()
        self.dropout = nn.Dropout(0.5)
        self.weight = nn.Parameter(torch.ones((2, 4)))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        **kwargs,
    ):
        del attention_mask, kwargs
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
        hidden = self.dropout(hidden)
        hidden, _ = self.feature_map(hidden, hidden)
        logits = hidden @ self.weight
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


def test_fixed_feature_maps_do_not_disable_policy_training() -> None:
    policy = _FeatureMapPolicy()
    model = OptimizedS3GFNModel(
        policy=policy,
        prior=_FeatureMapPolicy(),
        tokenizer=FakeTokenizer(),
        fidelity_head=FidelityActionHead(hidden_size=2, n_fidelities=2),
        fixed_feature_maps=True,
    )

    model.policy.train()
    feature_map = model.policy.feature_map
    weight_id = id(feature_map.weight)
    model.policy(
        input_ids=torch.tensor([[1, 3, 2]]),
        output_hidden_states=True,
    )

    assert model.policy.training is True
    assert model.policy.dropout.training is True
    assert feature_map.training is False
    assert id(feature_map.weight) == weight_id
    assert model.feature_map_redraw_count == 0
    assert policy.weight.grad is None


def test_cpu_precision_path_does_not_enable_autocast() -> None:
    _, model = _build_models(
        device=torch.device("cpu"),
        hidden_size=8,
        vocabulary_size=16,
    )
    model.precision = "cuda_auto"

    assert model.precision_dtype is None
    assert model.uses_grad_scaler is False


def test_carried_prior_scores_match_direct_prior_scores() -> None:
    _, model = _build_models(
        device=torch.device("cpu"),
        hidden_size=8,
        vocabulary_size=16,
    )
    input_ids = torch.tensor([[1, 3, 2, 0], [1, 4, 2, 0]])
    actions = torch.tensor([0, 1])
    direct = model.prior_trajectory_log_probabilities(input_ids, actions)
    sequence_scores = model.prior_sequence_log_probabilities_uncached(input_ids)
    carried = model.prior_trajectory_log_probabilities(
        input_ids,
        actions,
        precomputed_sequence_log_probabilities=sequence_scores,
    )

    assert torch.allclose(direct, carried)


def test_combined_auxiliary_batch_uses_one_policy_forward() -> None:
    _, model = _build_models(
        device=torch.device("cpu"),
        hidden_size=8,
        vocabulary_size=16,
    )
    model.combined_aux_policy_batch = True
    model.pad_token_id = 9
    positive = torch.tensor([[1, 3, 2, 9], [1, 4, 2, 9]])
    negative = torch.tensor([[1, 5, 2]])
    calls = 0
    forwarded_inputs: list[Tensor] = []

    def count_forward(module, args, kwargs):
        nonlocal calls
        del module, args
        calls += 1
        forwarded_inputs.append(kwargs["input_ids"].detach().clone())

    handle = model.policy.register_forward_pre_hook(
        count_forward,
        with_kwargs=True,
    )
    try:
        loss = model.replay_loss(
            positive_input_ids=positive,
            reward_scores=torch.tensor([0.0, 1.0]),
            beta=2.0,
            negative_input_ids=negative,
            aux_coefficient=0.1,
            positive_fidelity_indices=torch.tensor([0, 1]),
            negative_fidelity_indices=torch.tensor([0]),
        )
    finally:
        handle.remove()

    assert loss is not None
    assert calls == 1
    assert forwarded_inputs[0].tolist() == [
        [1, 3, 2, 9],
        [1, 4, 2, 9],
        [1, 5, 2, 9],
    ]


def test_optimized_replay_buffer_preserves_scores_and_missing_mask() -> None:
    buffer = OptimizedReplayBuffer(
        pad_token_id=0,
        capacity=3,
        policy="fifo",
    )
    tokens = torch.tensor([[1, 2, 0], [1, 3, 2]])
    buffer.add_batch(
        tokens,
        ["a", "b"],
        reward_scores=[0.1, 0.2],
        prior_log_probabilities=[-2.0, -3.0],
    )
    buffer.add_batch(
        tokens[:1],
        ["c"],
        reward_scores=[0.3],
    )

    batch = buffer.sample(3, "cpu")

    assert batch.prior_log_probabilities is not None
    assert batch.prior_score_mask is not None
    assert batch.prior_score_mask.tolist().count(False) == 1
    assert batch.prior_score_mask.tolist().count(True) == 2


def test_eager_rollout_transition_masks_finished_rows() -> None:
    _, model = _build_models(
        device=torch.device("cpu"),
        hidden_size=8,
        vocabulary_size=16,
    )
    token_state = torch.zeros((2, 4), dtype=torch.long)
    finished = torch.tensor([True, False])

    updated, updated_finished = model._eager_rollout_transition(
        token_state,
        torch.tensor([7, 2]),
        finished,
        1,
        model.pad_token_id,
        model.eos_token_id,
    )

    assert updated[:, 1].tolist() == [model.pad_token_id, model.eos_token_id]
    assert updated_finished.tolist() == [True, True]


def test_optimized_sampler_exposes_independent_ablation_switches() -> None:
    sampler = OptimizedS3GFNSampler(
        n_samples=1,
        fidelities=[1, 2],
        precision="cuda_auto",
        fixed_feature_maps=True,
        parallel_cuda_rollout=True,
        compile_mode="default",
        deferred_sync=True,
        carried_prior_scores=True,
        overlap_online_prior=True,
        combined_aux_policy_batch=True,
        stop_check_interval=4,
        prior_cache_enabled=False,
        prior_cache_capacity=32,
    )

    assert sampler.precision == "cuda_auto"
    assert sampler.fixed_feature_maps is True
    assert sampler.parallel_cuda_rollout is True
    assert sampler.compile_mode == "default"
    assert sampler.deferred_sync is True
    assert sampler.carried_prior_scores is True
    assert sampler.overlap_online_prior is True
    assert sampler.combined_aux_policy_batch is True
    assert sampler.stop_check_interval == 4
    assert sampler.prior_cache_enabled is False
    assert sampler.prior_cache_capacity == 32

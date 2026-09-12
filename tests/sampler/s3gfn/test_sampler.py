from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from activelearning.runtime import RuntimeContext
import activelearning.sampler.s3gfn.sampler as sampler_module
from activelearning.utils.types import Candidate

from tests.sampler.s3gfn.conftest import (
    FakeAcquisition,
    FakeChem,
    FakeModel,
    FakeSynthesizability,
    FakeTokenizer,
)


class _FakeBatchAcquisition:
    """Acquisition that cannot score candidates one at a time."""

    supports_singleton_scoring = False


class _GradientTrainingModel:
    def __init__(self) -> None:
        self.policy = nn.Linear(1, 1, bias=False)
        self.fidelity_head = nn.Linear(1, 1, bias=False)
        self.log_z = nn.Parameter(torch.tensor(0.5))
        self.pad_token_id = 0
        self.device = torch.device("cpu")
        self.last_auxiliary_loss = None

    def generate(self, count, max_length, temperature):
        assert count == 1
        return SimpleNamespace(
            smiles=("CC",),
            input_ids=torch.ones((1, 3), dtype=torch.long),
            fidelity_indices=torch.tensor([0], dtype=torch.long),
        )

    def encode_smiles(self, smiles):
        return torch.ones((len(smiles), 3), dtype=torch.long)

    def on_policy_loss(self, input_ids, reward_scores, beta, fidelity_indices=None):
        return (
            self.policy.weight.square().sum()
            + self.fidelity_head.weight.square().sum()
            + self.log_z.square()
        )


def test_sampler_uses_explicit_model_dtype_for_rewards(
    make_sampler,
    fake_model,
):
    sampler = make_sampler(model_dtype="bfloat16")

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC",),
        fidelity_indices=torch.tensor([0]),
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    assert prepared.reward_scores.dtype is torch.bfloat16


def test_training_uses_batch_size_for_generation(make_sampler):
    class _TrainingModel(_GradientTrainingModel):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls = []

        def generate(self, count, max_length, temperature):
            self.generate_calls.append(count)
            return SimpleNamespace(
                smiles=tuple("CC" for _ in range(count)),
                input_ids=torch.ones((count, 3), dtype=torch.long),
                fidelity_indices=torch.zeros(count, dtype=torch.long),
            )

    model = _TrainingModel()
    sampler = make_sampler(
        n_samples=1,
        batch_size=3,
        replay_batch_size=4,
        model_dtype="float32",
    )
    optimizer = torch.optim.SGD(
        [
            *model.policy.parameters(),
            *model.fidelity_head.parameters(),
            model.log_z,
        ],
        lr=0.1,
    )

    sampler._train_step(
        model=model,
        synthesizability=FakeSynthesizability(),
        positive_buffer=sampler_module.ReplayBuffer(pad_token_id=0, capacity=4),
        negative_buffer=None,
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
        optimizer=optimizer,
    )

    assert model.generate_calls == [3]


def test_replay_uses_replay_batch_size(make_sampler):
    class _RecordingBuffer:
        def __init__(self, size: int) -> None:
            self.size = size
            self.calls: list[dict] = []

        def __len__(self) -> int:
            return self.size

        def sample(self, **kwargs):
            self.calls.append(kwargs)
            count = kwargs["count"]
            return SimpleNamespace(
                input_ids=torch.ones((count, 3), dtype=torch.long),
                reward_scores=torch.ones(count, dtype=kwargs["dtype"]),
                fidelity_indices=None,
            )

    sampler = make_sampler(n_samples=1, replay_batch_size=3)
    positive_buffer = _RecordingBuffer(size=5)
    negative_buffer = _RecordingBuffer(size=5)

    class _ReplayModel:
        def replay_loss(self, **kwargs):
            del kwargs
            return None

    model = _ReplayModel()

    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )

    assert positive_buffer.calls[0]["count"] == 3
    assert negative_buffer.calls[0]["count"] == 3


def test_final_generation_uses_independent_batch_size_and_strict_cap(
    make_sampler,
):
    class _FinalGenerationModel:
        def __init__(self) -> None:
            self.policy = nn.Linear(1, 1)
            self.prior = nn.Linear(1, 1)
            self.calls: list[int] = []
            self.next_smiles = 0

        def generate(self, count, max_length, temperature):
            del max_length, temperature
            self.calls.append(count)
            smiles = tuple(f"C{self.next_smiles + index}" for index in range(count))
            self.next_smiles += count
            return SimpleNamespace(
                smiles=smiles,
                fidelity_indices=torch.zeros(count, dtype=torch.long),
            )

    sampler = make_sampler(
        n_samples=4,
        batch_size=99,
        generation_batch_size=3,
        max_generation_attempts=5,
    )
    model = _FinalGenerationModel()

    candidates = sampler._generate_final_candidates(
        model=model,
        molecule_chem=FakeChem,
    )

    assert len(candidates) == 4
    assert model.calls == [3, 2]


def test_final_generation_stops_at_strict_attempt_cap(make_sampler):
    class _EmptyGenerationModel:
        def __init__(self) -> None:
            self.policy = nn.Linear(1, 1)
            self.prior = nn.Linear(1, 1)
            self.calls: list[int] = []

        def generate(self, count, max_length, temperature):
            del max_length, temperature
            self.calls.append(count)
            return SimpleNamespace(smiles=(), fidelity_indices=None)

    sampler = make_sampler(
        n_samples=1,
        batch_size=99,
        generation_batch_size=8,
        max_generation_attempts=3,
    )
    model = _EmptyGenerationModel()

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        sampler._generate_final_candidates(model=model, molecule_chem=FakeChem)

    assert model.calls == [3]


def test_sampler_returns_canonical_smiles_with_conditionally_sampled_fidelities(
    make_sampler,
    fake_model,
    fake_acquisition,
    patch_molecule_dependencies,
):
    sampler = make_sampler()
    sampler._new_round_model = lambda: fake_model
    sampler._train_round = lambda **kwargs: None

    candidates = sampler.sample(
        acquisition=fake_acquisition,
        cost_fn=lambda candidates: [
            float(candidate.fidelity) for candidate in candidates
        ],
    )

    assert len(candidates) == 2
    assert {candidate.x for candidate in candidates} == {"CC", "CO"}
    assert [candidate.fidelity for candidate in candidates] == [2, 1]
    assert fake_acquisition.seen_fidelities == []
    assert fake_model.generate_calls[0] == {
        "count": 2,
        "max_length": 8,
        "temperature": 1.0,
    }


def test_sampler_compiles_policy_before_training(
    make_sampler,
    patch_molecule_dependencies,
) -> None:
    """Compilation must accelerate policy forwards in training and generation."""
    events: list[tuple[str, object]] = []

    class CompileAwareFakeModel(FakeModel):
        """Record compilation without invoking TorchInductor."""

        def compile_policy(
            self,
            *,
            mode: str,
            dynamic: bool | None,
            training_only: bool,
        ) -> None:
            """Record the requested compilation mode."""
            events.append(("compile", mode, dynamic, training_only))

    model = CompileAwareFakeModel()
    model.policy.train()
    sampler = make_sampler(
        compile_strategy="training_and_generation",
        torch_compile_mode="max-autotune",
        torch_compile_dynamic=None,
    )
    sampler._new_round_model = lambda: model
    sampler._train_round = lambda **kwargs: events.append(
        ("train", kwargs["model"].policy.training)
    )

    sampler.sample(acquisition=FakeAcquisition())

    assert events == [
        ("compile", "max-autotune", None, False),
        ("train", True),
    ]


def test_round_model_enables_attention_adapter_after_copy(
    make_sampler,
    monkeypatch,
) -> None:
    """The adapter must bind to the fresh round policy, not its CPU template."""
    sampler = make_sampler(attention_mask_adapter=True)
    template = SimpleNamespace(
        policy=nn.Linear(1, 1),
        prior=nn.Linear(1, 1),
        tokenizer=FakeTokenizer(),
        fidelity_head=None,
    )
    sampler._pretrained_model = template
    sampler._keep_pretrained_template_on_cpu = lambda: None
    enabled_policies = []

    def record_enable(model):
        enabled_policies.append(model.policy)
        return 1

    monkeypatch.setattr(
        sampler_module.S3GFNModel,
        "enable_attention_mask_adapter",
        record_enable,
    )
    round_model = sampler._new_round_model()

    assert enabled_policies == [round_model.policy]
    assert round_model.policy is not template.policy


def test_sampler_advances_round_state_across_consecutive_samples(
    make_sampler,
    patch_molecule_dependencies,
) -> None:
    models: list[FakeModel] = []
    sampler = make_sampler()

    def build_model() -> FakeModel:
        model = FakeModel()
        models.append(model)
        return model

    sampler._new_round_model = build_model
    sampler._train_round = lambda **kwargs: None

    sampler.sample(acquisition=FakeAcquisition())
    sampler.sample(acquisition=FakeAcquisition())

    assert len(models) == 2
    assert sampler._round_index == 2


def test_sampler_resets_round_metrics_at_sample_start(
    make_sampler,
    patch_molecule_dependencies,
) -> None:
    sampler = make_sampler()
    previous_metrics = sampler.round_metrics
    previous_metrics.record_training_step(
        generated_count=1,
        valid_count=1,
        synthesizable_count=1,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[1.0],
    )
    observed_metrics = []
    sampler._new_round_model = lambda: FakeModel()
    sampler._train_round = lambda **kwargs: observed_metrics.append(
        sampler.round_metrics
    )

    sampler.sample(acquisition=FakeAcquisition())

    assert observed_metrics == [sampler.round_metrics]
    assert observed_metrics[0] is not previous_metrics
    assert observed_metrics[0].generated_counts == []


def test_prepare_batch_scores_the_selected_fidelity(
    make_sampler,
    fake_model,
    fake_acquisition,
    patch_molecule_dependencies,
):
    sampler = make_sampler()

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC", "CO"),
        fidelity_indices=torch.tensor([1, 0]),
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=fake_acquisition,
        cost_fn=None,
    )

    assert prepared.reward_scores.tolist() == [1.0, 0.0]
    assert fake_acquisition.seen_fidelities == [2, 1]


def test_single_fidelity_preparation_omits_terminal_action(
    make_sampler,
    fake_model,
    patch_molecule_dependencies,
):
    sampler = make_sampler(n_samples=1, fidelities=[7], batch_size=1)

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC",),
        fidelity_indices=None,
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    assert prepared.fidelity_indices is None
    assert prepared.reward_scores.tolist() == [0.0]


def test_sampler_rejects_batch_only_acquisitions(make_sampler):
    sampler = make_sampler(n_samples=1, fidelities=[1])

    with pytest.raises(ValueError, match="singleton scoring"):
        sampler.sample(acquisition=_FakeBatchAcquisition())


def test_canonicalization_rejects_disconnected_molecules():
    assert (
        sampler_module._canonicalize_to_smiles(
            "C.C",
            molecule_chem=FakeChem,
        )
        is None
    )


def test_reward_scores_are_normalized_to_the_rtb_range() -> None:
    normalized = sampler_module._normalize_reward_scores([-2.0, 0.0, 4.0])
    assert normalized[0] == 0.0
    assert normalized[1] == pytest.approx(1.0 / 3.0)
    assert normalized[2] == 1.0
    assert sampler_module._normalize_reward_scores([3.0, 3.0]) == [0.0, 0.0]


def test_training_updates_policy_and_fidelity_parameters(
    make_sampler,
    make_replay_buffer,
    fake_acquisition,
) -> None:
    model = _GradientTrainingModel()
    policy_before = model.policy.weight.detach().clone()
    fidelity_before = model.fidelity_head.weight.detach().clone()
    sampler = make_sampler(
        n_samples=1,
        batch_size=1,
        replay_batch_size=2,
        num_warmup_steps=0,
        learning_rate=0.1,
        log_z_learning_rate=0.1,
    )
    positive_buffer = make_replay_buffer()

    sampler._train_round(
        model=model,
        synthesizability=FakeSynthesizability(),
        positive_buffer=positive_buffer,
        negative_buffer=None,
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    assert not torch.equal(model.policy.weight, policy_before)
    assert not torch.equal(model.fidelity_head.weight, fidelity_before)


def test_replay_updates_wait_for_a_full_replay_batch(
    make_sampler,
    make_replay_buffer,
) -> None:
    class _ReplayModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def replay_loss(self, **kwargs):
            self.calls.append(kwargs)
            return None

    sampler = make_sampler(n_samples=1, fidelities=[1], replay_batch_size=2)
    positive_buffer = make_replay_buffer()
    negative_buffer = make_replay_buffer()
    tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    positive_buffer.add_batch(tokens[:1], ["positive-a"], [0.5])
    model = _ReplayModel()
    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )
    assert model.calls == []

    positive_buffer.add_batch(tokens[1:], ["positive-b"], [1.0])
    negative_buffer.add_batch(tokens[:1], ["negative-a"], [0.0])
    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )
    assert model.calls[-1]["negative_input_ids"] is None

    negative_buffer.add_batch(tokens[1:], ["negative-b"], [0.0])
    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )
    assert model.calls[-1]["negative_input_ids"] is not None


def test_generated_batch_routes_synthesizable_and_negative_replays(
    make_sampler,
    make_replay_buffer,
) -> None:
    class _OnPolicyModel:
        def on_policy_loss(self, *args, **kwargs):
            return None

    sampler = make_sampler(n_samples=1, fidelities=[1], aux_coefficient=1.0e-4)
    positive_buffer = make_replay_buffer()
    negative_buffer = make_replay_buffer()
    prepared = sampler_module._PreparedMoleculeBatch(
        smiles=("positive", "negative"),
        input_ids=torch.tensor([[1, 2, 0], [1, 3, 0]]),
        reward_scores=torch.tensor([1.0, 0.0]),
        synthesizable=(True, False),
    )

    sampler._update_generated_batch(
        model=_OnPolicyModel(),
        prepared=prepared,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )

    assert [entry.smiles for entry in positive_buffer.entries] == ["positive"]
    assert [entry.smiles for entry in negative_buffer.entries] == ["negative"]


def test_optimizer_clips_fidelity_head_gradients(monkeypatch, make_sampler) -> None:
    policy = nn.Linear(1, 1)
    fidelity_head = nn.Linear(1, 1)
    model = SimpleNamespace(
        policy=policy,
        fidelity_head=fidelity_head,
        log_z=nn.Parameter(torch.zeros(())),
    )
    sampler = make_sampler(n_samples=1)
    optimizer = torch.optim.SGD(
        [
            *policy.parameters(),
            *fidelity_head.parameters(),
            model.log_z,
        ],
        lr=0.1,
    )
    captured_parameters = []
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda parameters, max_norm: captured_parameters.extend(parameters),
    )

    sampler._optimize(
        optimizer,
        sum(parameter.sum() for parameter in model.policy.parameters())
        + sum(parameter.sum() for parameter in model.fidelity_head.parameters())
        + model.log_z,
        model,
    )

    assert any(
        parameter is list(fidelity_head.parameters())[0]
        for parameter in captured_parameters
    )


def test_round_metrics_aggregate_scalars_and_figures(
    make_sampler,
    make_replay_buffer,
) -> None:
    sampler = make_sampler(n_train_steps=2)
    metrics = sampler.round_metrics
    metrics.record_training_step(
        generated_count=4,
        valid_count=3,
        synthesizable_count=2,
        online_loss=2.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.1,
        raw_reward_scores=[2.0, 4.0, 6.0],
    )
    metrics.record_training_step(
        generated_count=2,
        valid_count=2,
        synthesizable_count=1,
        online_loss=4.0,
        replay_loss=3.0,
        auxiliary_loss=0.5,
        log_z=0.2,
        raw_reward_scores=[8.0, 10.0],
    )
    metrics.record_generation_batch(
        attempts=5,
        invalid_count=1,
        duplicate_count=1,
    )
    metrics.record_final_candidates(
        [
            Candidate(x="CC", fidelity=1),
            Candidate(x="CO", fidelity=2),
        ]
    )
    metrics.training_duration_s = 1.25
    metrics.generation_duration_s = 0.75
    metrics.positive_buffer_size = 3

    logged, figure_calls = sampler.drain_round_diagnostics(
        include_figures=True,
        max_points=1000,
    )

    assert logged["sampler/s3gfn/train/generated_total"] == 6
    assert logged["sampler/s3gfn/train/valid_total"] == 5
    assert logged["sampler/s3gfn/train/synthesizable_total"] == 3
    assert logged["sampler/s3gfn/train/validity_rate"] == pytest.approx(5 / 6)
    assert logged["sampler/s3gfn/train/synthesizable_rate"] == pytest.approx(3 / 5)
    assert logged["sampler/s3gfn/train/online_updates"] == 2
    assert logged["sampler/s3gfn/train/replay_updates"] == 1
    assert logged["sampler/s3gfn/train/online_rtb_loss_mean"] == pytest.approx(3.0)
    assert logged["sampler/s3gfn/train/online_rtb_loss_final"] == pytest.approx(4.0)
    assert logged["sampler/s3gfn/train/replay_loss_mean"] == pytest.approx(3.0)
    assert logged["sampler/s3gfn/train/contrastive_loss_mean"] == pytest.approx(0.5)
    assert logged["sampler/s3gfn/train/contrastive_loss_final"] == pytest.approx(0.5)
    assert logged["sampler/s3gfn/train/log_z_final"] == pytest.approx(0.2)
    assert logged["sampler/s3gfn/reward/raw_mean"] == pytest.approx(6.0)
    assert logged["sampler/s3gfn/reward/raw_max"] == pytest.approx(10.0)
    assert logged["sampler/s3gfn/generation/yield"] == pytest.approx(0.4)
    assert logged["sampler/s3gfn/generation/invalid_rate"] == pytest.approx(0.2)
    assert logged["sampler/s3gfn/generation/duplicate_rate"] == pytest.approx(0.2)
    assert all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in logged.values()
    )
    assert set(figure_calls) == {
        "sampler/s3gfn/training_losses",
        "sampler/s3gfn/log_z",
        "sampler/s3gfn/reward/trajectory",
    }
    assert all(isinstance(figure, Figure) for figure in figure_calls.values())
    training_axis = figure_calls["sampler/s3gfn/training_losses"].axes[0]
    assert training_axis.get_ylabel() == "S3-GFN training loss"
    for figure in figure_calls.values():
        plt.close(figure)


def test_sampler_drains_round_metrics_without_touching_runtime_logger(
    make_sampler,
    fake_model,
    patch_molecule_dependencies,
):
    runtime_logger = Mock()
    sampler = make_sampler()
    sampler.bind_runtime_context(RuntimeContext(logger=runtime_logger))
    sampler._new_round_model = lambda: fake_model
    sampler._train_round = lambda **kwargs: None

    sampler.sample(acquisition=FakeAcquisition())

    metrics, figures = sampler.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    )

    assert metrics["sampler/s3gfn/generation/yield"] == pytest.approx(1.0)
    assert figures == {}
    runtime_logger.log_metric.assert_not_called()
    runtime_logger.log_step.assert_not_called()
    runtime_logger.end.assert_not_called()


def test_round_metrics_drain_without_runtime_logger(
    make_sampler,
    make_replay_buffer,
) -> None:
    """Metric emission must be skipped, not crash, when no logger is bound."""
    sampler = make_sampler(n_samples=1, fidelities=[1])
    assert sampler.logger is None
    metrics = sampler.round_metrics
    metrics.record_training_step(
        generated_count=1,
        valid_count=1,
        synthesizable_count=1,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[1.0],
    )

    logged, figures = sampler.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    )

    assert logged["sampler/s3gfn/train/generated_total"] == 1
    assert figures == {}
    assert sampler.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    ) == ({}, {})


def test_prepare_batch_retains_raw_reward_scores(
    make_sampler,
    fake_model,
    patch_molecule_dependencies,
):
    sampler = make_sampler()

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC", "CO"),
        fidelity_indices=torch.tensor([1, 0]),
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    assert prepared.raw_reward_scores == (2.0, 1.0)
    assert prepared.reward_scores.tolist() == [1.0, 0.0]

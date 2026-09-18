"""Focused tests for configurable DKL training controls."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.dkl.config import (
    DKLTrainingConfig,
    VariationalDKLSurrogateConfig,
)
from activelearning.surrogate.dkl.variational import VariationalDKLSurrogate
from activelearning.surrogate.encoder import LatentEncoder
from activelearning.utils.types import Observation


class _NumericEncoder(LatentEncoder):
    """Small deterministic encoder for DKL-control tests."""

    latent_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, self.latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project numeric inputs to the latent representation."""
        return self.projection(inputs)


def _observations(count: int = 8) -> list[Observation]:
    """Create a deterministic numeric regression dataset."""
    return [
        Observation(x=[float(index), float(index + 1)], y=float(index))
        for index in range(count)
    ]


def _build_surrogate(training: DKLTrainingConfig) -> VariationalDKLSurrogate:
    """Build a small variational DKL surrogate for unit tests."""
    return VariationalDKLSurrogate(
        encoder=_NumericEncoder(),
        training_params=training,
        num_inducing=2,
    )


def test_variational_config_propagates_initial_likelihood_noise() -> None:
    """The variational config should pass likelihood noise to its surrogate."""
    config = VariationalDKLSurrogateConfig.model_construct(
        encoder=None,
        training_params=DKLTrainingConfig(),
        num_inducing=2,
        initial_likelihood_noise=0.23,
    )

    surrogate = VariationalDKLSurrogate(
        encoder=_NumericEncoder(),
        training_params=config.training_params,
        **config._additional_build_kwargs(),
    )

    assert surrogate._initial_likelihood_noise == pytest.approx(0.23)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"betas": (-0.1, 0.9)},
        {"betas": (0.9, 1.0)},
        {"batch_size": 0},
        {"validation_fraction": -0.01},
        {"validation_fraction": 1.0},
        {"validation_seed": -1},
        {"early_stopping_patience": -1},
    ],
)
def test_dkl_training_config_rejects_invalid_controls(
    kwargs: dict[str, object],
) -> None:
    """New training controls should reject values outside their valid bounds."""
    with pytest.raises(ValueError):
        DKLTrainingConfig(**kwargs)


def test_validation_split_and_minibatches_are_seeded() -> None:
    """Equal seeds should produce equal validation and minibatch indices."""
    training = DKLTrainingConfig(
        batch_size=2,
        validation_fraction=0.25,
        validation_seed=17,
    )
    first = _build_surrogate(training)
    second = _build_surrogate(training)
    train_X = torch.arange(20, dtype=torch.float64).reshape(10, 2)
    train_Y = torch.arange(10, dtype=torch.float64).unsqueeze(-1)

    first._fit_train_X, first._fit_train_Y, _, _ = first._split_training_data(
        train_X,
        train_Y,
    )
    second._fit_train_X, second._fit_train_Y, _, _ = second._split_training_data(
        train_X,
        train_Y,
    )

    assert torch.equal(first._fit_train_indices, second._fit_train_indices)
    assert torch.equal(first._validation_indices, second._validation_indices)
    first_batches = [batch.tolist() for batch in first._iter_batch_indices(8, epoch=2)]
    second_batches = [
        batch.tolist() for batch in second._iter_batch_indices(8, epoch=2)
    ]
    assert first_batches == second_batches
    assert sorted(index for batch in first_batches for index in batch) == list(range(8))


def test_runtime_seed_controls_validation_when_not_overridden() -> None:
    """The runtime seed should provide deterministic split state by default."""
    training = DKLTrainingConfig(validation_fraction=0.25)
    first = _build_surrogate(training)
    second = _build_surrogate(training)
    runtime = RuntimeContext(seed=31)
    first.bind_runtime_context(runtime)
    second.bind_runtime_context(runtime)
    train_X = torch.arange(20, dtype=torch.float64).reshape(10, 2)
    train_Y = torch.arange(10, dtype=torch.float64).unsqueeze(-1)

    first._split_training_data(train_X, train_Y)
    second._split_training_data(train_X, train_Y)

    assert torch.equal(first._fit_train_indices, second._fit_train_indices)
    assert torch.equal(first._validation_indices, second._validation_indices)


def test_adam_uses_configured_betas() -> None:
    """The variational optimizer should receive configured Adam betas."""
    surrogate = _build_surrogate(
        DKLTrainingConfig(epochs=1, betas=(0.75, 0.88)),
    )
    surrogate._build_model(
        torch.zeros((2, 2), dtype=torch.float64),
        torch.zeros((2, 1), dtype=torch.float64),
    )

    optimizer = surrogate._make_optimizer()

    assert optimizer.defaults["betas"] == pytest.approx((0.75, 0.88))


def test_variational_likelihood_uses_configured_initial_noise() -> None:
    """The variational likelihood should initialize at the configured noise."""
    surrogate = VariationalDKLSurrogate(
        encoder=_NumericEncoder(),
        training_params=DKLTrainingConfig(epochs=1),
        num_inducing=2,
        initial_likelihood_noise=0.23,
    )
    surrogate._build_model(
        torch.zeros((2, 2), dtype=torch.float64),
        torch.zeros((2, 1), dtype=torch.float64),
    )

    assert surrogate._likelihood is not None
    assert surrogate._likelihood.noise.item() == pytest.approx(0.23)
    surrogate._remove_noise_prior()
    assert surrogate._likelihood.noise.item() == pytest.approx(0.23)


def test_validation_early_stopping_restores_best_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patience should stop training and restore the best validation state."""
    surrogate = _build_surrogate(
        DKLTrainingConfig(
            epochs=4,
            lr=1e-2,
            batch_size=2,
            validation_fraction=0.25,
            early_stopping_patience=1,
            validation_seed=5,
        ),
    )
    validation_losses = iter([1.0, 2.0, 3.0])
    captured_states: list[dict[str, dict[str, Tensor]]] = []
    original_snapshot = surrogate._snapshot_trainable_state

    def capture_snapshot() -> dict[str, dict[str, Tensor]]:
        state = original_snapshot()
        captured_states.append(deepcopy(state))
        return state

    monkeypatch.setattr(surrogate, "_snapshot_trainable_state", capture_snapshot)
    monkeypatch.setattr(
        surrogate,
        "_validation_loss",
        lambda mll: next(validation_losses),
    )

    surrogate.fit(_observations())

    assert surrogate._epochs_trained == 2
    assert surrogate._validation_losses == [1.0, 2.0]
    assert surrogate._best_validation_loss == pytest.approx(1.0)
    assert len(captured_states) == 1
    assert surrogate._gp_model is not None
    restored = False
    for name, parameter in surrogate._gp_model.named_parameters():
        if parameter.requires_grad and name in captured_states[0]["model"]:
            assert torch.equal(parameter, captured_states[0]["model"][name])
            restored = True
            break
    assert restored

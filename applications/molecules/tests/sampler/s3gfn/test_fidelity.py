from __future__ import annotations

import math

import pytest
import torch

from activelearning_molecules.samplers.s3gfn.fidelity import FidelityActionHead


def test_fidelity_action_head_starts_uniform() -> None:
    head = FidelityActionHead(hidden_size=2, n_fidelities=3)
    hidden_states = torch.ones((2, 2))

    log_probabilities = head.log_prob(
        hidden_states,
        torch.tensor([0, 2]),
    )

    assert log_probabilities.detach().tolist() == pytest.approx(
        [-math.log(3.0), -math.log(3.0)]
    )


def test_fidelity_action_head_samples_valid_indices() -> None:
    head = FidelityActionHead(hidden_size=2, n_fidelities=3)

    sampled = head.sample(torch.ones((10, 2)))

    assert sampled.shape == (10,)
    assert sampled.dtype == torch.long
    assert torch.all((sampled >= 0) & (sampled < 3))


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([-1]),
        torch.tensor([3]),
        torch.tensor([[0]]),
        torch.tensor([0.0]),
    ],
)
def test_fidelity_action_head_rejects_invalid_indices(indices: torch.Tensor) -> None:
    head = FidelityActionHead(hidden_size=2, n_fidelities=3)

    with pytest.raises((TypeError, ValueError)):
        head.log_prob(torch.ones((1, 2)), indices)


def test_uniform_prior_log_probability_is_constant() -> None:
    head = FidelityActionHead(hidden_size=2, n_fidelities=3)

    result = head.uniform_prior_log_prob(
        torch.tensor([0, 2]),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert result.dtype == torch.float64
    assert result == pytest.approx([-math.log(3.0), -math.log(3.0)])

"""Fake-backed active-learning integration tests for SMILES DKL surrogates."""

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from activelearning.acquisition.botorch.botorch_analytic import (
    UpperConfidenceBound,
)
from activelearning.active_learning import active_learning
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.dkl.exact import ExactDKLSurrogate
from activelearning.surrogate.sequence.huggingface_encoder import (
    HuggingFaceSequenceEncoder,
)
from activelearning.utils.types import Candidate, Observation


class _FakeTokenizer:
    """Tiny deterministic tokenizer with distinct special-token IDs."""

    vocab_size = 32
    padding_idx = 0
    eos_idx = 1
    cls_idx = 2
    mask_idx = 3

    def batch_from_strings(
        self,
        strings: list[str],
        max_tokens: int,
        device: torch.device | None = None,
    ) -> Tensor:
        rows: list[list[int]] = []
        for string in strings:
            content = [4 + (ord(character) % 20) for character in string]
            row = [self.cls_idx, *content, self.eos_idx][:max_tokens]
            row.extend([self.padding_idx] * (max_tokens - len(row)))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def attention_mask_from_batch(self, token_batch: Tensor) -> Tensor:
        return token_batch.ne(self.padding_idx).long()


class _FakeBackbone(nn.Module):
    """Small frozen backbone exposing a Hugging Face-like output."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask
        del return_dict
        hidden = input_ids.to(dtype=self.scale.dtype).unsqueeze(-1)
        hidden = hidden.mul(self.scale).expand(-1, -1, 4)
        return SimpleNamespace(last_hidden_state=hidden)


def _make_surrogate(*, multi_fidelity: bool) -> ExactDKLSurrogate:
    """Build a tiny exact DKL surrogate with a fake SMILES backbone."""
    encoder = HuggingFaceSequenceEncoder(
        backbone=_FakeBackbone(),
        tokenizer=_FakeTokenizer(),
        max_tokens=8,
        latent_dim=2,
        cache_size=16,
    )
    return ExactDKLSurrogate(
        encoder=encoder,
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        is_multi_fidelity=multi_fidelity,
        target_fidelity=3 if multi_fidelity else None,
        standardize_outputs=False,
    )


def _make_oracle() -> MultiFidelityOracle:
    """Build an inexpensive oracle for canonical SMILES candidates."""
    return MultiFidelityOracle(
        fidelity_configs={
            1: {
                "cost_per_sample": 1.0,
                "fidelity_confidence": 0.25,
                "score_fn": lambda value: float(len(value)),
            },
            2: {
                "cost_per_sample": 1.0,
                "fidelity_confidence": 0.6,
                "score_fn": lambda value: float(len(value)) + 0.1,
            },
            3: {
                "cost_per_sample": 1.0,
                "fidelity_confidence": 1.0,
                "score_fn": lambda value: float(len(value)) + 0.2,
            },
        }
    )


def _run_smiles_active_learning(
    surrogate: ExactDKLSurrogate,
    *,
    multi_fidelity: bool,
) -> tuple[ListDataset, UpperConfidenceBound]:
    """Run one active-learning round over a fake S3-GFN-shaped pool."""
    dataset = ListDataset()
    if multi_fidelity:
        initial_observations = [
            Observation(x="CC", y=2.0, fidelity=1),
            Observation(x="CCC", y=3.0, fidelity=2),
            Observation(x="CCCC", y=4.0, fidelity=3),
        ]
        pool = [
            Candidate("C1CC1", fidelity=1),
            Candidate("C1CCC1", fidelity=2),
            Candidate("C1CCCC1", fidelity=3),
        ]
        oracle = _make_oracle()
    else:
        initial_observations = [
            Observation(x="CC", y=2.0, fidelity=1),
            Observation(x="CCC", y=3.0, fidelity=1),
        ]
        pool = [
            Candidate("C1CC1", fidelity=1),
            Candidate("C1CCC1", fidelity=1),
        ]
        oracle = MultiFidelityOracle(
            fidelity_configs={
                1: {
                    "cost_per_sample": 1.0,
                    "fidelity_confidence": 1.0,
                    "score_fn": lambda value: float(len(value)),
                }
            }
        )
    dataset.add_observations(initial_observations)

    acquisition = UpperConfidenceBound(beta=1.0)
    active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=PoolScoreSampler(candidate_pool=pool, num_samples=2),
        selector=TopKAcquisitionSelector(num_samples=1),
        oracle=oracle,
        budget=Budget(available_budget=1.0, schedule=lambda _: 1.0, max_rounds=1),
    )
    return dataset, acquisition


def test_smiles_dkl_runs_single_fidelity_active_learning() -> None:
    """A fake S3-GFN-shaped SMILES pool completes an exact DKL round."""
    torch.manual_seed(0)
    dataset, _acquisition = _run_smiles_active_learning(
        _make_surrogate(multi_fidelity=False),
        multi_fidelity=False,
    )

    assert len(dataset.get_observations_iterable()) == 3
    assert all(
        torch.isfinite(torch.tensor(observation.y))
        for observation in dataset.get_observations_iterable()
    )


def test_smiles_dkl_resolves_multifidelity_projection_in_token_space() -> None:
    """Target fidelity metadata projects the final token-space column."""
    torch.manual_seed(0)
    surrogate = _make_surrogate(multi_fidelity=True)
    _dataset, acquisition = _run_smiles_active_learning(
        surrogate,
        multi_fidelity=True,
    )

    assert surrogate.get_target_fidelity_value() == 1.0
    fidelity_dimension = surrogate.get_fidelity_dimension()
    assert fidelity_dimension == 8
    encoded = surrogate.encode_candidates([Candidate("CO", fidelity=1)])
    projection = acquisition._resolved_project_to_target_fidelity_fn
    assert projection is not None
    projected = projection(encoded)
    assert projected is not None
    assert projected[0, fidelity_dimension].item() == 1.0
    assert encoded[0, fidelity_dimension].item() == 0.25

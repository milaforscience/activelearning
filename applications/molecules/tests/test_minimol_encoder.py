"""Fake-backed tests for the MiniMol SMILES DKL encoder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import activelearning_molecules.encoders.minimol as minimol_module
from activelearning.acquisition.botorch.botorch_analytic import (
    UpperConfidenceBound,
)
from activelearning.active_learning import active_learning
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dkl import ExactDKLSurrogate, VariationalDKLSurrogate
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.utils.types import Candidate, Observation


class _FakeMiniMol:
    """Deterministic MiniMol replacement returning 512-value fingerprints."""

    instances: list["_FakeMiniMol"] = []

    def __init__(
        self,
        *,
        batch_size: int,
        checkpoint_path: object = None,
    ) -> None:
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.calls: list[list[str]] = []
        self.grad_enabled: list[bool] = []
        self.__class__.instances.append(self)

    def __call__(self, smiles: list[str]) -> list[Tensor]:
        self.calls.append(list(smiles))
        self.grad_enabled.append(torch.is_grad_enabled())
        return [
            torch.arange(512, dtype=torch.float64) + sum(map(ord, value))
            for value in smiles
        ]


@pytest.fixture
def fake_minimol(monkeypatch: pytest.MonkeyPatch) -> list[_FakeMiniMol]:
    """Replace the lazy MiniMol loader and return constructed fake models."""
    _FakeMiniMol.instances = []
    monkeypatch.setattr(minimol_module, "_load_minimol", lambda: _FakeMiniMol)
    return _FakeMiniMol.instances


def test_minimol_encoder_validates_constructor_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid extraction and projection sizes should fail early."""
    monkeypatch.setattr(minimol_module, "_load_minimol", lambda: _FakeMiniMol)

    with pytest.raises(ValueError, match="batch_size"):
        minimol_module.MiniMolSmilesEncoder(batch_size=0)
    with pytest.raises(ValueError, match="latent_dim"):
        minimol_module.MiniMolSmilesEncoder(latent_dim=0)
    with pytest.raises(ValueError, match="cache_size"):
        minimol_module.MiniMolSmilesEncoder(cache_size=-1)
    with pytest.raises(FileNotFoundError, match="checkpoint_path"):
        minimol_module.MiniMolSmilesEncoder(
            checkpoint_path=tmp_path / "missing.pth",
        )


def test_minimol_encoder_passes_checkpoint_path_to_loader(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """The configured checkpoint path reaches the MiniMol loader."""
    checkpoint_path = tmp_path / "minimol.pth"
    checkpoint_path.touch()

    encoder = minimol_module.MiniMolSmilesEncoder(
        checkpoint_path=checkpoint_path,
    )

    assert encoder.checkpoint_path == checkpoint_path
    assert fake_minimol[0].checkpoint_path == checkpoint_path


@pytest.mark.parametrize("wrapped", [False, True])
def test_minimol_checkpoint_loads_predictor_state_dict(
    wrapped: bool,
    tmp_path: Path,
) -> None:
    """Raw and wrapped predictor state dicts replace MiniMol weights."""
    source = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(-2.0)
    checkpoint: object = source.state_dict()
    if wrapped:
        checkpoint = {"state_dict": checkpoint}
    checkpoint_path = tmp_path / "minimol.pth"
    torch.save(checkpoint, checkpoint_path)

    target = SimpleNamespace(
        predictor=SimpleNamespace(predictor=nn.Linear(2, 1)),
    )
    minimol_module._load_minimol_checkpoint(target, checkpoint_path)

    assert torch.equal(target.predictor.predictor.weight, source.weight)
    assert torch.equal(target.predictor.predictor.bias, source.bias)


def test_prepare_inputs_extracts_ordered_fingerprints(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """Raw SMILES become ordered float32 fingerprint rows."""
    encoder = minimol_module.MiniMolSmilesEncoder(batch_size=2, latent_dim=4)

    prepared = encoder.prepare_inputs(["CC", "CO"], device=torch.device("cpu"))

    assert prepared.shape == (2, 512)
    assert prepared.dtype == torch.float32
    assert prepared[:, 0].tolist() == pytest.approx(
        [float(sum(map(ord, value))) for value in ["CC", "CO"]]
    )
    assert fake_minimol[0].batch_size == 2
    assert fake_minimol[0].calls == [["CC", "CO"]]
    assert fake_minimol[0].grad_enabled == [False]


def test_fixed_encoder_returns_raw_fingerprints(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """The fixed encoder returns pooled512 without a projection module."""
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        batch_size=2,
        cache_size=8,
    )

    features = encoder.encode(["CC", "CO"], device=torch.device("cpu"))

    assert encoder.feature_dim == 512
    assert features.shape == (2, 512)
    assert not hasattr(encoder, "projection")
    assert fake_minimol[0].calls == [["CC", "CO"]]


def test_prepare_inputs_handles_empty_batches_without_model_call(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """An empty candidate batch should preserve its shape without extraction."""
    encoder = minimol_module.MiniMolSmilesEncoder()

    prepared = encoder.prepare_inputs([], device=torch.device("cpu"))

    assert encoder.latent_dim == 32
    assert prepared.shape == (0, 512)
    assert prepared.dtype == torch.float32
    assert fake_minimol[0].calls == []


def test_prepare_inputs_rejects_non_string_values(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """MiniMol's raw SMILES boundary should reject non-string values."""
    encoder = minimol_module.MiniMolSmilesEncoder()

    with pytest.raises(ValueError, match="string inputs"):
        encoder.prepare_inputs(["CC", 1], device=torch.device("cpu"))

    assert fake_minimol[0].calls == []


def test_fingerprint_cache_deduplicates_requests_and_evicts_lru_entries(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """The bounded cache should deduplicate inputs and evict least-recent rows."""
    encoder = minimol_module.MiniMolSmilesEncoder(cache_size=2)

    encoder.prepare_inputs(["CC", "CC", "CO"], device=torch.device("cpu"))
    encoder.prepare_inputs(["CC", "CN"], device=torch.device("cpu"))
    encoder.prepare_inputs(["CO"], device=torch.device("cpu"))

    assert fake_minimol[0].calls == [["CC", "CO"], ["CN"], ["CO"]]


def test_projection_is_trainable_while_minimol_inference_is_frozen(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """Backbone extraction runs without gradients while projection receives them."""
    encoder = minimol_module.MiniMolSmilesEncoder(latent_dim=3)
    inputs = encoder.prepare_inputs(["CC"], device=torch.device("cpu"))

    output = encoder(inputs)
    output.sum().backward()

    assert output.shape == (1, 3)
    assert encoder.projection.weight.grad is not None
    assert encoder.projection.bias.grad is not None
    assert fake_minimol[0].grad_enabled == [False]


@pytest.mark.parametrize(
    ("output", "exception", "message"),
    [
        (torch.zeros(1, 512), TypeError, "list of fingerprint"),
        ([torch.zeros(512), torch.zeros(512)], ValueError, "fingerprints"),
        ([torch.zeros(511)], ValueError, "must have shape"),
        ([torch.full((512,), float("nan"))], ValueError, "non-finite"),
    ],
)
def test_prepare_inputs_validates_minimol_outputs(
    monkeypatch: pytest.MonkeyPatch,
    output: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Malformed or non-finite upstream fingerprints should fail explicitly."""

    class _MalformedMiniMol:
        def __init__(
            self,
            *,
            batch_size: int,
            checkpoint_path: object = None,
        ) -> None:
            del batch_size
            del checkpoint_path

        def __call__(self, smiles: list[str]) -> object:
            del smiles
            return output

    monkeypatch.setattr(
        minimol_module,
        "_load_minimol",
        lambda: _MalformedMiniMol,
    )
    encoder = minimol_module.MiniMolSmilesEncoder()

    with pytest.raises(exception, match=message):
        encoder.prepare_inputs(["CC"], device=torch.device("cpu"))


@pytest.mark.parametrize("surrogate_type", [ExactDKLSurrogate, VariationalDKLSurrogate])
def test_minimol_encoder_trains_exact_and_variational_dkl(
    fake_minimol: list[_FakeMiniMol],
    surrogate_type: type[ExactDKLSurrogate] | type[VariationalDKLSurrogate],
) -> None:
    """Both DKL variants should fit and predict through MiniMol fingerprints."""
    encoder = minimol_module.MiniMolSmilesEncoder(latent_dim=4, cache_size=8)
    kwargs: dict[str, object] = {}
    if surrogate_type is VariationalDKLSurrogate:
        kwargs["num_inducing"] = 2
    surrogate = surrogate_type(
        encoder=encoder,
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        standardize_outputs=False,
        **kwargs,
    )

    surrogate.fit(
        [
            Observation(x="CC", y=1.0),
            Observation(x="CO", y=2.0),
            Observation(x="CN", y=3.0),
        ]
    )
    prediction = surrogate.predict(
        [Candidate(x="CC"), Candidate(x="CCC")],
    )

    assert surrogate.is_fitted()
    assert len(prediction["mean"]) == 2
    assert len(prediction["std"]) == 2
    assert torch.isfinite(torch.tensor(prediction["mean"])).all()
    assert torch.isfinite(torch.tensor(prediction["std"])).all()


def test_minimol_encoder_runs_a_smiles_active_learning_round(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """A string candidate pool can complete one active-learning round."""
    surrogate = ExactDKLSurrogate(
        encoder=minimol_module.MiniMolSmilesEncoder(latent_dim=4),
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        standardize_outputs=False,
    )
    dataset = ListDataset()
    dataset.add_observations(
        [
            Observation(x="CC", y=2.0, fidelity=1),
            Observation(x="CO", y=2.0, fidelity=1),
        ]
    )

    active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=UpperConfidenceBound(beta=1.0),
        sampler=PoolScoreSampler(
            candidate_pool=[
                Candidate(x="CCC", fidelity=1),
                Candidate(x="CN", fidelity=1),
            ],
            num_samples=2,
        ),
        selector=TopKAcquisitionSelector(num_samples=1),
        oracle=MultiFidelityOracle(
            fidelity_configs={
                1: {
                    "cost_per_sample": 1.0,
                    "fidelity_confidence": 1.0,
                    "score_fn": lambda value: float(len(value)),
                }
            }
        ),
        budget=Budget(
            available_budget=1.0,
            schedule=lambda _: 1.0,
            max_rounds=1,
        ),
    )

    assert len(dataset.get_observations_iterable()) == 3

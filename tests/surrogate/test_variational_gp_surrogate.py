"""Tests for the fixed-feature variational GP surrogate."""

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.config import VariationalGPTrainingConfig
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.variational_gp import VariationalGPSurrogate
from activelearning.utils.types import Candidate, Observation

FIDELITY_CONFIDENCES = {1: 0.25, 2: 0.5, 3: 1.0}
TRAINING = VariationalGPTrainingConfig(epochs=2, lr=1e-2)
POINTS = [[0.0, 0.1, 0.2], [1.0, 0.9, 0.8], [0.5, 0.4, 0.6]]
VALUES = [100.0, 102.0, 104.0]


class _VectorEncoder(FixedEncoder):
    """Return raw three-dimensional vectors as float32 features."""

    feature_dim = 3

    def encode(self, values: Sequence[Any], *, device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=device)


def _make_surrogate(is_multi_fidelity: bool = False) -> VariationalGPSurrogate:
    surrogate = VariationalGPSurrogate(
        encoder=_VectorEncoder(),
        training_params=TRAINING,
        num_inducing=4,
        is_multi_fidelity=is_multi_fidelity,
        target_fidelity=3 if is_multi_fidelity else None,
    )
    if is_multi_fidelity:
        surrogate.set_fidelity_confidences(FIDELITY_CONFIDENCES)
    return surrogate


def _observations(fidelities: Sequence[int] = (0, 0, 0)) -> list[Observation]:
    return [
        Observation(x=x, y=y, fidelity=fidelity)
        for x, y, fidelity in zip(POINTS, VALUES, fidelities)
    ]


def test_predict_before_fit_raises() -> None:
    surrogate = _make_surrogate()
    assert not surrogate.is_fitted()
    assert surrogate.get_state_dict() is None
    with pytest.raises(RuntimeError):
        surrogate.predict([Candidate(x=POINTS[0])])


def test_multi_fidelity_requires_target_fidelity() -> None:
    with pytest.raises(ValueError, match="target_fidelity"):
        VariationalGPSurrogate(
            encoder=_VectorEncoder(),
            training_params=TRAINING,
            is_multi_fidelity=True,
        )


def test_fit_predicts_on_original_target_scale() -> None:
    surrogate = _make_surrogate()
    surrogate.fit(_observations())
    candidates = [Candidate(x=x) for x in POINTS[:2]]

    prediction = surrogate.predict(candidates)

    assert surrogate.is_fitted()
    assert surrogate._y_mean == pytest.approx(102.0)
    assert surrogate._y_std == pytest.approx(2.0)
    # A standardized prediction would sit near 0 instead of near the data.
    assert all(abs(mean - 102.0) < 10.0 for mean in prediction["mean"])
    assert all(std > 0.0 for std in prediction["std"])

    train_X, train_Y = surrogate.get_train_data()
    assert train_X.shape == (3, 3)
    assert train_Y.squeeze(-1).tolist() == pytest.approx(VALUES)
    with torch.no_grad():
        posterior = surrogate.get_model().posterior(
            surrogate.encode_candidates(candidates), observation_noise=True
        )
    assert posterior.mean.squeeze(-1).tolist() == pytest.approx(prediction["mean"])
    assert posterior.variance.sqrt().squeeze(-1).tolist() == pytest.approx(
        prediction["std"]
    )


def test_state_dict_restores_predictions_on_next_fit() -> None:
    fitted = _make_surrogate()
    fitted.fit(_observations())
    candidates = [Candidate(x=[0.2, 0.3, 0.4])]

    restored = _make_surrogate()
    # Loading before the model exists defers the restore to the next fit.
    restored.load_state_dict(fitted.get_state_dict())
    assert not restored.is_fitted()
    restored.fit(_observations())

    expected, actual = fitted.predict(candidates), restored.predict(candidates)
    assert actual["mean"] == pytest.approx(expected["mean"])
    assert actual["std"] == pytest.approx(expected["std"])
    with torch.no_grad():
        posterior = restored.get_model().posterior(
            restored.encode_candidates(candidates), observation_noise=True
        )
    assert posterior.mean.squeeze(-1).tolist() == pytest.approx(expected["mean"])


def test_runtime_context_controls_dtype() -> None:
    surrogate = _make_surrogate()
    context = RuntimeContext(dtype=torch.float32)
    surrogate.bind_runtime_context(context)
    surrogate.fit(_observations())

    assert surrogate._encoder._runtime_context is context
    assert surrogate.encode_candidates([Candidate(x=POINTS[0])]).dtype == torch.float32
    assert next(surrogate._gp_model.parameters()).dtype == torch.float32
    assert next(surrogate._likelihood.parameters()).dtype == torch.float32
    assert all(
        math.isfinite(mean) for mean in surrogate.predict(_observations())["mean"]
    )


def test_default_dtype_casts_float32_features_to_float64() -> None:
    surrogate = _make_surrogate()
    surrogate.fit(_observations())

    assert surrogate.get_train_data()[0].dtype == torch.float64
    assert next(surrogate._gp_model.parameters()).dtype == torch.float64


def test_multi_fidelity_appends_fidelity_confidence() -> None:
    surrogate = _make_surrogate(is_multi_fidelity=True)
    surrogate.fit(_observations([1, 2, 3]))

    encoded = surrogate.encode_candidates(
        [Candidate(x=POINTS[0], fidelity=1), Candidate(x=POINTS[1], fidelity=3)]
    )

    assert encoded.shape == (2, 4)
    assert encoded[:, -1].tolist() == pytest.approx([0.25, 1.0])
    assert surrogate.get_fidelity_dimension() == 3
    assert surrogate.get_target_fidelity_value() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Missing fidelity confidence"):
        surrogate.encode_candidates([Candidate(x=POINTS[0], fidelity=7)])


def test_multi_fidelity_fit_requires_confidences() -> None:
    surrogate = VariationalGPSurrogate(
        encoder=_VectorEncoder(),
        training_params=TRAINING,
        is_multi_fidelity=True,
        target_fidelity=3,
    )
    with pytest.raises(ValueError, match="fidelity confidences"):
        surrogate.fit(_observations([1, 2, 3]))


def test_multi_fidelity_max_value_entropy_scores_are_finite() -> None:
    from activelearning.acquisition.botorch.botorch_multifidelity import (
        QMultiFidelityLowerBoundMaxValueEntropy,
    )
    from activelearning.acquisition.botorch.candidate_set import (
        TrainDataCandidateSetSpec,
    )

    surrogate = _make_surrogate(is_multi_fidelity=True)
    observations = _observations([1, 2, 3])
    surrogate.fit(observations)
    acquisition = QMultiFidelityLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(),
        num_fantasies=2,
        num_mv_samples=5,
        num_y_samples=16,
    )
    acquisition.update(surrogate, observations)

    scores = acquisition.score(
        [Candidate(x=POINTS[0], fidelity=2), Candidate(x=POINTS[1], fidelity=3)]
    )

    assert len(scores) == 2
    assert all(math.isfinite(score) for score in scores)

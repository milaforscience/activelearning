from typing import Any

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Observation, Candidate
from activelearning.dataset.list_dataset import ListDataset


@pytest.fixture
def single_fidelity_observations():
    return [
        Observation(x=[1.0, 2.0], y=5.0),
        Observation(x=[2.0, 3.0], y=6.0),
        Observation(x=[3.0, 4.0], y=7.0),
    ]


@pytest.fixture
def multi_fidelity_observations():
    return [
        Observation(x=[1.0, 2.0], y=4.5, fidelity=0),
        Observation(x=[1.0, 2.0], y=5.0, fidelity=1),
        Observation(x=[2.0, 3.0], y=5.5, fidelity=0),
        Observation(x=[2.0, 3.0], y=6.0, fidelity=1),
    ]


@pytest.fixture
def scalar_single_fidelity_observations():
    return [
        Observation(x=1.0, y=5.0),
        Observation(x=2.0, y=6.0),
        Observation(x=3.0, y=7.0),
    ]


@pytest.fixture
def scalar_multi_fidelity_observations():
    return [
        Observation(x=1.0, y=4.5, fidelity=0),
        Observation(x=1.0, y=5.0, fidelity=1),
        Observation(x=2.0, y=5.5, fidelity=0),
        Observation(x=2.0, y=6.0, fidelity=1),
    ]


@pytest.fixture
def multi_output_observations():
    return [
        Observation(x=[1.0, 2.0], y=[5.0, 50.0]),
        Observation(x=[2.0, 3.0], y=[6.0, 60.0]),
        Observation(x=[3.0, 4.0], y=[7.0, 70.0]),
    ]


@pytest.fixture
def dataset_with_observations(single_fidelity_observations):
    """Create a dataset with observations."""
    dataset = ListDataset()
    dataset.add_observations(single_fidelity_observations)
    return dataset


def test_single_fidelity_fit_and_predict(single_fidelity_observations):
    surrogate = BoTorchGPSurrogate()

    # Test Fitting
    surrogate.fit(single_fidelity_observations)
    assert isinstance(surrogate.model, SingleTaskGP), "Should route to SingleTaskGP"

    # Test Prediction
    candidates = [Candidate(x=[1.5, 2.5]), Candidate(x=[2.5, 3.5])]
    predictions = surrogate.predict(candidates)

    assert "mean" in predictions
    assert "std" in predictions
    assert "posterior" in predictions

    assert len(predictions["mean"]) == 2
    assert len(predictions["std"]) == 2
    assert isinstance(predictions["mean"], list)
    assert isinstance(predictions["std"], list)


def test_multi_fidelity_fit_and_predict(multi_fidelity_observations):
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})

    # Test Fitting
    surrogate.fit(multi_fidelity_observations)
    assert isinstance(surrogate.model, SingleTaskMultiFidelityGP), (
        "Should route to SingleTaskMultiFidelityGP"
    )

    # Test Prediction
    candidates = [
        Candidate(x=[1.5, 2.5], fidelity=1),
        Candidate(x=[2.5, 3.5], fidelity=1),
    ]
    predictions = surrogate.predict(candidates)

    assert len(predictions["mean"]) == 2


def test_tensor_parsing_shapes(multi_fidelity_observations):
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.0, 1: 1.0})

    # Manually trigger parsing to check internal tensor shapes
    train_X, train_Y, is_multi_fidelity = surrogate._parse_observations(
        multi_fidelity_observations
    )

    # 4 observations. x has 2 dims, plus 1 dim for the fidelity column
    assert train_X.shape == (4, 3)
    assert train_Y.shape == (4, 1)

    # Check that the last column of train_X matches the mapped confidences (0.0, 1.0, 0.0, 1.0)
    expected_fidelities = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.all(train_X[:, -1] == expected_fidelities)


def test_scalar_observation_parsing_uses_column_vector(
    scalar_single_fidelity_observations,
):
    """Test that scalar observations are reshaped to ``(N, 1)``."""
    surrogate = BoTorchGPSurrogate()

    train_X, train_Y, is_multi_fidelity = surrogate._parse_observations(
        scalar_single_fidelity_observations
    )

    assert train_X.shape == (3, 1)
    assert train_Y.shape == (3, 1)
    assert is_multi_fidelity is False


def test_scalar_multi_fidelity_observation_parsing_uses_column_vector(
    scalar_multi_fidelity_observations,
):
    """Test that scalar multi-fidelity observations keep the observation axis."""
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.25, 1: 0.95})

    train_X, train_Y, is_multi_fidelity = surrogate._parse_observations(
        scalar_multi_fidelity_observations
    )

    assert train_X.shape == (4, 2)
    assert train_Y.shape == (4, 1)
    assert is_multi_fidelity is True
    assert torch.allclose(
        train_X[:, -1], torch.tensor([0.25, 0.95, 0.25, 0.95], dtype=torch.float64)
    )


def test_multi_output_observation_parsing_preserves_output_dimension(
    multi_output_observations,
):
    """Test that vector-valued outputs keep their output dimension."""
    surrogate = BoTorchGPSurrogate()

    train_X, train_Y, is_multi_fidelity = surrogate._parse_observations(
        multi_output_observations
    )

    assert train_X.shape == (3, 2)
    assert train_Y.shape == (3, 2)
    assert is_multi_fidelity is False
    assert torch.allclose(
        train_Y,
        torch.tensor(
            [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]],
            dtype=torch.float64,
        ),
    )


def test_mixed_fidelity_raises_error():
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    bad_observations = [
        Observation(x=[1.0], y=2.0, fidelity=0),
        Observation(x=[2.0], y=3.0),  # Missing fidelity!
    ]

    with pytest.raises(ValueError, match="Mixed fidelity specification detected"):
        surrogate.fit(bad_observations)


def test_update_unfitted_model(single_fidelity_observations):
    """Test that updates_from_latest() returns False when model is None, causing full fit."""
    surrogate = BoTorchGPSurrogate()

    dataset = ListDataset()
    dataset.add_observations(single_fidelity_observations)

    # When model is None, updates_from_latest() must return False (no model to update)
    assert not surrogate.updates_from_latest()
    surrogate.fit(dataset.get_observations_iterable())

    assert surrogate.model is not None, "Model should be initialized"
    assert isinstance(surrogate.model, SingleTaskGP), "Should route to SingleTaskGP"


def test_update_fitted_model(single_fidelity_observations):
    """Test that update works on already fitted model."""
    surrogate = BoTorchGPSurrogate()

    # 1. Fit on the first two observations
    initial_train = single_fidelity_observations[:2]
    surrogate.fit(initial_train)
    initial_model = surrogate.model

    # 2. Update with the third observation using dataset
    dataset = ListDataset()
    dataset.add_observations(initial_train)
    dataset.add_observations([single_fidelity_observations[2]])

    # use_partial_updates=False (default) → updates_from_latest() is False → full refit
    assert not surrogate.updates_from_latest()
    surrogate.fit(dataset.get_observations_iterable())

    # With use_partial_updates=False (default), it should refit from scratch
    # Model will be a new instance
    assert surrogate.model is not initial_model, "Update should create a new model"

    # 3. Verify the updated model can still predict accurately
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)

    assert "mean" in predictions
    assert len(predictions["mean"]) == 1


def test_update_with_dataset_full_refit(dataset_with_observations):
    """Test updates_from_latest()=False with use_partial_updates=False always does full refit."""
    surrogate = BoTorchGPSurrogate(use_partial_updates=False)

    # Initial fit — loop always calls fit() when updates_from_latest() is False
    assert not surrogate.updates_from_latest()
    surrogate.fit(dataset_with_observations.get_observations_iterable())
    assert surrogate.model is not None
    assert isinstance(surrogate.model, SingleTaskGP)

    # Store reference to model
    first_model = surrogate.model

    # Add more observations
    dataset_with_observations.add_observations([Observation(x=[4.0, 5.0], y=8.0)])

    # Full refit again
    assert not surrogate.updates_from_latest()
    surrogate.fit(dataset_with_observations.get_observations_iterable())

    # Model should be a new instance (full refit creates new model)
    assert surrogate.model is not first_model

    # Verify predictions work
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1


def test_direct_update_full_refits_when_partial_updates_disabled(
    single_fidelity_observations, monkeypatch
):
    """Test that direct update() honors use_partial_updates=False."""
    surrogate = BoTorchGPSurrogate(use_partial_updates=False)
    surrogate.fit(single_fidelity_observations[:2])
    initial_model = surrogate.model
    assert initial_model is not None

    def fail_condition_on_observations(*args, **kwargs):
        raise AssertionError("condition_on_observations should not be called")

    monkeypatch.setattr(
        initial_model,
        "condition_on_observations",
        fail_condition_on_observations,
    )

    surrogate.update([single_fidelity_observations[2]])

    assert surrogate.model is not initial_model
    assert surrogate._train_X.shape == (3, 2)
    assert surrogate._train_Y.shape == (3, 1)


def test_train_data_after_multi_output_fit(multi_output_observations):
    """Test that multi-output training tensors retain their output width."""
    surrogate = BoTorchGPSurrogate()
    surrogate.fit(multi_output_observations)

    assert surrogate._train_X.shape == (3, 2)
    assert surrogate._train_Y.shape == (3, 2)


def test_multi_output_predict_preserves_nested_output_shape(multi_output_observations):
    """Test that predict preserves the output dimension for multi-output models."""
    surrogate = BoTorchGPSurrogate()
    surrogate.fit(multi_output_observations)

    predictions = surrogate.predict([Candidate(x=[1.5, 2.5]), Candidate(x=[2.5, 3.5])])

    assert len(predictions["mean"]) == 2
    assert len(predictions["std"]) == 2
    assert len(predictions["mean"][0]) == 2
    assert len(predictions["std"][0]) == 2
    assert isinstance(predictions["mean"][0], list)
    assert isinstance(predictions["std"][0], list)


def test_direct_update_full_refit_runs_custom_optimizer(single_fidelity_observations):
    """Test that full-refit updates reuse the configured optimization path."""
    call_tracker: dict[str, list[Any]] = {"models": []}

    def my_mock_optimizer(mll, **kwargs):
        call_tracker["models"].append(mll.model)

    surrogate = BoTorchGPSurrogate(
        use_partial_updates=False,
        custom_fit_function=my_mock_optimizer,
    )
    surrogate.fit(single_fidelity_observations[:2])

    assert len(call_tracker["models"]) == 1

    surrogate.update([single_fidelity_observations[2]])

    assert len(call_tracker["models"]) == 2
    assert surrogate.model is call_tracker["models"][-1]


def test_update_with_dataset_partial_updates(dataset_with_observations):
    """Test use_partial_updates=True: first call does full fit, subsequent calls use update()."""
    surrogate = BoTorchGPSurrogate(use_partial_updates=True)

    # First call: model is None → updates_from_latest() returns False → full fit
    assert not surrogate.updates_from_latest()
    surrogate.fit(dataset_with_observations.get_observations_iterable())
    assert surrogate.model is not None

    # Store reference to model
    first_model = surrogate.model

    # Add more observations
    new_obs = [Observation(x=[4.0, 5.0], y=8.0)]
    dataset_with_observations.add_observations(new_obs)

    # Second call: model exists → updates_from_latest() returns True → incremental update
    assert surrogate.updates_from_latest()
    surrogate.update(dataset_with_observations.get_latest_observations_iterable())

    # With partial updates, condition_on_observations returns a new model
    assert surrogate.model is not first_model

    # Verify predictions still work
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1


def test_partial_update_refreshes_mll(single_fidelity_observations):
    """Test that partial updates keep the stored MLL synchronized with the model."""
    surrogate = BoTorchGPSurrogate(use_partial_updates=True)
    surrogate.fit(single_fidelity_observations[:2])

    old_mll = surrogate.mll
    surrogate.update([single_fidelity_observations[2]])

    assert surrogate.mll is not None
    assert surrogate.mll is not old_mll
    assert surrogate.model is surrogate.mll.model
    assert surrogate.model is not None
    assert surrogate.model.likelihood is surrogate.mll.likelihood
    assert surrogate.model.training is False


def test_update_first_call_always_fits(single_fidelity_observations):
    """Test that updates_from_latest() returns False when model is None, forcing full fit."""
    # Test with use_partial_updates=True — first call: model is None
    surrogate_partial = BoTorchGPSurrogate(use_partial_updates=True)
    dataset = ListDataset()
    dataset.add_observations(single_fidelity_observations)

    assert not surrogate_partial.updates_from_latest(), (
        "No model yet → must do full fit"
    )
    surrogate_partial.fit(dataset.get_observations_iterable())
    assert surrogate_partial.model is not None

    # Test with use_partial_updates=False — always full fit
    surrogate_full = BoTorchGPSurrogate(use_partial_updates=False)
    dataset2 = ListDataset()
    dataset2.add_observations(single_fidelity_observations)

    assert not surrogate_full.updates_from_latest()
    surrogate_full.fit(dataset2.get_observations_iterable())
    assert surrogate_full.model is not None


def test_update_predictions_correct_both_modes(single_fidelity_observations):
    """Test that predictions are reasonable with both partial and full update modes."""
    # Create two surrogates with different update strategies
    surrogate_full = BoTorchGPSurrogate(use_partial_updates=False)
    surrogate_partial = BoTorchGPSurrogate(use_partial_updates=True)

    # Initial data
    initial_obs = single_fidelity_observations[:2]
    dataset_full = ListDataset()
    dataset_full.add_observations(initial_obs)
    dataset_partial = ListDataset()
    dataset_partial.add_observations(initial_obs)

    # Initial fit for both (model is None → updates_from_latest() is False)
    surrogate_full.fit(dataset_full.get_observations_iterable())
    surrogate_partial.fit(dataset_partial.get_observations_iterable())

    # Add new observations
    new_obs = [single_fidelity_observations[2]]
    dataset_full.add_observations(new_obs)
    dataset_partial.add_observations(new_obs)

    # Full surrogate: always fit on all observations
    assert not surrogate_full.updates_from_latest()
    surrogate_full.fit(dataset_full.get_observations_iterable())

    # Partial surrogate: model exists → incremental update on latest only
    assert surrogate_partial.updates_from_latest()
    surrogate_partial.update(dataset_partial.get_latest_observations_iterable())

    # Both should be able to make predictions
    candidates = [Candidate(x=[2.5, 3.5])]
    pred_full = surrogate_full.predict(candidates)
    pred_partial = surrogate_partial.predict(candidates)

    assert "mean" in pred_full and "mean" in pred_partial
    assert len(pred_full["mean"]) == 1
    assert len(pred_partial["mean"]) == 1

    # Both should return reasonable values (not NaN or extreme)
    assert not torch.isnan(torch.tensor(pred_full["mean"])).any()
    assert not torch.isnan(torch.tensor(pred_partial["mean"])).any()


def test_fidelity_confidence_mapping(multi_fidelity_observations):
    """Test that integer fidelity IDs are correctly mapped to continuous confidences."""
    surrogate = BoTorchGPSurrogate()

    # Map fidelity 0 -> 0.5, fidelity 1 -> 0.95
    surrogate.set_fidelity_confidences({0: 0.5, 1: 0.95})

    train_X, _, is_multi_fidelity = surrogate._parse_observations(
        multi_fidelity_observations
    )

    # The last column should now contain the mapped floats, not the 0 and 1 IDs
    expected_confidences = torch.tensor([0.5, 0.95, 0.5, 0.95], dtype=torch.float64)
    assert torch.allclose(train_X[:, -1], expected_confidences)


def test_multi_fidelity_fit_without_confidences_raises_value_error(
    multi_fidelity_observations,
):
    """Test that missing explicit fidelity mappings fail fast with ValueError."""
    surrogate = BoTorchGPSurrogate()

    with pytest.raises(ValueError, match="no fidelity_confidences mapping"):
        surrogate.fit(multi_fidelity_observations)


def test_custom_covar_module_routing(multi_fidelity_observations):
    """Test that providing a covar_module forces routing to SingleTaskGP in MF setting."""
    from gpytorch.kernels import RBFKernel

    surrogate = BoTorchGPSurrogate(covar_module=RBFKernel())
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)

    # covar_module provided → should use SingleTaskGP, not SingleTaskMultiFidelityGP
    assert isinstance(surrogate.model, SingleTaskGP), (
        "Should use SingleTaskGP when covar_module is provided"
    )
    assert surrogate._is_multi_fidelity is True, (
        "Should still track that the data has fidelity columns"
    )


def test_state_dict_extraction_and_injection(single_fidelity_observations):
    """Test that hyperparameters can be extracted and perfectly reloaded."""
    # 1. Train a base model
    base_surrogate = BoTorchGPSurrogate()
    base_surrogate.fit(single_fidelity_observations)
    saved_state = base_surrogate.state_dict()

    assert saved_state is not None

    # 2. Initialize a new model, explicitly freezing hyperparameters
    new_surrogate = BoTorchGPSurrogate(optimize_hyperparameters=False)

    # Inject the state dict BEFORE fitting (tests the _pending_state_dict logic)
    new_surrogate.load_state_dict(saved_state)
    new_surrogate.fit(single_fidelity_observations)

    # 3. Verify predictions are perfectly identical
    candidates = [Candidate(x=[2.5, 3.5])]
    pred_base = base_surrogate.predict(candidates)
    pred_new = new_surrogate.predict(candidates)

    assert torch.allclose(
        torch.tensor(pred_base["mean"]), torch.tensor(pred_new["mean"])
    ), "Predictions should match exactly when state_dict is loaded"


def test_custom_fit_function(single_fidelity_observations):
    """Test that a custom optimization loop is called instead of BoTorch defaults."""
    call_tracker: dict[str, Any] = {"called": False, "kwargs": None}

    # Create a dummy fit function
    def my_mock_optimizer(mll, **kwargs):
        call_tracker["called"] = True
        call_tracker["kwargs"] = kwargs

    surrogate = BoTorchGPSurrogate(
        custom_fit_function=my_mock_optimizer,
        fit_kwargs={"learning_rate": 0.01, "epochs": 50},
    )

    surrogate.fit(single_fidelity_observations)

    assert call_tracker["called"] is True, "Custom fit function was not triggered"
    assert call_tracker["kwargs"] == {"learning_rate": 0.01, "epochs": 50}, (
        "fit_kwargs were not passed correctly"
    )


def test_custom_fit_function_with_optimize_disabled_raises():
    """Test that providing custom_fit_function with optimize_hyperparameters=False raises."""
    with pytest.raises(
        ValueError,
        match="custom_fit_function is provided but optimize_hyperparameters=False",
    ):
        BoTorchGPSurrogate(
            custom_fit_function=lambda mll: None,
            optimize_hyperparameters=False,
        )


def test_predict_before_fit_raises():
    """Test that predict raises RuntimeError when called before the model is fitted."""
    surrogate = BoTorchGPSurrogate()
    candidates = [Candidate(x=[1.0, 2.0])]

    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        surrogate.predict(candidates)


def test_parse_candidates_missing_fidelity_raises(multi_fidelity_observations):
    """Test that predict raises ValueError when multi-fidelity model gets candidates without fidelity."""
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)

    # Candidates missing fidelity values
    candidates = [Candidate(x=[1.5, 2.5]), Candidate(x=[2.5, 3.5])]
    with pytest.raises(ValueError, match="All candidates must provide a fidelity"):
        surrogate.predict(candidates)


def test_encode_candidates_mixed_fidelity_candidates_raise(
    multi_fidelity_observations,
):
    """Test that mixed candidate fidelity specification is rejected explicitly."""
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)

    candidates = [
        Candidate(x=[1.5, 2.5], fidelity=1),
        Candidate(x=[2.5, 3.5]),
    ]

    with pytest.raises(ValueError, match="Mixed fidelity specification detected"):
        surrogate.encode_candidates(candidates)


def test_encode_candidates_unknown_fidelity_raises(multi_fidelity_observations):
    """Test that unknown candidate fidelities raise a clear validation error."""
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)

    with pytest.raises(ValueError, match="Invalid candidate indices: \\[0\\]"):
        surrogate.encode_candidates([Candidate(x=[1.5, 2.5], fidelity=999)])


def test_parse_candidates_fidelity_confidence_mapping(multi_fidelity_observations):
    """Test that fidelity confidence mapping is applied to candidates, not just observations."""
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 0.95})
    surrogate.fit(multi_fidelity_observations)

    candidates = [
        Candidate(x=[1.5, 2.5], fidelity=0),
        Candidate(x=[2.5, 3.5], fidelity=1),
    ]
    test_X = surrogate.encode_candidates(candidates)

    expected_confidences = torch.tensor([0.5, 0.95], dtype=torch.float64)
    assert torch.allclose(test_X[:, -1], expected_confidences)


def test_scalar_candidate_encoding_uses_column_vector(
    scalar_single_fidelity_observations,
):
    """Test that scalar candidates are reshaped to ``(N, 1)``."""
    surrogate = BoTorchGPSurrogate()
    surrogate.fit(scalar_single_fidelity_observations)

    test_X = surrogate.encode_candidates([Candidate(x=1.5), Candidate(x=2.5)])

    assert test_X.shape == (2, 1)


def test_scalar_multi_fidelity_candidate_encoding_uses_column_vector(
    scalar_multi_fidelity_observations,
):
    """Test that scalar multi-fidelity candidates preserve the candidate axis."""
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.25, 1: 0.95})
    surrogate.fit(scalar_multi_fidelity_observations)

    test_X = surrogate.encode_candidates(
        [Candidate(x=1.5, fidelity=0), Candidate(x=2.5, fidelity=1)]
    )

    assert test_X.shape == (2, 2)
    assert torch.allclose(
        test_X[:, -1], torch.tensor([0.25, 0.95], dtype=torch.float64)
    )


def test_state_dict_before_fit_returns_none():
    """Test that state_dict returns None when the model has not been fitted."""
    surrogate = BoTorchGPSurrogate()
    assert surrogate.state_dict() is None


def test_load_state_dict_after_fit(single_fidelity_observations):
    """Test that load_state_dict updates an already-fitted model in-place."""
    base = BoTorchGPSurrogate()
    base.fit(single_fidelity_observations)
    saved_state = base.state_dict()
    assert saved_state is not None  # state_dict() returns None only before fitting

    # Fit a second surrogate independently, then overwrite its state
    other = BoTorchGPSurrogate(optimize_hyperparameters=False)
    other.fit(single_fidelity_observations)
    other.load_state_dict(saved_state)

    # Predictions should now match the base model exactly
    candidates = [Candidate(x=[2.5, 3.5])]
    pred_base = base.predict(candidates)
    pred_other = other.predict(candidates)

    assert torch.allclose(
        torch.tensor(pred_base["mean"]),
        torch.tensor(pred_other["mean"]),
    ), "Loaded state dict should make predictions identical to the source model"


def test_scale_inputs_false(single_fidelity_observations):
    """Test that scale_inputs=False skips input scaling without errors."""
    surrogate = BoTorchGPSurrogate(scale_inputs=False)
    surrogate.fit(single_fidelity_observations)

    assert surrogate.model is not None
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1
    assert not torch.isnan(torch.tensor(predictions["mean"])).any()


def test_standardize_outputs_false(single_fidelity_observations):
    """Test that standardize_outputs=False skips output standardization without errors."""
    surrogate = BoTorchGPSurrogate(standardize_outputs=False)
    surrogate.fit(single_fidelity_observations)

    assert surrogate.model is not None
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1
    assert not torch.isnan(torch.tensor(predictions["mean"])).any()


def test_set_fidelity_confidences_propagates_to_covar_module():
    """Test that set_fidelity_confidences calls update_confidences on the covar_module if available."""
    import gpytorch

    received = {}

    class MockKernel(gpytorch.Module):
        def update_confidences(self, confidences: dict) -> None:
            received.update(confidences)

    surrogate = BoTorchGPSurrogate(covar_module=MockKernel())
    surrogate.set_fidelity_confidences({0: 0.3, 1: 0.9})

    assert received == {0: 0.3, 1: 0.9}, (
        "Confidences were not propagated to the covar_module"
    )


def test_set_fidelity_confidences_after_fit_rejects_changes(
    multi_fidelity_observations,
):
    """Test that fidelity encodings cannot change after fitting."""
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.3, 1: 0.9})
    surrogate.fit(multi_fidelity_observations)

    with pytest.raises(RuntimeError, match="Cannot change fidelity confidences"):
        surrogate.set_fidelity_confidences({0: 0.2, 1: 1.0})


def test_update_fallback_when_model_none(single_fidelity_observations):
    """Test that update falls back to a full fit when model is None."""
    surrogate = BoTorchGPSurrogate(use_partial_updates=True)

    # Call update directly before any fit
    surrogate.update(single_fidelity_observations)

    assert surrogate.model is not None, "Fallback full fit should initialise the model"
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1


def test_update_ignores_empty_observations(single_fidelity_observations):
    """Test that update is a no-op when the observation list is empty."""
    surrogate = BoTorchGPSurrogate(use_partial_updates=True)
    surrogate.fit(single_fidelity_observations)
    model_before = surrogate.model

    surrogate.update([])  # should not raise or change the model

    assert surrogate.model is model_before, (
        "Empty update should leave the model unchanged"
    )


def test_is_multi_fidelity_resets_between_fits(
    single_fidelity_observations, multi_fidelity_observations
):
    """Test that _is_multi_fidelity resets correctly when re-fitting on different data.

    Regression test: previously the flag was never cleared, so fitting on MF data
    followed by SF data would attempt to build a SingleTaskMultiFidelityGP without
    a fidelity column and crash.
    """
    surrogate = BoTorchGPSurrogate()

    # First fit: multi-fidelity
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)
    assert surrogate._is_multi_fidelity is True

    # Second fit: single-fidelity — must reset the flag and build SingleTaskGP
    surrogate.fit(single_fidelity_observations)
    assert surrogate._is_multi_fidelity is False
    assert isinstance(surrogate.model, SingleTaskGP)

    # Predictions must work without fidelity values
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1


def test_fit_empty_observations_is_noop():
    """fit([]) must be a no-op: no exception and surrogate stays unfitted."""
    surrogate = BoTorchGPSurrogate()
    surrogate.fit([])  # Must not raise
    assert not surrogate.is_fitted()
    assert surrogate.model is None


def test_update_raises_on_fidelity_mismatch(multi_fidelity_observations):
    """Test that update raises when new observations are missing fidelities."""
    surrogate = BoTorchGPSurrogate(use_partial_updates=True)
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)

    # New observations without fidelity values — should raise before any tensor ops
    bad_obs = [Observation(x=[1.5, 2.5], y=5.0)]
    with pytest.raises(ValueError, match="missing fidelity values"):
        surrogate.update(bad_obs)


def test_scale_inputs_excludes_fidelity_column(multi_fidelity_observations):
    """Test that Normalize is applied only to feature columns, not the fidelity column."""
    surrogate = BoTorchGPSurrogate(scale_inputs=True)
    surrogate.set_fidelity_confidences({0: 0.1, 1: 0.95})
    surrogate.fit(multi_fidelity_observations)

    assert surrogate.model is not None
    assert surrogate._train_X is not None

    transform = surrogate.model.input_transform
    n_dims = surrogate._train_X.shape[-1]  # feature dims + fidelity col
    expected_indices = list(range(n_dims - 1))  # all columns except the last

    # Use getattr to avoid ty confusing Normalize.indices (Tensor) with Tensor.indices()
    transform_indices = getattr(transform, "indices")
    assert hasattr(transform, "indices"), "Normalize should have an indices attribute"
    assert transform_indices.tolist() == expected_indices, (
        "Normalize should only cover feature columns, not the fidelity column"
    )

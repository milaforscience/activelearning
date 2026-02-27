import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

from activelearning.surrogate.botorch_surrogate import BoTorchSurrogate
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
def dataset_with_observations(single_fidelity_observations):
    """Create a dataset with observations."""
    dataset = ListDataset()
    dataset.add_observations(single_fidelity_observations)
    return dataset


def test_single_fidelity_fit_and_predict(single_fidelity_observations):
    surrogate = BoTorchSurrogate()

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
    surrogate = BoTorchSurrogate()

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
    surrogate = BoTorchSurrogate()

    # Manually trigger parsing to check internal tensor shapes
    train_X, train_Y = surrogate._parse_observations(multi_fidelity_observations)

    # 4 observations. x has 2 dims, plus 1 dim for the fidelity column
    assert train_X.shape == (4, 3)
    assert train_Y.shape == (4, 1)

    # Check that the last column of train_X matches the fidelities (0, 1, 0, 1)
    expected_fidelities = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.all(train_X[:, -1] == expected_fidelities)


def test_mixed_fidelity_raises_error():
    surrogate = BoTorchSurrogate()
    bad_observations = [
        Observation(x=[1.0], y=2.0, fidelity=0),
        Observation(x=[2.0], y=3.0),  # Missing fidelity!
    ]

    with pytest.raises(ValueError, match="all observations must have a fidelity"):
        surrogate.fit(bad_observations)


def test_update_unfitted_model(single_fidelity_observations):
    """Test that update falls back to fit when model is None."""
    surrogate = BoTorchSurrogate()

    # Create a dataset and use the new update interface
    dataset = ListDataset()
    dataset.add_observations(single_fidelity_observations)
    surrogate.update(dataset)

    assert surrogate.model is not None, "Model should be initialized"
    assert isinstance(surrogate.model, SingleTaskGP), "Should route to SingleTaskGP"


def test_update_fitted_model(single_fidelity_observations):
    """Test that update works on already fitted model."""
    surrogate = BoTorchSurrogate()

    # 1. Fit on the first two observations
    initial_train = single_fidelity_observations[:2]
    surrogate.fit(initial_train)
    initial_model = surrogate.model

    # 2. Update with the third observation using dataset
    dataset = ListDataset()
    dataset.add_observations(initial_train)
    dataset.add_observations([single_fidelity_observations[2]])
    surrogate.update(dataset)

    # With use_partial_updates=False (default), it should refit from scratch
    # Model will be a new instance
    assert surrogate.model is not initial_model, "Update should create a new model"

    # 3. Verify the updated model can still predict accurately
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)

    assert "mean" in predictions
    assert len(predictions["mean"]) == 1


def test_update_with_dataset_full_refit(dataset_with_observations):
    """Test update() with use_partial_updates=False does full refit."""
    surrogate = BoTorchSurrogate(use_partial_updates=False)

    # Initial fit
    surrogate.update(dataset_with_observations)
    assert surrogate.model is not None
    assert isinstance(surrogate.model, SingleTaskGP)

    # Store reference to model
    first_model = surrogate.model

    # Add more observations
    dataset_with_observations.add_observations([Observation(x=[4.0, 5.0], y=8.0)])

    # Update should do full refit
    surrogate.update(dataset_with_observations)

    # Model should be a new instance (full refit creates new model)
    assert surrogate.model is not first_model

    # Verify predictions work
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1


def test_update_with_dataset_partial_updates(dataset_with_observations):
    """Test update() with use_partial_updates=True uses only latest observations."""
    surrogate = BoTorchSurrogate(use_partial_updates=True)

    # Initial update (should do full fit since model doesn't exist)
    surrogate.update(dataset_with_observations)
    assert surrogate.model is not None

    # Store reference to model
    first_model = surrogate.model

    # Add more observations
    new_obs = [Observation(x=[4.0, 5.0], y=8.0)]
    dataset_with_observations.add_observations(new_obs)

    # Update should use partial update (condition_on_observations)
    surrogate.update(dataset_with_observations)

    # With partial updates, condition_on_observations returns a new model
    assert surrogate.model is not first_model

    # Verify predictions still work
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1


def test_update_first_call_always_fits(single_fidelity_observations):
    """Test that first update call always does full fit regardless of use_partial_updates flag."""
    # Test with use_partial_updates=True
    surrogate_partial = BoTorchSurrogate(use_partial_updates=True)
    dataset = ListDataset()
    dataset.add_observations(single_fidelity_observations)

    surrogate_partial.update(dataset)
    assert surrogate_partial.model is not None

    # Test with use_partial_updates=False
    surrogate_full = BoTorchSurrogate(use_partial_updates=False)
    dataset2 = ListDataset()
    dataset2.add_observations(single_fidelity_observations)

    surrogate_full.update(dataset2)
    assert surrogate_full.model is not None


def test_update_predictions_correct_both_modes(single_fidelity_observations):
    """Test that predictions are reasonable with both partial and full update modes."""
    # Create two surrogates with different update strategies
    surrogate_full = BoTorchSurrogate(use_partial_updates=False)
    surrogate_partial = BoTorchSurrogate(use_partial_updates=True)

    # Initial data
    initial_obs = single_fidelity_observations[:2]
    dataset_full = ListDataset()
    dataset_full.add_observations(initial_obs)
    dataset_partial = ListDataset()
    dataset_partial.add_observations(initial_obs)

    # Initial fit for both
    surrogate_full.update(dataset_full)
    surrogate_partial.update(dataset_partial)

    # Add new observations
    new_obs = [single_fidelity_observations[2]]
    dataset_full.add_observations(new_obs)
    dataset_partial.add_observations(new_obs)

    # Update both
    surrogate_full.update(dataset_full)
    surrogate_partial.update(dataset_partial)

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
    surrogate = BoTorchSurrogate()

    # Map fidelity 0 -> 0.5, fidelity 1 -> 0.95
    surrogate.set_fidelity_confidences({0: 0.5, 1: 0.95})

    train_X, _ = surrogate._parse_observations(multi_fidelity_observations)

    # The last column should now contain the mapped floats, not the 0 and 1 IDs
    expected_confidences = torch.tensor([0.5, 0.95, 0.5, 0.95], dtype=torch.float64)
    assert torch.allclose(train_X[:, -1], expected_confidences)


def test_custom_fidelity_kernel_routing(multi_fidelity_observations):
    """Test that custom_fidelity_kernel=True forces routing to SingleTaskGP."""
    surrogate = BoTorchSurrogate(custom_fidelity_kernel=True)
    surrogate.fit(multi_fidelity_observations)

    # Even though data is multi-fidelity, the custom kernel flag should force the fallback
    assert isinstance(surrogate.model, SingleTaskGP), (
        "Should fallback to SingleTaskGP for custom kernels"
    )
    assert surrogate._is_multi_fidelity is True, (
        "Should still track that the data has fidelity columns"
    )


def test_state_dict_extraction_and_injection(single_fidelity_observations):
    """Test that hyperparameters can be extracted and perfectly reloaded."""
    # 1. Train a base model
    base_surrogate = BoTorchSurrogate()
    base_surrogate.fit(single_fidelity_observations)
    saved_state = base_surrogate.state_dict()

    assert saved_state is not None

    # 2. Initialize a new model, explicitly freezing hyperparameters
    new_surrogate = BoTorchSurrogate(optimize_hyperparameters=False)

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
    call_tracker = {"called": False, "kwargs": None}

    # Create a dummy fit function
    def my_mock_optimizer(mll, **kwargs):
        call_tracker["called"] = True
        call_tracker["kwargs"] = kwargs

    surrogate = BoTorchSurrogate(
        custom_fit_function=my_mock_optimizer,
        fit_kwargs={"learning_rate": 0.01, "epochs": 50},
    )

    surrogate.fit(single_fidelity_observations)

    assert call_tracker["called"] is True, "Custom fit function was not triggered"
    assert call_tracker["kwargs"] == {"learning_rate": 0.01, "epochs": 50}, (
        "fit_kwargs were not passed correctly"
    )


def test_predict_before_fit_raises():
    """Test that predict raises RuntimeError when called before the model is fitted."""
    surrogate = BoTorchSurrogate()
    candidates = [Candidate(x=[1.0, 2.0])]

    with pytest.raises(RuntimeError, match="fitted before calling predict"):
        surrogate.predict(candidates)


def test_parse_candidates_missing_fidelity_raises(multi_fidelity_observations):
    """Test that predict raises ValueError when multi-fidelity model gets candidates without fidelity."""
    surrogate = BoTorchSurrogate()
    surrogate.fit(multi_fidelity_observations)

    # Candidates missing fidelity values
    candidates = [Candidate(x=[1.5, 2.5]), Candidate(x=[2.5, 3.5])]
    with pytest.raises(ValueError, match="Candidates require fidelities"):
        surrogate.predict(candidates)


def test_parse_candidates_fidelity_confidence_mapping(multi_fidelity_observations):
    """Test that fidelity confidence mapping is applied to candidates, not just observations."""
    surrogate = BoTorchSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 0.95})
    surrogate.fit(multi_fidelity_observations)

    candidates = [
        Candidate(x=[1.5, 2.5], fidelity=0),
        Candidate(x=[2.5, 3.5], fidelity=1),
    ]
    test_X = surrogate._parse_candidates(candidates)

    expected_confidences = torch.tensor([0.5, 0.95], dtype=torch.float64)
    assert torch.allclose(test_X[:, -1], expected_confidences)


def test_state_dict_before_fit_returns_none():
    """Test that state_dict returns None when the model has not been fitted."""
    surrogate = BoTorchSurrogate()
    assert surrogate.state_dict() is None


def test_load_state_dict_after_fit(single_fidelity_observations):
    """Test that load_state_dict updates an already-fitted model in-place."""
    base = BoTorchSurrogate()
    base.fit(single_fidelity_observations)
    saved_state = base.state_dict()
    assert saved_state is not None  # state_dict() returns None only before fitting

    # Fit a second surrogate independently, then overwrite its state
    other = BoTorchSurrogate(optimize_hyperparameters=False)
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


def test_normalize_inputs_false(single_fidelity_observations):
    """Test that normalize_inputs=False skips input normalization without errors."""
    surrogate = BoTorchSurrogate(normalize_inputs=False)
    surrogate.fit(single_fidelity_observations)

    assert surrogate.model is not None
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1
    assert not torch.isnan(torch.tensor(predictions["mean"])).any()


def test_standardize_outputs_false(single_fidelity_observations):
    """Test that standardize_outputs=False skips output standardization without errors."""
    surrogate = BoTorchSurrogate(standardize_outputs=False)
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

    surrogate = BoTorchSurrogate(covar_module=MockKernel())
    surrogate.set_fidelity_confidences({0: 0.3, 1: 0.9})

    assert received == {0: 0.3, 1: 0.9}, (
        "Confidences were not propagated to the covar_module"
    )


def test_partial_update_fallback_when_model_none(single_fidelity_observations):
    """Test that _partial_update falls back to a full fit when model is None."""
    surrogate = BoTorchSurrogate(use_partial_updates=True)

    # Call _partial_update directly before any fit
    surrogate._partial_update(single_fidelity_observations)

    assert surrogate.model is not None, "Fallback full fit should initialise the model"
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
    assert len(predictions["mean"]) == 1


def test_partial_update_ignores_empty_observations(single_fidelity_observations):
    """Test that _partial_update is a no-op when the observation list is empty."""
    surrogate = BoTorchSurrogate(use_partial_updates=True)
    surrogate.fit(single_fidelity_observations)
    model_before = surrogate.model

    surrogate._partial_update([])  # should not raise or change the model

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
    surrogate = BoTorchSurrogate()

    # First fit: multi-fidelity
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


def test_fit_empty_observations_raises():
    """Test that fit raises a clear ValueError when given an empty observation list."""
    surrogate = BoTorchSurrogate()
    with pytest.raises(ValueError, match="empty observation list"):
        surrogate.fit([])


def test_partial_update_raises_on_fidelity_mismatch(multi_fidelity_observations):
    """Test that _partial_update raises when new observations are missing fidelities."""
    surrogate = BoTorchSurrogate(use_partial_updates=True)
    surrogate.fit(multi_fidelity_observations)

    # New observations without fidelity values — should raise before any tensor ops
    bad_obs = [Observation(x=[1.5, 2.5], y=5.0)]
    with pytest.raises(ValueError, match="missing fidelity values"):
        surrogate._partial_update(bad_obs)


def test_normalize_excludes_fidelity_column(multi_fidelity_observations):
    """Test that Normalize is applied only to feature columns, not the fidelity column."""
    surrogate = BoTorchSurrogate(normalize_inputs=True)
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

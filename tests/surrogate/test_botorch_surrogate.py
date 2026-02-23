import pytest
import torch
from botorch.models import SingleTaskGP, MultiTaskGP

# Adjust import paths based on your actual src structure
from activelearning.surrogate.botorch_surrogate import BoTorchSurrogate
from activelearning.utils.types import Observation, Candidate

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
    assert type(predictions["mean"]) == list

def test_multi_fidelity_fit_and_predict(multi_fidelity_observations):
    surrogate = BoTorchSurrogate()
   
    # Test Fitting
    surrogate.fit(multi_fidelity_observations)
    assert isinstance(surrogate.model, MultiTaskGP), "Should route to MultiTaskGP"
   
    # Test Prediction
    candidates = [
        Candidate(x=[1.5, 2.5], fidelity=1),
        Candidate(x=[2.5, 3.5], fidelity=1)
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
        Observation(x=[2.0], y=3.0)  # Missing fidelity!
    ]
   
    with pytest.raises(ValueError, match="all observations must have a fidelity"):
        surrogate.fit(bad_observations)

def test_update_unfitted_model(single_fidelity_observations):
    surrogate = BoTorchSurrogate()
   
    # Update should automatically fall back to fit() if model is None
    surrogate.update(single_fidelity_observations)
   
    assert surrogate.model is not None, "Model should be initialized"
    assert isinstance(surrogate.model, SingleTaskGP), "Should route to SingleTaskGP"

def test_update_fitted_model(single_fidelity_observations):
    surrogate = BoTorchSurrogate()
   
    # 1. Fit on the first two observations
    initial_train = single_fidelity_observations[:2]
    surrogate.fit(initial_train)
    initial_model = surrogate.model
   
    # 2. Update with the third observation
    new_obs = [single_fidelity_observations[2]]
    surrogate.update(new_obs)
   
    # BoTorch's condition_on_observations returns a new model instance
    assert surrogate.model is not initial_model, "Update should create a new conditioned model"
   
    # 3. Verify the updated model can still predict accurately
    candidates = [Candidate(x=[1.5, 2.5])]
    predictions = surrogate.predict(candidates)
   
    assert "mean" in predictions
    assert len(predictions["mean"]) == 1
import torch
from typing import Iterable, Sequence, Mapping, Any, Tuple
from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation

class BoTorchSurrogate(Surrogate):
    """A Gaussian Process surrogate using BoTorch. Automatically handles multi-fidelity."""
   
    def __init__(self):
        self.model = None
        self.mll = None
        self._is_multi_fidelity = False
        self._fidelity_col_index = -1  # BoTorch expects the task index as a column in X

    def _parse_observations(self, observations: Iterable[Observation]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts generic Observations into BoTorch-ready tensors."""
        x_data, y_data, fidelities = [], [], []
       
        # We need to materialize the iterable to iterate over it safely
        obs_list = list(observations)

        for obs in obs_list:
            # Safely convert x (which could be float, list, or array) to a 1D tensor
            x_tensor = torch.atleast_1d(torch.as_tensor(obs.x, dtype=torch.float64))
            x_data.append(x_tensor)
            y_data.append(torch.as_tensor(obs.y, dtype=torch.float64))
           
            if obs.fidelity is not None:
                fidelities.append(obs.fidelity)

        train_X = torch.stack(x_data)
        train_Y = torch.stack(y_data).view(-1, 1)  # BoTorch expects shape (n, 1)

        # Handle multi-fidelity concatenation
        if fidelities:
            if len(fidelities) != len(obs_list):
                raise ValueError("If using multi-fidelity, all observations must have a fidelity.")
           
            self._is_multi_fidelity = True
            fid_tensor = torch.tensor(fidelities, dtype=torch.float64).view(-1, 1)
            # Append fidelity as the final column of the training data
            train_X = torch.cat([train_X, fid_tensor], dim=-1)

        return train_X, train_Y

    def _parse_candidates(self, candidates: Sequence[Candidate]) -> torch.Tensor:
        """Converts Candidates into BoTorch-ready tensors."""
        x_data, fidelities = [], []

        for cand in candidates:
            x_tensor = torch.atleast_1d(torch.as_tensor(cand.x, dtype=torch.float64))
            x_data.append(x_tensor)
           
            if cand.fidelity is not None:
                fidelities.append(cand.fidelity)

        test_X = torch.stack(x_data)

        if self._is_multi_fidelity:
            if len(fidelities) != len(candidates):
                raise ValueError("Surrogate was fitted on multi-fidelity data. Candidates require fidelities.")
           
            fid_tensor = torch.tensor(fidelities, dtype=torch.float64).view(-1, 1)
            test_X = torch.cat([test_X, fid_tensor], dim=-1)

        return test_X

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fits the exact GP model using GPyTorch."""
        train_X, train_Y = self._parse_observations(observations)
       
        if self._is_multi_fidelity:
            # MultiTaskGP uses the fidelity column to learn cross-task correlation
            self.model = MultiTaskGP(train_X, train_Y, task_feature=self._fidelity_col_index)
        else:
            self.model = SingleTaskGP(train_X, train_Y)
           
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        # Force GPyTorch to build the predictive  (Cholesky) caches
        # This is especially important for the update() method to work correctly 
        
        self.model.eval()
        with torch.no_grad(): #
            self.model(train_X)

    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, Any]:
        """Returns predictions including the BoTorch posterior."""
        if self.model is None:
            raise RuntimeError("Surrogate model must be fitted before calling predict().")
           
        test_X = self._parse_candidates(candidates)
       
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_X)
           
        return {
            "mean": posterior.mean.squeeze(-1).tolist(),
            "std": posterior.variance.sqrt().squeeze(-1).tolist(),
            "posterior": posterior
        }
    
    def update(self, new_observations: Iterable[Observation]) -> None:
        """Updates the surrogate with new obervations by refitting the model."""
        if self.model is None:
            self.fit(new_observations)
            return
        new_X, new_Y = self._parse_observations(new_observations)
        # Fast update of the posterior without retraining from scratch
        # Note: This creates a new model instance in BoTorch
        self.model = self.model.condition_on_observations(new_X, new_Y)
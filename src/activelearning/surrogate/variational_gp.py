"""Sparse variational Gaussian process over fixed feature representations."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from typing import Any

import gpytorch
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.mlls import VariationalELBO
from torch.optim import Adam

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.utils.types import Candidate, Observation


class _VariationalGP(gpytorch.models.ApproximateGP):
    """Single-output sparse variational GP with learned inducing locations."""

    def __init__(
        self,
        input_dim: int,
        num_inducing: int = 64,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
        inducing_points: torch.Tensor | None = None,
        initial_lengthscale: float | None = None,
    ) -> None:
        """Initialize the variational GP in a fixed-dimensional feature space."""
        if inducing_points is None:
            inducing_points = torch.randn(
                num_inducing,
                input_dim,
                dtype=dtype,
                device=device,
            )
        elif inducing_points.shape != (num_inducing, input_dim):
            raise ValueError(
                "inducing_points must have shape "
                f"({num_inducing}, {input_dim}), got {tuple(inducing_points.shape)}."
            )
        else:
            inducing_points = inducing_points.to(device=device, dtype=dtype)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=input_dim)
        )
        if initial_lengthscale is not None:
            self.covar_module.base_kernel.lengthscale = initial_lengthscale

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate the latent GP distribution at feature-space inputs."""
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


class _VariationalBoTorchAdapter(Model):
    """Expose a GPyTorch approximate GP through BoTorch's model contract."""

    def __init__(
        self,
        gp_model: _VariationalGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        outcome_mean: float = 0.0,
        outcome_std: float = 1.0,
    ) -> None:
        """Initialize the adapter around a GP and Gaussian likelihood."""
        super().__init__()
        self._gp = gp_model
        self._likelihood = likelihood
        self._outcome_mean = outcome_mean
        self._outcome_std = outcome_std

    @property
    def num_outputs(self) -> int:
        """Return the single modeled output."""
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        """Return the empty model batch shape."""
        return torch.Size([])

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        output_indices: list[int] | None = None,
        posterior_transform: Any | None = None,
    ) -> Any:
        """Return a BoTorch-compatible posterior over feature-space inputs."""
        if output_indices is not None and list(output_indices) != [0]:
            raise NotImplementedError(
                "Variational GP supports only the single output index [0]."
            )
        self._gp.eval()
        self._likelihood.eval()
        distribution = self._gp(X)
        if observation_noise:
            distribution = self._likelihood(distribution)
        if self._outcome_mean != 0.0 or self._outcome_std != 1.0:
            distribution = gpytorch.distributions.MultivariateNormal(
                distribution.mean * self._outcome_std + self._outcome_mean,
                distribution.lazy_covariance_matrix * self._outcome_std**2,
            )
        posterior = GPyTorchPosterior(distribution=distribution)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class VariationalGPSurrogate(BoTorchGPSurrogate):
    """Sparse variational GP fitted on fixed, non-trainable features."""

    def __init__(
        self,
        encoder: FixedEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: int | None = None,
        num_inducing: int = 64,
        standardize_outputs: bool = True,
    ) -> None:
        """Initialize a fixed-feature variational GP surrogate.

        Parameters
        ----------
        encoder : FixedEncoder
            Non-trainable mapping from raw values to fixed-width features.
        training_params : object
            Training settings exposing positive ``epochs`` and ``lr`` values.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidence to each feature row.
        target_fidelity : int, optional
            Discrete target fidelity used by multi-fidelity acquisitions.
        num_inducing : int, default=64
            Number of learned inducing-point locations.
        standardize_outputs : bool, default=True
            Whether to standardize targets during GP fitting.
        """
        if is_multi_fidelity and target_fidelity is None:
            raise ValueError("target_fidelity must be set when is_multi_fidelity=True.")
        if encoder.feature_dim < 1:
            raise ValueError("encoder.feature_dim must be positive.")
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive.")

        self._encoder = encoder
        self._training = training_params
        self._target_fidelity_level = target_fidelity if is_multi_fidelity else None
        self._num_inducing = num_inducing
        self._standardize_outputs = standardize_outputs
        self._gp_model: _VariationalGP | None = None
        self._likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self._botorch_adapter: _VariationalBoTorchAdapter | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        super().__init__(
            scale_inputs=False,
            standardize_outputs=False,
            optimize_hyperparameters=False,
            is_multi_fidelity=is_multi_fidelity,
        )

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind runtime settings to the encoder and variational GP stack."""
        super().bind_runtime_context(runtime_context)
        self._encoder.bind_runtime_context(runtime_context)
        self._apply_runtime_context()

    def fit(self, observations: Iterable[Observation]) -> None:
        """Rebuild and fit the variational GP on all supplied observations."""
        observation_list = list(observations)
        if not observation_list:
            return
        if self._is_multi_fidelity and not self._fidelity_confidences:
            raise ValueError(
                "Multi-fidelity mode requires fidelity confidences before fitting."
            )

        self._train_X = self._encode_items(observation_list)
        train_y = torch.as_tensor(
            [observation.y for observation in observation_list],
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(-1)
        self._train_Y = train_y
        targets = self._standardize_targets(train_y).squeeze(-1)
        pending_state = self._pending_state_dict
        self._pending_state_dict = None
        self._build_variational_model()
        self._apply_runtime_context()
        # A 0.1 initial noise keeps early Cholesky factorizations conditioned.
        self._likelihood.noise = 0.1
        if pending_state is not None:
            self.load_state_dict(pending_state)
        else:
            self._train_variational_gp(targets)

    def updates_from_latest(self) -> bool:
        """Return false because the model is rebuilt from all observations."""
        return False

    def update(self, observations: Iterable[Observation]) -> None:
        """Refit from the supplied complete observation collection."""
        self.fit(observations)

    def get_state_dict(self) -> dict[str, torch.Tensor] | None:
        """Return GP, likelihood, and output-scaling state."""
        if self._gp_model is None or self._likelihood is None:
            return None

        state = {
            f"{prefix}.{name}": value
            for prefix, module in (
                ("model", self._gp_model),
                ("likelihood", self._likelihood),
            )
            for name, value in module.state_dict().items()
        }
        for name, value in (("mean", self._y_mean), ("std", self._y_std)):
            state[f"outcome_{name}"] = torch.tensor(
                value, dtype=self.dtype, device=self.device
            )
        return state

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore fitted state immediately or defer until model construction."""
        if self._gp_model is None or self._likelihood is None:
            self._pending_state_dict = state_dict
            return

        model_state = {
            name.removeprefix("model."): value
            for name, value in state_dict.items()
            if name.startswith("model.")
        }
        likelihood_state = {
            name.removeprefix("likelihood."): value
            for name, value in state_dict.items()
            if name.startswith("likelihood.")
        }
        self._gp_model.load_state_dict(model_state)
        self._likelihood.load_state_dict(likelihood_state)
        self._y_mean = float(state_dict["outcome_mean"].item())
        self._y_std = float(state_dict["outcome_std"].item())
        self._botorch_adapter = _VariationalBoTorchAdapter(
            self._gp_model,
            self._likelihood,
            outcome_mean=self._y_mean,
            outcome_std=self._y_std,
        ).to(device=self.device, dtype=self.dtype)
        self._gp_model.eval()
        self._likelihood.eval()

    def get_model(self) -> _VariationalBoTorchAdapter:
        """Return the fitted BoTorch-compatible variational model."""
        if self._botorch_adapter is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        return self._botorch_adapter

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Encode candidates into fixed feature space with optional fidelity."""
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("Cannot encode an empty candidate iterable.")
        return self._encode_items(candidate_list)

    def predict(self, candidates: Iterable[Candidate]) -> dict[str, Any]:
        """Predict means and standard deviations on the original target scale."""
        if self._gp_model is None or self._likelihood is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        encoded = self.encode_candidates(candidates)
        self._gp_model.eval()
        self._likelihood.eval()
        with torch.no_grad():
            prediction = self._likelihood(self._gp_model(encoded))
        mean = prediction.mean * self._y_std + self._y_mean
        std = prediction.variance.sqrt() * self._y_std
        return {
            "mean": mean.cpu().tolist(),
            "std": std.cpu().tolist(),
        }

    def get_fidelity_dimension(self) -> int | None:
        """Return the appended fidelity coordinate in feature space."""
        if not self._is_multi_fidelity:
            return None
        return self._encoder.feature_dim

    def get_target_fidelity_value(self) -> float | None:
        """Return the encoded confidence for the configured target fidelity."""
        if not self._is_multi_fidelity or self._target_fidelity_level is None:
            return None
        return self._encode_fidelity_level(self._target_fidelity_level)

    def _build_variational_model(self) -> None:
        """Build the GP, likelihood, and BoTorch adapter."""
        assert self._train_X is not None
        input_dim = self._train_X.shape[-1]
        sample_indices = torch.randint(
            self._train_X.shape[0],
            (self._num_inducing,),
            device=self.device,
        )
        inducing_points = self._train_X[sample_indices].clone()
        inducing_points += 1e-3 * torch.randn_like(inducing_points)
        self._gp_model = _VariationalGP(
            input_dim=input_dim,
            num_inducing=self._num_inducing,
            dtype=self.dtype,
            device=self.device,
            inducing_points=inducing_points,
            initial_lengthscale=sqrt(input_dim),
        )
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._botorch_adapter = _VariationalBoTorchAdapter(
            self._gp_model,
            self._likelihood,
            outcome_mean=self._y_mean,
            outcome_std=self._y_std,
        )
        self.model = self._gp_model

    def _apply_runtime_context(self) -> None:
        """Move all fitted GP modules to the active runtime device and dtype."""
        for module in (
            self._gp_model,
            self._likelihood,
            self._botorch_adapter,
        ):
            if module is not None:
                module.to(device=self.device, dtype=self.dtype)

    def _encode_items(
        self,
        items: list[Candidate | Observation],
    ) -> torch.Tensor:
        """Encode fixed features and append fidelity confidence when enabled."""
        features = self._encoder.encode(
            [item.x for item in items],
            device=self.device,
        ).to(device=self.device, dtype=self.dtype)
        expected_shape = (len(items), self._encoder.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "Fixed encoder output must have shape "
                f"{expected_shape}, got {tuple(features.shape)}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("Fixed encoder output contains non-finite values.")
        if not self._is_multi_fidelity:
            return features

        fidelities = torch.tensor(
            [self._encode_fidelity_level(item.fidelity) for item in items],
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(-1)
        return torch.cat([features, fidelities], dim=-1)

    def _encode_fidelity_level(self, fidelity_level: int) -> float:
        """Map a discrete fidelity id to its configured confidence."""
        if fidelity_level not in self._fidelity_confidences:
            raise ValueError(f"Missing fidelity confidence for level {fidelity_level}.")
        return float(self._fidelity_confidences[fidelity_level])

    def _standardize_targets(self, train_y: torch.Tensor) -> torch.Tensor:
        """Standardize targets and retain statistics for prediction."""
        if not self._standardize_outputs:
            return train_y
        self._y_mean = float(train_y.mean())
        self._y_std = float(train_y.std().nan_to_num(nan=1.0).clamp_min(1e-6))
        return (train_y - self._y_mean) / self._y_std

    def _train_variational_gp(self, targets: torch.Tensor) -> None:
        """Optimize the variational ELBO and learned inducing locations."""
        assert self._gp_model is not None
        assert self._likelihood is not None

        objective = VariationalELBO(
            self._likelihood,
            self._gp_model,
            num_data=targets.shape[0],
        )
        parameters = [
            parameter
            for module in (self._gp_model, self._likelihood)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        optimizer = Adam(parameters, lr=self._training.lr)
        for _ in range(self._training.epochs):
            self._gp_model.train()
            self._likelihood.train()
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(1e-1):
                loss = -objective(self._gp_model(self._train_X), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
        self._gp_model.eval()
        self._likelihood.eval()

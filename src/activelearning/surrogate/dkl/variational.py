"""Variational DKL surrogate implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import gpytorch
import torch
from botorch.models.model import Model
from gpytorch.mlls import VariationalELBO
from torch.optim import Adam

from activelearning.surrogate.dkl.surrogate import DeepKernelSurrogate
from activelearning.surrogate.encoder import LatentEncoder
from activelearning.utils.types import Candidate


class _VariationalDKLGP(gpytorch.models.ApproximateGP):
    """Sparse variational GP head used by :class:`VariationalDKLSurrogate`.

    Operates in **latent feature space** — it receives encoder output vectors,
    not raw token IDs.  The encoder is kept separate and called explicitly by
    the surrogate before invoking this GP.
    """

    def __init__(
        self,
        input_dim: int,
        num_inducing: int = 64,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the variational GP head.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the latent feature vectors.
        num_inducing : int, default=64
            Number of inducing points used by the variational strategy.
        dtype : torch.dtype, default=torch.float64
            Dtype of the initial inducing-point locations.
        device : torch.device, optional
            Device on which to create the inducing-point locations.
        """
        # The surrogate passes RuntimeContext.dtype; this fallback matches the
        # default runtime precision for standalone construction.
        inducing_points = torch.randn(
            num_inducing, input_dim, dtype=dtype, device=device
        )
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

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate the variational GP head on latent feature inputs.

        Parameters
        ----------
        x : torch.Tensor
            Latent feature tensor with final dimension ``input_dim``.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Variational GP predictive distribution at ``x``.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class _VariationalBoTorchAdapter(Model):
    """BoTorch-compatible wrapper around the variational DKL GP.

    BoTorch acquisition functions (including qMF-MES) require a model that
    satisfies the :class:`botorch.models.model.Model` contract: it must
    implement :meth:`posterior`, :attr:`num_outputs`, and :attr:`batch_shape`.
    Plain GPyTorch ``ApproximateGP`` does not satisfy this contract; this
    adapter bridges the gap without modifying ``_VariationalDKLGP``.

    The adapter operates in **latent feature space**: ``X`` passed to
    :meth:`posterior` must already be encoded latent vectors (+ optional fidelity
    column), matching the input that ``_VariationalDKLGP.forward`` expects.

    Parameters
    ----------
    gp_model : _VariationalDKLGP
        The trained variational GP head.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The GP likelihood used for observation noise during prediction.
    """

    def __init__(
        self,
        gp_model: _VariationalDKLGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        """Initialize an adapter around a trained variational GP and likelihood.

        Parameters
        ----------
        gp_model : _VariationalDKLGP
            Sparse variational GP head operating on latent feature vectors.
        likelihood : gpytorch.likelihoods.GaussianLikelihood
            Likelihood used to include observation noise in posterior queries.
        """
        super().__init__()
        self._gp = gp_model
        self._likelihood = likelihood

    @property
    def num_outputs(self) -> int:
        """Return the number of modeled outputs.

        Returns
        -------
        int
            Always ``1`` because this adapter wraps a single-output GP.
        """
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        """Return the batch shape of the single-output GP.

        Returns
        -------
        torch.Size
            Empty batch shape because the wrapped GP has no model batch
            dimensions.
        """
        return torch.Size([])

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        output_indices: list[int] | None = None,
        posterior_transform: Any | None = None,
    ) -> Any:
        """Return a BoTorch-compatible GP posterior over latent-space inputs.

        Parameters
        ----------
        X : torch.Tensor
            Latent feature tensor of shape ``(..., d)`` where ``d`` is the
            encoder latent dimension (+ 1 for fidelity in multi-fidelity mode).
        observation_noise : bool, default=False
            If True, adds likelihood noise to the predictive variance.
        output_indices : list[int], optional
            Output indices to select. Only the single output index ``[0]`` is
            supported.
        posterior_transform : object, optional
            BoTorch posterior transform applied to the resulting posterior.

        Returns
        -------
        GPyTorchPosterior
            BoTorch posterior backed by the variational GP distribution.
        """
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        if output_indices is not None and list(output_indices) != [0]:
            raise NotImplementedError(
                "Variational DKL supports only the single output index [0]."
            )
        self._gp.eval()
        self._likelihood.eval()
        dist = self._gp(X)
        if observation_noise:
            dist = self._likelihood(dist)
        posterior = GPyTorchPosterior(distribution=dist)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class VariationalDKLSurrogate(DeepKernelSurrogate):
    """DKL surrogate with a sparse variational GP head — paper-faithful variant.

    Reproduces the architecture from the reference ``DeepKernelRegressor``:

    - Encoder and GP are **separate** components; encoding is explicit.
    - Training minimises ``VariationalELBO + MLM loss`` via Adam (ELBO, not MLL).
    - GP operates in **latent feature space** (encoder output + optional fidelity).
    - Compatible with all BoTorch acquisition functions via
      :class:`_VariationalBoTorchAdapter`, including qMF-MES.

    Parameters
    ----------
    encoder : LatentEncoder
    training_params : object
    is_multi_fidelity : bool
        Append fidelity scalar to latent feature vectors.
    target_fidelity : int, optional
        Required when ``is_multi_fidelity=True``.
    num_inducing : int
        Number of variational inducing points.
    standardize_outputs : bool
        Normalise targets before training and denormalise predictions.
    """

    def __init__(
        self,
        encoder: LatentEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        num_inducing: int = 64,
        standardize_outputs: bool = True,
    ) -> None:
        """Initialize a sparse variational DKL surrogate.

        Parameters
        ----------
        encoder : LatentEncoder
            Feature encoder whose output is modeled by the variational GP.
        training_params : object
            DKL training settings, including the epoch count and learning rate.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidences to latent features.
        target_fidelity : int, optional
            Fidelity level used for target-fidelity projections. Required when
            ``is_multi_fidelity`` is true.
        num_inducing : int, default=64
            Number of inducing points in the sparse variational GP.
        standardize_outputs : bool, default=True
            Whether to standardize regression targets before GP training.
        """
        self._num_inducing = num_inducing
        self._standardize_outputs = standardize_outputs
        self._gp_model: Optional[_VariationalDKLGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._botorch_adapter: Optional[_VariationalBoTorchAdapter] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        super().__init__(
            encoder=encoder,
            training_params=training_params,
            is_multi_fidelity=is_multi_fidelity,
            target_fidelity=target_fidelity,
            scale_inputs=False,
            standardize_outputs=False,  # handled manually via _prepare_targets
        )

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        gp_input_dim = self._encoder.latent_dim + (1 if self._is_multi_fidelity else 0)
        self._gp_model = _VariationalDKLGP(
            gp_input_dim,
            self._num_inducing,
            dtype=self.dtype,
            device=self.device,
        ).to(device=self.device, dtype=self.dtype)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self.device, dtype=self.dtype
        )
        self._botorch_adapter = _VariationalBoTorchAdapter(
            self._gp_model, self._likelihood
        )
        self.model = self._gp_model  # exposes is_fitted() / state-dict helpers

    def _runtime_modules(self) -> tuple[torch.nn.Module, ...]:
        """Return variational GP modules that follow the runtime context."""
        return tuple(
            module
            for module in (
                self._gp_model,
                self._likelihood,
                self._botorch_adapter,
            )
            if module is not None
        )

    def get_model(self) -> _VariationalBoTorchAdapter:
        """Return the BoTorch-compatible adapter wrapping the variational GP.

        Overrides :meth:`BoTorchGPSurrogate.get_model` so that acquisition
        functions receive a model that implements the BoTorch ``posterior()``
        interface and operates in latent feature space.

        Returns
        -------
        _VariationalBoTorchAdapter
            Fitted variational GP adapter for use by acquisition functions.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        """
        if self._botorch_adapter is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        return self._botorch_adapter

    def get_fidelity_dimension(self) -> Optional[int]:
        """Return the fidelity column index in latent feature space.

        Overrides the base method, which computes the index from ``train_X``
        shape (token space).  In the variational surrogate, acquisition
        functions receive **latent features** from :meth:`encode_candidates`,
        so the fidelity column sits at index ``latent_dim`` — the last column
        of the ``(latent_dim + 1)``-dimensional latent tensor.

        Returns
        -------
        int or None
            Fidelity column index in multi-fidelity mode, or ``None`` for
            single-fidelity surrogates.
        """
        if not self._is_multi_fidelity:
            return None
        return self._encoder.latent_dim

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Return encoder latent features (+ fidelity) for each candidate.

        Overrides the base input path. The variational GP head operates
        in latent feature space, so acquisition functions must receive encoded
        vectors rather than raw token IDs. This keeps the candidate set and
        the projection in MF-MES consistent with the GP's input space.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates to encode.

        Returns
        -------
        torch.Tensor
            Batched latent features, with an optional fidelity confidence as
            the final column.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        """
        if self._gp_model is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        # Encode via the parent, then map inputs to latent features.
        input_X = super().encode_candidates(candidates).to(self.device)
        self._set_eval_mode()
        with torch.no_grad():
            return self._encode_with_fidelity(input_X)

    def _prepare_targets(self, train_Y: torch.Tensor) -> torch.Tensor:
        if self._standardize_outputs:
            self._y_mean = float(train_Y.mean())
            self._y_std = float(train_Y.std().nan_to_num(nan=1.0).clamp_min(1e-6))
            return (train_Y - self._y_mean) / self._y_std
        return train_Y

    def _make_mll(self, num_data: int) -> VariationalELBO:
        return VariationalELBO(self._likelihood, self._gp_model, num_data=num_data)

    def _gp_forward(self, model_X: torch.Tensor) -> Any:
        return self._gp_model(self._encode_with_fidelity(model_X))

    def _make_optimizer(self) -> Adam:
        return Adam(
            [
                parameter
                for parameter in (
                    list(self._encoder.parameters())
                    + list(self._gp_model.parameters())
                    + list(self._likelihood.parameters())
                )
                if parameter.requires_grad
            ],
            lr=self._training.lr,
        )

    def predict(self, candidates: Iterable[Candidate]) -> dict[str, Any]:
        """Predict target means and standard deviations for candidates.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates whose values are converted by
            ``encoder.prepare_inputs()``.

        Returns
        -------
        dict[str, Any]
            Dictionary containing CPU lists under ``"mean"`` and ``"std"``.
            Values are returned on the original, pre-standardization target
            scale.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        """
        if self._gp_model is None or self._likelihood is None:
            raise RuntimeError("Surrogate has not been fitted yet.")

        # encode_candidates() now returns latent features directly
        latent_X = self.encode_candidates(list(candidates)).to(self.device)
        with torch.no_grad():
            pred = self._likelihood(self._gp_model(latent_X))

        mean = pred.mean * self._y_std + self._y_mean
        std = pred.variance.sqrt() * self._y_std
        return {"mean": mean.cpu().tolist(), "std": std.cpu().tolist()}

    def _encode_with_fidelity(self, input_X: torch.Tensor) -> torch.Tensor:
        """Encode model inputs to latent features, appending fidelity if active."""
        if self._is_multi_fidelity:
            features = self._encoder(input_X[:, :-1])
            return torch.cat([features, input_X[:, -1:].to(features.dtype)], dim=-1)
        return self._encoder(input_X)

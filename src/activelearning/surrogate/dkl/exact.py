"""Exact DKL surrogate implementation."""

from __future__ import annotations

from typing import Any, Optional

import gpytorch
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

from activelearning.surrogate.dkl.surrogate import DeepKernelSurrogate
from activelearning.surrogate.encoder import LatentEncoder
from activelearning.surrogate.dkl.kernel import EncoderKernel


class ExactDKLSurrogate(DeepKernelSurrogate):
    """DKL surrogate with an exact GP backed by BoTorch SingleTaskGP.

    The encoder is embedded inside an encoder kernel passed to SingleTaskGP as
    its covar_module, making all BoTorch acquisition functions work out of the
    box. Training jointly optimises encoder, GP kernel, and likelihood noise
    via Adam (ExactMarginalLogLikelihood + MLM loss).

    Parameters
    ----------
    encoder : LatentEncoder
    training_params : object
    is_multi_fidelity : bool
    target_fidelity : int, optional
        Required when ``is_multi_fidelity=True``.
    standardize_outputs : bool
        Normalise GP outputs to mean 0 / variance 1.
    scale_inputs : bool
        Whether BoTorch should normalize the model-space inputs. Defaults to
        ``False`` because tokenized sequence inputs are not continuous features.
    """

    def __init__(
        self,
        encoder: LatentEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        standardize_outputs: bool = True,
        scale_inputs: bool = False,
    ) -> None:
        """Initialize an exact-GP DKL surrogate.

        Parameters
        ----------
        encoder : LatentEncoder
            Feature encoder used inside the exact GP kernel.
        training_params : object
            DKL training settings, including the epoch count and learning rate.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidences to the GP inputs.
        target_fidelity : int, optional
            Fidelity level used for target-fidelity projections. Required when
            ``is_multi_fidelity`` is true.
        standardize_outputs : bool, default=True
            Whether to standardize regression targets before GP training.
        scale_inputs : bool, default=False
            Whether BoTorch should normalize model-space inputs.
        """
        super().__init__(
            encoder=encoder,
            training_params=training_params,
            is_multi_fidelity=is_multi_fidelity,
            target_fidelity=target_fidelity,
            scale_inputs=scale_inputs,
            standardize_outputs=standardize_outputs,
        )

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        gp_input_dim = self._encoder.latent_dim + (1 if self._is_multi_fidelity else 0)
        self.covar_module = EncoderKernel(
            encoder=self._encoder,
            base_kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=gp_input_dim)
            ),
            include_fidelity=self._is_multi_fidelity,
        )
        super()._build_model(train_X, train_Y)

    def _make_mll(self, num_data: int) -> ExactMarginalLogLikelihood:
        return ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def _gp_forward(self, model_X: torch.Tensor) -> Any:
        return self.model(model_X.to(device=self.device, dtype=self.dtype))

    def _make_optimizer(self) -> Adam:
        # model already contains the likelihood as a submodule
        return Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=self._training.lr,
        )

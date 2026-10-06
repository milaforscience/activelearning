"""GPyTorch kernel that wraps an encoder for exact DKL.

The encoder kernel runs each input through an encoder and applies a base kernel
to the resulting latent vectors.

When ``include_fidelity=True`` the **last column** of each input tensor is
treated as a fidelity scalar (e.g. 1.0, 2.0, 3.0) that is concatenated to
the encoder's latent output before the base kernel is applied. This mirrors
the feature construction used by the variational DKL surrogate.

This design means the exact DKL surrogate can be used with *any* BoTorch
acquisition function without modification.
"""

from __future__ import annotations

import gpytorch
import torch
from torch import Tensor, nn


class EncoderKernel(gpytorch.kernels.Kernel):
    """Covariance kernel for inputs represented by an encoder.

    Computes ``k(x1, x2)`` by encoding inputs and then applying a base kernel
    on the resulting latent feature vectors.
    """

    def __init__(
        self,
        encoder: nn.Module,
        base_kernel: gpytorch.kernels.Kernel,
        include_fidelity: bool = False,
    ) -> None:
        """Initialize an encoder-wrapping covariance kernel.

        Parameters
        ----------
        encoder : torch.nn.Module
            Feature encoder applied to each input before the base kernel.
        base_kernel : gpytorch.kernels.Kernel
            Kernel evaluated on encoded feature vectors.
        include_fidelity : bool, default=False
            Whether to treat the final input column as a fidelity value and
            append it to the encoded features.
        """
        super().__init__()
        self.encoder = encoder
        self.base_kernel = base_kernel
        self.include_fidelity = include_fidelity

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        """Compute the kernel matrix between two sets of encoded inputs.

        Parameters
        ----------
        x1 : Tensor
            Shape ``(N, input_dim)`` or ``(N, input_dim+1)``, with an optional
            fidelity scalar in the last column when ``include_fidelity=True``.
        x2 : Tensor
            Shape ``(M, input_dim)`` or ``(M, input_dim+1)`` — same convention.
        diag : bool, default=False
            If ``True``, return only the diagonal entries of the covariance
            instead of the full pairwise covariance matrix.
        last_dim_is_batch : bool, default=False
            If ``True``, interpret the final input dimension as a batch of
            independent dimensions, as supported by the wrapped kernel.
        **params
            Additional keyword arguments forwarded to ``base_kernel``.

        Returns
        -------
        Tensor
            Kernel matrix of shape ``(N, M)`` or a diagonal tensor of shape
            ``(N,)``, depending on ``diag``. The concrete return type is
            provided by the base kernel.
        """
        return self.base_kernel(
            self._encode(x1).to(x1.dtype),
            self._encode(x2).to(x1.dtype),
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )

    def _encode(self, x: Tensor) -> Tensor:
        """Encode ``(*leading, d)`` inputs, re-appending the fidelity column if any."""
        inputs = x[..., :-1] if self.include_fidelity else x
        # The encoder expects (N, input_dim); BoTorch may add leading batch/q
        # dimensions (e.g. (batch, q, d) during acquisition scoring), so
        # flatten them, encode, then restore.
        features = self.encoder(inputs.reshape(-1, inputs.shape[-1]))
        features = features.reshape(*x.shape[:-1], -1)
        if self.include_fidelity:
            features = torch.cat([features, x[..., -1:].to(features.dtype)], dim=-1)
        return features

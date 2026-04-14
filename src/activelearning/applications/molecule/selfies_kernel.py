"""GPyTorch kernel that wraps a SelfiesTransformerEncoder for exact DKL.

The SelfiesKernel accepts float-valued token-ID tensors as inputs (how
BoTorch stores training data), casts them to ``long`` internally, runs them
through the encoder, and applies a base kernel on the resulting latent vectors.

When ``include_fidelity=True`` the **last column** of each input tensor is
treated as a fidelity scalar (e.g. 1.0, 2.0, 3.0) that is concatenated to
the encoder's latent output before the base kernel is applied.  This mirrors
the reference ``DeepKernelMoleculeRegressor.build_features()`` design.

This design means :class:`~activelearning.applications.molecule.dkl_surrogate.ExactSelfiesDKLSurrogate`
can be used with *any* BoTorch acquisition function without modification.
"""

from __future__ import annotations

import gpytorch
import torch
from torch import Tensor

from activelearning.applications.molecule.selfies_transformer_encoder import (
    SelfiesTransformerEncoder,
)


class SelfiesKernel(gpytorch.kernels.Kernel):
    """Covariance kernel for molecule sequences.

    Computes k(x₁, x₂) by encoding token-ID sequences through a
    :class:`SelfiesTransformerEncoder` and then applying a base kernel on
    the resulting latent feature vectors.

    Parameters
    ----------
    encoder : SelfiesTransformerEncoder
        Shared encoder module whose parameters are optimised jointly with
        the GP hyperparameters during surrogate training.
    base_kernel : gpytorch.kernels.Kernel
        Stationary kernel (e.g. ``ScaleKernel(MaternKernel(...))``) applied
        in latent feature space.
    include_fidelity : bool
        If ``True``, the **last column** of each input is treated as a
        fidelity scalar and concatenated to the latent encoding before the
        base kernel.  The base kernel must have ``ard_num_dims`` set to
        ``encoder.latent_dim + 1`` in this case.
    """

    def __init__(
        self,
        encoder: SelfiesTransformerEncoder,
        base_kernel: gpytorch.kernels.Kernel,
        include_fidelity: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.base_kernel = base_kernel
        self.include_fidelity = include_fidelity

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        """Compute the kernel matrix between two sets of token sequences.

        Parameters
        ----------
        x1 : Tensor
            Shape ``(N, seq_len)`` or ``(N, seq_len+1)`` — float-valued token
            IDs, with an optional fidelity scalar in the last column when
            ``include_fidelity=True``.
        x2 : Tensor
            Shape ``(M, seq_len)`` or ``(M, seq_len+1)`` — same convention.

        Returns
        -------
        Tensor
            Kernel matrix of shape ``(N, M)`` or a ``gpytorch.lazy.LazyTensor``
            from the base kernel.
        """
        if self.include_fidelity:
            # Peel off the fidelity column (last feature dim) before tokenisation.
            # Use `...` so this works for any leading batch/q dimensions that
            # BoTorch may add (e.g. shape (batch, q, d) during acquisition scoring).
            fid1, fid2 = x1[..., -1:], x2[..., -1:]
            tok1, tok2 = x1[..., :-1], x2[..., :-1]
        else:
            tok1, tok2 = x1, x2
            fid1 = fid2 = None

        # The encoder expects (N, seq_len); BoTorch may pass (*batch, q, seq_len).
        # Flatten all leading dims, encode, then restore.
        f1 = self._encode_flat(tok1)
        f2 = self._encode_flat(tok2)

        if fid1 is not None:
            f1 = torch.cat([f1, fid1.to(f1.dtype)], dim=-1)
            f2 = torch.cat([f2, fid2.to(f2.dtype)], dim=-1)

        return self.base_kernel(f1.to(x1.dtype), f2.to(x1.dtype))

    def _encode_flat(self, tok: torch.Tensor) -> torch.Tensor:
        """Encode token IDs with any leading batch dims.

        Parameters
        ----------
        tok : Tensor
            Shape ``(*leading, seq_len)`` — float-valued token IDs.

        Returns
        -------
        Tensor
            Shape ``(*leading, latent_dim)``.
        """
        leading = tok.shape[:-1]
        seq_len = tok.shape[-1]
        # Flatten to (N, seq_len), encode, restore leading dims
        flat = tok.reshape(-1, seq_len).long()
        features = self.encoder(flat)  # (N, latent_dim)
        return features.reshape(*leading, -1)  # (*leading, latent_dim)

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
        if self.include_fidelity:
            # Peel off the fidelity column before encoding.
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

        return self.base_kernel(
            f1.to(x1.dtype),
            f2.to(x1.dtype),
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )

    def _encode_flat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs with any leading batch dimensions.

        Parameters
        ----------
        inputs : Tensor
            Shape ``(*leading, input_dim)``.

        Returns
        -------
        Tensor
            Shape ``(*leading, latent_dim)``.
        """
        leading = inputs.shape[:-1]
        input_dim = inputs.shape[-1]
        # Flatten to (N, input_dim), encode, then restore leading dimensions.
        flat = inputs.reshape(-1, input_dim)
        features = self.encoder(flat)  # (N, latent_dim)
        return features.reshape(*leading, -1)  # (*leading, latent_dim)

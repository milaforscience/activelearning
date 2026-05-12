"""Shared sampling utilities used across the active learning library."""

import torch


def latin_hypercube(
    n_points: int,
    n_dims: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a Latin Hypercube Sample (LHS) in the unit hypercube ``[0, 1]^d``.

    LHS divides each dimension into ``n_points`` equal-width bins and ensures
    that each bin contains exactly one sample. This produces better
    space-filling coverage than i.i.d. uniform sampling while remaining fast
    to generate.

    Parameters
    ----------
    n_points : int
        Number of sample points to generate.
    n_dims : int
        Dimensionality of the hypercube.
    dtype : torch.dtype
        Floating-point dtype for the returned tensor. Defaults to ``torch.float64``.
    device : torch.device, optional
        Device on which to allocate the sample tensor. Defaults to the current
        PyTorch default device behavior.

    Returns
    -------
    samples : torch.Tensor
        Tensor of shape ``(n_points, n_dims)`` with values in ``[0, 1)``.
    """
    float_kwargs: dict[str, torch.dtype | torch.device] = {"dtype": dtype}
    int_kwargs: dict[str, torch.device] = {}
    if device is not None:
        float_kwargs["device"] = device
        int_kwargs["device"] = device

    offsets = torch.rand(n_points, n_dims, **float_kwargs)
    perms = torch.stack([torch.randperm(n_points, **int_kwargs) for _ in range(n_dims)], dim=1)
    return (perms.to(dtype) + offsets) / n_points

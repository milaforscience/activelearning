"""Shared sampling utilities used across the active learning library."""

import torch


def latin_hypercube(n_points: int, n_dims: int) -> torch.Tensor:
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

    Returns
    -------
    samples : torch.Tensor
        Float64 tensor of shape ``(n_points, n_dims)`` with values in ``[0, 1)``.
    """
    offsets = torch.rand(n_points, n_dims, dtype=torch.float64)
    perms = torch.stack([torch.randperm(n_points) for _ in range(n_dims)], dim=1)
    return (perms.to(torch.float64) + offsets) / n_points

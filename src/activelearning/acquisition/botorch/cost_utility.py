"""Cost-aware utility for multi-fidelity acquisition functions.

Provides :class:`FidelityCostUtility`, a domain-agnostic
:class:`botorch.acquisition.cost_aware.CostAwareUtility` that scales
information gain by the inverse oracle cost at each fidelity level.
"""

from __future__ import annotations

import torch
from botorch.acquisition.cost_aware import CostAwareUtility
from torch import Tensor


class FidelityCostUtility(CostAwareUtility):
    """Scale information gain by inverse oracle cost.

    For each candidate, the fidelity value is read from the last column of the
    input tensor ``X``.  The delta (information gain) is divided by the
    corresponding oracle cost so that the acquisition favours cheap fidelities
    when their information gain is comparable.

    Parameters
    ----------
    fidelity_costs : dict[int, float]
        Mapping from discrete fidelity level (as stored in the last column of the
        encoded input) to oracle cost.  Example: ``{1: 1.0, 2: 5.0, 3: 25.0}``.
    fixed_cost : float, default=0.0
        Optional additive fixed cost added to each scaled delta.
    """

    def __init__(
        self,
        fidelity_costs: dict[int, float],
        fixed_cost: float = 0.0,
    ) -> None:
        super().__init__()
        self.fidelity_costs = fidelity_costs
        self.register_buffer(
            "_fixed_cost", torch.tensor([fixed_cost], dtype=torch.float64)
        )

    def forward(self, X: Tensor, deltas: Tensor, **kwargs: object) -> Tensor:
        """Compute cost-scaled information gain.

        Parameters
        ----------
        X : Tensor
            Candidate tensor of shape ``(q, batch, d)``.  The fidelity value
            is in ``X[:, 0, -1]``.
        deltas : Tensor
            Information-gain tensor of shape ``(1, batch)``.

        Returns
        -------
        scaled : Tensor
            Cost-scaled deltas of shape ``(1, batch)``.
        """
        fidelity = X[:, 0, -1]  # (batch,)
        scaled = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        for fid, cost in self.fidelity_costs.items():
            idx = torch.where(fidelity == fid)[0]
            if idx.numel() == 0:
                continue
            scaled[idx] = (deltas[0, idx] / cost) + self._fixed_cost
        return scaled.unsqueeze(0)

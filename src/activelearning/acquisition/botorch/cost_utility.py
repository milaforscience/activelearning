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
        Mapping from discrete fidelity level to oracle cost.
    fixed_cost : float, default=0.0
        Optional additive fixed cost added to each scaled delta.
    """

    def __init__(
        self,
        fidelity_costs: dict[int, float],
        fixed_cost: float = 0.0,
    ) -> None:
        super().__init__()
        invalid_costs = {
            fidelity: cost for fidelity, cost in fidelity_costs.items() if cost <= 0.0
        }
        if invalid_costs:
            raise ValueError(
                f"Fidelity costs must be strictly positive. Got: {invalid_costs}."
            )
        self.fidelity_costs = dict(fidelity_costs)
        self._encoded_fidelity_costs = {
            float(fidelity): cost for fidelity, cost in self.fidelity_costs.items()
        }
        self.register_buffer(
            "_fixed_cost", torch.tensor([fixed_cost], dtype=torch.float64)
        )

    def set_fidelity_confidences(self, fidelity_confidences: dict[int, float]) -> None:
        """Resolve discrete fidelity levels to the encoded values stored in X."""
        missing = sorted(set(self.fidelity_costs) - set(fidelity_confidences))
        if missing:
            raise ValueError(
                "Missing fidelity confidences for configured costs at levels "
                f"{missing}."
            )
        self._encoded_fidelity_costs = {
            float(fidelity_confidences[level]): cost
            for level, cost in self.fidelity_costs.items()
        }

    def forward(self, X: Tensor, deltas: Tensor, **kwargs: object) -> Tensor:
        """Compute cost-scaled information gain.

        Parameters
        ----------
        X : Tensor
            Candidate tensor of shape ``batch_shape x q x d``. The fidelity
            value is read from the last column.
        deltas : Tensor
            Information-gain tensor of shape ``sample_shape x batch_shape``.

        Returns
        -------
        scaled : Tensor
            Cost-scaled deltas with the same shape as ``deltas``.
        """
        if X.ndim < 2:
            raise ValueError(
                "Expected X to have shape batch_shape x q x d; "
                f"got shape {tuple(X.shape)}."
            )

        q_batch_size = X.shape[-2]
        if q_batch_size != 1:
            raise ValueError(
                "FidelityCostUtility currently supports only singleton q-batches "
                f"(q=1); got q={q_batch_size}."
            )

        batch_shape = X.shape[:-2]
        if batch_shape and deltas.shape[-len(batch_shape) :] != batch_shape:
            raise ValueError(
                "deltas must end with the same batch_shape as X. "
                f"Got X batch_shape={batch_shape} and deltas shape={tuple(deltas.shape)}."
            )

        fidelities = X[..., 0, -1].to(device=deltas.device)
        costs = torch.zeros_like(fidelities, dtype=deltas.dtype, device=deltas.device)
        matched = torch.zeros_like(fidelities, dtype=torch.bool, device=deltas.device)

        for fidelity, cost in self._encoded_fidelity_costs.items():
            mask = fidelities == float(fidelity)
            if mask.any():
                costs = torch.where(mask, costs.new_full((), float(cost)), costs)
                matched |= mask

        if not torch.all(matched):
            unknown_fidelities = (
                torch.unique(fidelities[~matched]).detach().cpu().tolist()
            )
            raise ValueError(
                "Encountered fidelities without configured costs: "
                f"{unknown_fidelities}. Known encoded fidelities: "
                f"{sorted(self._encoded_fidelity_costs)}."
            )

        while costs.ndim < deltas.ndim:
            costs = costs.unsqueeze(0)

        fixed_cost = self._fixed_cost.to(device=deltas.device, dtype=deltas.dtype)
        return (deltas / costs) + fixed_cost

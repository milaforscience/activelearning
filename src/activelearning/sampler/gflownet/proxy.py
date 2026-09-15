import numpy.typing as npt
from typing import Any, Callable, List, Optional, Sequence, Union
import torch
from gflownet.proxy.base import Proxy
from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.cost_utility import cost_weighting_from_cost_fn
from activelearning.sampler.gflownet.utils import proxy_states_to_candidates
from activelearning.utils.types import Candidate


class AcquisitionProxy(Proxy):
    """GFlowNet proxy that wraps an active learning acquisition function.

    Bridges between GFlowNet's proxy interface (tensor states → tensor values)
    and the AL acquisition interface (Candidate objects → float values).

    States arrive in proxy format (output of ``env.states2proxy()``), are
    converted to ``Candidate`` objects, scored
    via :meth:`~activelearning.acquisition.acquisition.Acquisition.score`,
    and returned as a tensor.

    For multi-fidelity environments, sub-environment indices are read from the
    env stored by :meth:`setup`, so the correct ``idx_base_env`` and
    ``idx_fidelity`` are used regardless of wrapper variant (SetFix, FidFirst,
    FidLast).

    Parameters
    ----------
    acquisition : Acquisition
        An active learning acquisition function implementing
        :meth:`~activelearning.acquisition.acquisition.Acquisition.score`.
        Set to ``None`` at init; must be set via :meth:`set_acquisition`
        before the proxy is used.
    cost_fn : callable, optional
        Optional candidate cost function. When provided, proxy scores are
        reweighted by inverse cost before they are returned to GFlowNet.
    reward_scale_beta : float, default=1.0
        Positive acquisition normalizer used by the MF-GFN reward scale.
    reward_scale_rho : float, default=1.0
        Positive per-round acquisition multiplier base used by the MF-GFN
        reward scale.
    **kwargs
        Forwarded to :class:`gflownet.proxy.base.Proxy` (device,
        float_precision, reward_function, reward_min, etc.).
    """

    def __init__(
        self,
        acquisition: Acquisition = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        reward_scale_beta: float = 1.0,
        reward_scale_rho: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if reward_scale_beta <= 0:
            raise ValueError("reward_scale_beta must be positive.")
        if reward_scale_rho <= 0:
            raise ValueError("reward_scale_rho must be positive.")
        super().__init__(**kwargs)
        self.acquisition = acquisition
        self.cost_fn = cost_fn
        self.reward_scale_beta = float(reward_scale_beta)
        self.reward_scale_rho = float(reward_scale_rho)
        self._env: Optional[Any] = None
        self._fidelity_map: Optional[list[int]] = None
        self._round_index = 0

    def setup(self, env: Any = None) -> None:
        """Store the environment for multi-fidelity index resolution."""
        self._env = env

    def set_acquisition(self, acquisition: Acquisition) -> None:
        """Replace the wrapped acquisition function."""
        self.acquisition = acquisition

    def set_cost_fn(
        self, cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]]
    ) -> None:
        """Replace the optional candidate cost function."""
        self.cost_fn = cost_fn

    def set_fidelity_map(self, fidelity_map: Optional[list[int]]) -> None:
        """Set the fidelity map for translating raw Choice env indices to domain values."""
        self._fidelity_map = fidelity_map

    def set_round_index(self, round_index: int) -> None:
        """Set the active-learning round used by the MF-GFN reward scale."""
        if round_index < 0:
            raise ValueError("round_index must be non-negative.")
        self._round_index = round_index

    def _round_reward_scale(self) -> float:
        """Return ``rho**round / beta`` for MF-GFN reward scaling."""
        return (self.reward_scale_rho**self._round_index) / self.reward_scale_beta

    def __call__(self, states: Union[torch.Tensor, List, npt.NDArray]) -> torch.Tensor:
        """Evaluate acquisition values for a batch of states in proxy format.

        MF-GFN round scaling is applied before the base-class reward
        transformation (``reward_function``, ``reward_min``, clipping, etc.).
        GFlowNet calls ``proxy.rewards()`` during training, so the full
        transformation pipeline is exercised automatically.

        Handles both single-fidelity (tensor/list of coord vectors) and
        multi-fidelity (list of dicts produced by a composite env's
        ``states2proxy``).

        Parameters
        ----------
        states : tensor, list, or ndarray
            Batch of states in proxy format.

        Returns
        -------
        values : torch.Tensor
            1-D tensor of scaled proxy values, one per state.

        Raises
        ------
        RuntimeError
            If no acquisition function has been set.
        """
        if self.acquisition is None:
            raise RuntimeError(
                "AcquisitionProxy has no acquisition function set. "
                "Call set_acquisition() before use."
            )

        candidates = proxy_states_to_candidates(
            states, self._env, fidelity_map=self._fidelity_map
        )
        if self.cost_fn is None:
            acq_values = self.acquisition.score(candidates)
        else:
            acq_values = self.acquisition.score(
                candidates,
                cost_weighting=cost_weighting_from_cost_fn(self.cost_fn),
            )
        values = torch.tensor(acq_values, dtype=self.float, device=self.device)
        return values * self._round_reward_scale()

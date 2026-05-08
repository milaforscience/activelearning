import torch
import numpy.typing as npt
from typing import Any, List, Optional, Union
from torchtyping import TensorType
from gflownet.proxy.base import Proxy
from activelearning.sampler.gflownet.utils import proxy_states_to_candidates
from activelearning.utils.types import Candidate


class AcquisitionProxy(Proxy):
    """GFlowNet proxy that wraps an active learning acquisition function.

    Bridges between GFlowNet's proxy interface (tensor states → tensor values)
    and the AL acquisition interface (Candidate objects → float values).

    States arrive in proxy format (continuous coordinates from
    ``env.states2proxy()``), are converted to ``Candidate`` objects, scored
    via :meth:`~activelearning.acquisition.acquisition.Acquisition.score`,
    and returned as a tensor.

    For multi-fidelity environments, sub-environment indices are read from the
    env stored by :meth:`setup`, so the correct ``idx_base_env`` and
    ``idx_fidelity`` are used regardless of wrapper variant (SetFix, FidFirst,
    FidLast).

    Parameters
    ----------
    acquisition : Any
        An active learning acquisition function implementing
        :meth:`~activelearning.acquisition.acquisition.Acquisition.score`.
        Set to ``None`` at init; must be set via :meth:`set_acquisition`
        before the proxy is used.
    reward_beta : float, default=1.0
        Positive denominator in the MF-GFN paper reward scale. A value of
        ``1e-6`` implements the paper's molecule reward temperature.
    reward_rho : float, default=1.0
        Positive per-round reward multiplier base. The proxy scales acquisition
        scores by ``reward_rho ** round_index / reward_beta``.
    **kwargs
        Forwarded to :class:`gflownet.proxy.base.Proxy` (device,
        float_precision, reward_function, reward_min, etc.).
    """

    def __init__(
        self,
        acquisition: Any = None,
        reward_beta: float = 1.0,
        reward_rho: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if reward_beta <= 0:
            raise ValueError("reward_beta must be positive.")
        if reward_rho <= 0:
            raise ValueError("reward_rho must be positive.")
        super().__init__(**kwargs)
        self.acquisition = acquisition
        self._env: Optional[Any] = None
        self.reward_beta = float(reward_beta)
        self.reward_rho = float(reward_rho)
        self._round_index = 0
        self._reward_fidelity: int | None = None

    def setup(self, env: Any = None) -> None:
        """Store the environment for multi-fidelity index resolution."""
        self._env = env

    def set_acquisition(self, acquisition: Any) -> None:
        """Replace the wrapped acquisition function."""
        self.acquisition = acquisition

    def set_round_index(self, round_index: int) -> None:
        """Set the zero-based active-learning round used for reward scaling."""
        if round_index < 0:
            raise ValueError("round_index must be non-negative.")
        self._round_index = round_index

    def set_reward_fidelity(self, reward_fidelity: int | None) -> None:
        """Override candidate fidelities before acquisition scoring."""
        self._reward_fidelity = reward_fidelity

    def _round_reward_scale(self) -> float:
        """Return ``rho**round / beta`` for MF-GFN reward scaling."""
        return (self.reward_rho**self._round_index) / self.reward_beta

    def __call__(self, states: Union[TensorType, List, npt.NDArray]) -> TensorType:
        """Evaluate proxy values for a batch of states in proxy format.

        Handles both single-fidelity (tensor/list of coord vectors) and
        multi-fidelity (list of dicts produced by a composite env's
        ``states2proxy``).

        Parameters
        ----------
        states : tensor, list, or ndarray
            Batch of states in proxy format (continuous coordinates).

        Returns
        -------
        values : TensorType
            1-D tensor of proxy values, one per state.

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

        candidates = proxy_states_to_candidates(states, self._env)
        if self._reward_fidelity is not None:
            candidates = [
                Candidate(
                    x=candidate.x,
                    fidelity=self._reward_fidelity,
                    metadata=candidate.metadata,
                )
                for candidate in candidates
            ]
        acq_values = self.acquisition.score(candidates)
        scaled_values = [value * self._round_reward_scale() for value in acq_values]
        return torch.tensor(scaled_values, dtype=self.float, device=self.device)

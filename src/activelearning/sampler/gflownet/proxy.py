import numpy.typing as npt
from typing import Any, Callable, List, Optional, Sequence, Union
import torch
from torchtyping import TensorType
from gflownet.proxy.base import Proxy
from activelearning.acquisition.cost_utility import cost_weighting_from_cost_fn
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
    cost_fn : callable, optional
        Optional candidate cost function. When provided, proxy scores are
        reweighted by inverse cost before they are returned to GFlowNet.
    **kwargs
        Forwarded to :class:`gflownet.proxy.base.Proxy` (device,
        float_precision, reward_function, reward_min, etc.).
    """

    def __init__(
        self,
        acquisition: Any = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.acquisition = acquisition
        self.cost_fn = cost_fn
        self._env: Optional[Any] = None

    def setup(self, env: Any = None) -> None:
        """Store the environment for multi-fidelity index resolution."""
        self._env = env

    def set_acquisition(self, acquisition: Any) -> None:
        """Replace the wrapped acquisition function."""
        self.acquisition = acquisition

    def set_cost_fn(
        self, cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]]
    ) -> None:
        """Replace the optional candidate cost function."""
        self.cost_fn = cost_fn

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
        if self.cost_fn is None:
            acq_values = self.acquisition.score(candidates)
        else:
            acq_values = self.acquisition.score(
                candidates,
                cost_weighting=cost_weighting_from_cost_fn(self.cost_fn),
            )
        return torch.tensor(acq_values, dtype=self.float, device=self.device)

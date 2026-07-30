import numpy.typing as npt
from typing import Any, Callable, List, Optional, Sequence, Union
import torch
from gflownet.proxy.base import Proxy
from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.cost_utility import cost_weighting_from_cost_fn
from activelearning.sampler.gflownet.utils import proxy_states_to_candidates
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY


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
    **kwargs
        Forwarded to :class:`gflownet.proxy.base.Proxy` (device,
        float_precision, reward_function, reward_min, etc.).
    """

    def __init__(
        self,
        acquisition: Acquisition = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.acquisition = acquisition
        self.cost_fn = cost_fn
        self._env: Optional[Any] = None
        self._fidelity_map: list[int] = [DEFAULT_FIDELITY]

    def setup(self, env: Any = None) -> None:
        """Store the environment for multi-fidelity index resolution.

        Parameters
        ----------
        env : Any, optional
            GFlowNet environment instance.  In multi-fidelity mode this must be
            a :class:`~activelearning.sampler.gflownet.multi_fidelity_env_wrapper.MultiFidelityGFlowNetEnvWrapperBase`
            so that ``idx_base_env`` and ``idx_fidelity`` are available.
        """
        self._env = env

    def set_acquisition(self, acquisition: Acquisition) -> None:
        """Replace the wrapped acquisition function.

        Parameters
        ----------
        acquisition : Acquisition
            The new acquisition function to use when scoring proxy states.
        """
        self.acquisition = acquisition

    def set_cost_fn(
        self, cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]]
    ) -> None:
        """Replace the optional candidate cost function.

        Parameters
        ----------
        cost_fn : callable or None
            A function that maps a sequence of :class:`~activelearning.utils.types.Candidate`
            objects to a list of costs.  When ``None``, proxy scores are
            returned without cost reweighting.
        """
        self.cost_fn = cost_fn

    def set_fidelity_map(self, fidelity_map: list[int]) -> None:
        """Set the fidelity map for translating raw Choice env indices to domain values.

        Parameters
        ----------
        fidelity_map : list[int]
            Maps 1-based ``Choice`` env states to domain fidelity values.
        """
        self._fidelity_map = fidelity_map

    def __call__(self, states: Union[torch.Tensor, List, npt.NDArray]) -> torch.Tensor:
        """Evaluate *raw* proxy values for a batch of states in proxy format.

        Returns raw acquisition scores without any reward transformation.
        Reward shaping (``reward_function``, ``reward_min``, clipping, etc.)
        is applied by the base-class :meth:`~gflownet.proxy.base.Proxy.rewards`
        method, which calls this method internally and then passes the result
        through ``proxy2reward`` / ``proxy2logreward``. GFlowNet always
        calls ``proxy.rewards()`` during training, so the full transformation
        pipeline is exercised automatically.

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
            1-D tensor of raw proxy values, one per state.

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
            states,
            self._env,
            fidelity_map=self._fidelity_map,
        )
        if self.cost_fn is None:
            acq_values = self.acquisition.score(candidates)
        else:
            acq_values = self.acquisition.score(
                candidates,
                cost_weighting=cost_weighting_from_cost_fn(self.cost_fn),
            )
        return torch.tensor(acq_values, dtype=self.float, device=self.device)

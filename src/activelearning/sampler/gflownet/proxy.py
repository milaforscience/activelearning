import torch
import numpy.typing as npt
from typing import Any, List, Optional, Union
from gflownet.proxy.base import Proxy
from activelearning.sampler.gflownet.utils import proxy_states_to_candidates


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
    acquisition : Any
        An active learning acquisition function implementing
        :meth:`~activelearning.acquisition.acquisition.Acquisition.score`.
        Set to ``None`` at init; must be set via :meth:`set_acquisition`
        before the proxy is used.
    **kwargs
        Forwarded to :class:`gflownet.proxy.base.Proxy` (device,
        float_precision, reward_function, reward_min, etc.).
    """

    def __init__(self, acquisition: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.acquisition = acquisition
        self._env: Optional[Any] = None

    def setup(self, env: Any = None) -> None:
        """Store the environment for multi-fidelity index resolution."""
        self._env = env

    def set_acquisition(self, acquisition: Any) -> None:
        """Replace the wrapped acquisition function."""
        self.acquisition = acquisition

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

        candidates = proxy_states_to_candidates(states, self._env)
        acq_values = self.acquisition.score(candidates)
        return torch.tensor(acq_values, dtype=self.float, device=self.device)

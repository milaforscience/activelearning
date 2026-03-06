import torch
import numpy.typing as npt
from typing import Any, List, Union
from torchtyping import TensorType
from gflownet.proxy.base import Proxy
from activelearning.utils.types import Candidate


class AcquisitionProxy(Proxy):
    """GFlowNet proxy that wraps an active learning acquisition function.

    Bridges between GFlowNet's proxy interface (tensor states → tensor values)
    and the AL acquisition interface (Candidate objects → float values).

    States arrive in proxy format (continuous coordinates from
    ``env.states2proxy()``), are converted to ``Candidate`` objects, evaluated
    by the acquisition function, and returned as a tensor.

    Parameters
    ----------
    acquisition : Any
        An active learning acquisition function compatible with
        ``Acquisition.__call__(Sequence[Candidate]) -> Sequence[float]``.
        Set to ``None`` at init; must be set via :meth:`set_acquisition`
        before the proxy is used.
    **kwargs
        Forwarded to :class:`gflownet.proxy.base.Proxy` (device,
        float_precision, reward_function, reward_min, etc.).
    """

    def __init__(self, acquisition: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.acquisition = acquisition

    def set_acquisition(self, acquisition: Any) -> None:
        """Replace the wrapped acquisition function.

        Parameters
        ----------
        acquisition : Any
            The new acquisition callable.
        """
        self.acquisition = acquisition

    def __call__(self, states: Union[TensorType, List, npt.NDArray]) -> TensorType:
        """Evaluate proxy values for a batch of states in proxy format.

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

        if torch.is_tensor(states):
            states_list = states.detach().cpu().tolist()
        else:
            states_list = [list(s) for s in states]

        candidates = [Candidate(x=tuple(s)) for s in states_list]
        acq_values = self.acquisition(candidates)

        return torch.tensor(acq_values, dtype=self.float, device=self.device)

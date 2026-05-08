"""GFlowNet sampler for continuous bounded grid domains."""

from typing import Any, Optional, Sequence

import torch
from hydra.utils import get_class
from gflownet.envs.grid import Grid
from omegaconf import DictConfig

from activelearning.sampler.fidelity_policy import SamplerFidelityPolicy
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler
from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.utils.types import Candidate


class GFlowNetGridSampler(GFlowNetSampler):
    """GFlowNet sampler that rescales grid coordinates to a target bounded domain.

    The GFlowNet :class:`gflownet.envs.grid.Grid` environment operates in a
    discrete grid whose cells span ``[cell_min, cell_max]^d``.  This sampler
    linearly maps the generated proxy coordinates to an arbitrary
    ``output_bounds`` domain before returning candidates.  If ``output_bounds``
    is ``None``, coordinates are returned in the native grid coordinate system.

    The configured env (``conf.env._target_``) must be a
    :class:`gflownet.envs.grid.Grid` or a subclass — a :exc:`ValueError` is
    raised at construction time if it is not.

    Parameters
    ----------
    n_samples : int
        Number of candidates to generate per :meth:`sample` call.
    conf : DictConfig
        Complete GFlowNet configuration tree (env, policy, gflownet, loss,
        buffer, evaluator, logger, proxy).
    output_bounds : sequence of (float, float), optional
        Per-dimension ``(lower, upper)`` bounds to which the grid coordinates
        are rescaled.  Must have one entry per grid dimension.  If ``None``,
        the native ``[cell_min, cell_max]`` coordinates are returned as-is.
    fidelity_policy : SamplerFidelityPolicy, optional
        Fidelity assignment policy shared with :class:`GFlowNetSampler`.
    reward_fidelity : int, optional
        Optional proxy-scoring fidelity override shared with
        :class:`GFlowNetSampler`.

    Raises
    ------
    ValueError
        If ``conf.env._target_`` does not resolve to a :class:`~gflownet.envs.grid.Grid`
        subclass.
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        output_bounds: Optional[Sequence[tuple[float, float]]] = None,
        fidelity_policy: SamplerFidelityPolicy | None = None,
        reward_fidelity: int | None = None,
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            conf=conf,
            fidelity_policy=fidelity_policy,
            reward_fidelity=reward_fidelity,
        )
        self._validate_grid_env(conf)
        if output_bounds is not None:
            self._out_lb = torch.tensor(
                [lo for lo, _ in output_bounds], dtype=torch.float64
            )
            self._out_ub = torch.tensor(
                [hi for _, hi in output_bounds], dtype=torch.float64
            )
        else:
            self._out_lb = None
            self._out_ub = None

    @staticmethod
    def _validate_grid_env(conf: DictConfig) -> None:
        """Raise ValueError if conf.env does not target a Grid subclass."""
        target = conf.env.get("_target_", "")
        try:
            cls = get_class(target)
        except Exception as exc:
            raise ValueError(
                f"GFlowNetGridSampler could not resolve conf.env._target_='{target}'."
            ) from exc
        if not issubclass(cls, Grid):
            raise ValueError(
                f"GFlowNetGridSampler requires a Grid environment "
                f"(gflownet.envs.grid.Grid or subclass), "
                f"but conf.env._target_='{target}' resolves to {cls.__name__}."
            )

    @staticmethod
    def _get_grid_bounds(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract cell bounds from the built Grid env.

        For multi-fidelity envs the base env is retrieved from
        ``env.env_base``. Grid stores its cell positions in ``env.cells``
        (a 1-D array from ``np.linspace(cell_min, cell_max, length)``), so
        the actual bounds are ``cells[0]`` and ``cells[-1]``.

        Parameters
        ----------
        env : GFlowNetEnv
            The environment (or MF wrapper) built by the GFlowNet agent.

        Returns
        -------
        grid_min, grid_max : torch.Tensor
            Scalar float64 tensors with the grid's lower and upper coordinate bounds.
        """
        base_env = (
            env.env_base
            if isinstance(env, MultiFidelityGFlowNetEnvWrapperBase)
            else env
        )
        grid_min = torch.tensor(float(base_env.cells[0]), dtype=torch.float64)
        grid_max = torch.tensor(float(base_env.cells[-1]), dtype=torch.float64)
        return grid_min, grid_max

    def _states_to_candidates(self, states: Any, env: Any) -> list[Candidate]:
        """Convert GFlowNet states to :class:`~activelearning.utils.types.Candidate` objects.

        If ``output_bounds`` was provided, linearly rescales each coordinate from
        the grid's native ``[cell_min, cell_max]`` range (read from the built env)
        to the target domain.

        Parameters
        ----------
        states : tensor or list
            Terminating states from the GFlowNet trajectory batch.
        env : GFlowNetEnv
            The environment used to convert states to proxy coordinates.
        """
        candidates = super()._states_to_candidates(states, env)
        if self._out_lb is None:
            return candidates
        grid_min, grid_max = self._get_grid_bounds(env)
        grid_range = grid_max - grid_min
        rescaled = []
        for c in candidates:
            coords = torch.tensor(c.x, dtype=torch.float64)
            normed = (coords - grid_min) / grid_range
            new_coords = normed * (self._out_ub - self._out_lb) + self._out_lb
            rescaled.append(
                Candidate(x=tuple(new_coords.tolist()), fidelity=c.fidelity)
            )
        return rescaled

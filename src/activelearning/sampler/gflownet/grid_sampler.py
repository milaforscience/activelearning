"""GFlowNet sampler for grid domains."""

from types import MethodType
from typing import List, Literal, Optional, Sequence

import numpy as np
import torch
from gflownet.envs.grid import Grid
from gflownet.utils.common import tfloat
from hydra.utils import get_class
from omegaconf import DictConfig

from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler
from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.utils.types import DEFAULT_FIDELITY


class GFlowNetGridSampler(GFlowNetSampler):
    """Convenience GFlowNet sampler for :class:`gflownet.envs.grid.Grid` environments.

    Validates at construction time that the configured env is a
    :class:`~gflownet.envs.grid.Grid` subclass, giving an early and clear error
    if not.

    Coordinate system
    -----------------
    The Grid env maps integer cell indices to continuous coordinates.  Use
    ``domain_bounds`` to set per-dimension coordinate ranges::

        sampler:
          type: GFlowNetGridSampler
          domain_bounds:
            - [-5.0, 10.0]   # x1
            - [0.0, 15.0]    # x2
          conf:
            env:
              n_dim: 2
              length: 100

    Without ``domain_bounds``, all dimensions share the range ``[cell_min, cell_max]``
    from ``conf.env`` (Grid defaults: ``-1`` to ``1``).

    Parameters
    ----------
    n_samples : int
        Number of candidates to generate per :meth:`sample` call.
    conf : DictConfig
        Complete GFlowNet configuration tree (env, policy, gflownet, loss,
        buffer, evaluator, logger, proxy).
    fidelities : Sequence[int]
        See :class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`.
    fidelity_action : {"any", "first", "last"}
        Controls when fidelity is chosen during a trajectory. Only used when
        more than one fidelity is configured. See
        :class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`
        for full semantics.
    domain_bounds : list of [lo, hi] pairs, optional
        Per-dimension coordinate bounds, one ``[lo, hi]`` pair per dimension.
        Length must equal ``conf.env.n_dim`` and each pair must satisfy
        ``lo < hi``.

    Raises
    ------
    ValueError
        If ``conf.env._target_`` does not resolve to a
        :class:`~gflownet.envs.grid.Grid` subclass.
    ValueError
        If ``domain_bounds`` length does not match ``conf.env.n_dim``, or any
        ``[lo, hi]`` pair has ``lo >= hi``.
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        fidelities: Sequence[int] = (DEFAULT_FIDELITY,),
        fidelity_action: Literal["any", "first", "last"] = "any",
        domain_bounds: Optional[List[List[float]]] = None,
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            conf=conf,
            fidelities=fidelities,
            fidelity_action=fidelity_action,
        )
        self._validate_grid_env(conf)
        if domain_bounds is not None:
            self._validate_domain_bounds(domain_bounds, conf)
        self.domain_bounds = domain_bounds

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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
    def _validate_domain_bounds(
        domain_bounds: List[List[float]], conf: DictConfig
    ) -> None:
        """Raise ValueError if domain_bounds is malformed or mismatches n_dim."""
        n_dim = conf.env.get("n_dim", 2)
        if len(domain_bounds) != n_dim:
            raise ValueError(
                f"domain_bounds has {len(domain_bounds)} entries but "
                f"conf.env.n_dim={n_dim}. Provide one [lo, hi] pair per dimension."
            )
        for i, bounds in enumerate(domain_bounds):
            lo, hi = bounds
            if lo >= hi:
                raise ValueError(
                    f"domain_bounds[{i}] has lo={lo} >= hi={hi}. "
                    "Each bound must satisfy lo < hi."
                )

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _build_agent(self, acquisition, cost_fn=None):
        """Build agent, applying per-dimension coordinate bounds to the env if set."""
        agent = super()._build_agent(acquisition, cost_fn=cost_fn)
        # For multi-fidelity, bounds are applied inside _build_multi_fidelity_env.
        if self.domain_bounds is not None and self._n_fidelities == 1:
            _apply_per_dimension_bounds(agent.env, self.domain_bounds)
        return agent

    def _build_multi_fidelity_env(
        self, conf: DictConfig, device: str, fp: int
    ) -> MultiFidelityGFlowNetEnvWrapperBase:
        """Construct the multi-fidelity env, applying per-dimension bounds if set."""
        env = super()._build_multi_fidelity_env(conf, device, fp)
        if self.domain_bounds is not None:
            _apply_per_dimension_bounds(env.env_base, self.domain_bounds)
        return env


def _apply_per_dimension_bounds(
    env: Grid,
    domain_bounds: List[List[float]],
) -> None:
    """Configure a Grid env to use per-dimension coordinate ranges.

    By default the Grid env maps all dimensions to the same ``[cell_min,
    cell_max]`` range.  This function overrides that mapping so each dimension
    uses its own independent linspace, making the grid cover an axis-aligned
    rectangular domain.

    Parameters
    ----------
    env : Grid
        The Grid env instance to configure.
    domain_bounds : list of [lo, hi] pairs
        Per-dimension coordinate ranges. Length must match ``env.n_dim``.
    """
    n_dim = env.n_dim
    length = env.length

    cells_matrix = torch.stack(
        [
            torch.tensor(
                np.linspace(lo, hi, length), device=env.device, dtype=env.float
            )
            for lo, hi in domain_bounds
        ],
        dim=0,
    )

    def states2proxy_per_dim(self, states):
        states = tfloat(states, device=self.device, float_type=self.float)
        # states2policy returns (batch, n_dim * length); reshape to (batch, n_dim, length)
        return (
            self.states2policy(states).reshape((states.shape[0], n_dim, length))
            * cells_matrix.to(states.device)[None, :, :]
        ).sum(axis=2)

    env.states2proxy = MethodType(states2proxy_per_dim, env)
    env.cells = np.linspace(domain_bounds[0][0], domain_bounds[0][1], length)
    env.cells_torch = cells_matrix[0]

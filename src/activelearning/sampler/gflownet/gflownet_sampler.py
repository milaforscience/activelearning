"""GFlowNet-based sampler for active learning candidate generation."""

import hydra
import torch
from typing import Any, Iterable, Optional
from omegaconf import DictConfig, OmegaConf
from gflownet.utils.common import gflownet_from_config
from activelearning.sampler.gflownet.logger_wrapper import RuntimeGFlowNetLoggerWrapper
from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapper,
)
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class GFlowNetSampler(Sampler):
    """Sampler that uses a GFlowNet agent to generate candidates.

    On each :meth:`sample` call a fresh agent is built via
    ``gflownet_from_config``, trained, and used to draw ``n_samples`` forward
    trajectories.  Device and precision come from the runtime context
    (:meth:`~activelearning.runtime.ALRuntimeMixin.bind_runtime_context`).

    Parameters
    ----------
    n_samples : int
        Candidates to generate per :meth:`sample` call.
    conf : DictConfig
        GFlowNet config with keys ``env``, ``policy``, ``gflownet``, ``loss``,
        ``buffer``, ``evaluator``, ``logger``, ``proxy`` — matching the
        ``gflownet_from_config`` contract.
    n_fidelities : int
        Fidelity levels.  When > 1 the env is wrapped with
        :class:`~activelearning.sampler.gflownet.multi_fidelity_env_wrapper.MultiFidelityGFlowNetEnvWrapper`.
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        n_fidelities: int = 1,
    ) -> None:
        self.n_samples = n_samples
        self.conf = conf
        self.n_fidelities = n_fidelities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device_str(self) -> str:
        """Return the device as a plain string (e.g. ``'cpu'``)."""
        return str(self.device)

    def _float_precision(self) -> int:
        """Return floating-point precision as an integer (32 or 64)."""
        return 32 if self.dtype == torch.float32 else 64

    def _build_agent(self, acquisition: Any) -> Any:
        """Build a ``GFlowNetAgent`` ready for training.

        Merges runtime device/precision into conf and calls
        ``gflownet_from_config``.  For multi-fidelity, builds the env as a
        factory (``_partial_=True``) so each ``env.copy()`` gets a fresh
        ``env_base``.  Acquisition and runtime logger are injected
        post-construction.

        Parameters
        ----------
        acquisition : Any
            Acquisition function used as the reward signal.

        Returns
        -------
        agent : GFlowNetAgent
        """

        device = self._device_str()
        fp = self._float_precision()
        conf = OmegaConf.merge(self.conf, {"device": device, "float_precision": fp})

        # When env=None, gflownet_from_config instantiates the env from conf.env
        # using the device/float_precision already merged into conf above.
        env = None
        if self.n_fidelities > 1:
            env_base_maker = hydra.utils.instantiate(
                conf.env, device=device, float_precision=fp, _partial_=True
            )
            env = MultiFidelityGFlowNetEnvWrapper(
                env_base_maker=env_base_maker, n_fidelities=self.n_fidelities
            )

        agent = gflownet_from_config(conf, env=env)
        agent.proxy.set_acquisition(acquisition)

        if self.logger is not None:
            agent.logger = RuntimeGFlowNetLoggerWrapper(
                runtime_logger=self.logger,
                config=conf,
                logger_conf=conf.logger,
            )

        return agent

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------

    def _states_to_candidates(self, states: Any, env: Any) -> list[Candidate]:
        """Convert GFlowNet terminating states to :class:`~activelearning.utils.types.Candidate` objects.

        Parameters
        ----------
        states : tensor or list
            Terminating states from a trajectory batch.
        env : GFlowNetEnv
            Environment used to map states to proxy coordinates.

        Returns
        -------
        list[Candidate]
        """
        if torch.is_tensor(states):
            proxy_coords = env.states2proxy(states)
            coords = proxy_coords.detach().cpu().to(torch.float64)
        elif isinstance(states, list) and len(states) > 0:
            proxy_coords = env.states2proxy(states)
            if torch.is_tensor(proxy_coords):
                coords = proxy_coords.detach().cpu().to(torch.float64)
            else:
                coords = torch.tensor(
                    [list(s) for s in proxy_coords], dtype=torch.float64
                )
        else:
            return []
        return [Candidate(x=tuple(row.tolist())) for row in coords]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(
        self,
        acquisition: Optional[Any] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        """Train a GFlowNet and return sampled candidates.

        Parameters
        ----------
        acquisition : Optional[Any]
            Acquisition function used as the reward signal. Must not be ``None``.
        observations : Optional[Iterable[Observation]]
            Unused; reserved for future warm-starting.

        Returns
        -------
        list[Candidate]
            ``n_samples`` candidates in proxy coordinates.

        Raises
        ------
        ValueError
            If ``acquisition`` is ``None``.
        """
        if acquisition is None:
            raise ValueError("GFlowNetSampler requires an acquisition function.")

        agent = self._build_agent(acquisition)
        agent.train()

        batch, _ = agent.sample_batch(n_forward=self.n_samples, train=False)
        raw_states = batch.get_terminating_states()

        return self._states_to_candidates(raw_states, agent.env)

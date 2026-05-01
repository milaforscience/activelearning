"""GFlowNet-based sampler for active learning candidate generation."""

from dataclasses import replace
import logging
import hydra
import torch
from typing import Any, Iterable, Literal, Optional
from omegaconf import DictConfig, OmegaConf
from gflownet.utils.common import gflownet_from_config
from activelearning.sampler.gflownet.logger_wrapper import RuntimeGFlowNetLoggerWrapper
from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    build_multi_fidelity_env_wrapper,
)
from activelearning.sampler.gflownet.utils import proxy_states_to_candidates
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation

logger = logging.getLogger(__name__)


class GFlowNetSampler(Sampler):
    """Sampler that trains a GFlowNet agent and returns sampled candidates.

    Each :meth:`sample` call builds a fresh agent via ``gflownet_from_config``,
    trains it, and draws ``n_samples`` forward trajectories. Device and
    precision are taken from the runtime context.

    Parameters
    ----------
    n_samples : int
        Number of candidates to generate per :meth:`sample` call.
    conf : DictConfig
        GFlowNet config (``env``, ``policy``, ``gflownet``, ``loss``,
        ``buffer``, ``evaluator``, ``logger``, ``proxy``).
    n_fidelities : int
        Number of fidelity levels. When > 1 the env is wrapped with a
        multi-fidelity wrapper chosen by *fidelity_action*.
    fidelity_action : {"any", "first", "last"}
        Controls when fidelity is chosen during a trajectory. Only used when
        ``n_fidelities > 1``.
        - ``"any"`` *(default)* — fidelity may be chosen at any point,
          interleaved with base-env actions (SetFix wrapper).
        - ``"first"`` — fidelity is chosen before any base-env action (Stack).
        - ``"last"`` — fidelity is chosen after all base-env actions (Stack).
    fixed_fidelity : int, optional
        Stamps the same fidelity onto every sampled candidate. Intended for
        single-fidelity tutorial runs that still query a multi-fidelity oracle.
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        n_fidelities: int = 1,
        fidelity_action: Literal["any", "first", "last"] = "any",
        fixed_fidelity: int | None = None,
    ) -> None:
        if fixed_fidelity is not None and n_fidelities != 1:
            raise ValueError("fixed_fidelity is only supported when n_fidelities=1.")
        self.n_samples = n_samples
        self.conf = conf
        self.n_fidelities = n_fidelities
        self.fidelity_action = fidelity_action
        self.fixed_fidelity = fixed_fidelity
        if n_fidelities == 1 and fidelity_action != "any":
            logger.warning(
                "fidelity_action=%r has no effect when n_fidelities=1.",
                fidelity_action,
            )

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
        """Build and return a ``GFlowNetAgent`` ready for training.

        Merges runtime device/precision into the config, then calls
        ``gflownet_from_config``. For multi-fidelity, the env is built as a
        factory so each copy gets a fresh base env. The acquisition function
        and runtime logger are injected after construction.

        Parameters
        ----------
        acquisition : Any
            Acquisition function used as the GFlowNet reward proxy.
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
            env = build_multi_fidelity_env_wrapper(
                fidelity_action=self.fidelity_action,
                env_base_maker=env_base_maker,
                n_fidelities=self.n_fidelities,
                float_precision=fp,
                device=device,
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

        Guards against empty or invalid ``states`` before calling
        ``env.states2proxy``; delegates the actual conversion to
        :func:`~activelearning.sampler.gflownet.utils.proxy_states_to_candidates`.

        Parameters
        ----------
        states : tensor or list
            Terminating states from a trajectory batch.
        env : GFlowNetEnv
            The environment used to map states to proxy coordinates.
        """
        if not isinstance(states, (list, torch.Tensor)) or len(states) == 0:
            return []
        return proxy_states_to_candidates(env.states2proxy(states), env)

    def _apply_fixed_fidelity(self, candidates: list[Candidate]) -> list[Candidate]:
        """Stamp a fixed fidelity onto sampled candidates when configured."""
        if self.fixed_fidelity is None:
            return candidates
        return [
            replace(candidate, fidelity=self.fixed_fidelity) for candidate in candidates
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(
        self,
        acquisition: Optional[Any] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        """Train a GFlowNet agent and return sampled candidates.

        Parameters
        ----------
        acquisition : Optional[Any]
            Acquisition function used as the GFlowNet reward proxy. Required.
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

        candidates = self._states_to_candidates(raw_states, agent.env)
        return self._apply_fixed_fidelity(candidates)

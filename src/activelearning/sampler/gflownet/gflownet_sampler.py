"""GFlowNet-based sampler for active learning candidate generation."""

import logging
import hydra
import torch
from typing import Any, Callable, Iterable, Literal, Optional, Sequence
from omegaconf import DictConfig, OmegaConf
from gflownet.gflownet import GFlowNetAgent
from gflownet.utils.common import gflownet_from_config
from activelearning.sampler.gflownet.logger_wrapper import RuntimeGFlowNetLoggerWrapper
from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
    build_multi_fidelity_env_wrapper,
)
from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.gflownet.utils import proxy_states_to_candidates
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY, Observation

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
    fidelities : Sequence[int]
        Fidelity levels to generate. Defaults to the single level
        :data:`~activelearning.utils.types.DEFAULT_FIDELITY`. More than one
        entry enables multi-fidelity mode: each sampled candidate is assigned
        one of these values as its ``fidelity``.
    fidelity_action : {"any", "first", "last"}
        Controls when fidelity is chosen during a trajectory.  Only relevant in
        multi-fidelity mode (``len(fidelities) > 1``).

        - ``"any"`` *(default)* — fidelity may be chosen at any point,
          interleaved with base-env actions (SetFix wrapper).
        - ``"first"`` — fidelity is chosen before any base-env action (Stack).
        - ``"last"`` — fidelity is chosen after all base-env actions (Stack).
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        fidelities: Sequence[int] = (DEFAULT_FIDELITY,),
        fidelity_action: Literal["any", "first", "last"] = "any",
    ) -> None:
        self.n_samples = n_samples
        self.conf = conf
        self.fidelities = list(fidelities)
        self._n_fidelities = len(self.fidelities)
        self.fidelity_action = fidelity_action
        if self._n_fidelities == 1 and fidelity_action != "any":
            logger.warning(
                "fidelity_action=%r has no effect in single-fidelity mode "
                "(fidelities has %d level).",
                fidelity_action,
                self._n_fidelities,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device_str(self) -> str:
        """Return the device as a plain string (e.g. ``'cpu'``).

        Returns
        -------
        str
            String representation of the runtime device.
        """
        return str(self.device)

    def _float_precision(self) -> int:
        """Return floating-point precision as an integer (32 or 64).

        Returns
        -------
        int
            ``32`` when the runtime dtype is :data:`torch.float32`,
            ``64`` otherwise.
        """
        return 32 if self.dtype == torch.float32 else 64

    def _build_agent(
        self,
        acquisition: Acquisition,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> GFlowNetAgent:
        """Build and return a ``GFlowNetAgent`` ready for training.

        Merges runtime device/precision into the config, then calls
        ``gflownet_from_config``. For single-fidelity, the environment is
        instantiated directly from the config. For multi-fidelity, ``conf.env``
        only describes the base environment — the multi-fidelity wrapper is not
        representable in a single Hydra config and must be built
        programmatically (see :meth:`_build_multi_fidelity_env`). The
        acquisition function and runtime logger are injected after construction.

        Parameters
        ----------
        acquisition : Acquisition
            Acquisition function used as the GFlowNet reward proxy.
        cost_fn : callable, optional
            Candidate cost function forwarded to the proxy so acquisition
            scores can be reweighted before GFlowNet uses them as rewards.

        Returns
        -------
        agent : GFlowNetAgent
            A fully configured agent with the proxy and logger injected,
            ready to be trained via ``agent.train()``.
        """
        device = self._device_str()
        fp = self._float_precision()
        conf = OmegaConf.merge(self.conf, {"device": device, "float_precision": fp})

        env = (
            self._build_multi_fidelity_env(conf, device, fp)
            if self._n_fidelities > 1
            else None
        )

        agent = gflownet_from_config(conf, env=env)
        agent.proxy.set_acquisition(acquisition)
        agent.proxy.set_cost_fn(cost_fn)
        agent.proxy.set_fidelity_map(self.fidelities)

        if self.logger is not None:
            agent.logger = RuntimeGFlowNetLoggerWrapper(
                runtime_logger=self.logger,
                config=conf,
                logger_conf=conf.logger,
            )

        return agent

    def _build_multi_fidelity_env(
        self, conf: DictConfig, device: str, fp: int
    ) -> MultiFidelityGFlowNetEnvWrapperBase:
        """Construct the multi-fidelity environment wrapper.

        The Hydra config (``conf.env``) only describes the base environment
        (e.g. Grid). The multi-fidelity wrapper composes the base env with a
        discrete fidelity-choice env, which cannot be expressed as a single
        Hydra target. This method creates a partial callable from ``conf.env``
        and passes it to the wrapper factory.

        Parameters
        ----------
        conf : DictConfig
            Merged GFlowNet config with device and float precision already set.
        device : str
            Device string (e.g. ``'cpu'`` or ``'cuda:0'``).
        fp : int
            Floating-point precision (32 or 64).

        Returns
        -------
        env : MultiFidelityGFlowNetEnvWrapperBase
            The assembled wrapper, ready to be passed to ``gflownet_from_config``.
        """
        env_base_maker = hydra.utils.instantiate(
            conf.env, device=device, float_precision=fp, _partial_=True
        )
        return build_multi_fidelity_env_wrapper(
            fidelity_action=self.fidelity_action,
            env_base_maker=env_base_maker,
            n_fidelities=self._n_fidelities,
            float_precision=fp,
            device=device,
        )

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
            The environment used to map states to proxy format.

        Returns
        -------
        list[Candidate]
            Candidates built from the proxy-format states, or an
            empty list if ``states`` is empty or of an unsupported type.
        """
        if not isinstance(states, (list, torch.Tensor)):
            raise TypeError(
                f"states must be a list or torch.Tensor, got {type(states).__name__}. "
                "This likely indicates a bug in the calling code."
            )
        if len(states) == 0:
            return []
        return proxy_states_to_candidates(
            env.states2proxy(states),
            env,
            fidelity_map=self.fidelities,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(
        self,
        acquisition: Optional[Any] = None,
        observations: Optional[Iterable[Observation]] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        """Train a GFlowNet agent and return sampled candidates.

        Parameters
        ----------
        acquisition : Optional[Any]
            Acquisition function used as the GFlowNet reward proxy. Required.
        observations : Optional[Iterable[Observation]]
            Unused; reserved for future warm-starting.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Optional candidate cost function forwarded to the proxy so
            acquisition scores can be reweighted before sampling.

        Returns
        -------
        list[Candidate]
            ``n_samples`` candidates in proxy format.

        Raises
        ------
        ValueError
            If ``acquisition`` is ``None``.
        """
        if acquisition is None:
            raise ValueError("GFlowNetSampler requires an acquisition function.")

        agent = self._build_agent(acquisition, cost_fn=cost_fn)
        agent.train()

        batch, _ = agent.sample_batch(n_forward=self.n_samples, train=False)
        states_term = batch.get_terminating_states()

        return self._states_to_candidates(states_term, agent.env)

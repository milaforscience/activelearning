"""GFlowNet-based sampler for active learning candidate generation."""

import logging
import random
import hydra
import torch
from typing import Any, Iterable, Optional
from omegaconf import DictConfig, OmegaConf
from gflownet.utils.common import gflownet_from_config
from activelearning.sampler.fidelity_policy import (
    FixedFidelityPolicy,
    JointSamplingFidelityPolicy,
    SamplerFidelityPolicy,
    apply_fidelity_policy,
)
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
    fidelity_policy : SamplerFidelityPolicy, optional
        Declares how the sampler should assign fidelities to returned candidates.
        A ``joint_sampling`` policy uses a multi-fidelity GFlowNet environment; other
        policies stamp fidelities after state decoding.
    reward_fidelity : int, optional
        Optional fidelity override used only while scoring candidates through the
        GFlowNet proxy. This preserves experiments that train proposals against a
        fixed-fidelity reward while returning candidates with a different final
        fidelity policy.
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        fidelity_policy: SamplerFidelityPolicy | None = None,
        reward_fidelity: int | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.conf = conf
        self.fidelity_policy = fidelity_policy
        self.reward_fidelity = reward_fidelity
        if isinstance(self.fidelity_policy, JointSamplingFidelityPolicy):
            if self.reward_fidelity is not None:
                raise ValueError(
                    "reward_fidelity is not supported when fidelity_policy is joint_sampling."
                )
        elif self.reward_fidelity is None and isinstance(
            self.fidelity_policy, FixedFidelityPolicy
        ):
            self.reward_fidelity = self.fidelity_policy.value

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
        joint_sampling_policy = self._joint_sampling_policy()
        if joint_sampling_policy is not None:
            env_base_maker = hydra.utils.instantiate(
                conf.env, device=device, float_precision=fp, _partial_=True
            )
            env = build_multi_fidelity_env_wrapper(
                fidelity_action=joint_sampling_policy.action,
                env_base_maker=env_base_maker,
                n_fidelities=joint_sampling_policy.n_fidelities,
                float_precision=fp,
                device=device,
            )

        agent = gflownet_from_config(conf, env=env)
        agent.proxy.set_acquisition(acquisition)
        if hasattr(agent.proxy, "set_reward_fidelity"):
            agent.proxy.set_reward_fidelity(self.reward_fidelity)
        if hasattr(agent.proxy, "set_round_index"):
            agent.proxy.set_round_index(self.active_learning_round)

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

    def _joint_sampling_policy(self) -> JointSamplingFidelityPolicy | None:
        """Return the joint-sampling policy when configured."""

        if isinstance(self.fidelity_policy, JointSamplingFidelityPolicy):
            return self.fidelity_policy
        return None

    def _apply_output_fidelity_policy(
        self, candidates: list[Candidate]
    ) -> list[Candidate]:
        """Apply the configured output fidelity policy to sampled candidates."""

        if self.fidelity_policy is None or isinstance(
            self.fidelity_policy, JointSamplingFidelityPolicy
        ):
            return candidates
        rng = random.Random(self.runtime_context.seed + self.active_learning_round)
        return apply_fidelity_policy(candidates, self.fidelity_policy, rng)

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
        return self._apply_output_fidelity_policy(candidates)

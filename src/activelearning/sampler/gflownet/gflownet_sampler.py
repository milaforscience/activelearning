import torch
from typing import Any, Iterable, Optional
from omegaconf import DictConfig
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class GFlowNetSampler(Sampler):
    """Sampler that uses a GFlowNet agent to generate candidates.

    At each active-learning iteration the sampler:
    1. Wraps the current acquisition function in a GFlowNet proxy.
    2. Builds a GFlowNet agent from the Hydra configs.
    3. Trains the agent to sample proportionally to the acquisition reward.
    4. Draws ``n_samples`` forward trajectories and converts the
       terminating states to ``Candidate`` objects.

    Parameters
    ----------
    n_samples : int
        Number of candidate samples to generate per ``sample()`` call.
    conf : DictConfig
        Hydra/OmegaConf config tree with keys ``env``, ``policy``,
        ``agent``, ``logger``, ``proxy``, and optionally ``state_flow``.
    device : str
        Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
    float_precision : int
        Floating-point precision (32 or 64).
    output_bounds : Sequence[tuple[float, float]] | None
        Per-dimension ``(lo, hi)`` bounds for the output domain.  If
        provided, grid coordinates in ``[cell_min, cell_max]`` are linearly
        rescaled to these bounds.  Length must equal ``n_dim``.
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        device: str,
        float_precision: int,
    ) -> None:
        import hydra

        self.n_samples = n_samples
        self.device = device
        self.float_precision = float_precision
        self.conf = conf

        self.env_maker = hydra.utils.instantiate(
            self.conf.env,
            device=device,
            float_precision=float_precision,
            _partial_=True,
        )
        env = self.env_maker()

        self.forward_policy = hydra.utils.instantiate(
            self.conf.policy.forward,
            env=env,
            device=device,
            float_precision=float_precision,
        )
        self.backward_policy = hydra.utils.instantiate(
            self.conf.policy.backward,
            env=env,
            device=device,
            float_precision=float_precision,
        )
        self.state_flow = (
            hydra.utils.instantiate(
                self.conf.state_flow,
                env=env,
                device=device,
                float_precision=float_precision,
                base=self.forward_policy,
            )
            if self.conf.state_flow is not None
            else None
        )

    def _build_agent(self, proxy: Any) -> Any:
        """Assemble and return a ``GFlowNetAgent`` ready for training."""
        import hydra

        logger = hydra.utils.instantiate(self.conf.logger, self.conf, _recursive_=False)

        env = self.env_maker()
        proxy.setup(env)

        loss = hydra.utils.instantiate(
            self.conf.agent.loss,
            forward_policy=self.forward_policy,
            backward_policy=self.backward_policy,
            state_flow=self.state_flow,
            device=self.device,
            float_precision=self.float_precision,
        )
        buffer = hydra.utils.instantiate(
            self.conf.agent.buffer,
            env=env,
            proxy=proxy,
            datadir=logger.datadir,
        )
        evaluator = hydra.utils.instantiate(self.conf.agent.evaluator)

        return hydra.utils.instantiate(
            self.conf.agent,
            env_maker=self.env_maker,
            proxy=proxy,
            device=self.device,
            float_precision=self.float_precision,
            loss=loss,
            buffer=buffer,
            forward_policy=self.forward_policy,
            backward_policy=self.backward_policy,
            state_flow=self.state_flow,
            logger=logger,
            evaluator=evaluator,
            _recursive_=False,
        )

    def _states_to_candidates(self, states: Any, env: Any) -> list[Candidate]:
        """Convert GFlowNet terminating states to ``Candidate`` objects."""
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

    def sample(
        self,
        acquisition: Optional[Any] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        """Train a GFlowNet and sample candidates proportional to the reward.

        Parameters
        ----------
        acquisition : Optional[Any]
            Acquisition function used as the GFlowNet reward signal.
        observations : Optional[Iterable[Observation]]
            Current observations (unused; reserved for future warm-starting).

        Returns
        -------
        candidates : list[Candidate]
            ``n_samples`` candidates with continuous proxy coordinates.
        """
        if acquisition is None:
            raise ValueError("GFlowNetSampler requires an acquisition function.")

        import hydra

        proxy = hydra.utils.instantiate(
            self.conf.proxy,
            acquisition=acquisition,
            device=self.device,
            float_precision=self.float_precision,
        )
        agent = self._build_agent(proxy)
        agent.train()

        batch, _ = agent.sample_batch(n_forward=self.n_samples, train=False)
        raw_states = batch.get_terminating_states()

        env = self.env_maker()
        return self._states_to_candidates(raw_states, env)

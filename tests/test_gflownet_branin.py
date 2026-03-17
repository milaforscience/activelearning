"""End-to-end test for GFlowNetSampler with the BraninOracle.

Validates the full pipeline: config loading → sampler construction → GFlowNet
training → candidate generation → active-learning loop integration using the
BraninOracle and the ``active_learning`` function.
"""

import shutil
import tempfile
from unittest.mock import Mock
import pytest
import torch
from typing import Sequence, Optional, Any
from omegaconf import DictConfig, OmegaConf
from botorch.test_functions.multi_fidelity import AugmentedBranin
from activelearning.acquisition.acquisition import Acquisition
from activelearning.active_learning import active_learning
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.augmented_function_oracle import BraninOracle
from activelearning.runtime import RuntimeContext
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.utils.types import Candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = "config"

_AUGMENTED_BRANIN = AugmentedBranin(negate=False)
# Branin natural bounds (excluding fidelity dim)
_BRANIN_BOUNDS = _AUGMENTED_BRANIN._bounds[:-1]  # [(-5, 10), (0, 15)]


def _load_yaml(path: str) -> OmegaConf:
    return OmegaConf.load(path)


def _compose_sampler_conf(
    n_train_steps: int = 20,
    grid_length: int = 20,
    cell_min: float = 0.0,
    cell_max: float = 1.0,
) -> tuple[OmegaConf, str]:
    """Manually compose configs (mirrors Hydra composition for the Branin experiment).

    Parameters
    ----------
    n_train_steps : int
        Number of GFlowNet training steps (keep small for tests).
    grid_length : int
        Number of cells per dimension.
    cell_min : float
        Lower bound of the continuous coordinate mapping.
    cell_max : float
        Upper bound of the continuous coordinate mapping.

    Returns
    -------
    conf : OmegaConf
        Merged sampler configuration.
    tmpdir : str
        Temporary directory for GFlowNet logs (caller should clean up).
    """
    env_cfg = _load_yaml(f"{CONFIG_DIR}/env/grid.yaml")
    if "defaults" in env_cfg:
        del env_cfg["defaults"]

    agent_cfg = _load_yaml(f"{CONFIG_DIR}/sampler/conf/agent/base.yaml")
    logger_cfg = _load_yaml(f"{CONFIG_DIR}/sampler/conf/logger/base.yaml")
    policy_cfg = _load_yaml(f"{CONFIG_DIR}/sampler/conf/policy/base.yaml")
    proxy_base = _load_yaml(f"{CONFIG_DIR}/sampler/conf/proxy/base.yaml")
    proxy_acq = _load_yaml(f"{CONFIG_DIR}/sampler/conf/proxy/acquisition.yaml")
    if "defaults" in proxy_acq:
        del proxy_acq["defaults"]
    proxy_cfg = OmegaConf.merge(proxy_base, proxy_acq)

    branin_cfg = _load_yaml(f"{CONFIG_DIR}/branin.yaml")
    if "defaults" in branin_cfg:
        del branin_cfg["defaults"]

    if "env" in branin_cfg:
        env_cfg = OmegaConf.merge(env_cfg, branin_cfg.env)

    # Override grid parameters for testing
    env_cfg.length = grid_length
    env_cfg.cell_min = cell_min
    env_cfg.cell_max = cell_max

    sampler_conf = OmegaConf.create(
        {
            "env": env_cfg,
            "policy": policy_cfg,
            "agent": agent_cfg,
            "logger": logger_cfg,
            "proxy": proxy_cfg,
            "state_flow": None,
        }
    )

    if "sampler" in branin_cfg and "conf" in branin_cfg.sampler:
        sampler_conf = OmegaConf.merge(sampler_conf, branin_cfg.sampler.conf)

    # Use a temporary log directory to avoid polluting the repo
    tmpdir = tempfile.mkdtemp(prefix="gfn_test_")
    sampler_conf.logger.logdir.root = tmpdir
    sampler_conf.logger.logdir.ckpts = "ckpts"

    sampler_conf.agent.optimizer.n_train_steps = n_train_steps
    return sampler_conf, tmpdir


class BraninAcquisition(Acquisition):
    """Acquisition that evaluates AugmentedBranin at full fidelity.

    Expects candidate coordinates in the natural Branin domain
    (x1 ∈ [-5, 10], x2 ∈ [0, 15]).  Appends fidelity s=1.0 for evaluation.
    """

    def __call__(self, candidates: Sequence[Candidate]) -> list[float]:
        if not candidates:
            return []
        rows = [[*c.x, 1.0] for c in candidates]
        x = torch.tensor(rows, dtype=torch.float64)
        return _AUGMENTED_BRANIN(x).tolist()


class FidelityAssigningSelector:
    """Selector wrapper that stamps a fixed fidelity on selected candidates.

    The GFlowNet sampler produces candidates without fidelity; the oracle
    requires one. This wrapper delegates selection then assigns the fidelity.

    Parameters
    ----------
    inner
        The actual selector to delegate candidate ranking to.
    fidelity : int
        Fidelity level to assign to every selected candidate.
    """

    def __init__(self, inner, fidelity: int = 0) -> None:
        self.inner = inner
        self.fidelity = fidelity

    def __call__(self, samples, **kwargs):
        selected = self.inner(samples, **kwargs)
        return [Candidate(x=c.x, fidelity=self.fidelity) for c in selected]


class GFlowNetGridSampler(GFlowNetSampler):
    """GFlowNetSampler that optionally rescales grid coordinates to a target domain.

    The GFlowNet grid environment operates in ``[cell_min, cell_max]^d``.
    If the downstream oracle/acquisition expects a different domain, pass
    ``output_bounds`` to rescale the grid coordinates to that domain.
    """

    def __init__(
        self,
        n_samples: int,
        conf: DictConfig,
        device: str,
        float_precision: int,
        output_bounds: Optional[Sequence[tuple[float, float]]] = None,
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            conf=conf,
            device=device,
            float_precision=float_precision,
        )
        self._grid_min = torch.tensor(conf.env.cell_min, dtype=torch.float64)
        self._grid_max = torch.tensor(conf.env.cell_max, dtype=torch.float64)
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

    def _states_to_candidates(self, states: Any, env: Any) -> list[Candidate]:
        """Convert GFlowNet states to ``Candidate`` objects, rescaling to ``output_bounds`` if set."""
        candidates = super()._states_to_candidates(states, env)
        if self._out_lb is None:
            return candidates
        grid_range = self._grid_max - self._grid_min
        rescaled = []
        for c in candidates:
            coords = torch.tensor(c.x, dtype=torch.float64)
            normed = (coords - self._grid_min) / grid_range
            new_coords = normed * (self._out_ub - self._out_lb) + self._out_lb
            rescaled.append(Candidate(x=tuple(new_coords.tolist())))
        return rescaled


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerSmoke:
    """Basic smoke tests for GFlowNetSampler."""

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self):
        self.conf, self.tmpdir = _compose_sampler_conf(
            n_train_steps=10,
            grid_length=10,
        )
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_instantiation(self):
        """Sampler can be created from composed config."""
        sampler = GFlowNetGridSampler(
            n_samples=3,
            conf=self.conf,
            device="cpu",
            float_precision=32,
            output_bounds=_BRANIN_BOUNDS,
        )
        assert sampler.forward_policy.is_model
        assert sampler.n_samples == 3

    def test_sample_returns_candidates(self):
        """sample() returns Candidate objects with 2-D coordinates in Branin domain."""
        sampler = GFlowNetGridSampler(
            n_samples=5,
            conf=self.conf,
            device="cpu",
            float_precision=32,
            output_bounds=_BRANIN_BOUNDS,
        )
        acq = BraninAcquisition()

        candidates = sampler.sample(acquisition=acq)

        assert len(candidates) == 5
        for c in candidates:
            assert isinstance(c, Candidate)
            assert len(c.x) == 2
            assert -5.0 <= c.x[0] <= 10.0, f"x1={c.x[0]} out of [-5, 10]"
            assert 0.0 <= c.x[1] <= 15.0, f"x2={c.x[1]} out of [0, 15]"

    def test_sample_requires_acquisition(self):
        """sample() raises ValueError when acquisition is None."""
        sampler = GFlowNetGridSampler(
            n_samples=3,
            conf=self.conf,
            device="cpu",
            float_precision=32,
            output_bounds=_BRANIN_BOUNDS,
        )
        with pytest.raises(ValueError, match="requires an acquisition"):
            sampler.sample(acquisition=None)

    def test_sample_mirrors_metrics_to_runtime_logger(self):
        """GFlowNet training metrics should flow through the bound runtime logger."""
        sampler = GFlowNetGridSampler(
            n_samples=5,
            conf=self.conf,
            device="cpu",
            float_precision=32,
            output_bounds=_BRANIN_BOUNDS,
        )
        runtime_logger = Mock()
        sampler.bind_runtime_context(RuntimeContext(logger=runtime_logger))

        candidates = sampler.sample(acquisition=BraninAcquisition())

        assert len(candidates) == 5
        runtime_logger.log_metric.assert_called()
        runtime_logger.log_step.assert_called()
        runtime_logger.end.assert_not_called()


class TestGFlowNetBraninEndToEnd:
    """End-to-end integration with BraninOracle and the active_learning loop."""

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self):
        self.conf, self.tmpdir = _compose_sampler_conf(
            n_train_steps=10,
            grid_length=10,
        )
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_active_learning_loop_with_gflownet(self):
        """Run a mini active-learning loop with GFlowNet sampler and BraninOracle.

        Uses the real BraninOracle (BoTorch AugmentedBranin) and the
        ``active_learning`` function to exercise the full pipeline.
        """
        sampler = GFlowNetGridSampler(
            n_samples=100,
            conf=self.conf,
            device="cpu",
            float_precision=32,
            output_bounds=_BRANIN_BOUNDS,
        )

        dataset = ListDataset()
        surrogate = DummyMeanSurrogate()
        acquisition = BraninAcquisition()
        selector = FidelityAssigningSelector(
            inner=TopKAcquisitionSelector(num_samples=3),
            fidelity=0,
        )

        oracle = BraninOracle(
            fidelity_costs={0: 1.0},
            fidelity_confidences={0: 1.0},
        )

        budget = Budget(available_budget=10.0, schedule=lambda r: 5.0)

        dataset_out, cost, num_rounds = active_learning(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
        )

        assert num_rounds >= 1, "Expected at least 1 AL round"
        assert cost > 0.0
        obs = list(dataset_out.get_observations_iterable())
        assert len(obs) > 0, "Expected observations in the dataset"
        for o in obs:
            assert len(o.x) == 2
            assert o.fidelity == 0
            assert torch.isfinite(torch.tensor(o.y))

    def test_active_learning_loop_ends_runtime_logger_once(self):
        """The outer active-learning loop should own runtime logger shutdown."""
        sampler = GFlowNetGridSampler(
            n_samples=100,
            conf=self.conf,
            device="cpu",
            float_precision=32,
            output_bounds=_BRANIN_BOUNDS,
        )
        runtime_logger = Mock()
        runtime_context = RuntimeContext(logger=runtime_logger)

        dataset = ListDataset()
        surrogate = DummyMeanSurrogate()
        acquisition = BraninAcquisition()
        selector = FidelityAssigningSelector(
            inner=TopKAcquisitionSelector(num_samples=3),
            fidelity=0,
        )
        oracle = BraninOracle(
            fidelity_costs={0: 1.0},
            fidelity_confidences={0: 1.0},
        )
        budget = Budget(available_budget=10.0, schedule=lambda r: 5.0)

        active_learning(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
            runtime_context=runtime_context,
        )

        runtime_logger.log_metric.assert_called()
        runtime_logger.end.assert_called_once()

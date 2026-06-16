"""Shared fixtures for GFlowNet sampler tests."""

import shutil
import tempfile
from typing import Tuple

import pytest
from omegaconf import DictConfig, OmegaConf


def _make_minimal_gflownet_conf(
    n_train_steps: int = 5,
    grid_length: int = 5,
    n_dim: int = 2,
    cell_min: float = 0.0,
    cell_max: float = 1.0,
) -> Tuple[DictConfig, str]:
    """Build a minimal GFlowNet DictConfig suitable for fast unit tests.

    Parameters
    ----------
    n_train_steps : int
        Number of GFlowNet training steps.  Keep small for tests.
    grid_length : int
        Number of cells per grid dimension.
    n_dim : int
        Number of grid dimensions.
    cell_min : float
        Lower bound of the grid coordinate system.
    cell_max : float
        Upper bound of the grid coordinate system.

    Returns
    -------
    conf : DictConfig
        Merged GFlowNet configuration.
    tmpdir : str
        Temporary directory created for GFlowNet logs.
    """
    tmpdir = tempfile.mkdtemp(prefix="gfn_test_")

    conf = OmegaConf.create(
        {
            "env": {
                "_target_": "gflownet.envs.grid.Grid",
                "id": "grid",
                "func": "corners",
                "n_dim": n_dim,
                "length": grid_length,
                "max_increment": 1,
                "max_dim_per_action": 1,
                "cell_min": cell_min,
                "cell_max": cell_max,
                "buffer": {"train": None, "test": None},
            },
            "policy": {
                "_target_": "gflownet.policy.base.Policy",
                "forward": {
                    "type": "mlp",
                    "n_hid": 16,
                    "n_layers": 1,
                    "checkpoint": None,
                    "reload_ckpt": False,
                    "is_model": False,
                },
                "backward": None,
                "shared": None,
            },
            "gflownet": {
                "_target_": "gflownet.gflownet.GFlowNetAgent",
                "seed": 0,
                "optimizer": {
                    "loss": "trajectorybalance",
                    "lr": 1e-3,
                    "lr_decay_period": 1000000,
                    "lr_decay_gamma": 0.5,
                    "z_dim": 4,
                    "lr_z_mult": 100,
                    "method": "adam",
                    "early_stopping": 0.0,
                    "ema_alpha": 0.5,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "sgd_momentum": 0.9,
                    "batch_size": {
                        "forward": 5,
                        "backward_dataset": 0,
                        "backward_replay": 0,
                    },
                    "train_to_sample_ratio": 1,
                    "n_train_steps": n_train_steps,
                    "bootstrap_tau": 0.0,
                    "clip_grad_norm": 0.0,
                },
                "state_flow": None,
                "batch_reward": True,
                "mask_invalid_actions": True,
                "temperature_logits": 1.0,
                "random_action_prob": 0.0,
                "pct_offline": 0.0,
                "replay_capacity": 0,
                "replay_sampling": "permutation",
                "train_sampling": "permutation",
                "num_empirical_loss": 200000,
                "use_context": False,
            },
            "loss": {
                "_target_": "gflownet.losses.trajectorybalance.TrajectoryBalance",
            },
            "buffer": {
                "_target_": "gflownet.buffer.base.BaseBuffer",
                "replay_capacity": 0,
                "train": None,
                "test": None,
            },
            "evaluator": {
                "_target_": "gflownet.evaluator.base.BaseEvaluator",
                "first_it": True,
                "period": 9999,
                "n": 10,
                "kde": {"bandwidth": 0.1, "kernel": "gaussian"},
                "n_top_k": 10,
                "top_k": 5,
                "top_k_period": -1,
                "n_trajs_logprobs": 2,
                "logprobs_batch_size": 10,
                "logprobs_bootstrap_size": 10,
                "max_data_logprobs": 100,
                "n_grid": 100,
                "train_log_period": 1,
                "checkpoints_period": 9999,
                "metrics": "all",
            },
            "logger": {
                "_target_": "gflownet.utils.logger.Logger",
                "do": {"online": False, "times": False},
                "project_name": "test_gflownet",
                "logdir": {
                    "root": tmpdir,
                    "ckpts": "ckpts",
                    "overwrite": True,
                },
                "debug": False,
                "lightweight": False,
                "progressbar": {"skip": True, "n_iters_mean": 100},
                "context": "0",
                "notes": None,
                "tags": ["gflownet"],
            },
            "proxy": {
                "_target_": "activelearning.sampler.gflownet.proxy.AcquisitionProxy",
                "reward_function": "power",
                "reward_min": 1e-8,
                "reward_function_kwargs": {"beta": 1.0},
            },
        }
    )
    return conf, tmpdir


@pytest.fixture()
def gflownet_conf_2d():
    """2-D grid GFlowNet config and log tmpdir for unit tests."""
    conf, tmpdir = _make_minimal_gflownet_conf(n_train_steps=5, grid_length=5, n_dim=2)
    yield conf, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)

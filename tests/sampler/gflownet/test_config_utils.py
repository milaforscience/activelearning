"""Tests for :mod:`activelearning.sampler.gflownet.config_utils`."""

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf

from activelearning.sampler.gflownet.config_utils import (
    _find_config_dir,
    compose_gflownet_conf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {
    "env",
    "policy",
    "gflownet",
    "loss",
    "buffer",
    "evaluator",
    "logger",
    "proxy",
}


# ---------------------------------------------------------------------------
# _find_config_dir
# ---------------------------------------------------------------------------


def test_find_config_dir_returns_existing_directory():
    """_find_config_dir must return an existing directory."""
    d = _find_config_dir()
    assert isinstance(d, Path)
    assert d.is_dir()


def test_find_config_dir_points_to_gflownet_config():
    """The directory returned by _find_config_dir must end with config/gflownet."""
    d = _find_config_dir()
    assert d.name == "gflownet"
    assert d.parent.name == "config"


def test_find_config_dir_raises_when_not_found(tmp_path):
    """_find_config_dir must raise FileNotFoundError when config/gflownet/ cannot be found."""
    fake_file = tmp_path / "fake_module.py"
    fake_file.write_text("")
    # Patch __file__ to point somewhere with no config/gflownet ancestor
    with patch(
        "activelearning.sampler.gflownet.config_utils.__file__",
        str(fake_file),
    ):
        # Import the module so we can access and modify its __file__ attribute directly
        from activelearning.sampler.gflownet import config_utils

        original_file = config_utils.__file__
        try:
            config_utils.__file__ = str(fake_file)  # type: ignore[attr-defined]
            with pytest.raises(FileNotFoundError, match="config/gflownet"):
                config_utils._find_config_dir()
        finally:
            config_utils.__file__ = original_file  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# compose_gflownet_conf — structure
# ---------------------------------------------------------------------------


def test_compose_gflownet_conf_returns_dictconfig():
    """compose_gflownet_conf must return an OmegaConf DictConfig."""
    conf = compose_gflownet_conf()
    assert isinstance(conf, DictConfig)


def test_compose_gflownet_conf_has_all_required_keys():
    """compose_gflownet_conf must produce a config with all keys expected by gflownet_from_config."""
    conf = compose_gflownet_conf()
    assert set(conf.keys()) >= REQUIRED_TOP_LEVEL_KEYS


def test_compose_gflownet_conf_env_target():
    """Default env must be the base GFlowNetEnv — no env class is hardcoded."""
    conf = compose_gflownet_conf()
    assert conf.env._target_ == "gflownet.envs.base.GFlowNetEnv"


def test_compose_gflownet_conf_env_no_grid_specific_keys():
    """Default env must not contain Grid-specific keys (n_dim, length, etc.)."""
    conf = compose_gflownet_conf()
    assert not hasattr(conf.env, "n_dim")
    assert not hasattr(conf.env, "length")
    assert not hasattr(conf.env, "cell_min")


def test_compose_gflownet_conf_env_grid_via_overrides():
    """Users must be able to configure the Grid env via conf_overrides."""
    conf = compose_gflownet_conf(
        conf_overrides={
            "env": {
                "_target_": "gflownet.envs.grid.Grid",
                "n_dim": 2,
                "length": 10,
            }
        }
    )
    assert conf.env._target_ == "gflownet.envs.grid.Grid"
    assert conf.env.n_dim == 2
    assert conf.env.length == 10


def test_compose_gflownet_conf_env_has_base_fields():
    """Env must include fields from env/base.yaml."""
    conf = compose_gflownet_conf()
    assert hasattr(conf.env, "conditional")
    assert hasattr(conf.env, "continuous")
    assert hasattr(conf.env, "skip_mask_check")


def test_compose_gflownet_conf_env_cell_min_default():
    """Env base config must NOT include cell_min — that is a Grid-specific field."""
    conf = compose_gflownet_conf()
    assert not hasattr(conf.env, "cell_min")
    assert not hasattr(conf.env, "cell_max")


def test_compose_gflownet_conf_proxy_target():
    """Default proxy must be the AL AcquisitionProxy."""
    conf = compose_gflownet_conf()
    assert (
        conf.proxy._target_ == "activelearning.sampler.gflownet.proxy.AcquisitionProxy"
    )


def test_compose_gflownet_conf_loss_target():
    """Default loss must be TrajectoryBalance."""
    conf = compose_gflownet_conf()
    assert conf.loss._target_ == "gflownet.losses.trajectorybalance.TrajectoryBalance"


def test_compose_gflownet_conf_loss_has_base_fields():
    """Loss must include fields from loss/base.yaml (merged via defaults: [base])."""
    conf = compose_gflownet_conf()
    # These fields come from loss/base.yaml
    assert hasattr(conf.loss, "early_stopping_th")
    assert hasattr(conf.loss, "ema_alpha")


def test_compose_gflownet_conf_gflownet_target():
    """Default gflownet must be GFlowNetAgent."""
    conf = compose_gflownet_conf()
    assert conf.gflownet._target_ == "gflownet.gflownet.GFlowNetAgent"


def test_compose_gflownet_conf_gflownet_has_tb_optimizer_keys():
    """GFlowNet config must include TrajectoryBalance-specific optimizer keys (lr_z_mult, z_dim)."""
    conf = compose_gflownet_conf()
    # These come from gflownet/trajectorybalance.yaml merged over gflownet/gflownet.yaml
    assert hasattr(conf.gflownet.optimizer, "lr_z_mult")
    assert hasattr(conf.gflownet.optimizer, "z_dim")


def test_compose_gflownet_conf_policy_n_hid_default():
    """Default policy n_hid must be 128, matching the original gflownet repo mlp.yaml."""
    conf = compose_gflownet_conf()
    assert conf.policy.forward.n_hid == 128


def test_compose_gflownet_conf_no_device_or_float_precision():
    """Device and float_precision must NOT be set — they are injected at runtime."""
    conf = compose_gflownet_conf()
    assert "device" not in conf
    assert "float_precision" not in conf


# ---------------------------------------------------------------------------
# compose_gflownet_conf — overrides
# ---------------------------------------------------------------------------


def test_compose_gflownet_conf_overrides_env_n_dim():
    """Overriding env fields via conf_overrides must take precedence over base defaults."""
    conf = compose_gflownet_conf(
        conf_overrides={
            "env": {
                "_target_": "gflownet.envs.grid.Grid",
                "n_dim": 6,
                "length": 3,
            }
        }
    )
    assert conf.env.n_dim == 6


def test_compose_gflownet_conf_overrides_preserve_other_env_keys():
    """An env override must deep-merge, not replace the entire env block."""
    conf = compose_gflownet_conf(
        conf_overrides={
            "env": {
                "_target_": "gflownet.envs.grid.Grid",
                "n_dim": 6,
                "length": 20,
            }
        }
    )
    # Base fields must survive the merge
    assert conf.env._target_ == "gflownet.envs.grid.Grid"
    assert conf.env.n_dim == 6
    assert conf.env.length == 20
    assert hasattr(conf.env, "conditional")


def test_compose_gflownet_conf_overrides_n_train_steps():
    """Overriding a nested optimizer key must apply cleanly without losing sibling keys."""
    conf = compose_gflownet_conf(
        conf_overrides={"gflownet": {"optimizer": {"n_train_steps": 10}}}
    )
    assert conf.gflownet.optimizer.n_train_steps == 10
    # Sibling keys must be preserved
    assert conf.gflownet.optimizer.lr is not None


def test_compose_gflownet_conf_no_overrides_uses_defaults():
    """Calling with no overrides must return the unmodified defaults."""
    conf_default = compose_gflownet_conf()
    conf_none = compose_gflownet_conf(conf_overrides=None)
    # Compare structure (log dirs will differ due to tempdir, exclude them)
    for key in REQUIRED_TOP_LEVEL_KEYS - {"logger"}:
        assert OmegaConf.to_container(conf_default[key]) == OmegaConf.to_container(
            conf_none[key]
        )


# ---------------------------------------------------------------------------
# compose_gflownet_conf — log_dir
# ---------------------------------------------------------------------------


def test_compose_gflownet_conf_log_dir_sets_logger_root(tmp_path):
    """Providing log_dir must set logger.logdir.root to that path."""
    conf = compose_gflownet_conf(log_dir=str(tmp_path))
    assert conf.logger.logdir.root == str(tmp_path)


def test_compose_gflownet_conf_log_dir_sets_ckpts_subdir(tmp_path):
    """Providing log_dir must set logger.logdir.ckpts to <log_dir>/ckpts."""
    conf = compose_gflownet_conf(log_dir=str(tmp_path))
    assert conf.logger.logdir.ckpts == str(tmp_path / "ckpts")


def test_compose_gflownet_conf_no_log_dir_creates_tempdir():
    """When log_dir is None, compose_gflownet_conf must create a temporary directory."""
    conf = compose_gflownet_conf(log_dir=None)
    root = conf.logger.logdir.root
    assert root is not None
    assert Path(root).exists()


def test_compose_gflownet_conf_log_dir_overrides_yaml_default(tmp_path):
    """log_dir must override whatever logdir.root is specified in logger/base.yaml."""
    conf = compose_gflownet_conf(log_dir=str(tmp_path))
    # Should not be the YAML default ("./logs" or similar)
    assert conf.logger.logdir.root == str(tmp_path)


# ---------------------------------------------------------------------------
# compose_gflownet_conf — integration: Pydantic config parsing
# ---------------------------------------------------------------------------


def test_pydantic_gflownet_grid_sampler_config_builds_with_compose(tmp_path):
    """GFlowNetGridSamplerConfig.build() must produce a sampler with a fully composed conf."""
    from activelearning.sampler.config import GFlowNetGridSamplerConfig

    cfg = GFlowNetGridSamplerConfig(
        n_samples=5,
        n_fidelities=1,
        log_dir=str(tmp_path),
        conf={
            "env": {
                "_target_": "gflownet.envs.grid.Grid",
                "n_dim": 2,
                "length": 10,
            }
        },
    )
    sampler = cfg.build()
    assert sampler.conf.env.n_dim == 2
    assert sampler.conf.env.length == 10
    assert (
        sampler.conf.proxy._target_
        == "activelearning.sampler.gflownet.proxy.AcquisitionProxy"
    )
    assert sampler.conf.logger.logdir.root == str(tmp_path)


def test_pydantic_gflownet_grid_sampler_config_no_conf_uses_defaults(tmp_path):
    """GFlowNetGridSamplerConfig requires env._target_ in conf — omitting it must raise."""
    from activelearning.sampler.config import GFlowNetGridSamplerConfig

    cfg = GFlowNetGridSamplerConfig(n_samples=5, log_dir=str(tmp_path))
    with pytest.raises(
        ValueError, match="GFlowNetGridSampler requires a Grid environment"
    ):
        cfg.build()

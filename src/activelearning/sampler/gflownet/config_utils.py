"""Config composition utilities for the GFlowNet sampler.

Provides :func:`compose_gflownet_conf`, which loads the bundled gflownet YAML
defaults from ``config/gflownet/`` and deep-merges any user overrides to
produce a ready-to-use :class:`~omegaconf.DictConfig` for
:class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`.

File layout mirrors the `original gflownet repository
<https://github.com/alexhernandezgarcia/gflownet/tree/main/config>`_:

- ``env/base.yaml`` + ``env/grid.yaml``
- ``gflownet/gflownet.yaml`` + ``gflownet/trajectorybalance.yaml``
- ``policy/mlp.yaml``
- ``loss/base.yaml`` + ``loss/trajectorybalance.yaml``
- ``buffer/base.yaml``
- ``evaluator/base.yaml``
- ``logger/base.yaml``
- ``proxy/base.yaml`` + ``proxy/acquisition.yaml``  (our custom AL proxy)

Hydra ``defaults:`` lists in the YAML files are resolved manually via
:func:`_strip_defaults` + :func:`OmegaConf.merge` so that no Hydra runtime
is required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _find_config_dir() -> Path:
    """Locate the bundled ``config/gflownet/`` directory.

    Walks up from this file's location until it finds the project root
    (identified by the presence of ``config/gflownet/``).

    Returns
    -------
    Path
        Absolute path to ``config/gflownet/``.

    Raises
    ------
    FileNotFoundError
        If the directory cannot be found in any parent of this file.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "gflownet"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate 'config/gflownet/' relative to "
        f"{__file__}. "
        "Ensure the repository is installed from its source root."
    )


def _load_yaml(path: Path) -> DictConfig:
    """Load a YAML file as an OmegaConf DictConfig."""
    return OmegaConf.load(path)  # type: ignore[return-value]


def _strip_defaults(cfg: DictConfig) -> DictConfig:
    """Return *cfg* with the Hydra ``defaults`` list removed.

    The original gflownet YAML files use a ``defaults:`` list that Hydra
    resolves at runtime.  Since we compose configs manually (without a Hydra
    runtime) we merge the base file first and then merge the child file with
    its ``defaults:`` key stripped.
    """
    if "defaults" in cfg:
        cfg = OmegaConf.masked_copy(cfg, [k for k in cfg if k != "defaults"])
    return cfg


def compose_gflownet_conf(
    conf_overrides: dict[str, Any] | None = None,
    log_dir: str | None = None,
) -> DictConfig:
    """Compose a full GFlowNet sampler ``DictConfig`` from bundled YAML defaults.

    Loads base component configs from the ``config/gflownet/`` directory that
    ships with this library, resolves their Hydra ``defaults:`` chains manually,
    assembles them into the structure expected by
    :func:`gflownet.utils.common.gflownet_from_config`, then deep-merges any
    user-supplied overrides on top.

    The resulting config has top-level keys ``env``, ``policy``, ``gflownet``,
    ``loss``, ``buffer``, ``evaluator``, ``logger``, and ``proxy``.
    ``device`` and ``float_precision`` are **not** set here — they are injected
    at runtime by
    :meth:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler._build_agent`.

    Parameters
    ----------
    conf_overrides : dict[str, Any] or None
        Nested dict of gflownet config overrides following the top-level key
        structure above (e.g. ``{"gflownet": {"optimizer": {"n_train_steps": 100}}}``).
        Deep-merged over the YAML defaults when provided; ignored when ``None``.
    log_dir : str or None
        Root directory for gflownet logs (sets ``logger.logdir.root`` and
        ``logger.logdir.ckpts``). A temporary directory is created automatically
        when ``None``.

    Returns
    -------
    DictConfig
        Fully composed OmegaConf config ready to be passed to
        :class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`.
    """
    cfg_dir = _find_config_dir()

    # --- env: base + grid (resolves defaults: [base] in grid.yaml) ---
    env_cfg = OmegaConf.merge(
        _load_yaml(cfg_dir / "env" / "base.yaml"),
        _strip_defaults(_load_yaml(cfg_dir / "env" / "grid.yaml")),
    )

    # --- policy: mlp (no defaults chain) ---
    policy_cfg = _load_yaml(cfg_dir / "policy" / "mlp.yaml")

    # --- gflownet agent: gflownet + trajectorybalance (resolves defaults: [gflownet]) ---
    gflownet_cfg = OmegaConf.merge(
        _load_yaml(cfg_dir / "gflownet" / "gflownet.yaml"),
        _strip_defaults(_load_yaml(cfg_dir / "gflownet" / "trajectorybalance.yaml")),
    )

    # --- loss: base + trajectorybalance (resolves defaults: [base]) ---
    loss_cfg = OmegaConf.merge(
        _load_yaml(cfg_dir / "loss" / "base.yaml"),
        _strip_defaults(_load_yaml(cfg_dir / "loss" / "trajectorybalance.yaml")),
    )

    # --- buffer, evaluator, logger: single base file (no defaults chain) ---
    buffer_cfg = _load_yaml(cfg_dir / "buffer" / "base.yaml")
    evaluator_cfg = _load_yaml(cfg_dir / "evaluator" / "base.yaml")
    logger_cfg = _load_yaml(cfg_dir / "logger" / "base.yaml")

    # --- proxy: base + acquisition (our custom AL proxy) ---
    proxy_cfg = OmegaConf.merge(
        _load_yaml(cfg_dir / "proxy" / "base.yaml"),
        _load_yaml(cfg_dir / "proxy" / "acquisition.yaml"),
    )

    # --- Assemble the top-level config ---
    conf: DictConfig = OmegaConf.create(
        {
            "env": OmegaConf.to_container(env_cfg),
            "policy": OmegaConf.to_container(policy_cfg),
            "gflownet": OmegaConf.to_container(gflownet_cfg),
            "loss": OmegaConf.to_container(loss_cfg),
            "buffer": OmegaConf.to_container(buffer_cfg),
            "evaluator": OmegaConf.to_container(evaluator_cfg),
            "logger": OmegaConf.to_container(logger_cfg),
            "proxy": OmegaConf.to_container(proxy_cfg),
        }
    )

    # --- Apply user-supplied overrides ---
    if conf_overrides is not None:
        conf = OmegaConf.merge(conf, OmegaConf.create(conf_overrides))

    # --- Set log directory ---
    if log_dir is None:
        log_dir = tempfile.mkdtemp(prefix="gfn_logs_")
    OmegaConf.update(conf, "logger.logdir.root", log_dir)
    OmegaConf.update(conf, "logger.logdir.ckpts", str(Path(log_dir) / "ckpts"))

    return conf

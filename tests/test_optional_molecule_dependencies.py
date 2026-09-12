"""Regression tests for optional molecules dependency boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_python_snippet_in_subprocess(source: str) -> subprocess.CompletedProcess[str]:
    """Run an arbitrary Python source string in a fresh subprocess.

    The repository root is prepended to ``PYTHONPATH`` so that ``activelearning``
    is importable. Any import blocking must be done inside ``source`` itself —
    see :func:`_blocked_imports_prelude` for the pattern used in these tests.
    """
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPOSITORY_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(REPOSITORY_ROOT)
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _blocked_imports_prelude() -> str:
    return textwrap.dedent(
        """
        import builtins

        _real_import = builtins.__import__

        def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in {
                "selfies",
                "rdkit",
                "transformers",
            }:
                raise ModuleNotFoundError(f"blocked optional dependency: {name}")
            return _real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _blocked_import
        """
    )


def test_core_config_import_and_branin_parse_work_without_molecule_extras() -> None:
    script = _blocked_imports_prelude() + textwrap.dedent(
        """
        from pathlib import Path

        from activelearning.config import ActiveLearningConfig
        from activelearning.utils.config_loader import load_and_parse

        config = load_and_parse(
            [
                Path("config/branin_benchmark/base.yaml"),
                Path("config/branin_benchmark/mf_gfn.yaml"),
            ],
            ActiveLearningConfig,
        )
        assert config.oracle.type == "BraninOracle"
        """
    )

    result = _run_python_snippet_in_subprocess(script)

    assert result.returncode == 0, result.stderr


def test_building_molecule_component_raises_helpful_error_without_extras() -> None:
    script = _blocked_imports_prelude() + textwrap.dedent(
        """
        from activelearning.surrogate.encoder_config import SelfiesTransformerEncoderConfig

        try:
            SelfiesTransformerEncoderConfig().build()
        except ImportError as error:
            message = str(error)
            assert "optional molecules dependencies" in message
            assert "uv sync --extra molecules" in message
        else:
            raise AssertionError("Expected ImportError when molecules extras are blocked")
        """
    )

    result = _run_python_snippet_in_subprocess(script)

    assert result.returncode == 0, result.stderr


def test_shared_surrogate_imports_stay_lazy_without_molecule_extras() -> None:
    script = _blocked_imports_prelude() + textwrap.dedent(
        """
        from activelearning.surrogate.dkl import ExactDKLSurrogate
        from activelearning.surrogate.sequence.config import HuggingFaceEncoderConfig
        from activelearning.surrogate.sequence import (
            HuggingFaceSequenceEncoder,
            HuggingFaceTokenizer,
            TransformerSequenceEncoder,
        )

        assert ExactDKLSurrogate
        assert HuggingFaceEncoderConfig
        assert HuggingFaceSequenceEncoder
        assert HuggingFaceTokenizer
        assert TransformerSequenceEncoder
        """
    )

    result = _run_python_snippet_in_subprocess(script)

    assert result.returncode == 0, result.stderr


def test_importing_s3gfn_sampler_stays_lazy_without_molecule_extras() -> None:
    script = _blocked_imports_prelude() + textwrap.dedent(
        """
        from activelearning.sampler.s3gfn.sampler import S3GFNSampler
        from activelearning.sampler.s3gfn._optional import require_rdkit

        try:
            require_rdkit()
        except ImportError as error:
            message = str(error)
            assert "molecules" in message
            assert "uv sync --extra molecules" in message
        else:
            raise AssertionError("Expected ImportError when molecule extras are blocked")
        """
    )

    result = _run_python_snippet_in_subprocess(script)

    assert result.returncode == 0, result.stderr

"""Tests for Dock3Oracle (ligbuild/dock64 subprocess calls are mocked)."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from activelearning.applications.molecules import dock3_oracle
from activelearning.applications.molecules.dock3_oracle import (
    Dock3Oracle,
    _build_indock,
    _classify_failure,
    _extract_db2,
    _is_usable_directory,
    _parse_outdock_score,
    _run_dock64,
    _run_ligbuild,
    _run_subprocess,
    _short_workdir_base,
    _warmup_dockenv,
)
from activelearning.logger.logger import Logger
from activelearning.oracle.config import Dock3OracleConfig
from activelearning.runtime import RuntimeContext
from activelearning.utils.types import Candidate

_MODULE = "activelearning.applications.molecules.dock3_oracle"

# A minimal but format-faithful OUTDOCK. The leading whitespace on the header
# and end markers is part of what the parser matches on.
_OUTDOCK_HEADER_LINE = (
    "  mol#           id_num  flexiblecode  matched  nscored  time  hac  setnum"
    "  matnum  rank  charge  elect  gist  vdW  psol  asol  tStrain  mStrain"
    "  rec_d  r_hyd  Total"
)


def _pose_line(total: str, r_hyd: str = "0.00") -> str:
    """Build a 21-field OUTDOCK pose line ending in ``total``."""
    fields = [
        "1",  # 0 mol#
        "lig",  # 1 id_num
        "0",  # 2 flexiblecode
        "100",  # 3 matched
        "50",  # 4 nscored
        "1.5",  # 5 time
        "13",  # 6 hac
        "1",  # 7 setnum
        "1",  # 8 matnum
        "1",  # 9 rank
        "-1.00",  # 10 charge
        "-20.00",  # 11 elect
        "0.00",  # 12 gist
        "-30.00",  # 13 vdW
        "10.00",  # 14 psol
        "-5.00",  # 15 asol
        "2.00",  # 16 tStrain
        "3.00",  # 17 mStrain
        "-1.00",  # 18 rec_d
        r_hyd,  # 19 r_hyd
        total,  # 20 Total
    ]
    return " " + "  ".join(fields)


def _write_outdock(tmp_path: Path, body: str) -> Path:
    """Write an OUTDOCK file containing ``body`` and return its path."""
    outdock = tmp_path / "OUTDOCK"
    outdock.write_text(body)
    return outdock


@pytest.fixture
def dock_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Return a valid (indock_template, dockfiles_dir) pair."""
    dockfiles = tmp_path / "dockfiles_src"
    dockfiles.mkdir()
    (dockfiles / "vdw.vdw").write_bytes(b"\x00grid")
    indock = dockfiles / "INDOCK"
    indock.write_text(
        "DOCK 3.7 parameter\nligand_atom_file          split_database_index\n"
    )
    return indock, dockfiles


class _RecordingLogger(Logger):
    """Minimal logger that records every metric it is handed."""

    def __init__(self) -> None:
        super().__init__(project_name="test")
        self.metrics: dict[str, Any] = {}

    def log_config(self, config: dict[str, Any]) -> None:
        """Ignore configuration."""

    def log_metric(self, key: str, value: Any) -> None:
        """Record one metric."""
        self.metrics[key] = value

    def log_figure(self, key: str, figure: Any) -> None:
        """Ignore figures."""

    def log_step(self, step: int) -> None:
        """Ignore step boundaries."""

    def end(self) -> None:
        """Ignore shutdown."""


@pytest.fixture(scope="session")
def hitrate_kwargs(tmp_path_factory) -> dict[str, Any]:
    """Return the hit-rate constructor arguments, backed by synthetic files.

    Built here rather than read from ``ampc_hitrate_fits/``, which is untracked
    and may move. The conversion itself is covered by ``test_hit_rate.py``.
    """
    directory = tmp_path_factory.mktemp("hitrate")

    table = directory / "full_scores.df"
    scores = [-120.0 + 0.5 * i for i in range(241)]
    pprops = [9.0 - (7.5 / 240) * i for i in range(241)]
    table.write_text(
        "score n cumul_n prop cumul_prop pprop\n"
        + "".join(f"{s:.4f} 1 1 0.0 0.0 {p:.4f}\n" for s, p in zip(scores, pprops))
    )

    params = directory / "fitted_params.json"
    params.write_text(
        json.dumps(
            {
                "ampc": {
                    "rho": -0.75,
                    "exp_mean": -1.5,
                    "exp_std": 1.4,
                    "artifact_freq": 1.2e-06,
                    "artifact_mean": -3.7,
                    "artifact_std": 1.0,
                }
            }
        )
    )

    return {
        "hitrate_params": str(params),
        "score_pprop_table": str(table),
        "pki_threshold": 6.5,
    }


@pytest.fixture
def oracle(dock_paths: tuple[Path, Path], hitrate_kwargs) -> Dock3Oracle:
    """Return a constructed oracle with the environment warmup skipped."""
    indock, dockfiles = dock_paths
    return Dock3Oracle(
        indock_template=indock,
        dockfiles_dir=dockfiles,
        fidelity_costs={0: 32.0},
        **hitrate_kwargs,
        warmup=False,
    )


class TestParseOutdockScore:
    """Tests for the OUTDOCK score parser."""

    def test_parses_total_from_21_field_line(self, tmp_path: Path) -> None:
        outdock = _write_outdock(
            tmp_path, f"preamble\n{_OUTDOCK_HEADER_LINE}\n{_pose_line('-67.85')}\n"
        )
        assert _parse_outdock_score(outdock) == pytest.approx(-67.85)

    def test_returns_minimum_across_poses(self, tmp_path: Path) -> None:
        body = "\n".join(
            [
                _OUTDOCK_HEADER_LINE,
                _pose_line("-40.00"),
                _pose_line("-67.85"),
                _pose_line("-12.30"),
            ]
        )
        assert _parse_outdock_score(_write_outdock(tmp_path, body)) == pytest.approx(
            -67.85
        )

    def test_ignores_lines_before_header(self, tmp_path: Path) -> None:
        """A pose-shaped line before the header must not be scored."""
        body = "\n".join(
            [_pose_line("-999.00"), _OUTDOCK_HEADER_LINE, _pose_line("-67.85")]
        )
        assert _parse_outdock_score(_write_outdock(tmp_path, body)) == pytest.approx(
            -67.85
        )

    def test_stops_at_end_marker(self, tmp_path: Path) -> None:
        body = "\n".join(
            [
                _OUTDOCK_HEADER_LINE,
                _pose_line("-67.85"),
                "  we reached the end of the file",
                _pose_line("-999.00"),
            ]
        )
        assert _parse_outdock_score(_write_outdock(tmp_path, body)) == pytest.approx(
            -67.85
        )

    @pytest.mark.parametrize("marker", [" close the file:", " open the file:"])
    def test_skips_line_after_file_marker(self, tmp_path: Path, marker: str) -> None:
        """The line following an open/close marker is a filename, not a pose."""
        body = "\n".join(
            [
                _OUTDOCK_HEADER_LINE,
                marker,
                _pose_line("-999.00"),  # actually a filename line in real output
                _pose_line("-67.85"),
            ]
        )
        assert _parse_outdock_score(_write_outdock(tmp_path, body)) == pytest.approx(
            -67.85
        )

    def test_recomputes_total_on_fortran_overflow(self, tmp_path: Path) -> None:
        """A 20-field line ending in '*' means Total overflowed its field."""
        # r_hyd and Total merge into one token: '-2.00**********'.
        overflow_line = _pose_line("", r_hyd="-2.00**********").rstrip()
        body = f"{_OUTDOCK_HEADER_LINE}\n{overflow_line}\n"
        # elect + gist + vdW + psol + asol + rec_d + r_hyd
        # -20 + 0 + -30 + 10 + -5 + -1 + -2
        assert _parse_outdock_score(_write_outdock(tmp_path, body)) == pytest.approx(
            -48.00
        )

    def test_overflow_excludes_charge_and_strain(self, tmp_path: Path) -> None:
        """charge, tStrain and mStrain must not enter the recomputed total."""
        overflow_line = _pose_line("", r_hyd="-2.00**********").rstrip()
        body = f"{_OUTDOCK_HEADER_LINE}\n{overflow_line}\n"
        result = _parse_outdock_score(_write_outdock(tmp_path, body))
        # Including charge (-1), tStrain (2) and mStrain (3) would give -44.00.
        assert result == pytest.approx(-48.00)
        assert result != pytest.approx(-44.00)

    def test_returns_none_when_no_pose(self, tmp_path: Path) -> None:
        body = f"preamble\n{_OUTDOCK_HEADER_LINE}\n  we reached the end of the file\n"
        assert _parse_outdock_score(_write_outdock(tmp_path, body)) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert _parse_outdock_score(tmp_path / "does_not_exist") is None

    def test_ignores_unparseable_pose_line(self, tmp_path: Path) -> None:
        body = f"{_OUTDOCK_HEADER_LINE}\n{_pose_line('not_a_number')}\n"
        assert _parse_outdock_score(_write_outdock(tmp_path, body)) is None


class TestClassifyFailure:
    """Tests for the failure label table."""

    @pytest.mark.parametrize(
        ("stage", "message", "expected"),
        [
            ("ligbuild", "Timeout reached for lig.db2", "ligbuild_db2_timeout"),
            ("ligbuild", "Command timed out", "ligbuild_subprocess_timeout"),
            ("ligbuild", "TimeoutExpired", "ligbuild_subprocess_timeout"),
            (
                "ligbuild",
                "Error in build_db2: list index out of range",
                "ligbuild_build_db2_index_error",
            ),
            (
                "ligbuild",
                "all protomer builds failed",
                "ligbuild_protomer_build_failed",
            ),
            ("ligbuild", "ligbuild produced no .tgz under /tmp", "ligbuild_no_tgz"),
            ("ligbuild", "something else", "ligbuild_failed"),
            ("db2_extract", "anything", "db2_extract_failed"),
            ("dock64", "dock64 produced no OUTDOCK.", "dock64_no_outdock"),
            ("dock64", "timed out", "dock64_timeout"),
            ("dock64", "segfault", "dock64_failed"),
            ("outdock_parse", "no pose or score", "dock64_no_pose_or_score"),
            ("mystery", "boom", "mystery_failed"),
        ],
    )
    def test_labels(self, stage: str, message: str, expected: str) -> None:
        assert _classify_failure(stage, message) == expected

    def test_db2_timeout_takes_priority_over_subprocess_timeout(self) -> None:
        """The more specific db2 label wins when both patterns are present."""
        message = "Timeout reached for lig.db2; process timed out"
        assert _classify_failure("ligbuild", message) == "ligbuild_db2_timeout"


class TestBuildIndock:
    """Tests for INDOCK template patching."""

    def test_bumps_version_and_sets_ligand_path(self, tmp_path: Path) -> None:
        template = tmp_path / "INDOCK"
        template.write_text(
            "DOCK 3.7 parameter\nligand_atom_file          split_database_index\n"
        )
        dest = tmp_path / "run"
        dest.mkdir()
        db2 = tmp_path / "bundle_lig_000" / "lig.db2"
        db2.parent.mkdir()
        db2.write_text("db2")

        out = _build_indock(template, db2, dest)

        text = out.read_text()
        assert out == dest / "INDOCK_run"
        assert "DOCK 3.8 parameter" in text
        assert "DOCK 3.7 parameter" not in text
        assert "split_database_index" not in text
        assert str(db2) in text

    def test_uses_relative_path_when_db2_is_under_dest(self, tmp_path: Path) -> None:
        template = tmp_path / "INDOCK"
        template.write_text("DOCK 3.7 parameter\nsplit_database_index\n")
        dest = tmp_path / "run"
        dest.mkdir()
        db2 = dest / "bundle" / "lig.db2"
        db2.parent.mkdir()
        db2.write_text("db2")

        text = _build_indock(template, db2, dest).read_text()

        assert "bundle/lig.db2" in text
        assert str(db2) not in text


class TestRunDock64:
    """Tests for the dock64 invocation."""

    @staticmethod
    def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
        """Build the dock-root layout _run_dock64 expects.

        The dockfiles copy is a sibling of run_dir, not inside it, because
        INDOCK refers to its grids as ``../dockfiles/...``. It is made once per
        oracle by ``Dock3Oracle._get_dock_root``.
        """
        dock_root = tmp_path / "k"
        run_dir = dock_root / "r"
        run_dir.mkdir(parents=True)
        indock = run_dir / "INDOCK_run"
        indock.write_text("DOCK 3.8 parameter\n")
        dockfiles = dock_root / "dockfiles"
        dockfiles.mkdir()
        (dockfiles / "vdw.vdw").write_bytes(b"\x00grid")
        dock64 = dock_root / "dock64"
        dock64.write_bytes(b"#!/bin/true\n")
        return run_dir, indock, dock64

    def test_succeeds_despite_nonzero_return_code(self, tmp_path: Path) -> None:
        """dock64 exits non-zero on success (ieee_inexact); OUTDOCK is the signal."""
        run_dir, indock, dock64 = self._setup(tmp_path)

        def fake_run(*args, **kwargs):
            (run_dir / "OUTDOCK").write_text("done")
            return subprocess.CompletedProcess(args[0], returncode=2)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run):
            _run_dock64(run_dir, indock, dock64_exe=dock64, timeout=300)

        assert (run_dir / "OUTDOCK").exists()

    def test_raises_when_no_outdock(self, tmp_path: Path) -> None:
        run_dir, indock, dock64 = self._setup(tmp_path)
        completed = subprocess.CompletedProcess(
            ["./dock64"], returncode=0, stdout="out", stderr="err"
        )

        with patch(f"{_MODULE}._run_subprocess", return_value=completed):
            with pytest.raises(RuntimeError, match="produced no OUTDOCK"):
                _run_dock64(run_dir, indock, dock64_exe=dock64, timeout=300)

    def test_runs_dock64_in_run_dir(self, tmp_path: Path) -> None:
        run_dir, indock, dock64 = self._setup(tmp_path)

        def fake_run(*args, **kwargs):
            (run_dir / "OUTDOCK").write_text("done")
            return subprocess.CompletedProcess(args[0], returncode=0)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run) as run_mock:
            _run_dock64(run_dir, indock, dock64_exe=dock64, timeout=123)

        assert run_mock.call_args.args[0] == [str(dock64), "INDOCK_run"]
        assert run_mock.call_args.kwargs["cwd"] == run_dir
        assert run_mock.call_args.kwargs["timeout"] == 123

    def test_dock_root_copies_dockfiles_rather_than_symlinking(
        self, dock_paths, hitrate_kwargs, tmp_path: Path, monkeypatch
    ) -> None:
        """Symlinked grid files silently mis-score on Lustre; must be a copy.

        The copy moved from per-molecule to per-oracle, so this now exercises
        ``_get_dock_root``; the invariant it guards is unchanged.
        """
        indock, dockfiles = dock_paths
        dock64 = tmp_path / "dock64_src"
        dock64.write_bytes(b"#!/bin/true\n")
        base = tmp_path / "base"
        base.mkdir()
        monkeypatch.setattr(f"{_MODULE}._short_workdir_base", lambda: base)

        oracle = Dock3Oracle(
            indock_template=indock,
            dockfiles_dir=dockfiles,
            fidelity_costs={0: 32.0},
            **hitrate_kwargs,
            dock64_exe=dock64,
            warmup=False,
        )
        dock_root = oracle._get_dock_root()

        copied = dock_root / "dockfiles"
        assert copied.is_dir()
        assert not copied.is_symlink()
        assert (copied / "vdw.vdw").read_bytes() == b"\x00grid"
        # One copy per oracle, reused by every worker.
        assert oracle._get_dock_root() == dock_root


class TestRunLigbuild:
    """Tests for the ligbuild invocation and bundle discovery."""

    @staticmethod
    def _setup(tmp_path: Path) -> tuple[Path, Path]:
        work_dir = tmp_path / "td"
        work_dir.mkdir()
        smi = work_dir / "lig.smi"
        smi.write_text("CCO lig\n")
        return smi, work_dir / "ligbuild_out"

    @pytest.mark.parametrize(
        "location", ["out_dir", "out_dir_archives", "parent_archives", "deep"]
    )
    def test_finds_tgz_in_each_search_location(
        self, tmp_path: Path, location: str
    ) -> None:
        smi, out_dir = self._setup(tmp_path)
        targets = {
            "out_dir": out_dir / "bundle.tgz",
            "out_dir_archives": out_dir / "db2_archives" / "bundle.tgz",
            "parent_archives": out_dir.parent / "db2_archives" / "bundle.tgz",
            "deep": out_dir.parent / "a" / "b" / "bundle.tgz",
        }
        target = targets[location]

        def fake_run(*args, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"tgz")
            return subprocess.CompletedProcess(args[0], returncode=0)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run):
            result = _run_ligbuild(
                smi,
                out_dir,
                dockenv_sh="/fake/dockenv.sh",
                ligbuild_exe="ligbuild",
                timeout=300,
                ligbuild_timeout=150,
            )

        assert result == target

    def test_succeeds_despite_nonzero_return_code(self, tmp_path: Path) -> None:
        """ligbuild exits rc=1 on a benign rmtree race after producing the tgz."""
        smi, out_dir = self._setup(tmp_path)

        def fake_run(*args, **kwargs):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "bundle.tgz").write_bytes(b"tgz")
            return subprocess.CompletedProcess(args[0], returncode=1)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run):
            result = _run_ligbuild(
                smi,
                out_dir,
                dockenv_sh="/fake/dockenv.sh",
                ligbuild_exe="ligbuild",
                timeout=300,
                ligbuild_timeout=150,
            )

        assert result == out_dir / "bundle.tgz"

    def test_raises_when_no_tgz(self, tmp_path: Path) -> None:
        smi, out_dir = self._setup(tmp_path)
        completed = subprocess.CompletedProcess(
            ["bash"], returncode=1, stdout="out", stderr="err"
        )

        with patch(f"{_MODULE}._run_subprocess", return_value=completed):
            with pytest.raises(RuntimeError, match=r"produced no \.tgz"):
                _run_ligbuild(
                    smi,
                    out_dir,
                    dockenv_sh="/fake/dockenv.sh",
                    ligbuild_exe="ligbuild",
                    timeout=300,
                    ligbuild_timeout=150,
                )

    def test_writes_custom_parms_with_verbose_and_timeout(self, tmp_path: Path) -> None:
        """verbose=1 is what makes build_db2 report errors instead of None."""
        import json

        smi, out_dir = self._setup(tmp_path)

        def fake_run(*args, **kwargs):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "bundle.tgz").write_bytes(b"tgz")
            return subprocess.CompletedProcess(args[0], returncode=0)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run):
            _run_ligbuild(
                smi,
                out_dir,
                dockenv_sh="/fake/dockenv.sh",
                ligbuild_exe="ligbuild",
                timeout=300,
                ligbuild_timeout=150,
            )

        custom_parms = json.loads((out_dir.parent / "custom_parms.json").read_text())
        assert custom_parms == {"verbose": 1, "timeout": 150}


class TestExtractDb2:
    """Tests for ligand bundle extraction."""

    def test_finds_nested_db2(self, tmp_path: Path) -> None:
        import tarfile

        payload = tmp_path / "bundle_lig_000"
        payload.mkdir()
        (payload / "lig.db2").write_text("db2 content")
        tgz = tmp_path / "bundle.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(payload, arcname="bundle_lig_000")

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        result = _extract_db2(tgz, extract_dir)

        assert result.name == "lig.db2"
        assert result.read_text() == "db2 content"

    def test_raises_when_no_db2(self, tmp_path: Path) -> None:
        import tarfile

        payload = tmp_path / "empty"
        payload.mkdir()
        (payload / "readme.txt").write_text("nothing here")
        tgz = tmp_path / "bundle.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(payload, arcname="empty")

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        with pytest.raises(RuntimeError, match="No .db2 file found"):
            _extract_db2(tgz, extract_dir)


class TestShortWorkdirBase:
    """Tests for the AMSOL short-path workdir alias."""

    def test_uses_job_id_without_slurm_tmpdir(self, monkeypatch) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)
        assert _short_workdir_base() == Path("/tmp/d.12345")

    def test_falls_back_to_run_id_then_cli(self, monkeypatch) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)
        monkeypatch.setenv("RUN_ID", "abc")
        assert _short_workdir_base() == Path("/tmp/d.abc")

        monkeypatch.delenv("RUN_ID")
        assert _short_workdir_base() == Path("/tmp/d.cli")

    def test_symlinks_to_slurm_tmpdir(self, tmp_path: Path, monkeypatch) -> None:
        scratch = tmp_path / "localscratch"
        scratch.mkdir()
        monkeypatch.setenv("SLURM_JOB_ID", "test_dock3_symlink")
        monkeypatch.setenv("SLURM_TMPDIR", str(scratch))

        alias = _short_workdir_base()
        try:
            assert alias.resolve() == scratch.resolve()
        finally:
            if alias.is_symlink():
                alias.unlink()

    def test_path_stays_short_enough_for_amsol(self, monkeypatch) -> None:
        """The whole point of the alias is a prefix AMSOL can live with."""
        monkeypatch.setenv("SLURM_JOB_ID", "62723188")
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)
        assert len(str(_short_workdir_base())) <= 25


class TestDock3OracleConstruction:
    """Tests for oracle construction and configuration validation."""

    def test_missing_indock_template_raises(
        self, tmp_path: Path, hitrate_kwargs
    ) -> None:
        dockfiles = tmp_path / "dockfiles"
        dockfiles.mkdir()
        with pytest.raises(FileNotFoundError, match="INDOCK template not found"):
            Dock3Oracle(
                indock_template=tmp_path / "nope" / "INDOCK",
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                **hitrate_kwargs,
                warmup=False,
            )

    def test_missing_dockfiles_dir_raises(self, dock_paths, hitrate_kwargs) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(FileNotFoundError, match="dockfiles_dir not found"):
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles / "not_a_dir",
                fidelity_costs={0: 32.0},
                **hitrate_kwargs,
                warmup=False,
            )

    @pytest.mark.parametrize("costs", [{}, {0: 1.0, 1: 2.0}])
    def test_rejects_non_single_fidelity(
        self, dock_paths, costs, hitrate_kwargs
    ) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValueError, match="exactly one"):
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs=costs,
                **hitrate_kwargs,
                warmup=False,
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"timeout": 0}, "timeout must be positive"),
            ({"ligbuild_timeout": -1}, "ligbuild_timeout must be positive"),
            ({"num_workers": 0}, "num_workers must be None or at least 1"),
        ],
    )
    def test_rejects_invalid_numeric_arguments(
        self, dock_paths, kwargs, match, hitrate_kwargs
    ) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValueError, match=match):
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                **hitrate_kwargs,
                warmup=False,
                **kwargs,
            )

    def test_warmup_is_skipped_when_disabled(self, dock_paths, hitrate_kwargs) -> None:
        indock, dockfiles = dock_paths
        with patch(f"{_MODULE}._warmup_dockenv") as warmup_mock:
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                **hitrate_kwargs,
                warmup=False,
            )
        warmup_mock.assert_not_called()

    def test_warmup_runs_once_during_construction(
        self, dock_paths, hitrate_kwargs
    ) -> None:
        indock, dockfiles = dock_paths
        with patch(f"{_MODULE}._warmup_dockenv") as warmup_mock:
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                **hitrate_kwargs,
                warmup=True,
            )
        warmup_mock.assert_called_once()

    def test_inherited_cost_and_confidence_accessors(self, oracle) -> None:
        assert oracle.get_supported_fidelities() == [0]
        assert oracle.get_fidelity_confidences() == {0: 1.0}
        assert oracle.get_min_query_cost() == 32.0
        assert oracle.get_costs([Candidate(x="CCO", fidelity=0)]) == [32.0]

    def test_num_workers_auto_resolves_from_slurm(
        self, dock_paths, monkeypatch, hitrate_kwargs
    ) -> None:
        indock, dockfiles = dock_paths
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
        built = Dock3Oracle(
            indock_template=indock,
            dockfiles_dir=dockfiles,
            fidelity_costs={0: 32.0},
            **hitrate_kwargs,
            num_workers=None,
            warmup=False,
        )
        assert built._num_workers == 64


class TestDock3OracleConfig:
    """Tests for the pydantic configuration model."""

    def test_build_round_trip(self, dock_paths, hitrate_kwargs) -> None:
        indock, dockfiles = dock_paths
        config = Dock3OracleConfig(
            indock_template=str(indock),
            dockfiles_dir=str(dockfiles),
            fidelity_costs={0: 32.0},
            **hitrate_kwargs,
            num_workers=8,
            timeout=111,
            ligbuild_timeout=77,
            warmup=False,
        )
        built = config.build()

        assert isinstance(built, Dock3Oracle)
        assert built._num_workers == 8
        assert built._timeout == 111
        assert built._ligbuild_timeout == 77
        assert built._pki_threshold == pytest.approx(6.5)
        assert built.get_supported_fidelities() == [0]

    def test_defaults_resolve_to_cluster_paths(
        self, dock_paths, hitrate_kwargs
    ) -> None:
        """A null path in config means 'use the oracle's default'."""
        indock, dockfiles = dock_paths
        built = Dock3OracleConfig(
            indock_template=str(indock),
            dockfiles_dir=str(dockfiles),
            fidelity_costs={0: 32.0},
            **hitrate_kwargs,
            warmup=False,
        ).build()

        assert built._dockenv_sh.endswith("dockenv.sh")
        assert built._dock64_exe.endswith("dock64")

    def test_rejects_extra_fields(self, dock_paths, hitrate_kwargs) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Dock3OracleConfig(
                indock_template=str(indock),
                dockfiles_dir=str(dockfiles),
                fidelity_costs={0: 32.0},
                **hitrate_kwargs,
                exhaustiveness=8,
            )

    def test_rejects_multi_fidelity(self, dock_paths, hitrate_kwargs) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValidationError, match="exactly one"):
            Dock3OracleConfig(
                indock_template=str(indock),
                dockfiles_dir=str(dockfiles),
                fidelity_costs={0: 32.0, 1: 64.0},
                **hitrate_kwargs,
            )

    def test_rejects_non_smiles_representation(
        self, dock_paths, hitrate_kwargs
    ) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValidationError):
            Dock3OracleConfig(
                indock_template=str(indock),
                dockfiles_dir=str(dockfiles),
                fidelity_costs={0: 32.0},
                **hitrate_kwargs,
                mol_repr="selfies",
            )


class TestDock3OracleQuery:
    """Tests for the query path, with docking itself mocked out."""

    def test_observes_a_probability_not_the_raw_score(self, oracle) -> None:
        """y is P(binding); the docking energy stays in metadata."""
        with patch.object(oracle, "_dock3_score", return_value=(-67.85, None)):
            observations = oracle.query([Candidate(x="CCO", fidelity=0)])

        observation = observations[0]
        expected = oracle._hit_rate_model.hit_rate(-67.85, 6.5)
        assert observation.y == pytest.approx(expected)
        assert 0.0 <= observation.y <= 1.0
        assert observation.metadata["dock3_raw_score"] == pytest.approx(-67.85)
        assert observation.x == "CCO"
        assert observation.fidelity == 0

    def test_better_scores_give_higher_probabilities(self, oracle) -> None:
        """Within the normal range, a stronger docking score must rank higher."""
        with patch.object(
            oracle,
            "_dock3_score",
            side_effect=lambda smiles: ({"good": -90.0, "weak": -30.0}[smiles], None),
        ):
            observations = oracle.query(
                [Candidate(x="good", fidelity=0), Candidate(x="weak", fidelity=0)]
            )

        assert observations[0].y > observations[1].y

    def test_preserves_order_with_multiple_workers(
        self, dock_paths, hitrate_kwargs
    ) -> None:
        indock, dockfiles = dock_paths
        built = Dock3Oracle(
            indock_template=indock,
            dockfiles_dir=dockfiles,
            fidelity_costs={0: 32.0},
            **hitrate_kwargs,
            num_workers=4,
            warmup=False,
        )
        scores = {"a": -10.0, "b": -20.0, "c": -30.0, "d": -40.0}
        candidates = [Candidate(x=key, fidelity=0) for key in "abcd"]

        with patch.object(
            built, "_dock3_score", side_effect=lambda smiles: (scores[smiles], None)
        ):
            observations = built.query(candidates)

        assert [obs.x for obs in observations] == ["a", "b", "c", "d"]
        assert [obs.metadata["dock3_raw_score"] for obs in observations] == [
            -10.0,
            -20.0,
            -30.0,
            -40.0,
        ]
        expected = [built._hit_rate_model.hit_rate(scores[k], 6.5) for k in "abcd"]
        assert [obs.y for obs in observations] == pytest.approx(expected)

    def test_failed_molecule_yields_nan_and_reason(self, oracle) -> None:
        with patch.object(
            oracle,
            "_dock3_score",
            side_effect=[(-67.85, None), (float("nan"), "ligbuild_no_tgz")],
        ):
            observations = oracle.query(
                [Candidate(x="CCO", fidelity=0), Candidate(x="bad", fidelity=0)]
            )

        assert observations[0].y == pytest.approx(
            oracle._hit_rate_model.hit_rate(-67.85, 6.5)
        )
        assert math.isnan(observations[1].y)
        assert observations[1].metadata["dock3_failure_reason"] == "ligbuild_no_tgz"

    def test_metadata_carries_raw_score_and_preserves_candidate_metadata(
        self, oracle
    ) -> None:
        candidate = Candidate(x="CCO", fidelity=0, metadata={"source": "pool"})
        with patch.object(oracle, "_dock3_score", return_value=(-67.85, None)):
            observations = oracle.query([candidate])

        metadata = observations[0].metadata
        assert metadata["source"] == "pool"
        assert metadata["dock3_raw_score"] == pytest.approx(-67.85)
        assert metadata["dock3_pprop"] == pytest.approx(
            oracle._hit_rate_model.pprop(-67.85)
        )
        assert metadata["dock3_failure_reason"] is None

    def test_empty_query_returns_empty_list(self, oracle) -> None:
        assert oracle.query([]) == []

    def test_non_string_candidate_raises(self, oracle) -> None:
        with pytest.raises(ValueError, match="Expected candidate.x to be a SMILES"):
            oracle.query([Candidate(x=1.23, fidelity=0)])

    def test_unsupported_fidelity_raises(self, oracle) -> None:
        with pytest.raises(ValueError, match="Unsupported fidelity"):
            oracle.query([Candidate(x="CCO", fidelity=7)])

    def test_validation_happens_before_any_docking(self, oracle) -> None:
        """A bad candidate must not leave earlier molecules half-docked."""
        with patch.object(oracle, "_dock3_score") as score_mock:
            with pytest.raises(ValueError):
                oracle.query(
                    [Candidate(x="CCO", fidelity=0), Candidate(x=None, fidelity=0)]
                )
        score_mock.assert_not_called()

    def test_unexpected_exception_becomes_nan(self, oracle, caplog) -> None:
        with patch.object(
            oracle, "_dock3_score", side_effect=RuntimeError("disk exploded")
        ):
            with caplog.at_level("WARNING"):
                observations = oracle.query([Candidate(x="CCO", fidelity=0)])

        assert math.isnan(observations[0].y)
        assert observations[0].metadata["dock3_failure_reason"] == "oracle_failed"
        assert any("disk exploded" in record.message for record in caplog.records)


class TestRunSubprocess:
    """Tests for the process-group-aware subprocess runner."""

    def test_captures_output_and_return_code(self) -> None:
        result = _run_subprocess(
            ["bash", "-c", "echo out; echo err >&2; exit 3"], timeout=30
        )

        assert result.returncode == 3
        assert result.stdout.strip() == "out"
        assert result.stderr.strip() == "err"

    def test_runs_in_the_requested_directory(self, tmp_path: Path) -> None:
        result = _run_subprocess(["bash", "-c", "pwd"], timeout=30, cwd=tmp_path)

        assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()

    def test_timeout_kills_grandchildren(self, tmp_path: Path) -> None:
        """Killing only the shell leaves ligbuild's children eating the allocation."""
        pid_file = tmp_path / "child.pid"
        script = f'sleep 300 & echo $! > "{pid_file}"; wait'

        with pytest.raises(subprocess.TimeoutExpired):
            _run_subprocess(["bash", "-c", script], timeout=2)

        child_pid = int(pid_file.read_text().strip())
        # SIGKILL is asynchronous and the process lingers as a zombie until it
        # is reparented and reaped, so poll rather than checking once.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except OSError:
                return
            time.sleep(0.05)
        pytest.fail(f"grandchild {child_pid} survived the timeout")


class TestWarmupDockenv:
    """Tests for the docking-environment probe."""

    def test_probes_with_the_same_shell_form_as_ligbuild(self, tmp_path: Path) -> None:
        """A login-shell probe could pass where the non-login real call fails."""
        completed = subprocess.CompletedProcess(
            ["bash"], returncode=0, stdout="", stderr=""
        )
        with patch(f"{_MODULE}._run_subprocess", return_value=completed) as probe_mock:
            _warmup_dockenv(dockenv_sh="/fake/dockenv.sh", ligbuild_exe="ligbuild")
        probe_argv = probe_mock.call_args.args[0]

        smi = tmp_path / "lig.smi"
        smi.write_text("CCO lig\n")
        out_dir = tmp_path / "ligbuild_out"

        def fake_run(*args, **kwargs):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "bundle.tgz").write_bytes(b"tgz")
            return subprocess.CompletedProcess(args[0], returncode=0)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run) as ligbuild_mock:
            _run_ligbuild(
                smi,
                out_dir,
                dockenv_sh="/fake/dockenv.sh",
                ligbuild_exe="ligbuild",
                timeout=300,
                ligbuild_timeout=150,
            )
        ligbuild_argv = ligbuild_mock.call_args.args[0]

        assert probe_argv[:2] == ["bash", "-c"]
        assert probe_argv[:2] == ligbuild_argv[:2]

    def test_probe_failure_is_logged_not_raised(self, caplog) -> None:
        """A healthy pre-built environment must not be blocked by a failed probe."""
        completed = subprocess.CompletedProcess(
            ["bash"], returncode=1, stdout="boom", stderr=""
        )
        with patch(f"{_MODULE}._run_subprocess", return_value=completed):
            with caplog.at_level("WARNING"):
                _warmup_dockenv(dockenv_sh="/fake/dockenv.sh", ligbuild_exe="ligbuild")

        assert any("warmup probe exited" in record.message for record in caplog.records)

    def test_probe_timeout_is_logged_not_raised(self, caplog) -> None:
        with patch(
            f"{_MODULE}._run_subprocess",
            side_effect=subprocess.TimeoutExpired(["bash"], 600),
        ):
            with caplog.at_level("WARNING"):
                _warmup_dockenv(dockenv_sh="/fake/dockenv.sh", ligbuild_exe="ligbuild")

        assert any("timed out" in record.message for record in caplog.records)


class TestIsUsableDirectory:
    """Tests for the workdir usability probe."""

    def test_accepts_a_writable_directory(self, tmp_path: Path) -> None:
        assert _is_usable_directory(tmp_path)

    def test_rejects_a_plain_file(self, tmp_path: Path) -> None:
        target = tmp_path / "file"
        target.write_text("x")
        assert not _is_usable_directory(target)

    def test_rejects_a_dangling_symlink(self, tmp_path: Path) -> None:
        link = tmp_path / "link"
        link.symlink_to(tmp_path / "gone", target_is_directory=True)
        assert not _is_usable_directory(link)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
    def test_rejects_an_unwritable_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "readonly"
        target.mkdir(mode=0o500)
        try:
            assert not _is_usable_directory(target)
        finally:
            target.chmod(0o700)


class TestShortWorkdirBaseFallback:
    """Tests for the fallback taken when the /tmp alias is unusable."""

    @pytest.fixture(autouse=True)
    def _reset_fallback(self, monkeypatch) -> Iterator[None]:
        """Isolate the memoized fallback base and remove it afterwards."""
        monkeypatch.setattr(dock3_oracle, "_fallback_base", None)
        yield
        created = dock3_oracle._fallback_base
        if created is not None:
            shutil.rmtree(created, ignore_errors=True)

    @staticmethod
    def _dangling_alias(tag: str, tmp_path: Path) -> Path:
        """Create a dangling symlink at the alias path for ``tag``."""
        alias = Path(f"/tmp/d.{tag}")
        alias.symlink_to(tmp_path / "does_not_exist", target_is_directory=True)
        return alias

    def test_dangling_alias_falls_back_to_a_short_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A stale alias must not turn every molecule into an unexplained NaN."""
        monkeypatch.setenv("SLURM_JOB_ID", "test_dock3_fallback")
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)
        alias = self._dangling_alias("test_dock3_fallback", tmp_path)
        try:
            base = _short_workdir_base()

            assert base != alias
            assert _is_usable_directory(base)
            # Still short enough for AMSOL's fixed-width path buffer.
            assert len(str(base)) <= 25
        finally:
            if alias.is_symlink():
                alias.unlink()

    def test_fallback_is_created_once_per_process(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """One shared base; a fresh directory per molecule would leak them."""
        monkeypatch.setenv("SLURM_JOB_ID", "test_dock3_fallback_once")
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)
        alias = self._dangling_alias("test_dock3_fallback_once", tmp_path)
        try:
            assert _short_workdir_base() == _short_workdir_base()
        finally:
            if alias.is_symlink():
                alias.unlink()


class TestQueryFailureSummary:
    """Tests for the aggregated docking-outcome metrics."""

    def test_logs_counts_and_reasons(self, oracle) -> None:
        """Failed molecules are dropped downstream, so the counts must be logged."""
        recorder = _RecordingLogger()
        oracle.bind_runtime_context(RuntimeContext(logger=recorder))
        with patch.object(
            oracle,
            "_dock3_score",
            side_effect=[
                (-67.85, None),
                (float("nan"), "ligbuild_no_tgz"),
                (float("nan"), "ligbuild_no_tgz"),
                (float("nan"), "dock64_timeout"),
            ],
        ):
            oracle.query([Candidate(x=key, fidelity=0) for key in "abcd"])

        assert recorder.metrics["dock3/queried"] == 4.0
        assert recorder.metrics["dock3/succeeded"] == 1.0
        assert recorder.metrics["dock3/success_rate"] == pytest.approx(0.25)
        assert recorder.metrics["dock3/failures/ligbuild_no_tgz"] == 2.0
        assert recorder.metrics["dock3/failures/dock64_timeout"] == 1.0

    def test_no_failure_keys_when_everything_succeeds(self, oracle) -> None:
        recorder = _RecordingLogger()
        oracle.bind_runtime_context(RuntimeContext(logger=recorder))
        with patch.object(oracle, "_dock3_score", return_value=(-10.0, None)):
            oracle.query([Candidate(x="CCO", fidelity=0)])

        assert recorder.metrics["dock3/success_rate"] == pytest.approx(1.0)
        assert not [key for key in recorder.metrics if "failures" in key]

    def test_is_a_no_op_without_a_bound_logger(self, oracle) -> None:
        with patch.object(oracle, "_dock3_score", return_value=(-10.0, None)):
            observations = oracle.query([Candidate(x="CCO", fidelity=0)])

        assert observations[0].y == pytest.approx(
            oracle._hit_rate_model.hit_rate(-10.0, 6.5)
        )


class TestMakeWorkdir:
    """Tests for per-call working directory creation."""

    def test_persistent_workdirs_are_named_stably_and_uniquely(
        self, dock_paths, tmp_path: Path, hitrate_kwargs
    ) -> None:
        """hash() is salted per process and would rename the same molecule."""
        indock, dockfiles = dock_paths
        built = Dock3Oracle(
            indock_template=indock,
            dockfiles_dir=dockfiles,
            fidelity_costs={0: 32.0},
            **hitrate_kwargs,
            tmp_dir=tmp_path / "wd",
            warmup=False,
        )

        first, auto_cleanup = built._make_workdir("CCO")
        second, _ = built._make_workdir("CCO")

        digest = hashlib.blake2b(b"CCO", digest_size=4).hexdigest()
        assert auto_cleanup is False
        assert first.name.startswith(f"dock3_{digest}_")
        assert second.name.startswith(f"dock3_{digest}_")
        assert first != second

    def test_ephemeral_workdirs_are_marked_for_cleanup(self, oracle) -> None:
        work_dir, auto_cleanup = oracle._make_workdir("CCO")
        try:
            assert auto_cleanup is True
            assert work_dir.is_dir()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

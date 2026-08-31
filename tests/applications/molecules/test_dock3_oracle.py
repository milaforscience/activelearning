"""Tests for Dock3Oracle (ligbuild/dock64 subprocess calls are mocked)."""

import math
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from activelearning.applications.molecules.dock3_oracle import (
    Dock3Oracle,
    _build_indock,
    _classify_failure,
    _extract_db2,
    _parse_outdock_score,
    _run_dock64,
    _run_ligbuild,
    _short_workdir_base,
)
from activelearning.oracle.config import Dock3OracleConfig
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


@pytest.fixture
def oracle(dock_paths: tuple[Path, Path]) -> Dock3Oracle:
    """Return a constructed oracle with the environment warmup skipped."""
    indock, dockfiles = dock_paths
    return Dock3Oracle(
        indock_template=indock,
        dockfiles_dir=dockfiles,
        fidelity_costs={0: 32.0},
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
    def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        work_dir = tmp_path / "td" / "run"
        work_dir.mkdir(parents=True)
        indock = work_dir / "INDOCK_run"
        indock.write_text("DOCK 3.8 parameter\n")
        dockfiles = tmp_path / "dockfiles_src"
        dockfiles.mkdir()
        (dockfiles / "vdw.vdw").write_bytes(b"\x00grid")
        dock64 = tmp_path / "dock64"
        dock64.write_bytes(b"#!/bin/true\n")
        return work_dir, indock, dockfiles, dock64

    def test_copies_dockfiles_rather_than_symlinking(self, tmp_path: Path) -> None:
        """Symlinked grid files silently mis-score on Lustre; must be a copy."""
        work_dir, indock, dockfiles, dock64 = self._setup(tmp_path)

        def fake_run(*args, **kwargs):
            (work_dir / "OUTDOCK").write_text("done")
            return subprocess.CompletedProcess(args[0], returncode=0)

        with patch(f"{_MODULE}.subprocess.run", side_effect=fake_run):
            _run_dock64(
                work_dir,
                indock,
                dock64_exe=str(dock64),
                dockfiles_dir=dockfiles,
                timeout=300,
            )

        copied = work_dir.parent / "dockfiles"
        assert copied.is_dir()
        assert not copied.is_symlink()
        assert (copied / "vdw.vdw").read_bytes() == b"\x00grid"

    def test_succeeds_despite_nonzero_return_code(self, tmp_path: Path) -> None:
        """dock64 exits non-zero on success (ieee_inexact); OUTDOCK is the signal."""
        work_dir, indock, dockfiles, dock64 = self._setup(tmp_path)

        def fake_run(*args, **kwargs):
            (work_dir / "OUTDOCK").write_text("done")
            return subprocess.CompletedProcess(args[0], returncode=2)

        with patch(f"{_MODULE}.subprocess.run", side_effect=fake_run):
            _run_dock64(
                work_dir,
                indock,
                dock64_exe=str(dock64),
                dockfiles_dir=dockfiles,
                timeout=300,
            )

        assert (work_dir / "OUTDOCK").exists()

    def test_raises_when_no_outdock(self, tmp_path: Path) -> None:
        work_dir, indock, dockfiles, dock64 = self._setup(tmp_path)
        completed = subprocess.CompletedProcess(
            ["./dock64"], returncode=0, stdout="out", stderr="err"
        )

        with patch(f"{_MODULE}.subprocess.run", return_value=completed):
            with pytest.raises(RuntimeError, match="produced no OUTDOCK"):
                _run_dock64(
                    work_dir,
                    indock,
                    dock64_exe=str(dock64),
                    dockfiles_dir=dockfiles,
                    timeout=300,
                )

    def test_runs_dock64_in_work_dir(self, tmp_path: Path) -> None:
        work_dir, indock, dockfiles, dock64 = self._setup(tmp_path)

        def fake_run(*args, **kwargs):
            (work_dir / "OUTDOCK").write_text("done")
            return subprocess.CompletedProcess(args[0], returncode=0)

        with patch(f"{_MODULE}.subprocess.run", side_effect=fake_run) as run_mock:
            _run_dock64(
                work_dir,
                indock,
                dock64_exe=str(dock64),
                dockfiles_dir=dockfiles,
                timeout=123,
            )

        assert run_mock.call_args.args[0] == ["./dock64", "INDOCK_run"]
        assert run_mock.call_args.kwargs["cwd"] == str(work_dir)
        assert run_mock.call_args.kwargs["timeout"] == 123


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

        with patch(f"{_MODULE}.subprocess.run", side_effect=fake_run):
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

        with patch(f"{_MODULE}.subprocess.run", side_effect=fake_run):
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

        with patch(f"{_MODULE}.subprocess.run", return_value=completed):
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

        with patch(f"{_MODULE}.subprocess.run", side_effect=fake_run):
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

    def test_missing_indock_template_raises(self, tmp_path: Path) -> None:
        dockfiles = tmp_path / "dockfiles"
        dockfiles.mkdir()
        with pytest.raises(FileNotFoundError, match="INDOCK template not found"):
            Dock3Oracle(
                indock_template=tmp_path / "nope" / "INDOCK",
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                warmup=False,
            )

    def test_missing_dockfiles_dir_raises(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(FileNotFoundError, match="dockfiles_dir not found"):
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles / "not_a_dir",
                fidelity_costs={0: 32.0},
                warmup=False,
            )

    @pytest.mark.parametrize("costs", [{}, {0: 1.0, 1: 2.0}])
    def test_rejects_non_single_fidelity(self, dock_paths, costs) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValueError, match="exactly one"):
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs=costs,
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
    def test_rejects_invalid_numeric_arguments(self, dock_paths, kwargs, match) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValueError, match=match):
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                warmup=False,
                **kwargs,
            )

    def test_warmup_is_skipped_when_disabled(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        with patch(f"{_MODULE}._warmup_dockenv") as warmup_mock:
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                warmup=False,
            )
        warmup_mock.assert_not_called()

    def test_warmup_runs_once_during_construction(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        with patch(f"{_MODULE}._warmup_dockenv") as warmup_mock:
            Dock3Oracle(
                indock_template=indock,
                dockfiles_dir=dockfiles,
                fidelity_costs={0: 32.0},
                warmup=True,
            )
        warmup_mock.assert_called_once()

    def test_inherited_cost_and_confidence_accessors(self, oracle) -> None:
        assert oracle.get_supported_fidelities() == [0]
        assert oracle.get_fidelity_confidences() == {0: 1.0}
        assert oracle.get_min_query_cost() == 32.0
        assert oracle.get_costs([Candidate(x="CCO", fidelity=0)]) == [32.0]

    def test_num_workers_auto_resolves_from_slurm(
        self, dock_paths, monkeypatch
    ) -> None:
        indock, dockfiles = dock_paths
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
        built = Dock3Oracle(
            indock_template=indock,
            dockfiles_dir=dockfiles,
            fidelity_costs={0: 32.0},
            num_workers=None,
            warmup=False,
        )
        assert built._num_workers == 64


class TestDock3OracleConfig:
    """Tests for the pydantic configuration model."""

    def test_build_round_trip(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        config = Dock3OracleConfig(
            indock_template=str(indock),
            dockfiles_dir=str(dockfiles),
            fidelity_costs={0: 32.0},
            num_workers=8,
            timeout=111,
            ligbuild_timeout=77,
            negate_score=False,
            warmup=False,
        )
        built = config.build()

        assert isinstance(built, Dock3Oracle)
        assert built._num_workers == 8
        assert built._timeout == 111
        assert built._ligbuild_timeout == 77
        assert built._negate_score is False
        assert built.get_supported_fidelities() == [0]

    def test_defaults_resolve_to_cluster_paths(self, dock_paths) -> None:
        """A null path in config means 'use the oracle's default'."""
        indock, dockfiles = dock_paths
        built = Dock3OracleConfig(
            indock_template=str(indock),
            dockfiles_dir=str(dockfiles),
            fidelity_costs={0: 32.0},
            warmup=False,
        ).build()

        assert built._dockenv_sh.endswith("dockenv.sh")
        assert built._dock64_exe.endswith("dock64")

    def test_rejects_extra_fields(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Dock3OracleConfig(
                indock_template=str(indock),
                dockfiles_dir=str(dockfiles),
                fidelity_costs={0: 32.0},
                exhaustiveness=8,
            )

    def test_rejects_multi_fidelity(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValidationError, match="exactly one"):
            Dock3OracleConfig(
                indock_template=str(indock),
                dockfiles_dir=str(dockfiles),
                fidelity_costs={0: 32.0, 1: 64.0},
            )

    def test_rejects_non_smiles_representation(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        with pytest.raises(ValidationError):
            Dock3OracleConfig(
                indock_template=str(indock),
                dockfiles_dir=str(dockfiles),
                fidelity_costs={0: 32.0},
                mol_repr="selfies",
            )


class TestDock3OracleQuery:
    """Tests for the query path, with docking itself mocked out."""

    def test_negates_score_by_default(self, oracle) -> None:
        with patch.object(oracle, "_dock3_score", return_value=(-67.85, None)):
            observations = oracle.query([Candidate(x="CCO", fidelity=0)])

        assert observations[0].y == pytest.approx(67.85)
        assert observations[0].x == "CCO"
        assert observations[0].fidelity == 0

    def test_raw_score_when_negation_disabled(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        built = Dock3Oracle(
            indock_template=indock,
            dockfiles_dir=dockfiles,
            fidelity_costs={0: 32.0},
            negate_score=False,
            warmup=False,
        )
        with patch.object(built, "_dock3_score", return_value=(-67.85, None)):
            observations = built.query([Candidate(x="CCO", fidelity=0)])

        assert observations[0].y == pytest.approx(-67.85)

    def test_preserves_order_with_multiple_workers(self, dock_paths) -> None:
        indock, dockfiles = dock_paths
        built = Dock3Oracle(
            indock_template=indock,
            dockfiles_dir=dockfiles,
            fidelity_costs={0: 32.0},
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
        assert [obs.y for obs in observations] == [10.0, 20.0, 30.0, 40.0]

    def test_failed_molecule_yields_nan_and_reason(self, oracle) -> None:
        with patch.object(
            oracle,
            "_dock3_score",
            side_effect=[(-67.85, None), (float("nan"), "ligbuild_no_tgz")],
        ):
            observations = oracle.query(
                [Candidate(x="CCO", fidelity=0), Candidate(x="bad", fidelity=0)]
            )

        assert observations[0].y == pytest.approx(67.85)
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

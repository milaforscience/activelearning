"""Tests for CxcalcOracle (the cxcalc subprocess is always mocked)."""

import math
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from activelearning.applications.molecules.cxcalc_oracle import (
    CxcalcOracle,
    _build_input_smi,
    _classify_failure,
    _parse_msdistr_sdf,
    _run_cxcalc,
    _warmup_cxcalc,
    anion_percent_to_probability,
)
from activelearning.logger.logger import Logger
from activelearning.oracle.config import CxcalcOracleConfig
from activelearning.runtime import RuntimeContext
from activelearning.utils.types import Candidate

_MODULE = "activelearning.applications.molecules.cxcalc_oracle"


def _record(
    title: str, *, charges: str = "", proportion: str = "50.00", ph: str = "7.4"
) -> str:
    """Build one msdistr SDF record.

    Only the parts the parser reads are format-faithful: the title line, any
    ``M  CHG`` lines, the ``DISTR`` property tag and its value, and the ``$$$$``
    terminator. The connection table in between is deliberately filler, since
    the parser skips it.

    Parameters
    ----------
    title : str
        Molecule id, written as the record's title line.
    charges : str, default=""
        A full ``M  CHG`` line, or an empty string for a neutral microspecies.
    proportion : str, default="50.00"
        The value written under the ``DISTR`` tag.
    ph : str, default="7.4"
        pH rendered into the ``DISTR`` tag name.

    Returns
    -------
    str
        The SDF record, terminated by ``$$$$``.
    """
    lines = [title, "  Marvin  ", "", "  0  0  0  0  0  0            999 V2000"]
    if charges:
        lines.append(charges)
    lines += ["M  END", f"> <DISTR[pH={ph}]>", proportion, "", "$$$$"]
    return "\n".join(lines) + "\n"


class TestAnionPercentToProbability:
    """Tests for the raw-percentage to probability-of-binding step."""

    @pytest.mark.parametrize("percent", [0.01, 0.5, 47.16, 99.98, 100.0])
    def test_any_positive_percent_maps_to_the_nonzero_probability(
        self, percent: float
    ) -> None:
        """Every molecule carrying charge gets the same small probability."""
        assert anion_percent_to_probability(percent) == pytest.approx(0.01)

    def test_zero_percent_maps_to_the_zero_probability(self) -> None:
        """An uncharged molecule is treated as a non-binder."""
        assert anion_percent_to_probability(0.0) == pytest.approx(0.0)

    def test_nan_propagates_rather_than_becoming_a_non_binder(self) -> None:
        """A failed evaluation must stay distinguishable from an uncharged one.

        With the default step, 0.0 is the value roughly three quarters of all
        molecules legitimately take, so collapsing failures onto it would turn
        every environment failure into a confident "does not bind".
        """
        assert math.isnan(anion_percent_to_probability(float("nan")))

    def test_threshold_moves_the_boundary(self) -> None:
        """Raising the threshold reclassifies barely-charged molecules."""
        assert anion_percent_to_probability(0.3, threshold=0.5) == pytest.approx(0.0)
        assert anion_percent_to_probability(0.7, threshold=0.5) == pytest.approx(0.01)

    def test_comparison_with_the_threshold_is_strict(self) -> None:
        """A molecule exactly at the threshold counts as neutral."""
        assert anion_percent_to_probability(0.5, threshold=0.5) == pytest.approx(0.0)

    def test_custom_probabilities_are_honoured(self) -> None:
        """Both output values are configurable."""
        assert anion_percent_to_probability(
            10.0, zero_probability=0.2, nonzero_probability=0.9
        ) == pytest.approx(0.9)
        assert anion_percent_to_probability(
            0.0, zero_probability=0.2, nonzero_probability=0.9
        ) == pytest.approx(0.2)


class TestParseMsdistrSdf:
    """Tests for the msdistr SDF state machine."""

    def _write(self, tmp_path: Path, text: str) -> Path:
        """Write SDF text to a file and return its path."""
        path = tmp_path / "out.sdf"
        path.write_text(text)
        return path

    def test_neutral_only_molecule_scores_zero(self, tmp_path: Path) -> None:
        """A record with no M CHG line is neutral, so nothing is summed."""
        path = self._write(tmp_path, _record("0", proportion="100.00"))
        assert _parse_msdistr_sdf(path) == {"0": 0.0}

    def test_anionic_microspecies_is_summed(self, tmp_path: Path) -> None:
        """A net -1 species contributes its proportion."""
        path = self._write(
            tmp_path, _record("0", charges="M  CHG  1   4  -1", proportion="99.98")
        )
        assert _parse_msdistr_sdf(path)["0"] == pytest.approx(99.98)

    def test_net_minus_two_species_is_counted(self, tmp_path: Path) -> None:
        """Two -1 atoms on one microspecies sum to a net charge of -2."""
        path = self._write(
            tmp_path,
            _record("0", charges="M  CHG  2   4  -1   7  -1", proportion="80.00"),
        )
        assert _parse_msdistr_sdf(path)["0"] == pytest.approx(80.0)

    def test_net_minus_three_species_is_excluded(self, tmp_path: Path) -> None:
        """Charges outside the target set do not contribute."""
        path = self._write(
            tmp_path,
            _record(
                "0", charges="M  CHG  3   4  -1   7  -1   9  -1", proportion="80.00"
            ),
        )
        assert _parse_msdistr_sdf(path) == {"0": 0.0}

    def test_cationic_species_is_excluded(self, tmp_path: Path) -> None:
        """A positively charged microspecies is not anionic."""
        path = self._write(
            tmp_path, _record("0", charges="M  CHG  1   4   1", proportion="60.00")
        )
        assert _parse_msdistr_sdf(path) == {"0": 0.0}

    def test_multiple_microspecies_of_one_molecule_are_summed(
        self, tmp_path: Path
    ) -> None:
        """Consecutive records sharing a title accumulate into one total."""
        text = _record("0", charges="M  CHG  1   4  -1", proportion="30.00") + _record(
            "0", charges="M  CHG  2   4  -1   7  -1", proportion="20.00"
        )
        assert _parse_msdistr_sdf(self._write(tmp_path, text))["0"] == pytest.approx(
            50.0
        )

    def test_charges_reset_between_records(self, tmp_path: Path) -> None:
        """A neutral record following an anionic one must not inherit its charge."""
        text = _record("0", charges="M  CHG  1   4  -1", proportion="40.00") + _record(
            "1", proportion="100.00"
        )
        parsed = _parse_msdistr_sdf(self._write(tmp_path, text))
        assert parsed["0"] == pytest.approx(40.0)
        assert parsed["1"] == pytest.approx(0.0)

    def test_every_molecule_seen_is_present_in_the_result(self, tmp_path: Path) -> None:
        """Uncharged molecules report 0.0 rather than being absent."""
        text = _record("0", proportion="100.00") + _record(
            "1", charges="M  CHG  1   4  -1", proportion="99.00"
        )
        assert sorted(_parse_msdistr_sdf(self._write(tmp_path, text))) == ["0", "1"]

    def test_blank_title_line_is_skipped(self, tmp_path: Path) -> None:
        """The production 10M run emitted a record with an empty id."""
        text = _record("", charges="M  CHG  1   4  -1", proportion="100.00") + _record(
            "7", charges="M  CHG  1   4  -1", proportion="99.97"
        )
        parsed = _parse_msdistr_sdf(self._write(tmp_path, text))
        assert "" not in parsed
        assert parsed["7"] == pytest.approx(99.97)

    def test_unparsable_proportion_is_skipped(self, tmp_path: Path) -> None:
        """One malformed record must not fail a whole chunk."""
        text = _record(
            "0", charges="M  CHG  1   4  -1", proportion="not-a-number"
        ) + _record("1", charges="M  CHG  1   4  -1", proportion="50.00")
        parsed = _parse_msdistr_sdf(self._write(tmp_path, text))
        assert parsed["0"] == pytest.approx(0.0)
        assert parsed["1"] == pytest.approx(50.0)

    def test_non_default_ph_still_matches_the_tag(self, tmp_path: Path) -> None:
        """The DISTR tag is matched by prefix, not by a formatted pH."""
        path = self._write(
            tmp_path,
            _record("0", charges="M  CHG  1   4  -1", proportion="70.00", ph="6.0"),
        )
        assert _parse_msdistr_sdf(path)["0"] == pytest.approx(70.0)

    def test_custom_anionic_charges(self, tmp_path: Path) -> None:
        """Restricting the target set changes which species count."""
        text = _record("0", charges="M  CHG  2   4  -1   7  -1", proportion="80.00")
        path = self._write(tmp_path, text)
        assert _parse_msdistr_sdf(path, anionic_charges=(-1,)) == {"0": 0.0}
        assert _parse_msdistr_sdf(path, anionic_charges=(-2,))["0"] == pytest.approx(
            80.0
        )


class TestBuildInputSmi:
    """Tests for the cxcalc input file writer."""

    def test_writes_smiles_then_id(self, tmp_path: Path) -> None:
        """cxcalc keys its output by the second column, so order matters."""
        path = tmp_path / "input.smi"
        _build_input_smi(path, [("0", "CCO"), ("1", "c1ccccc1")])
        assert path.read_text() == "CCO 0\nc1ccccc1 1\n"

    def test_duplicate_smiles_keep_distinct_ids(self, tmp_path: Path) -> None:
        """A batch may propose the same molecule twice; ids must stay unique."""
        path = tmp_path / "input.smi"
        _build_input_smi(path, [("0", "CCO"), ("1", "CCO")])
        assert path.read_text().splitlines() == ["CCO 0", "CCO 1"]


class TestRunCxcalc:
    """Tests for the cxcalc invocation."""

    def _completed(self, command: Any) -> subprocess.CompletedProcess:
        """Return a successful CompletedProcess for a command."""
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    def test_command_carries_the_msdistr_arguments(self, tmp_path: Path) -> None:
        """The ported command line is msdistr at an explicit pH."""
        input_smi = tmp_path / "input.smi"
        input_smi.write_text("CCO 0\n")
        output_sdf = tmp_path / "out.sdf"

        def fake_run(command, **kwargs):
            output_sdf.write_text(_record("0"))
            return self._completed(command)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run) as run_mock:
            _run_cxcalc(
                input_smi,
                output_sdf,
                cxcalc_exe="/opt/cxcalc",
                env_setup="module load java",
                ph=7.4,
                timeout=60,
            )

        command = run_mock.call_args.args[0]
        assert command[:2] == ["bash", "-c"]
        assert "module load java && " in command[2]
        assert "msdistr" in command[2]
        assert '"-H" "7.4"' in command[2]
        assert "/opt/cxcalc" in command[2]

    def test_env_setup_none_invokes_the_binary_directly(self, tmp_path: Path) -> None:
        """With java already loaded there is no need for a shell."""
        input_smi = tmp_path / "input.smi"
        input_smi.write_text("CCO 0\n")
        output_sdf = tmp_path / "out.sdf"

        def fake_run(command, **kwargs):
            output_sdf.write_text(_record("0"))
            return self._completed(command)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run) as run_mock:
            _run_cxcalc(
                input_smi,
                output_sdf,
                cxcalc_exe="/opt/cxcalc",
                env_setup=None,
                ph=7.4,
                timeout=60,
            )

        command = run_mock.call_args.args[0]
        assert command[0] == "/opt/cxcalc"
        assert command[3] == "msdistr"

    def test_missing_sdf_raises(self, tmp_path: Path) -> None:
        """Presence of a non-empty SDF is the only reliable success signal."""
        input_smi = tmp_path / "input.smi"
        input_smi.write_text("CCO 0\n")

        with patch(
            f"{_MODULE}._run_subprocess",
            return_value=self._completed(["cxcalc"]),
        ):
            with pytest.raises(RuntimeError, match="produced no SDF"):
                _run_cxcalc(
                    input_smi,
                    tmp_path / "out.sdf",
                    cxcalc_exe="/opt/cxcalc",
                    env_setup=None,
                    ph=7.4,
                    timeout=60,
                )

    def test_empty_sdf_raises(self, tmp_path: Path) -> None:
        """An empty SDF is the failure mode the shell scripts test for."""
        input_smi = tmp_path / "input.smi"
        input_smi.write_text("CCO 0\n")
        output_sdf = tmp_path / "out.sdf"
        output_sdf.write_text("")

        with patch(
            f"{_MODULE}._run_subprocess",
            return_value=self._completed(["cxcalc"]),
        ):
            with pytest.raises(RuntimeError, match="produced no SDF"):
                _run_cxcalc(
                    input_smi,
                    output_sdf,
                    cxcalc_exe="/opt/cxcalc",
                    env_setup=None,
                    ph=7.4,
                    timeout=60,
                )

    def test_nonzero_return_code_with_an_sdf_still_succeeds(
        self, tmp_path: Path
    ) -> None:
        """The install4j launcher's return code carries no information."""
        input_smi = tmp_path / "input.smi"
        input_smi.write_text("CCO 0\n")
        output_sdf = tmp_path / "out.sdf"

        def fake_run(command, **kwargs):
            output_sdf.write_text(_record("0"))
            return subprocess.CompletedProcess(command, returncode=1)

        with patch(f"{_MODULE}._run_subprocess", side_effect=fake_run):
            _run_cxcalc(
                input_smi,
                output_sdf,
                cxcalc_exe="/opt/cxcalc",
                env_setup=None,
                ph=7.4,
                timeout=60,
            )


class TestWarmupCxcalc:
    """Tests for the single-threaded environment probe."""

    def test_probe_goes_through_the_real_invocation_path(self) -> None:
        """A probe that differed could pass for an environment the run never sees."""
        with patch(f"{_MODULE}._run_cxcalc") as run_mock:
            _warmup_cxcalc(
                cxcalc_exe="/opt/cxcalc", env_setup="module load java", ph=7.4
            )

        run_mock.assert_called_once()
        assert run_mock.call_args.kwargs["cxcalc_exe"] == "/opt/cxcalc"
        assert run_mock.call_args.kwargs["env_setup"] == "module load java"
        assert run_mock.call_args.kwargs["ph"] == 7.4

    def test_failure_warns_and_does_not_raise(self, caplog: Any) -> None:
        """Warmup is best effort; a healthy node must still proceed."""
        with patch(f"{_MODULE}._run_cxcalc", side_effect=RuntimeError("no java")):
            _warmup_cxcalc(cxcalc_exe="/opt/cxcalc", env_setup=None, ph=7.4)

        assert any(
            "cxcalc warmup failed" in record.message for record in caplog.records
        )


class TestClassifyFailure:
    """Tests for the compact failure labels."""

    @pytest.mark.parametrize(
        ("stage", "message", "expected"),
        [
            ("smiles", "", "cxcalc_bad_smiles"),
            ("lookup", "", "cxcalc_no_result"),
            ("parse", "boom", "cxcalc_parse_failed"),
            ("cxcalc", "Command timed out after 600s", "cxcalc_timeout"),
            ("cxcalc", "TimeoutExpired", "cxcalc_timeout"),
            ("cxcalc", "cxcalc produced no SDF at /tmp/out.sdf", "cxcalc_no_sdf"),
            ("cxcalc", "something else", "cxcalc_failed"),
            ("mystery", "boom", "mystery_failed"),
        ],
    )
    def test_labels(self, stage: str, message: str, expected: str) -> None:
        """Each stage and message maps to its documented label."""
        assert _classify_failure(stage, message) == expected


@pytest.fixture
def oracle() -> CxcalcOracle:
    """Return a constructed oracle with the environment warmup skipped."""
    return CxcalcOracle(fidelity_costs={0: 0.2}, warmup=False)


class TestCxcalcOracleConstruction:
    """Tests for constructor validation and config round-tripping."""

    def test_rejects_multiple_fidelities(self) -> None:
        """cxcalc computes one property at one pH, so there is no ladder."""
        with pytest.raises(ValueError, match="single-fidelity"):
            CxcalcOracle(fidelity_costs={0: 0.2, 1: 1.0}, warmup=False)

    def test_rejects_no_fidelity(self) -> None:
        """A fidelity level must be declared."""
        with pytest.raises(ValueError, match="single-fidelity"):
            CxcalcOracle(fidelity_costs={}, warmup=False)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"timeout": 0}, "timeout must be positive"),
            ({"chunk_size": 0}, "chunk_size must be at least 1"),
            ({"num_workers": 0}, "num_workers must be None or at least 1"),
            ({"ph": float("nan")}, "ph must be a finite number"),
            ({"anionic_charges": ()}, "at least one charge"),
            ({"anionic_charges": (-1.5,)}, "must be integers"),
            ({"zero_probability": 1.5}, r"zero_probability must lie in \[0, 1\]"),
            ({"nonzero_probability": -0.1}, r"nonzero_probability must lie in"),
            ({"anion_percent_threshold": -1.0}, "must be a non-negative number"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs: dict, match: str) -> None:
        """Domain errors raise at construction, not per molecule."""
        with pytest.raises(ValueError, match=match):
            CxcalcOracle(fidelity_costs={0: 0.2}, warmup=False, **kwargs)

    def test_does_not_warm_up_when_disabled(self) -> None:
        """Tests and dry runs must never touch the real binary."""
        with patch(f"{_MODULE}._warmup_cxcalc") as warmup_mock:
            CxcalcOracle(fidelity_costs={0: 0.2}, warmup=False)
        warmup_mock.assert_not_called()

    def test_warms_up_once_at_construction(self) -> None:
        """The probe must run single-threaded, before any parallel call."""
        with patch(f"{_MODULE}._warmup_cxcalc") as warmup_mock:
            CxcalcOracle(fidelity_costs={0: 0.2}, ph=6.0, warmup=True)
        warmup_mock.assert_called_once()
        assert warmup_mock.call_args.kwargs["ph"] == 6.0

    def test_config_round_trip(self) -> None:
        """Config fields land on the matching oracle attributes."""
        config = CxcalcOracleConfig(
            fidelity_costs={0: 0.2},
            fidelity_confidences={0: 0.3},
            cxcalc_exe="/opt/cxcalc",
            env_setup=None,
            ph=6.5,
            anionic_charges=[-1],
            anion_percent_threshold=0.5,
            zero_probability=0.0,
            nonzero_probability=0.02,
            timeout=120,
            chunk_size=64,
            num_workers=4,
            warmup=False,
        )
        built = config.build()

        assert isinstance(built, CxcalcOracle)
        assert built.get_fidelity_confidences() == {0: 0.3}
        assert built._cxcalc_exe == "/opt/cxcalc"
        assert built._env_setup is None
        assert built._ph == pytest.approx(6.5)
        assert built._anionic_charges == (-1,)
        assert built._anion_percent_threshold == pytest.approx(0.5)
        assert built._nonzero_probability == pytest.approx(0.02)
        assert built._timeout == 120
        assert built._chunk_size == 64
        assert built._num_workers == 4

    def test_config_defaults_match_the_oracle(self) -> None:
        """A bare config builds the documented default step."""
        built = CxcalcOracleConfig(fidelity_costs={0: 0.2}, warmup=False).build()
        assert built._ph == pytest.approx(7.4)
        assert built._anionic_charges == (-1, -2)
        assert built._zero_probability == pytest.approx(0.0)
        assert built._nonzero_probability == pytest.approx(0.01)

    def test_config_rejects_multiple_fidelities(self) -> None:
        """The single-fidelity contract is enforced before anything is built."""
        with pytest.raises(ValidationError, match="single-fidelity"):
            CxcalcOracleConfig(fidelity_costs={0: 0.2, 1: 1.0})

    def test_config_rejects_unknown_fields(self) -> None:
        """extra='forbid' catches typos in YAML."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CxcalcOracleConfig(fidelity_costs={0: 0.2}, negate_score=True)

    def test_config_rejects_non_smiles_representation(self) -> None:
        """cxcalc reads SMILES only."""
        with pytest.raises(ValidationError):
            CxcalcOracleConfig(fidelity_costs={0: 0.2}, mol_repr="selfies")


class TestCxcalcOracleQuery:
    """Tests for the batched query path."""

    def test_observes_a_probability_not_the_raw_percentage(
        self, oracle: CxcalcOracle
    ) -> None:
        """y is P(binding); the anionic percentage stays in metadata."""
        with patch.object(oracle, "_run_chunk", return_value=({"0": 99.98}, None)):
            observations = oracle.query([Candidate(x="CCO", fidelity=0)])

        observation = observations[0]
        assert observation.y == pytest.approx(0.01)
        assert observation.metadata["cxcalc_anion_percent"] == pytest.approx(99.98)
        assert observation.metadata["cxcalc_ph"] == pytest.approx(7.4)
        assert observation.metadata["cxcalc_failure_reason"] is None

    def test_uncharged_molecule_observes_the_zero_probability(
        self, oracle: CxcalcOracle
    ) -> None:
        """A real 0.0 is an ordinary result, not a failure."""
        with patch.object(oracle, "_run_chunk", return_value=({"0": 0.0}, None)):
            observations = oracle.query([Candidate(x="CCO", fidelity=0)])

        assert observations[0].y == pytest.approx(0.0)
        assert observations[0].metadata["cxcalc_failure_reason"] is None

    def test_whole_batch_goes_through_a_single_invocation(
        self, oracle: CxcalcOracle
    ) -> None:
        """Batching is the entire cost advantage over per-molecule scoring."""
        percentages = {str(index): 10.0 for index in range(5)}
        with patch.object(
            oracle, "_run_chunk", return_value=(percentages, None)
        ) as chunk_mock:
            oracle.query([Candidate(x="CCO", fidelity=0) for _ in range(5)])

        chunk_mock.assert_called_once()

    def test_molecule_missing_from_the_sdf_yields_nan(
        self, oracle: CxcalcOracle
    ) -> None:
        """cxcalc silently drops roughly 1% of molecules."""
        with patch.object(oracle, "_run_chunk", return_value=({"0": 42.0}, None)):
            observations = oracle.query(
                [Candidate(x="CCO", fidelity=0), Candidate(x="CCC", fidelity=0)]
            )

        assert observations[0].y == pytest.approx(0.01)
        assert math.isnan(observations[1].y)
        assert observations[1].metadata["cxcalc_failure_reason"] == "cxcalc_no_result"

    def test_chunk_failure_marks_every_molecule_in_it(
        self, oracle: CxcalcOracle
    ) -> None:
        """A broken environment fails the whole chunk, not one molecule."""
        with patch.object(oracle, "_run_chunk", return_value=({}, "cxcalc_no_sdf")):
            observations = oracle.query(
                [Candidate(x="CCO", fidelity=0), Candidate(x="CCC", fidelity=0)]
            )

        assert all(math.isnan(obs.y) for obs in observations)
        assert all(
            obs.metadata["cxcalc_failure_reason"] == "cxcalc_no_sdf"
            for obs in observations
        )

    def test_whitespace_smiles_is_held_out_of_the_input_file(self) -> None:
        """An embedded space would shift every id and misalign the chunk."""
        oracle = CxcalcOracle(fidelity_costs={0: 0.2}, warmup=False)
        with patch.object(
            oracle, "_run_chunk", return_value=({"0": 55.0}, None)
        ) as chunk_mock:
            observations = oracle.query(
                [
                    Candidate(x="CCO", fidelity=0),
                    Candidate(x="CC O", fidelity=0),
                ]
            )

        # Only the well-formed molecule is sent.
        assert chunk_mock.call_args.args[0] == [(0, "CCO")]
        assert observations[0].y == pytest.approx(0.01)
        assert math.isnan(observations[1].y)
        assert observations[1].metadata["cxcalc_failure_reason"] == "cxcalc_bad_smiles"

    def test_empty_smiles_is_held_out(self) -> None:
        """An empty molecule cannot be written as a two-column line."""
        oracle = CxcalcOracle(fidelity_costs={0: 0.2}, warmup=False)
        with patch.object(oracle, "_run_chunk") as chunk_mock:
            observations = oracle.query([Candidate(x="", fidelity=0)])

        chunk_mock.assert_not_called()
        assert math.isnan(observations[0].y)
        assert observations[0].metadata["cxcalc_failure_reason"] == "cxcalc_bad_smiles"

    def test_order_is_preserved_across_parallel_chunks(self) -> None:
        """executor.map ordering must survive chunking and threading."""
        oracle = CxcalcOracle(
            fidelity_costs={0: 0.2}, chunk_size=1, num_workers=4, warmup=False
        )

        def fake_chunk(chunk):
            index, _ = chunk[0]
            # Every other molecule is anionic, so the pattern is checkable.
            return ({str(index): 90.0 if index % 2 else 0.0}, None)

        with patch.object(oracle, "_run_chunk", side_effect=fake_chunk):
            observations = oracle.query(
                [Candidate(x=f"C{'C' * i}O", fidelity=0) for i in range(8)]
            )

        assert [obs.y for obs in observations] == pytest.approx([0.0, 0.01] * 4)

    def test_candidate_metadata_is_preserved(self, oracle: CxcalcOracle) -> None:
        """Oracle metadata is merged over the candidate's, not replacing it."""
        with patch.object(oracle, "_run_chunk", return_value=({"0": 5.0}, None)):
            observations = oracle.query(
                [Candidate(x="CCO", fidelity=0, metadata={"source": "sampler"})]
            )

        assert observations[0].metadata["source"] == "sampler"
        assert observations[0].metadata["cxcalc_anion_percent"] == pytest.approx(5.0)

    def test_empty_query_returns_no_observations(self, oracle: CxcalcOracle) -> None:
        """An empty round must not start a subprocess."""
        with patch.object(oracle, "_run_chunk") as chunk_mock:
            assert oracle.query([]) == []
        chunk_mock.assert_not_called()

    def test_non_string_candidate_raises_before_any_work(
        self, oracle: CxcalcOracle
    ) -> None:
        """A bad candidate must not leave earlier molecules half-evaluated."""
        with patch.object(oracle, "_run_chunk") as chunk_mock:
            with pytest.raises(ValueError, match="SMILES string"):
                oracle.query(
                    [Candidate(x="CCO", fidelity=0), Candidate(x=None, fidelity=0)]
                )
        chunk_mock.assert_not_called()

    def test_unsupported_fidelity_raises(self, oracle: CxcalcOracle) -> None:
        """Only the declared fidelity level is accepted."""
        with pytest.raises(ValueError, match="Unsupported fidelity"):
            oracle.query([Candidate(x="CCO", fidelity=3)])

    def test_logs_failure_summary(self, oracle: CxcalcOracle) -> None:
        """Per-molecule reasons never reach the dataset, so aggregate them."""
        recorder = _RecordingLogger()
        oracle.bind_runtime_context(RuntimeContext(logger=recorder))

        with patch.object(oracle, "_run_chunk", return_value=({"0": 12.0}, None)):
            oracle.query([Candidate(x=f"C{'C' * i}O", fidelity=0) for i in range(4)])

        assert recorder.metrics["cxcalc/queried"] == pytest.approx(4.0)
        assert recorder.metrics["cxcalc/succeeded"] == pytest.approx(1.0)
        assert recorder.metrics["cxcalc/success_rate"] == pytest.approx(0.25)
        assert recorder.metrics["cxcalc/failures/cxcalc_no_result"] == pytest.approx(
            3.0
        )

    def test_no_logger_is_not_an_error(self, oracle: CxcalcOracle) -> None:
        """Metrics are optional; an unbound logger must not raise."""
        with patch.object(oracle, "_run_chunk", return_value=({"0": 1.0}, None)):
            assert len(oracle.query([Candidate(x="CCO", fidelity=0)])) == 1


class TestRunChunk:
    """Tests for the per-chunk orchestration."""

    def test_timeout_is_classified(self, oracle: CxcalcOracle) -> None:
        """A killed process tree surfaces as a timeout, not a generic failure."""
        with patch(
            f"{_MODULE}._run_cxcalc",
            side_effect=subprocess.TimeoutExpired(cmd="cxcalc", timeout=600),
        ):
            percentages, reason = oracle._run_chunk([(0, "CCO")])

        assert percentages == {}
        assert reason == "cxcalc_timeout"

    def test_missing_sdf_is_classified(self, oracle: CxcalcOracle) -> None:
        """The no-SDF failure keeps its own label for aggregation."""
        with patch(
            f"{_MODULE}._run_cxcalc",
            side_effect=RuntimeError("cxcalc produced no SDF at /tmp/out.sdf"),
        ):
            percentages, reason = oracle._run_chunk([(0, "CCO")])

        assert percentages == {}
        assert reason == "cxcalc_no_sdf"

    def test_successful_chunk_parses_the_sdf(self, oracle: CxcalcOracle) -> None:
        """The chunk writes its input, runs cxcalc and parses what comes back."""

        def fake_run(input_smi, output_sdf, **kwargs):
            assert input_smi.read_text() == "CCO 0\nCCC 1\n"
            output_sdf.write_text(
                _record("0", charges="M  CHG  1   4  -1", proportion="99.90")
                + _record("1", proportion="100.00")
            )

        with patch(f"{_MODULE}._run_cxcalc", side_effect=fake_run):
            percentages, reason = oracle._run_chunk([(0, "CCO"), (1, "CCC")])

        assert reason is None
        assert percentages["0"] == pytest.approx(99.90)
        assert percentages["1"] == pytest.approx(0.0)

    def test_objective_score_returns_a_probability(self, oracle: CxcalcOracle) -> None:
        """The score_fn contract still holds for single-molecule scoring."""
        with patch.object(oracle, "_run_chunk", return_value=({"0": 80.0}, None)):
            assert oracle._objective_score("CCO") == pytest.approx(0.01)


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

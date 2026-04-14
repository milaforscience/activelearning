"""Tests for XTBIPEAOracle (xtb subprocess calls are mocked)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from activelearning.applications.molecule.xtb_oracle import (
    XTBIPEAOracle,
    hartree_to_ev,
    _decode_to_smiles,
    _parse_vertical_ipea,
    _parse_total_energy,
)
from activelearning.utils.types import Candidate, Observation

BENZENE_SELFIES = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
BENZENE_SMILES = "c1ccccc1"

# Minimal fake xtb output snippets used for parser tests
_FAKE_EA_OUTPUT = "delta SCC EA (eV):    2.3456\n"
_FAKE_IP_OUTPUT = "delta SCC IP (eV):    9.8765\n"
_FAKE_OPT_OUTPUT = (
    "TOTAL ENERGY     -10.123456789012 Eh\n"
    "TOTAL ENERGY     -10.234567890123 Eh\n"  # last value is used
)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_hartree_to_ev(self):
        assert abs(hartree_to_ev(1.0) - 27.2114) < 1e-6

    def test_hartree_to_ev_zero(self):
        assert hartree_to_ev(0.0) == pytest.approx(0.0)

    def test_hartree_to_ev_negative(self):
        """Negative energy differences are common in adiabatic IP/EA computations."""
        assert hartree_to_ev(-1.0) == pytest.approx(-27.2114)

    def test_decode_selfies_to_smiles(self):
        smiles = _decode_to_smiles(BENZENE_SELFIES, mol_repr="selfies")
        assert isinstance(smiles, str) and len(smiles) > 0

    def test_decode_smiles_passthrough(self):
        assert _decode_to_smiles(BENZENE_SMILES, mol_repr="smiles") == BENZENE_SMILES

    def test_decode_bad_repr_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _decode_to_smiles("C", mol_repr="inchi")

    def test_decode_selfies_decodes_to_empty_raises(self):
        """ValueError is raised when the selfies library returns an empty SMILES."""
        with patch("selfies.decoder", return_value=""):
            with pytest.raises(ValueError, match="Failed to decode"):
                _decode_to_smiles("[C]", mol_repr="selfies")


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


class TestParsers:
    def test_parse_vertical_ea(self):
        val = _parse_vertical_ipea(_FAKE_EA_OUTPUT, task="ea")
        assert val == pytest.approx(2.3456)

    def test_parse_vertical_ip(self):
        val = _parse_vertical_ipea(_FAKE_IP_OUTPUT, task="ip")
        assert val == pytest.approx(9.8765)

    def test_parse_vertical_negative_ea(self):
        """EA can be negative for molecules with low electron affinity."""
        output = "delta SCC EA (eV):    -0.5432\n"
        assert _parse_vertical_ipea(output, task="ea") == pytest.approx(-0.5432)

    def test_parse_vertical_bad_task_raises(self):
        with pytest.raises(ValueError, match="Unsupported task"):
            _parse_vertical_ipea(_FAKE_EA_OUTPUT, task="gibbs")

    def test_parse_vertical_missing_raises(self):
        with pytest.raises(RuntimeError, match="Could not parse"):
            _parse_vertical_ipea("no match here", task="ea")

    def test_parse_total_energy_last_value(self):
        energy = _parse_total_energy(_FAKE_OPT_OUTPUT)
        assert energy == pytest.approx(-10.234567890123)

    def test_parse_total_energy_missing_raises(self):
        with pytest.raises(RuntimeError, match="TOTAL ENERGY"):
            _parse_total_energy("no energy here")

    def test_parse_total_energy_single_line(self):
        output = "TOTAL ENERGY     -42.987654321098 Eh\n"
        assert _parse_total_energy(output) == pytest.approx(-42.987654321098)


# ---------------------------------------------------------------------------
# XTBIPEAOracle construction
# ---------------------------------------------------------------------------


class TestXTBIPEAOracleConstruction:
    def test_valid_construction(self):
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0})
        assert oracle is not None

    def test_bad_task_raises(self):
        with pytest.raises(ValueError, match="task"):
            XTBIPEAOracle(task="free_energy", fidelity_costs={1: 1.0})

    def test_default_confidences_normalised(self):
        oracle = XTBIPEAOracle(task="ip", fidelity_costs={1: 1.0, 2: 10.0})
        confidences = oracle.get_fidelity_confidences()
        assert confidences[2] == pytest.approx(1.0)
        assert confidences[1] == pytest.approx(0.1)

    def test_custom_confidences(self):
        oracle = XTBIPEAOracle(
            task="ea", fidelity_costs={1: 1.0}, fidelity_confidences={1: 0.8}
        )
        assert oracle.get_fidelity_confidences()[1] == pytest.approx(0.8)

    def test_get_costs(self):
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0, 2: 5.0})
        candidates = [Candidate(x=BENZENE_SELFIES, fidelity=1)]
        assert oracle.get_costs(candidates) == [1.0]

    def test_single_fidelity_confidence_is_one(self):
        """With a single fidelity level the normalised confidence must be 1.0."""
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 7.5})
        assert oracle.get_fidelity_confidences()[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# XTBIPEAOracle.query (_xtb_score mocked)
# ---------------------------------------------------------------------------


class TestXTBIPEAOracleQuery:
    @pytest.fixture
    def oracle(self):
        return XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0})

    def test_query_returns_observations(self, oracle: XTBIPEAOracle):
        with patch.object(oracle, "_xtb_score", return_value=2.5):
            obs = oracle.query([Candidate(x=BENZENE_SELFIES, fidelity=1)])
        assert len(obs) == 1
        assert isinstance(obs[0], Observation)
        assert obs[0].y == pytest.approx(2.5)

    def test_query_preserves_x(self, oracle: XTBIPEAOracle):
        with patch.object(oracle, "_xtb_score", return_value=1.0):
            obs = oracle.query([Candidate(x=BENZENE_SELFIES, fidelity=1)])
        assert obs[0].x == BENZENE_SELFIES

    def test_query_preserves_fidelity(self, oracle: XTBIPEAOracle):
        with patch.object(oracle, "_xtb_score", return_value=1.0):
            obs = oracle.query([Candidate(x=BENZENE_SELFIES, fidelity=2)])
        assert obs[0].fidelity == 2

    def test_query_from_metadata_raw(self, oracle: XTBIPEAOracle):
        """Pre-embed path: original string in metadata['raw']."""
        import torch

        candidates = [
            Candidate(x=torch.zeros(4), fidelity=1, metadata={"raw": BENZENE_SELFIES})
        ]
        with patch.object(oracle, "_xtb_score", return_value=3.0) as mock_score:
            obs = oracle.query(candidates)
        mock_score.assert_called_once_with(BENZENE_SELFIES, 1)
        assert obs[0].y == pytest.approx(3.0)

    def test_query_missing_string_raises(self, oracle: XTBIPEAOracle):
        import torch

        candidates = [Candidate(x=torch.zeros(4), fidelity=1)]
        with pytest.raises(ValueError, match="molecule string"):
            oracle.query(candidates)

    def test_query_multiple_candidates(self, oracle: XTBIPEAOracle):
        with patch.object(oracle, "_xtb_score", side_effect=[1.0, 2.0]):
            obs = oracle.query(
                [
                    Candidate(x=BENZENE_SELFIES, fidelity=1),
                    Candidate(x="[C][N]", fidelity=1),
                ]
            )
        assert [o.y for o in obs] == pytest.approx([1.0, 2.0])

    def test_query_mixed_fidelities(self, oracle: XTBIPEAOracle):
        """Each candidate is scored at its own fidelity level."""
        with patch.object(oracle, "_xtb_score", side_effect=[1.0, 3.5]):
            obs = oracle.query(
                [
                    Candidate(x=BENZENE_SELFIES, fidelity=1),
                    Candidate(x=BENZENE_SELFIES, fidelity=2),
                ]
            )
        assert obs[0].fidelity == 1
        assert obs[1].fidelity == 2
        assert obs[0].y == pytest.approx(1.0)
        assert obs[1].y == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# _xtb_score fidelity routing (subprocess/filesystem mocked)
# ---------------------------------------------------------------------------


class TestXTBScoreFidelityRouting:
    """Verify that each fidelity calls the correct xtb helpers."""

    @pytest.fixture
    def oracle(self):
        return XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0})

    def _mock_xtb_helpers(self, oracle, vertical_return=2.0, adiabatic_return=1.5):
        """Return context-manager patches for all filesystem/subprocess helpers."""
        fake_xyz = MagicMock(spec=Path)
        fake_log = MagicMock(spec=Path)

        rdkit_patch = patch(
            "activelearning.applications.molecule.xtb_oracle._write_best_rdkit_xyz",
            return_value=fake_xyz,
        )
        opt_patch = patch(
            "activelearning.applications.molecule.xtb_oracle._run_xtb_optimize",
            return_value=(fake_xyz, fake_log),
        )
        vertical_patch = patch.object(
            oracle, "_vertical_score", return_value=vertical_return
        )
        adiabatic_patch = patch.object(
            oracle, "_adiabatic_score", return_value=adiabatic_return
        )
        return rdkit_patch, opt_patch, vertical_patch, adiabatic_patch

    def test_fidelity1_calls_vertical_no_opt(self, oracle):
        rdkit_p, opt_p, vert_p, adiab_p = self._mock_xtb_helpers(oracle)
        with rdkit_p, opt_p as opt_mock, vert_p as vert_mock, adiab_p:
            result = oracle._xtb_score(BENZENE_SELFIES, fidelity=1)
        opt_mock.assert_not_called()
        vert_mock.assert_called_once()
        assert result == pytest.approx(2.0)

    def test_fidelity2_calls_opt_then_vertical(self, oracle):
        rdkit_p, opt_p, vert_p, adiab_p = self._mock_xtb_helpers(oracle)
        with rdkit_p, opt_p as opt_mock, vert_p as vert_mock, adiab_p:
            result = oracle._xtb_score(BENZENE_SELFIES, fidelity=2)
        opt_mock.assert_called_once()
        vert_mock.assert_called_once()
        assert result == pytest.approx(2.0)

    def test_fidelity3_calls_two_opts_then_adiabatic(self, oracle):
        rdkit_p, opt_p, vert_p, adiab_p = self._mock_xtb_helpers(oracle)
        with rdkit_p, opt_p as opt_mock, vert_p, adiab_p as adiab_mock:
            result = oracle._xtb_score(BENZENE_SELFIES, fidelity=3)
        assert opt_mock.call_count == 2
        adiab_mock.assert_called_once()
        assert result == pytest.approx(1.5)

    def test_fidelity3_does_not_call_vertical(self, oracle):
        """Fidelity 3 uses the adiabatic path; vertical scoring must not be called."""
        rdkit_p, opt_p, vert_p, adiab_p = self._mock_xtb_helpers(oracle)
        with rdkit_p, opt_p, vert_p as vert_mock, adiab_p:
            oracle._xtb_score(BENZENE_SELFIES, fidelity=3)
        vert_mock.assert_not_called()

    def test_bad_fidelity_raises(self, oracle):
        with pytest.raises(ValueError, match="fidelity"):
            oracle._xtb_score(BENZENE_SELFIES, fidelity=99)


# ---------------------------------------------------------------------------
# _adiabatic_score arithmetic (sign convention and energy arithmetic)
# ---------------------------------------------------------------------------


class TestAdiabatic:
    """Unit-test _adiabatic_score energy arithmetic for both tasks.

    The sign conventions are:
      IP = E(cation) - E(neutral)  → cation is higher in energy → positive
      EA = E(neutral) - E(anion)   → bound anion is lower in energy → positive
    """

    def _log_mock(self, energy_hartree: float) -> MagicMock:
        """Return a mock Path whose read_text() returns a single TOTAL ENERGY line."""
        log = MagicMock(spec=Path)
        log.read_text.return_value = f"TOTAL ENERGY     {energy_hartree:.12f} Eh\n"
        return log

    def test_adiabatic_ip_sign_convention(self):
        """IP = hartree_to_ev(E_cation - E_neutral) - correction."""
        oracle = XTBIPEAOracle(task="ip", fidelity_costs={3: 25.0})
        neutral = self._log_mock(-10.0)
        ionic = self._log_mock(-9.5)  # cation is higher in energy than neutral
        expected = hartree_to_ev(-9.5 - (-10.0)) - oracle._correction_factor
        assert oracle._adiabatic_score(neutral, ionic) == pytest.approx(expected)

    def test_adiabatic_ea_sign_convention(self):
        """EA = hartree_to_ev(E_neutral - E_anion) - correction."""
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={3: 25.0})
        neutral = self._log_mock(-10.0)
        ionic = self._log_mock(-10.1)  # anion is lower in energy (more stable)
        expected = hartree_to_ev(-10.0 - (-10.1)) - oracle._correction_factor
        assert oracle._adiabatic_score(neutral, ionic) == pytest.approx(expected)

    def test_adiabatic_ip_and_ea_differ_in_sign(self):
        """Swapping task while keeping the same energies must change the sign."""
        neutral_e, ionic_e = -10.0, -9.8
        oracle_ip = XTBIPEAOracle(
            task="ip", fidelity_costs={3: 25.0}, correction_factor=0.0
        )
        oracle_ea = XTBIPEAOracle(
            task="ea", fidelity_costs={3: 25.0}, correction_factor=0.0
        )
        ip = oracle_ip._adiabatic_score(
            self._log_mock(neutral_e), self._log_mock(ionic_e)
        )
        ea = oracle_ea._adiabatic_score(
            self._log_mock(neutral_e), self._log_mock(ionic_e)
        )
        assert ip == pytest.approx(-ea)

    def test_adiabatic_custom_correction_factor(self):
        """A custom correction_factor is correctly subtracted from the result."""
        oracle = XTBIPEAOracle(
            task="ip", fidelity_costs={3: 25.0}, correction_factor=0.0
        )
        neutral = self._log_mock(-10.0)
        ionic = self._log_mock(-9.0)  # exactly 1.0 Hartree difference
        assert oracle._adiabatic_score(neutral, ionic) == pytest.approx(
            hartree_to_ev(1.0)
        )

    def test_adiabatic_uses_last_total_energy_line(self):
        """When multiple TOTAL ENERGY lines appear, only the last is used."""
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={3: 25.0})
        neutral = MagicMock(spec=Path)
        neutral.read_text.return_value = (
            "TOTAL ENERGY     -10.000000000000 Eh\n"
            "TOTAL ENERGY     -10.100000000000 Eh\n"  # last → -10.1
        )
        ionic = self._log_mock(-10.200000000000)
        expected = hartree_to_ev(-10.1 - (-10.2)) - oracle._correction_factor
        assert oracle._adiabatic_score(neutral, ionic) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Ionic charge routing for fidelity 3
# ---------------------------------------------------------------------------


class TestIonicChargeRouting:
    """Verify _xtb_score passes the correct ionic charge to _run_xtb_optimize."""

    def _mock_helpers(self, oracle):
        fake_xyz = MagicMock(spec=Path)
        fake_log = MagicMock(spec=Path)
        rdkit_p = patch(
            "activelearning.applications.molecule.xtb_oracle._write_best_rdkit_xyz",
            return_value=fake_xyz,
        )
        opt_p = patch(
            "activelearning.applications.molecule.xtb_oracle._run_xtb_optimize",
            return_value=(fake_xyz, fake_log),
        )
        adiab_p = patch.object(oracle, "_adiabatic_score", return_value=1.0)
        return rdkit_p, opt_p, adiab_p

    def test_ea_fidelity3_ionic_charge_is_minus1(self):
        """EA at fidelity 3 optimises the anion: charge=-1 must be passed."""
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0})
        rdkit_p, opt_p, adiab_p = self._mock_helpers(oracle)
        with rdkit_p, opt_p as opt_mock, adiab_p:
            oracle._xtb_score(BENZENE_SELFIES, fidelity=3)
        ionic_call_kwargs = opt_mock.call_args_list[1].kwargs
        assert ionic_call_kwargs.get("charge") == -1

    def test_ip_fidelity3_ionic_charge_is_plus1(self):
        """IP at fidelity 3 optimises the cation: charge=+1 must be passed."""
        oracle = XTBIPEAOracle(task="ip", fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0})
        rdkit_p, opt_p, adiab_p = self._mock_helpers(oracle)
        with rdkit_p, opt_p as opt_mock, adiab_p:
            oracle._xtb_score(BENZENE_SELFIES, fidelity=3)
        ionic_call_kwargs = opt_mock.call_args_list[1].kwargs
        assert ionic_call_kwargs.get("charge") == 1

    def test_fidelity1_and_2_pass_no_charge(self):
        """Fidelities 1 and 2 never call _run_xtb_optimize with a charge argument."""
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0})
        fake_xyz = MagicMock(spec=Path)
        fake_log = MagicMock(spec=Path)
        rdkit_p = patch(
            "activelearning.applications.molecule.xtb_oracle._write_best_rdkit_xyz",
            return_value=fake_xyz,
        )
        opt_p = patch(
            "activelearning.applications.molecule.xtb_oracle._run_xtb_optimize",
            return_value=(fake_xyz, fake_log),
        )
        vert_p = patch.object(oracle, "_vertical_score", return_value=1.0)
        with rdkit_p, opt_p as opt_mock, vert_p:
            oracle._xtb_score(BENZENE_SELFIES, fidelity=2)
        assert opt_mock.call_count == 1
        assert opt_mock.call_args_list[0].kwargs.get("charge") is None

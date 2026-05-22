"""Tests for XTBIPEAOracle (xtb subprocess calls are mocked)."""

import math
import subprocess

import numpy as np
import pytest
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

from activelearning.applications.molecules.plotting import (
    build_xtb_query_molecule_figure,
)
from activelearning.applications.molecules.xtb_oracle import (
    ConformerConfig,
    XTBIPEAOracle,
    XTBUnavailableError,
    hartree_to_ev,
    _decode_to_smiles,
    _parse_vertical_ipea,
    _parse_total_energy,
    _run_xtb,
    _write_best_rdkit_xyz,
)
from activelearning.oracle.config import XTBIPEAOracleConfig
from activelearning.runtime import RuntimeContext
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

    def test_decode_selfies_decodes_to_empty_string(self):
        """An empty decode is returned as an empty SMILES string."""
        with patch("selfies.decoder", return_value=""):
            assert _decode_to_smiles("[C]", mol_repr="selfies") == ""

    def test_write_best_rdkit_xyz_mmff_failure_is_clean(self, tmp_path):
        """Unsupported MMFF parameterization should raise a descriptive error."""
        with (
            patch(
                "activelearning.applications.molecules.xtb_oracle.AllChem.MMFFHasAllMoleculeParams",
                return_value=False,
            ),
            patch(
                "activelearning.applications.molecules.xtb_oracle.AllChem.MMFFGetMoleculeProperties",
                return_value=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="MMFF parameters unavailable"):
                _write_best_rdkit_xyz(
                    smiles="C",
                    xyz_path=tmp_path / "methane.xyz",
                    conformer_cfg=ConformerConfig(num_conformers=1),
                    ff="mmff",
                )


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

    def test_parse_vertical_scientific_notation(self):
        output = "delta SCC EA (eV):    2.3456e-01\n"
        assert _parse_vertical_ipea(output, task="ea") == pytest.approx(0.23456)

    def test_parse_vertical_bad_task_raises(self):
        with pytest.raises(ValueError, match="Unsupported task"):
            _parse_vertical_ipea(_FAKE_EA_OUTPUT, task="gibbs")

    def test_parse_vertical_missing_raises(self):
        with pytest.raises(RuntimeError, match="Could not parse"):
            _parse_vertical_ipea("no match here", task="ea")

    def test_parse_total_energy_last_value(self):
        energy = _parse_total_energy(_FAKE_OPT_OUTPUT)
        assert energy == pytest.approx(-10.234567890123)

    def test_parse_total_energy_scientific_notation(self):
        output = (
            "TOTAL ENERGY     -1.012345678901e+01 Eh\n"
            "TOTAL ENERGY     -1.023456789012e+01 Eh\n"
        )
        assert _parse_total_energy(output) == pytest.approx(-10.23456789012)

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
    def test_conformer_config_rejects_non_positive_num_conformers(self):
        with pytest.raises(ValueError, match="num_conformers"):
            ConformerConfig(num_conformers=0)

    def test_valid_construction(self):
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0})
        assert oracle is not None
        assert oracle.log_molecule_visualizations is False
        assert oracle._molecule_visualization_limit == 25


class TestRunXTB:
    def test_successful_return_code_returns_completed_process(self, tmp_path):
        xyz_path = tmp_path / "mol.xyz"
        xyz_path.write_text("3\n\nH 0 0 0\nH 0 0 1\nH 0 1 0\n")
        output_path = tmp_path / "xtb.out"
        completed = subprocess.CompletedProcess(["xtb", str(xyz_path)], returncode=0)

        with patch(
            "activelearning.applications.molecules.xtb_oracle.subprocess.run",
            return_value=completed,
        ) as run_mock:
            result = _run_xtb(
                xyz_path=xyz_path,
                args=["--gfn", "2", "--vea"],
                output_path=output_path,
                cwd=tmp_path,
            )

        assert result is completed
        run_mock.assert_called_once()

    def test_non_zero_return_code_raises_with_output_tail(self, tmp_path):
        xyz_path = tmp_path / "mol.xyz"
        xyz_path.write_text("3\n\nH 0 0 0\nH 0 0 1\nH 0 1 0\n")
        output_path = tmp_path / "xtb.out"

        def fake_run(*args, **kwargs):
            kwargs["stdout"].write("xtb failed to converge\n")
            kwargs["stdout"].flush()
            return subprocess.CompletedProcess(args[0], returncode=2)

        with patch(
            "activelearning.applications.molecules.xtb_oracle.subprocess.run",
            side_effect=fake_run,
        ):
            with pytest.raises(RuntimeError, match="return code 2") as error_info:
                _run_xtb(
                    xyz_path=xyz_path,
                    args=["--gfn", "2", "--vea"],
                    output_path=output_path,
                    cwd=tmp_path,
                )

        assert "xtb failed to converge" in str(error_info.value)
        assert "xtb " in str(error_info.value)

    def test_missing_binary_raises_clear_install_message(self, tmp_path):
        xyz_path = tmp_path / "mol.xyz"
        xyz_path.write_text("3\n\nH 0 0 0\nH 0 0 1\nH 0 1 0\n")
        output_path = tmp_path / "xtb.out"

        with (
            patch(
                "activelearning.applications.molecules.xtb_oracle.subprocess.run",
                side_effect=FileNotFoundError("xtb"),
            ),
            pytest.raises(RuntimeError, match="Install xtb"),
        ):
            _run_xtb(
                xyz_path=xyz_path,
                args=["--gfn", "2", "--vea"],
                output_path=output_path,
                cwd=tmp_path,
            )


class TestXTBIPEAOracleConstructionValidation:
    def test_empty_fidelity_costs_raises(self):
        with pytest.raises(ValueError, match="fidelity_costs"):
            XTBIPEAOracle(task="ea", fidelity_costs={})

    def test_bad_task_raises(self):
        with pytest.raises(ValueError, match="task"):
            XTBIPEAOracle(task="free_energy", fidelity_costs={1: 1.0})

    def test_bad_visualization_limit_raises(self):
        with pytest.raises(ValueError, match="molecule_visualization_limit"):
            XTBIPEAOracle(
                task="ea",
                fidelity_costs={1: 1.0},
                molecule_visualization_limit=0,
            )

    def test_config_build_passes_visualization_options(self):
        config = XTBIPEAOracleConfig(
            task="ip",
            fidelity_costs={1: 1.0},
            log_molecule_visualizations=True,
            molecule_visualization_limit=7,
        )

        oracle = config.build()

        assert isinstance(oracle, XTBIPEAOracle)
        assert oracle.log_molecule_visualizations is True
        assert oracle._molecule_visualization_limit == 7

    def test_config_build_passes_negate_score(self):
        config = XTBIPEAOracleConfig(
            task="ip",
            fidelity_costs={1: 1.0},
            negate_score=True,
        )

        oracle = config.build()

        assert isinstance(oracle, XTBIPEAOracle)
        assert oracle._negate_score is True

    def test_config_build_defaults_ip_to_negated_objective(self):
        config = XTBIPEAOracleConfig(
            task="ip",
            fidelity_costs={1: 1.0},
        )

        oracle = config.build()

        assert isinstance(oracle, XTBIPEAOracle)
        assert oracle._negate_score is True

    def test_config_build_passes_conformer_options(self):
        config = XTBIPEAOracleConfig(
            task="ea",
            fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0},
            num_conformers=3,
            per_fidelity_num_conformers={1: 1, 3: 4},
        )

        oracle = config.build()

        assert isinstance(oracle, XTBIPEAOracle)
        assert oracle._conformer_cfg.num_conformers == 3
        assert oracle._per_fidelity_num_conformers == {1: 1, 3: 4}

    def test_config_defaults_to_global_num_conformers(self):
        config = XTBIPEAOracleConfig(task="ea", fidelity_costs={1: 1.0})

        oracle = config.build()

        assert oracle._conformer_cfg.num_conformers == 2
        assert oracle._per_fidelity_num_conformers == {}

    def test_config_rejects_non_positive_per_fidelity_num_conformers(self):
        with pytest.raises(ValidationError, match="per_fidelity_num_conformers"):
            XTBIPEAOracleConfig(
                task="ea",
                fidelity_costs={1: 1.0, 2: 5.0},
                per_fidelity_num_conformers={1: 0},
            )

    def test_config_rejects_unknown_per_fidelity_num_conformers(self):
        with pytest.raises(ValidationError, match="unsupported fidelities"):
            XTBIPEAOracleConfig(
                task="ea",
                fidelity_costs={1: 1.0, 2: 5.0},
                per_fidelity_num_conformers={3: 4},
            )

    def test_config_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            XTBIPEAOracleConfig(
                task="ea",
                fidelity_costs={1: 1.0},
                unexpected_field=True,
            )

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

    def test_missing_custom_confidence_key_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            XTBIPEAOracle(
                task="ea",
                fidelity_costs={1: 1.0, 2: 5.0},
                fidelity_confidences={1: 0.8},
            )

    def test_extra_custom_confidence_key_raises(self):
        with pytest.raises(ValueError, match="unexpected keys"):
            XTBIPEAOracle(
                task="ea",
                fidelity_costs={1: 1.0},
                fidelity_confidences={1: 0.8, 2: 0.9},
            )

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

    def test_query_negates_score_when_configured(self):
        oracle = XTBIPEAOracle(
            task="ip",
            fidelity_costs={1: 1.0},
            negate_score=True,
        )

        with patch.object(oracle, "_xtb_score", return_value=9.0):
            obs = oracle.query([Candidate(x=BENZENE_SELFIES, fidelity=1)])

        assert obs[0].y == pytest.approx(-9.0)

    def test_query_negates_ip_by_default(self):
        oracle = XTBIPEAOracle(task="ip", fidelity_costs={1: 1.0})

        with patch.object(oracle, "_xtb_score", return_value=9.0):
            obs = oracle.query([Candidate(x=BENZENE_SELFIES, fidelity=1)])

        assert obs[0].y == pytest.approx(-9.0)

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
        with pytest.raises(ValueError, match="molecules string"):
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

    def test_query_does_not_log_visualization_by_default(self, oracle: XTBIPEAOracle):
        logger = MagicMock()
        oracle.bind_runtime_context(RuntimeContext(logger=logger))

        with (
            patch.object(oracle, "_xtb_score", return_value=2.5),
            patch(
                "activelearning.applications.molecules.xtb_oracle."
                "build_xtb_query_molecule_figure"
            ) as build_figure,
        ):
            oracle.query([Candidate(x=BENZENE_SELFIES, fidelity=1)])

        build_figure.assert_not_called()
        logger.log_figure.assert_not_called()

    def test_query_visualization_without_logger_is_noop(self):
        oracle = XTBIPEAOracle(
            task="ea",
            fidelity_costs={1: 1.0},
            log_molecule_visualizations=True,
        )

        with (
            patch.object(oracle, "_xtb_score", return_value=2.5),
            patch(
                "activelearning.applications.molecules.xtb_oracle."
                "build_xtb_query_molecule_figure"
            ) as build_figure,
        ):
            oracle.query([Candidate(x=BENZENE_SELFIES, fidelity=1)])

        build_figure.assert_not_called()

    def test_query_logs_visualization_when_enabled(self):
        oracle = XTBIPEAOracle(
            task="ea",
            fidelity_costs={1: 1.0},
            log_molecule_visualizations=True,
            molecule_visualization_limit=7,
        )
        logger = MagicMock()
        oracle.bind_runtime_context(RuntimeContext(logger=logger))
        candidates = [Candidate(x=BENZENE_SELFIES, fidelity=1)]
        figure = MagicMock()

        with (
            patch.object(oracle, "_xtb_score", return_value=2.5),
            patch(
                "activelearning.applications.molecules.xtb_oracle."
                "build_xtb_query_molecule_figure",
                return_value=figure,
            ) as build_figure,
            patch(
                "activelearning.applications.molecules.xtb_oracle.plt.close"
            ) as close,
        ):
            observations = oracle.query(candidates)

        build_figure.assert_called_once_with(
            candidates=candidates,
            observations=observations,
            task="ea",
            mol_repr="selfies",
            limit=7,
        )
        logger.log_figure.assert_called_once_with("xtb_ea_query_molecules", figure)
        close.assert_called_once_with(figure)


class TestXTBMoleculeVisualization:
    class _FakeGridImage:
        """Small array-like image test double returned by RDKit drawing."""

        size = (260, 220)

        def __array__(self, dtype=None):
            image = np.zeros((220, 260, 3), dtype=np.uint8)
            if dtype is not None:
                return image.astype(dtype)
            return image

    def test_query_grid_is_capped_and_score_ranked(self):
        candidates = [
            Candidate(x="C", fidelity=1),
            Candidate(x="CC", fidelity=1),
            Candidate(x="CCC", fidelity=2),
        ]
        observations = [
            Observation(x="C", y=1.0, fidelity=1),
            Observation(x="CC", y=3.0, fidelity=1),
            Observation(x="CCC", y=2.0, fidelity=2),
        ]
        captured: dict[str, object] = {}

        def fake_grid(*args, **kwargs):
            captured["molecules"] = args[0]
            captured["legends"] = kwargs["legends"]
            captured["mols_per_row"] = kwargs["molsPerRow"]
            return self._FakeGridImage()

        with patch(
            "activelearning.applications.molecules.plotting.Draw.MolsToGridImage",
            side_effect=fake_grid,
        ):
            figure = build_xtb_query_molecule_figure(
                candidates,
                observations,
                task="ip",
                mol_repr="smiles",
                limit=2,
            )

        try:
            legends = captured["legends"]
            assert len(captured["molecules"]) == 2
            assert captured["mols_per_row"] == 5
            assert legends[0].startswith("#2 IP@fid=1: 3.000 eV")
            assert legends[1].startswith("#3 IP@fid=2: 2.000 eV")
            assert figure.axes[0].get_title() == "xTB IP queried molecules (top 2 of 3)"
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_query_grid_is_score_ranked_without_capping(self):
        candidates = [
            Candidate(x="C", fidelity=1),
            Candidate(x="CC", fidelity=1),
            Candidate(x="CCC", fidelity=2),
        ]
        observations = [
            Observation(x="C", y=1.0, fidelity=1),
            Observation(x="CC", y=3.0, fidelity=1),
            Observation(x="CCC", y=2.0, fidelity=2),
        ]
        captured: dict[str, object] = {}

        def fake_grid(*args, **kwargs):
            captured["molecules"] = args[0]
            captured["legends"] = kwargs["legends"]
            return self._FakeGridImage()

        with patch(
            "activelearning.applications.molecules.plotting.Draw.MolsToGridImage",
            side_effect=fake_grid,
        ):
            figure = build_xtb_query_molecule_figure(
                candidates,
                observations,
                task="ea",
                mol_repr="smiles",
            )

        try:
            legends = captured["legends"]
            assert len(captured["molecules"]) == 3
            assert legends[0].startswith("#2 EA@fid=1: 3.000 eV")
            assert legends[1].startswith("#3 EA@fid=2: 2.000 eV")
            assert legends[2].startswith("#1 EA@fid=1: 1.000 eV")
            assert figure.axes[0].get_title() == "xTB EA queried molecules"
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_query_grid_labels_invalid_molecules(self):
        candidates = [Candidate(x="[Ring1]", fidelity=1)]
        observations = [Observation(x="[Ring1]", y=math.nan, fidelity=1)]
        captured: dict[str, object] = {}

        def fake_grid(*args, **kwargs):
            captured["legends"] = kwargs["legends"]
            return self._FakeGridImage()

        with (
            patch(
                "activelearning.applications.molecules.plotting.sf.decoder",
                return_value="",
            ),
            patch(
                "activelearning.applications.molecules.plotting.Draw.MolsToGridImage",
                side_effect=fake_grid,
            ),
        ):
            figure = build_xtb_query_molecule_figure(
                candidates,
                observations,
                task="ea",
                mol_repr="selfies",
            )

        try:
            legend = captured["legends"][0]
            assert "EA@fid=1: nan" in legend
            assert "invalid: empty molecule" in legend
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


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
            "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz",
            return_value=fake_xyz,
        )
        opt_patch = patch(
            "activelearning.applications.molecules.xtb_oracle._run_xtb_optimize",
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

    def test_per_fidelity_conformer_override_is_used(self):
        oracle = XTBIPEAOracle(
            task="ea",
            fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0},
            conformer_cfg=ConformerConfig(num_conformers=3),
            per_fidelity_num_conformers={1: 1, 3: 4},
        )
        rdkit_p, opt_p, vert_p, adiab_p = self._mock_xtb_helpers(oracle)
        with rdkit_p as rdkit_mock, opt_p, vert_p, adiab_p:
            oracle._xtb_score(BENZENE_SELFIES, fidelity=3)

        conformer_cfg = rdkit_mock.call_args.kwargs["conformer_cfg"]
        assert conformer_cfg.num_conformers == 4

    def test_unlisted_fidelity_uses_global_num_conformers(self):
        oracle = XTBIPEAOracle(
            task="ea",
            fidelity_costs={1: 1.0, 2: 5.0, 3: 25.0},
            conformer_cfg=ConformerConfig(num_conformers=3),
            per_fidelity_num_conformers={1: 1, 3: 4},
        )
        rdkit_p, opt_p, vert_p, adiab_p = self._mock_xtb_helpers(oracle)
        with rdkit_p as rdkit_mock, opt_p, vert_p, adiab_p:
            oracle._xtb_score(BENZENE_SELFIES, fidelity=2)

        conformer_cfg = rdkit_mock.call_args.kwargs["conformer_cfg"]
        assert conformer_cfg.num_conformers == 3

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

    def test_empty_decoded_molecule_returns_nan_without_xtb(self, oracle):
        with (
            patch(
                "activelearning.applications.molecules.xtb_oracle.sf.decoder",
                return_value="",
            ),
            patch(
                "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz"
            ) as rdkit_mock,
            patch.object(oracle, "_vertical_score") as vertical_mock,
            patch(
                "activelearning.applications.molecules.xtb_oracle._run_xtb_optimize"
            ) as optimize_mock,
        ):
            result = oracle._xtb_score("[Ring1]", fidelity=1)

        rdkit_mock.assert_not_called()
        vertical_mock.assert_not_called()
        optimize_mock.assert_not_called()
        assert math.isnan(result)

    def test_rdkit_geometry_failure_returns_nan(self, oracle):
        with patch(
            "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz",
            side_effect=AttributeError("mmff failure"),
        ):
            result = oracle._xtb_score(BENZENE_SELFIES, fidelity=1)

        assert math.isnan(result)

    def test_rdkit_geometry_failure_logs_compact_warning(self, oracle, caplog):
        with (
            caplog.at_level("WARNING"),
            patch(
                "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz",
                side_effect=RuntimeError("MMFF parameters unavailable for 'B=O'"),
            ),
        ):
            result = oracle._xtb_score("[B][=O]", fidelity=1)

        assert math.isnan(result)
        warning_record = next(
            record
            for record in caplog.records
            if "Returning NaN for molecule" in record.message
        )
        assert "MMFF parameters unavailable" in warning_record.message
        assert warning_record.exc_info is None

    def test_xtb_subprocess_failure_returns_nan(self, tmp_path):
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0}, mol_repr="smiles")
        fake_xyz = tmp_path / "neutral.xyz"

        with (
            patch(
                "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz",
                return_value=fake_xyz,
            ),
            patch(
                "activelearning.applications.molecules.xtb_oracle._run_xtb",
                side_effect=RuntimeError("xTB command failed with return code 1"),
            ),
        ):
            result = oracle._xtb_score(BENZENE_SMILES, fidelity=1)

        assert math.isnan(result)

    def test_missing_xtb_binary_raises_instead_of_returning_nan(self, tmp_path):
        oracle = XTBIPEAOracle(task="ea", fidelity_costs={1: 1.0}, mol_repr="smiles")
        fake_xyz = tmp_path / "neutral.xyz"

        with (
            patch(
                "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz",
                return_value=fake_xyz,
            ),
            patch(
                "activelearning.applications.molecules.xtb_oracle._run_xtb",
                side_effect=XTBUnavailableError("xtb executable not found"),
            ),
            pytest.raises(XTBUnavailableError, match="xtb executable not found"),
        ):
            oracle._xtb_score(BENZENE_SMILES, fidelity=1)


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
            "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz",
            return_value=fake_xyz,
        )
        opt_p = patch(
            "activelearning.applications.molecules.xtb_oracle._run_xtb_optimize",
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
            "activelearning.applications.molecules.xtb_oracle._write_best_rdkit_xyz",
            return_value=fake_xyz,
        )
        opt_p = patch(
            "activelearning.applications.molecules.xtb_oracle._run_xtb_optimize",
            return_value=(fake_xyz, fake_log),
        )
        vert_p = patch.object(oracle, "_vertical_score", return_value=1.0)
        with rdkit_p, opt_p as opt_mock, vert_p:
            oracle._xtb_score(BENZENE_SELFIES, fidelity=2)
        assert opt_mock.call_count == 1
        assert opt_mock.call_args_list[0].kwargs.get("charge") is None

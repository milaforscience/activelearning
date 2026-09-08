"""Tests for the DOCK3 score -> probability-of-binding conversion.

Fixtures build small synthetic inputs rather than reading ``ampc_hitrate_fits/``,
which is untracked and may move.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from activelearning.applications.molecules.hit_rate import (
    HitRateModel,
    load_score_pprop_table,
)

# Shaped like the fitted AmpC parameters: a strong negative correlation between
# docking score and experimental potency, plus a rare artifact population that
# scores very well without binding.
_PARAMS = {
    "rho": -0.75,
    "exp_mean": -1.5,
    "exp_std": 1.4,
    "artifact_freq": 1.2e-06,
    "artifact_mean": -3.7,
    "artifact_std": 1.0,
}


@pytest.fixture
def table_path(tmp_path: Path) -> Path:
    """Write a monotone score/pProp table spanning -120 .. 0."""
    scores = np.linspace(-120.0, 0.0, 241)
    # pProp falls as scores get worse, like a real cumulative rank curve.
    pprops = np.linspace(9.0, 1.5, 241)
    lines = ["score n cumul_n prop cumul_prop pprop"]
    lines += [f"{s:.4f} 1 1 0.0 0.0 {p:.4f}" for s, p in zip(scores, pprops)]
    path = tmp_path / "full_scores.df"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def params_path(tmp_path: Path) -> Path:
    """Write a fitted-parameters JSON holding one target."""
    path = tmp_path / "fitted_params.json"
    path.write_text(json.dumps({"ampc": _PARAMS}))
    return path


@pytest.fixture
def model(params_path: Path, table_path: Path) -> HitRateModel:
    """Return a model built from the synthetic fixtures."""
    return HitRateModel.from_files(params_path, table_path, target="ampc")


class TestLoadScorePpropTable:
    """Tests for the score/pProp table loader."""

    def test_reads_and_sorts(self, tmp_path: Path) -> None:
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.0 5.0\n-30.0 7.0\n-20.0 6.0\n")

        scores, pprops = load_score_pprop_table(path)

        assert list(scores) == [-30.0, -20.0, -10.0]
        assert list(pprops) == [7.0, 6.0, 5.0]

    def test_uses_full_precision(self, tmp_path: Path) -> None:
        """Upstream rounded scores to 2dp as dict keys, losing 42% of rows."""
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.001 5.0\n-10.002 6.0\n-10.003 7.0\n")

        scores, _ = load_score_pprop_table(path)

        assert len(scores) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Score/pProp table not found"):
            load_score_pprop_table(tmp_path / "nope.df")

    def test_too_few_rows_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.0 5.0\n")
        with pytest.raises(ValueError, match="at least 2 are required"):
            load_score_pprop_table(path)


class TestConstruction:
    """Tests for model construction and parameter validation."""

    def test_from_files_reads_target(self, model: HitRateModel) -> None:
        assert model.rho == pytest.approx(-0.75)
        assert model.score_min == pytest.approx(-120.0)
        assert model.score_max == pytest.approx(0.0)

    def test_unknown_target_raises(self, params_path: Path, table_path: Path) -> None:
        with pytest.raises(KeyError, match="not found in"):
            HitRateModel.from_files(params_path, table_path, target="cdk2")

    def test_missing_params_file_raises(self, tmp_path: Path, table_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Fitted parameters not found"):
            HitRateModel.from_files(tmp_path / "nope.json", table_path)

    def test_missing_required_parameter_raises(self, table_path: Path) -> None:
        with pytest.raises(KeyError, match="missing required key"):
            HitRateModel({"rho": -0.5, "exp_mean": -1.5}, table_path)

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ({"rho": 1.0}, "rho must lie in"),
            ({"exp_std": 0.0}, "exp_std must be positive"),
        ],
    )
    def test_out_of_range_parameters_raise(self, table_path, bad, match) -> None:
        params = dict(_PARAMS)
        params.update(bad)
        with pytest.raises(ValueError, match=match):
            HitRateModel(params, table_path)

    def test_artifact_terms_are_optional(self, table_path: Path) -> None:
        params = {"rho": -0.75, "exp_mean": -1.5, "exp_std": 1.4}
        built = HitRateModel(params, table_path)

        assert built.artifact_freq == 0.0
        # With no artifact component the hit rate is the plain bivariate result.
        assert 0.0 <= built.hit_rate(-100.0, 5.0) <= 1.0


class TestHitRate:
    """Tests for the score -> probability conversion."""

    def test_returns_a_probability(self, model: HitRateModel) -> None:
        value = model.hit_rate(-90.0, 5.0)
        assert 0.0 <= value <= 1.0

    def test_better_scores_beat_worse_ones_in_the_normal_range(
        self, model: HitRateModel
    ) -> None:
        assert model.hit_rate(-80.0, 5.0) > model.hit_rate(-40.0, 5.0)

    def test_stricter_threshold_lowers_the_hit_rate(self, model: HitRateModel) -> None:
        assert model.hit_rate(-80.0, 6.5) < model.hit_rate(-80.0, 4.0)

    def test_peaks_at_an_interior_score(self, model: HitRateModel) -> None:
        """The artifact term makes very good scores score worse, by design.

        Pinning this means a future parameter change that removes the turnover
        shows up as a failure rather than silently altering what the loop
        optimizes toward.
        """
        grid = np.arange(-120.0, 0.0, 0.5)
        rates = np.array([model.hit_rate(float(s), 6.5) for s in grid])
        peak = float(grid[rates.argmax()])

        assert -120.0 < peak < 0.0, "expected an interior maximum"
        assert model.hit_rate(-120.0, 6.5) < rates.max()

    def test_nan_propagates(self, model: HitRateModel) -> None:
        assert math.isnan(model.hit_rate(float("nan"), 6.5))
        assert math.isnan(model.pprop(float("nan")))

    @pytest.mark.parametrize("score", [-500.0, 500.0])
    def test_out_of_domain_clamps_rather_than_raising(
        self, model: HitRateModel, score: float, caplog
    ) -> None:
        """Raising would turn a real observation into an oracle failure."""
        with caplog.at_level("WARNING"):
            value = model.hit_rate(score, 6.5)

        assert 0.0 <= value <= 1.0
        assert any("outside the score/pProp table" in r.message for r in caplog.records)

    def test_clamped_scores_match_the_endpoints(self, model: HitRateModel) -> None:
        assert model.pprop(-500.0) == pytest.approx(model.pprop(-120.0))
        assert model.pprop(500.0) == pytest.approx(model.pprop(0.0))


class TestThreadSafety:
    """The oracle shares one model across a thread pool."""

    def test_concurrent_calls_with_different_thresholds_agree_with_serial(
        self, model: HitRateModel
    ) -> None:
        """Upstream stored the threshold on the instance, so calls interfered."""
        jobs = [(-100.0 + 5 * i, 3.0 + 0.25 * i) for i in range(24)] * 8
        expected = [model.hit_rate(s, d) for s, d in jobs]

        with ThreadPoolExecutor(max_workers=16) as executor:
            got = list(executor.map(lambda job: model.hit_rate(*job), jobs))

        assert got == expected

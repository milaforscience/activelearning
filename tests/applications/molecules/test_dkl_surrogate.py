"""Tests for ExactSelfiesDKLSurrogate and VariationalSelfiesDKLSurrogate."""

import math
import pytest
import torch

from activelearning.applications.molecules.config import (
    SelfiesTrainingConfig,
    SelfiesTransformerEncoderConfig,
)
from activelearning.applications.molecules.dkl_surrogate import (
    ExactSelfiesDKLSurrogate,
    VariationalSelfiesDKLSurrogate,
)
from activelearning.utils.types import Candidate, Observation

BENZENE = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
ALANINE = "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]"
ETHANOL = "[C][C][O]"
FIDELITY_CONFIDENCES = {1: 0.25, 2: 0.5, 3: 1.0}

TRAINING = SelfiesTrainingConfig(epochs=2, lr=1e-3, mask_ratio=0.15, pretrain_epochs=1)
ENCODER_CFG = SelfiesTransformerEncoderConfig(
    max_length=32,
    embed_dim=8,
    ff_dim=16,
    num_heads=2,
    num_layers=1,
    latent_dim=4,
)


def _make_observations(
    selfies_strings: list[str], values: list[float]
) -> list[Observation]:
    return [Observation(x=s, y=v) for s, v in zip(selfies_strings, values)]


def _make_candidates(selfies_strings: list[str]) -> list[Candidate]:
    return [Candidate(x=s) for s in selfies_strings]


def _make_mf_observations(
    selfies_strings: list[str], values: list[float], fidelities: list[int]
) -> list[Observation]:
    return [
        Observation(x=s, y=v, fidelity=f)
        for s, v, f in zip(selfies_strings, values, fidelities)
    ]


def _make_mf_candidates(
    selfies_strings: list[str], fidelities: list[int]
) -> list[Candidate]:
    return [Candidate(x=s, fidelity=f) for s, f in zip(selfies_strings, fidelities)]


# ---------------------------------------------------------------------------
# ExactSelfiesDKLSurrogate
# ---------------------------------------------------------------------------


@pytest.fixture
def exact_surrogate() -> ExactSelfiesDKLSurrogate:
    encoder = ENCODER_CFG.build()
    return ExactSelfiesDKLSurrogate(encoder=encoder, training_params=TRAINING)


@pytest.fixture
def exact_mf_surrogate() -> ExactSelfiesDKLSurrogate:
    encoder = ENCODER_CFG.build()
    surrogate = ExactSelfiesDKLSurrogate(
        encoder=encoder,
        training_params=TRAINING,
        multi_fidelity=True,
        target_fidelity=3,
    )
    surrogate.set_fidelity_confidences(FIDELITY_CONFIDENCES)
    return surrogate


class TestExactSelfiesDKLSurrogate:
    def test_not_fitted_initially(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        assert not exact_surrogate.is_fitted()

    def test_fit_single_observation(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        obs = _make_observations([BENZENE], [1.5])
        exact_surrogate.fit(obs)
        assert exact_surrogate.is_fitted()

    def test_fit_multiple_observations(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
        exact_surrogate.fit(obs)
        assert exact_surrogate.is_fitted()

    def test_fit_noop_on_empty(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        exact_surrogate.fit([])
        assert not exact_surrogate.is_fitted()

    def test_predict_mean_std_keys(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = exact_surrogate.predict(_make_candidates([ETHANOL]))
        assert "mean" in result
        assert "std" in result

    def test_predict_length(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = exact_surrogate.predict(_make_candidates([BENZENE, ETHANOL]))
        assert len(result["mean"]) == 2
        assert len(result["std"]) == 2

    def test_predict_std_positive(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = exact_surrogate.predict(_make_candidates([ETHANOL]))
        for s in result["std"]:
            assert s >= 0.0

    def test_encode_candidates_shape(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE], [1.0]))
        X = exact_surrogate.encode_candidates(_make_candidates([BENZENE, ALANINE]))
        assert X.ndim == 2
        assert X.shape[0] == 2

    def test_encode_candidates_dtype(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE], [1.0]))
        X = exact_surrogate.encode_candidates(_make_candidates([BENZENE]))
        assert X.dtype == torch.float64

    def test_updates_from_latest_false(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        assert not exact_surrogate.updates_from_latest()

    def test_fit_two_observations(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        """Fitting with two observations should store both training points."""
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, 2.0]))
        assert exact_surrogate.is_fitted()
        X, Y = exact_surrogate.get_train_data()
        assert X.shape[0] == 2

    def test_multi_fidelity_fit_and_encode(
        self, exact_mf_surrogate: ExactSelfiesDKLSurrogate
    ):
        """Multi-fidelity observations should work and add +1 to train_X width."""
        obs = _make_mf_observations(
            [BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3]
        )
        exact_mf_surrogate.fit(obs)
        assert exact_mf_surrogate.is_fitted()
        # Token IDs + fidelity column
        X, _ = exact_mf_surrogate.get_train_data()
        assert X.shape == (3, exact_mf_surrogate._encoder.max_length + 1)

    def test_multi_fidelity_encode_candidates(
        self, exact_mf_surrogate: ExactSelfiesDKLSurrogate
    ):
        """encode_candidates should append confidence values for BoTorch MF helpers."""
        obs = _make_mf_observations([BENZENE, ALANINE], [1.0, 2.0], [1, 2])
        exact_mf_surrogate.fit(obs)
        candidates = _make_mf_candidates([BENZENE, ALANINE], [2, 3])
        tokens = exact_mf_surrogate.encode_candidates(candidates)
        # Shape: (2, seq_len + 1 for fidelity)
        assert tokens.shape == (2, exact_mf_surrogate._encoder.max_length + 1)
        # Last column should contain encoded confidence values.
        assert tokens[:, -1].tolist() == pytest.approx([0.5, 1.0])

    def test_multi_fidelity_requires_confidence_mapping(self) -> None:
        surrogate = ExactSelfiesDKLSurrogate(
            encoder=ENCODER_CFG.build(),
            training_params=TRAINING,
            multi_fidelity=True,
            target_fidelity=3,
        )

        with pytest.raises(ValueError, match="Missing fidelity confidence"):
            surrogate.fit(_make_mf_observations([BENZENE], [1.0], [1]))

    def test_target_fidelity_value_uses_encoded_confidence(
        self, exact_mf_surrogate: ExactSelfiesDKLSurrogate
    ) -> None:
        assert exact_mf_surrogate.get_target_fidelity_value() == pytest.approx(1.0)

    def test_empty_candidates_raises(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        with pytest.raises(ValueError):
            exact_surrogate.encode_candidates([])

    def test_pretrain_epochs(self, exact_surrogate: ExactSelfiesDKLSurrogate):
        """Ensure pretrain_epochs runs without error."""
        # TRAINING already has pretrain_epochs=1
        obs = _make_observations([BENZENE, ALANINE], [0.5, -0.5])
        exact_surrogate.fit(obs)
        assert exact_surrogate.is_fitted()


# ---------------------------------------------------------------------------
# VariationalSelfiesDKLSurrogate
# ---------------------------------------------------------------------------


@pytest.fixture
def var_surrogate() -> VariationalSelfiesDKLSurrogate:
    encoder = ENCODER_CFG.build()
    return VariationalSelfiesDKLSurrogate(
        encoder=encoder, training_params=TRAINING, num_inducing=8
    )


@pytest.fixture
def var_mf_surrogate() -> VariationalSelfiesDKLSurrogate:
    encoder = ENCODER_CFG.build()
    surrogate = VariationalSelfiesDKLSurrogate(
        encoder=encoder,
        training_params=TRAINING,
        num_inducing=8,
        multi_fidelity=True,
        target_fidelity=3,
    )
    surrogate.set_fidelity_confidences(FIDELITY_CONFIDENCES)
    return surrogate


class TestVariationalSelfiesDKLSurrogate:
    def test_not_fitted_initially(self, var_surrogate: VariationalSelfiesDKLSurrogate):
        assert not var_surrogate.is_fitted()

    def test_fit(self, var_surrogate: VariationalSelfiesDKLSurrogate):
        obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
        var_surrogate.fit(obs)
        assert var_surrogate.is_fitted()

    def test_predict_keys(self, var_surrogate: VariationalSelfiesDKLSurrogate):
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = var_surrogate.predict(_make_candidates([ETHANOL]))
        assert "mean" in result
        assert "std" in result

    def test_predict_length(self, var_surrogate: VariationalSelfiesDKLSurrogate):
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = var_surrogate.predict(_make_candidates([BENZENE, ETHANOL]))
        assert len(result["mean"]) == 2

    def test_predict_std_nonnegative(
        self, var_surrogate: VariationalSelfiesDKLSurrogate
    ):
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = var_surrogate.predict(_make_candidates([ETHANOL]))
        assert all(s >= 0.0 for s in result["std"])

    def test_updates_from_latest_false(
        self, var_surrogate: VariationalSelfiesDKLSurrogate
    ):
        assert not var_surrogate.updates_from_latest()

    def test_fit_multiple_observations(
        self, var_surrogate: VariationalSelfiesDKLSurrogate
    ):
        """Fitting with two observations after a first fit should work."""
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, 2.0]))
        assert var_surrogate.is_fitted()

    def test_predict_before_fit_raises(
        self, var_surrogate: VariationalSelfiesDKLSurrogate
    ):
        with pytest.raises(RuntimeError):
            var_surrogate.predict(_make_candidates([BENZENE]))

    def test_encode_candidates_returns_latent_features(
        self, var_surrogate: VariationalSelfiesDKLSurrogate
    ):
        """encode_candidates() must return latent features, not token IDs."""
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        latent = var_surrogate.encode_candidates(_make_candidates([BENZENE, ETHANOL]))
        assert latent.ndim == 2
        assert latent.shape[0] == 2
        # Latent dim, not seq_len
        assert latent.shape[1] == ENCODER_CFG.latent_dim

    def test_encode_candidates_mf_returns_latent_plus_fidelity(
        self, var_mf_surrogate: VariationalSelfiesDKLSurrogate
    ):
        """Multi-fidelity encode_candidates must return (latent_dim + 1) features."""
        obs = _make_mf_observations([BENZENE, ALANINE], [1.0, 2.0], [1, 2])
        var_mf_surrogate.fit(obs)
        candidates = _make_mf_candidates([BENZENE, ALANINE], [1, 2])
        latent = var_mf_surrogate.encode_candidates(candidates)
        assert latent.shape == (2, ENCODER_CFG.latent_dim + 1)
        # Last column is the encoded fidelity confidence.
        assert latent[:, -1].tolist() == pytest.approx([0.25, 0.5])

    def test_get_model_returns_botorch_adapter(
        self, var_surrogate: VariationalSelfiesDKLSurrogate
    ):
        """get_model() must return the BoTorch-compatible adapter after fitting."""
        from activelearning.applications.molecules.dkl_surrogate import (
            _VariationalBoTorchAdapter,
        )

        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        model = var_surrogate.get_model()
        assert isinstance(model, _VariationalBoTorchAdapter)

    def test_get_fidelity_dimension_uses_latent_dim(
        self, var_mf_surrogate: VariationalSelfiesDKLSurrogate
    ):
        """get_fidelity_dimension() must index into latent space, not token space."""
        obs = _make_mf_observations([BENZENE, ALANINE], [1.0, 2.0], [1, 2])
        var_mf_surrogate.fit(obs)
        assert var_mf_surrogate.get_fidelity_dimension() == ENCODER_CFG.latent_dim

    def test_multi_fidelity_fit_and_predict(
        self, var_mf_surrogate: VariationalSelfiesDKLSurrogate
    ):
        """Variational surrogate should accept multi-fidelity observations."""
        obs = _make_mf_observations(
            [BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3]
        )
        var_mf_surrogate.fit(obs)
        assert var_mf_surrogate.is_fitted()
        candidates = _make_mf_candidates([BENZENE], [1])
        result = var_mf_surrogate.predict(candidates)
        assert "mean" in result and "std" in result


# ---------------------------------------------------------------------------
# Acquisition compatibility — single-fidelity
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_exact(exact_surrogate: ExactSelfiesDKLSurrogate) -> ExactSelfiesDKLSurrogate:
    exact_surrogate.fit(
        _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
    )
    return exact_surrogate


@pytest.fixture
def fitted_var(
    var_surrogate: VariationalSelfiesDKLSurrogate,
) -> VariationalSelfiesDKLSurrogate:
    var_surrogate.fit(_make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0]))
    return var_surrogate


@pytest.fixture
def fitted_exact_mf(
    exact_mf_surrogate: ExactSelfiesDKLSurrogate,
) -> ExactSelfiesDKLSurrogate:
    exact_mf_surrogate.fit(
        _make_mf_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3])
    )
    return exact_mf_surrogate


@pytest.fixture
def fitted_var_mf(
    var_mf_surrogate: VariationalSelfiesDKLSurrogate,
) -> VariationalSelfiesDKLSurrogate:
    var_mf_surrogate.fit(
        _make_mf_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3])
    )
    return var_mf_surrogate


def _scores_finite(scores: list[float], n: int) -> None:
    """Assert that scores is a list of n finite floats."""
    assert len(scores) == n
    assert all(math.isfinite(s) for s in scores)


class TestExactDKLAcquisitions:
    """ExactSelfiesDKLSurrogate is compatible with BoTorch acquisition functions."""

    _sf_obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
    _sf_cands = _make_candidates([BENZENE, ETHANOL])
    _mf_obs = _make_mf_observations(
        [BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3]
    )
    _mf_cands = _make_mf_candidates([BENZENE, ETHANOL], [2, 3])

    def test_ucb_scores_finite(self, fitted_exact: ExactSelfiesDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            UpperConfidenceBound,
        )

        acq = UpperConfidenceBound(beta=2.0)
        acq.update(fitted_exact, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_ei_scores_finite(self, fitted_exact: ExactSelfiesDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            ExpectedImprovement,
        )

        acq = ExpectedImprovement()
        acq.update(fitted_exact, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_log_ei_scores_finite(self, fitted_exact: ExactSelfiesDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            LogExpectedImprovement,
        )

        acq = LogExpectedImprovement()
        acq.update(fitted_exact, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_qmfmes_scores_finite(
        self, fitted_exact_mf: ExactSelfiesDKLSurrogate
    ) -> None:
        from activelearning.acquisition.botorch.botorch_multifidelity import (
            QMultiFidelityMaxValueEntropy,
        )
        from activelearning.acquisition.botorch.candidate_set import (
            TrainDataCandidateSetSpec,
        )

        acq = QMultiFidelityMaxValueEntropy(
            candidate_set_spec=TrainDataCandidateSetSpec(),
            num_fantasies=2,
            num_mv_samples=5,
            num_y_samples=16,
        )
        acq.update(fitted_exact_mf, self._mf_obs)
        _scores_finite(acq.score(self._mf_cands), len(self._mf_cands))

    def test_qmflbmes_scores_finite(
        self, fitted_exact_mf: ExactSelfiesDKLSurrogate
    ) -> None:
        from activelearning.acquisition.botorch.botorch_multifidelity import (
            QMultiFidelityLowerBoundMaxValueEntropy,
        )
        from activelearning.acquisition.botorch.candidate_set import (
            TrainDataCandidateSetSpec,
        )

        acq = QMultiFidelityLowerBoundMaxValueEntropy(
            candidate_set_spec=TrainDataCandidateSetSpec(),
            num_fantasies=2,
            num_mv_samples=5,
            num_y_samples=16,
        )
        acq.update(fitted_exact_mf, self._mf_obs)
        _scores_finite(acq.score(self._mf_cands), len(self._mf_cands))


class TestVariationalDKLAcquisitions:
    """VariationalSelfiesDKLSurrogate is compatible with BoTorch acquisition functions."""

    _sf_obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
    _sf_cands = _make_candidates([BENZENE, ETHANOL])
    _mf_obs = _make_mf_observations(
        [BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3]
    )
    _mf_cands = _make_mf_candidates([BENZENE, ETHANOL], [2, 3])

    def test_ucb_scores_finite(
        self, fitted_var: VariationalSelfiesDKLSurrogate
    ) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            UpperConfidenceBound,
        )

        acq = UpperConfidenceBound(beta=2.0)
        acq.update(fitted_var, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_ei_scores_finite(self, fitted_var: VariationalSelfiesDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            ExpectedImprovement,
        )

        acq = ExpectedImprovement()
        acq.update(fitted_var, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_log_ei_scores_finite(
        self, fitted_var: VariationalSelfiesDKLSurrogate
    ) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            LogExpectedImprovement,
        )

        acq = LogExpectedImprovement()
        acq.update(fitted_var, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_qmfmes_scores_finite(
        self, fitted_var_mf: VariationalSelfiesDKLSurrogate
    ) -> None:
        from activelearning.acquisition.botorch.botorch_multifidelity import (
            QMultiFidelityMaxValueEntropy,
        )
        from activelearning.acquisition.botorch.candidate_set import (
            TrainDataCandidateSetSpec,
        )

        acq = QMultiFidelityMaxValueEntropy(
            candidate_set_spec=TrainDataCandidateSetSpec(),
            num_fantasies=2,
            num_mv_samples=5,
            num_y_samples=16,
        )
        acq.update(fitted_var_mf, self._mf_obs)
        _scores_finite(acq.score(self._mf_cands), len(self._mf_cands))

    def test_qmflbmes_scores_finite(
        self, fitted_var_mf: VariationalSelfiesDKLSurrogate
    ) -> None:
        from activelearning.acquisition.botorch.botorch_multifidelity import (
            QMultiFidelityLowerBoundMaxValueEntropy,
        )
        from activelearning.acquisition.botorch.candidate_set import (
            TrainDataCandidateSetSpec,
        )

        acq = QMultiFidelityLowerBoundMaxValueEntropy(
            candidate_set_spec=TrainDataCandidateSetSpec(),
            num_fantasies=2,
            num_mv_samples=5,
            num_y_samples=16,
        )
        acq.update(fitted_var_mf, self._mf_obs)
        _scores_finite(acq.score(self._mf_cands), len(self._mf_cands))


class TestSelfiesKernelBatchDims:
    """Verify the kernel handles BoTorch's (batch, q, seq_len) input shape."""

    def test_kernel_3d_input_no_fidelity(self):
        """Kernel must work when BoTorch adds a q-dimension."""
        from activelearning.applications.molecules.selfies_kernel import SelfiesKernel
        import gpytorch

        encoder = ENCODER_CFG.build()
        base_kernel = gpytorch.kernels.RBFKernel()
        kernel = SelfiesKernel(encoder, base_kernel, include_fidelity=False)

        seq_len = encoder.max_length
        # BoTorch-style: (batch=2, q=1, seq_len)
        x = torch.zeros(2, 1, seq_len, dtype=torch.float64)
        result = kernel(x, x).evaluate()
        assert result.shape[-1] == 1  # no shape/encoding error

    def test_kernel_3d_input_with_fidelity(self):
        """Kernel must correctly split fidelity from last column with 3-D input."""
        from activelearning.applications.molecules.selfies_kernel import SelfiesKernel
        import gpytorch

        encoder = ENCODER_CFG.build()
        gp_input_dim = encoder.latent_dim + 1
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=gp_input_dim)
        kernel = SelfiesKernel(encoder, base_kernel, include_fidelity=True)

        seq_len = encoder.max_length
        # Last column = encoded fidelity value; shape (batch=2, q=1, seq_len+1)
        x = torch.zeros(2, 1, seq_len + 1, dtype=torch.float64)
        x[..., -1] = 1.0  # encoded target fidelity confidence
        result = kernel(x, x).evaluate()
        assert result.shape[0] == 2  # no shape error


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestDKLSurrogateConfigs:
    def test_exact_config_builds(self):
        from activelearning.applications.molecules.config import (
            ExactSelfiesDKLSurrogateConfig,
        )

        cfg = ExactSelfiesDKLSurrogateConfig(encoder=ENCODER_CFG)
        surrogate = cfg.build()
        assert isinstance(surrogate, ExactSelfiesDKLSurrogate)

    def test_variational_config_builds(self):
        from activelearning.applications.molecules.config import (
            VariationalSelfiesDKLSurrogateConfig,
        )

        cfg = VariationalSelfiesDKLSurrogateConfig(encoder=ENCODER_CFG, num_inducing=8)
        surrogate = cfg.build()
        assert isinstance(surrogate, VariationalSelfiesDKLSurrogate)

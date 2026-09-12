"""Tests for the exact and variational molecular DKL surrogates."""

import math
import pytest
import torch

from activelearning.surrogate.encoder_config import (
    SelfiesTransformerEncoderConfig,
)
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.dkl.kernel import EncoderKernel
from activelearning.surrogate.dkl import (
    DeepKernelSurrogate,
    ExactDKLSurrogate,
    VariationalDKLSurrogate,
)
from activelearning.runtime import RuntimeContext
from activelearning.utils.types import Candidate, Observation

BENZENE = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
ALANINE = "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]"
ETHANOL = "[C][C][O]"
FIDELITY_CONFIDENCES = {1: 0.25, 2: 0.5, 3: 1.0}

TRAINING = DKLTrainingConfig(epochs=2, lr=1e-3, mask_ratio=0.15, pretrain_epochs=1)
ENCODER_CFG = SelfiesTransformerEncoderConfig(
    max_mol_tokens=32,
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
# ExactDKLSurrogate
# ---------------------------------------------------------------------------


@pytest.fixture
def exact_surrogate() -> ExactDKLSurrogate:
    encoder = ENCODER_CFG.build()
    return ExactDKLSurrogate(
        encoder=encoder,
        training_params=TRAINING,
    )


@pytest.fixture
def exact_mf_surrogate() -> ExactDKLSurrogate:
    encoder = ENCODER_CFG.build()
    surrogate = ExactDKLSurrogate(
        encoder=encoder,
        training_params=TRAINING,
        is_multi_fidelity=True,
        target_fidelity=3,
    )
    surrogate.set_fidelity_confidences(FIDELITY_CONFIDENCES)
    return surrogate


@pytest.fixture(params=["exact", "variational"])
def molecule_dkl_surrogate(
    request: pytest.FixtureRequest,
) -> DeepKernelSurrogate:
    """Yield both molecular DKL variants for shared correctness tests."""
    encoder = ENCODER_CFG.build()
    if request.param == "exact":
        return ExactDKLSurrogate(
            encoder=encoder,
            training_params=TRAINING,
        )
    return VariationalDKLSurrogate(
        encoder=encoder,
        training_params=TRAINING,
        num_inducing=8,
    )


class TestExactDKLSurrogate:
    def test_not_fitted_initially(self, exact_surrogate: ExactDKLSurrogate):
        assert not exact_surrogate.is_fitted()

    def test_fit_single_observation(self, exact_surrogate: ExactDKLSurrogate):
        obs = _make_observations([BENZENE], [1.5])
        exact_surrogate.fit(obs)
        assert exact_surrogate.is_fitted()

    def test_fit_multiple_observations(self, exact_surrogate: ExactDKLSurrogate):
        obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
        exact_surrogate.fit(obs)
        assert exact_surrogate.is_fitted()

    def test_fit_noop_on_empty(self, exact_surrogate: ExactDKLSurrogate):
        exact_surrogate.fit([])
        assert not exact_surrogate.is_fitted()

    def test_predict_mean_std_keys(self, exact_surrogate: ExactDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = exact_surrogate.predict(_make_candidates([ETHANOL]))
        assert "mean" in result
        assert "std" in result

    def test_predict_length(self, exact_surrogate: ExactDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = exact_surrogate.predict(_make_candidates([BENZENE, ETHANOL]))
        assert len(result["mean"]) == 2
        assert len(result["std"]) == 2

    def test_predict_std_positive(self, exact_surrogate: ExactDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = exact_surrogate.predict(_make_candidates([ETHANOL]))
        for s in result["std"]:
            assert s >= 0.0

    def test_encode_candidates_shape(self, exact_surrogate: ExactDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE], [1.0]))
        X = exact_surrogate.encode_candidates(_make_candidates([BENZENE, ALANINE]))
        assert X.ndim == 2
        assert X.shape[0] == 2

    def test_encode_candidates_dtype(self, exact_surrogate: ExactDKLSurrogate):
        exact_surrogate.fit(_make_observations([BENZENE], [1.0]))
        X = exact_surrogate.encode_candidates(_make_candidates([BENZENE]))
        assert X.dtype == torch.float64

    def test_runtime_float32_controls_exact_surrogate_dtype(
        self, exact_surrogate: ExactDKLSurrogate
    ) -> None:
        exact_surrogate.bind_runtime_context(RuntimeContext(dtype=torch.float32))
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, 2.0]))

        X, Y = exact_surrogate.get_train_data()
        assert X.dtype == torch.float32
        assert Y.dtype == torch.float32
        assert exact_surrogate.model.train_inputs[0].dtype == torch.float32
        assert next(exact_surrogate._encoder.parameters()).dtype == torch.float32

    def test_updates_from_latest_false(self, exact_surrogate: ExactDKLSurrogate):
        assert not exact_surrogate.updates_from_latest()

    def test_fit_two_observations(self, exact_surrogate: ExactDKLSurrogate):
        """Fitting with two observations should store both training points."""
        exact_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, 2.0]))
        assert exact_surrogate.is_fitted()
        X, Y = exact_surrogate.get_train_data()
        assert X.shape[0] == 2

    def test_multi_fidelity_fit_and_encode(self, exact_mf_surrogate: ExactDKLSurrogate):
        """Multi-fidelity observations should work and add +1 to train_X width."""
        obs = _make_mf_observations(
            [BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3]
        )
        exact_mf_surrogate.fit(obs)
        assert exact_mf_surrogate.is_fitted()
        # Token IDs + fidelity column
        X, _ = exact_mf_surrogate.get_train_data()
        assert X.shape == (3, exact_mf_surrogate._encoder.max_tokens + 1)

    def test_multi_fidelity_encode_candidates(
        self, exact_mf_surrogate: ExactDKLSurrogate
    ):
        """encode_candidates should append confidence values for BoTorch MF helpers."""
        obs = _make_mf_observations([BENZENE, ALANINE], [1.0, 2.0], [1, 2])
        exact_mf_surrogate.fit(obs)
        candidates = _make_mf_candidates([BENZENE, ALANINE], [2, 3])
        tokens = exact_mf_surrogate.encode_candidates(candidates)
        # Shape: (2, seq_len + 1 for fidelity)
        assert tokens.shape == (2, exact_mf_surrogate._encoder.max_tokens + 1)
        # Last column should contain encoded confidence values.
        assert tokens[:, -1].tolist() == pytest.approx([0.5, 1.0])

    def test_multi_fidelity_requires_confidence_mapping(self) -> None:
        encoder = ENCODER_CFG.build()
        surrogate = ExactDKLSurrogate(
            encoder=encoder,
            training_params=TRAINING,
            is_multi_fidelity=True,
            target_fidelity=3,
        )

        with pytest.raises(ValueError, match="Missing fidelity confidence"):
            surrogate.fit(_make_mf_observations([BENZENE], [1.0], [1]))

    def test_target_fidelity_value_uses_encoded_confidence(
        self, exact_mf_surrogate: ExactDKLSurrogate
    ) -> None:
        assert exact_mf_surrogate.get_target_fidelity_value() == pytest.approx(1.0)

    def test_empty_candidates_raises(self, exact_surrogate: ExactDKLSurrogate):
        with pytest.raises(ValueError):
            exact_surrogate.encode_candidates([])

    def test_pretrain_epochs(self, exact_surrogate: ExactDKLSurrogate):
        """Ensure pretrain_epochs runs without error."""
        # TRAINING already has pretrain_epochs=1
        obs = _make_observations([BENZENE, ALANINE], [0.5, -0.5])
        exact_surrogate.fit(obs)
        assert exact_surrogate.is_fitted()


# ---------------------------------------------------------------------------
# VariationalDKLSurrogate
# ---------------------------------------------------------------------------


@pytest.fixture
def var_surrogate() -> VariationalDKLSurrogate:
    encoder = ENCODER_CFG.build()
    return VariationalDKLSurrogate(
        encoder=encoder,
        training_params=TRAINING,
        num_inducing=8,
    )


@pytest.fixture
def var_mf_surrogate() -> VariationalDKLSurrogate:
    encoder = ENCODER_CFG.build()
    surrogate = VariationalDKLSurrogate(
        encoder=encoder,
        training_params=TRAINING,
        num_inducing=8,
        is_multi_fidelity=True,
        target_fidelity=3,
    )
    surrogate.set_fidelity_confidences(FIDELITY_CONFIDENCES)
    return surrogate


class TestVariationalDKLSurrogate:
    def test_not_fitted_initially(self, var_surrogate: VariationalDKLSurrogate):
        assert not var_surrogate.is_fitted()

    def test_fit(self, var_surrogate: VariationalDKLSurrogate):
        obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
        var_surrogate.fit(obs)
        assert var_surrogate.is_fitted()

    def test_predict_keys(self, var_surrogate: VariationalDKLSurrogate):
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = var_surrogate.predict(_make_candidates([ETHANOL]))
        assert "mean" in result
        assert "std" in result

    def test_predict_length(self, var_surrogate: VariationalDKLSurrogate):
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = var_surrogate.predict(_make_candidates([BENZENE, ETHANOL]))
        assert len(result["mean"]) == 2

    def test_predict_std_nonnegative(self, var_surrogate: VariationalDKLSurrogate):
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        result = var_surrogate.predict(_make_candidates([ETHANOL]))
        assert all(s >= 0.0 for s in result["std"])

    def test_updates_from_latest_false(self, var_surrogate: VariationalDKLSurrogate):
        assert not var_surrogate.updates_from_latest()

    def test_fit_multiple_observations(self, var_surrogate: VariationalDKLSurrogate):
        """Fitting with two observations after a first fit should work."""
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, 2.0]))
        assert var_surrogate.is_fitted()

    def test_predict_before_fit_raises(self, var_surrogate: VariationalDKLSurrogate):
        with pytest.raises(RuntimeError):
            var_surrogate.predict(_make_candidates([BENZENE]))

    def test_encode_candidates_returns_latent_features(
        self, var_surrogate: VariationalDKLSurrogate
    ):
        """encode_candidates() must return latent features, not token IDs."""
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        latent = var_surrogate.encode_candidates(_make_candidates([BENZENE, ETHANOL]))
        assert latent.ndim == 2
        assert latent.shape[0] == 2
        # Latent dim, not seq_len
        assert latent.shape[1] == ENCODER_CFG.latent_dim

    def test_encode_candidates_mf_returns_latent_plus_fidelity(
        self, var_mf_surrogate: VariationalDKLSurrogate
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
        self, var_surrogate: VariationalDKLSurrogate
    ):
        """get_model() must return the BoTorch-compatible adapter after fitting."""
        from activelearning.surrogate.dkl.variational import (
            _VariationalBoTorchAdapter,
        )

        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))
        model = var_surrogate.get_model()
        assert isinstance(model, _VariationalBoTorchAdapter)

    def test_get_fidelity_dimension_uses_latent_dim(
        self, var_mf_surrogate: VariationalDKLSurrogate
    ):
        """get_fidelity_dimension() must index into latent space, not token space."""
        obs = _make_mf_observations([BENZENE, ALANINE], [1.0, 2.0], [1, 2])
        var_mf_surrogate.fit(obs)
        assert var_mf_surrogate.get_fidelity_dimension() == ENCODER_CFG.latent_dim

    def test_multi_fidelity_fit_and_predict(
        self, var_mf_surrogate: VariationalDKLSurrogate
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

    def test_runtime_float32_controls_variational_surrogate_dtype(
        self, var_surrogate: VariationalDKLSurrogate
    ) -> None:
        var_surrogate.bind_runtime_context(RuntimeContext(dtype=torch.float32))
        var_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, 2.0]))

        latent = var_surrogate.encode_candidates(_make_candidates([ETHANOL]))
        assert latent.dtype == torch.float32
        assert next(var_surrogate._encoder.parameters()).dtype == torch.float32
        assert next(var_surrogate._gp_model.parameters()).dtype == torch.float32
        assert next(var_surrogate._likelihood.parameters()).dtype == torch.float32


# ---------------------------------------------------------------------------
# Acquisition compatibility — single-fidelity
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_exact(
    exact_surrogate: ExactDKLSurrogate,
) -> ExactDKLSurrogate:
    exact_surrogate.fit(
        _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
    )
    return exact_surrogate


@pytest.fixture
def fitted_var(
    var_surrogate: VariationalDKLSurrogate,
) -> VariationalDKLSurrogate:
    var_surrogate.fit(_make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0]))
    return var_surrogate


@pytest.fixture
def fitted_exact_mf(
    exact_mf_surrogate: ExactDKLSurrogate,
) -> ExactDKLSurrogate:
    exact_mf_surrogate.fit(
        _make_mf_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3])
    )
    return exact_mf_surrogate


@pytest.fixture
def fitted_var_mf(
    var_mf_surrogate: VariationalDKLSurrogate,
) -> VariationalDKLSurrogate:
    var_mf_surrogate.fit(
        _make_mf_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3])
    )
    return var_mf_surrogate


def _scores_finite(scores: list[float], n: int) -> None:
    """Assert that scores is a list of n finite floats."""
    assert len(scores) == n
    assert all(math.isfinite(s) for s in scores)


class TestExactDKLAcquisitions:
    """ExactDKLSurrogate is compatible with BoTorch acquisitions."""

    _sf_obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
    _sf_cands = _make_candidates([BENZENE, ETHANOL])
    _mf_obs = _make_mf_observations(
        [BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3]
    )
    _mf_cands = _make_mf_candidates([BENZENE, ETHANOL], [2, 3])

    def test_ucb_scores_finite(self, fitted_exact: ExactDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            UpperConfidenceBound,
        )

        acq = UpperConfidenceBound(beta=2.0)
        acq.update(fitted_exact, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_ei_scores_finite(self, fitted_exact: ExactDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            ExpectedImprovement,
        )

        acq = ExpectedImprovement()
        acq.update(fitted_exact, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_log_ei_scores_finite(self, fitted_exact: ExactDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            LogExpectedImprovement,
        )

        acq = LogExpectedImprovement()
        acq.update(fitted_exact, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_qmfmes_scores_finite(self, fitted_exact_mf: ExactDKLSurrogate) -> None:
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

    def test_qmflbmes_scores_finite(self, fitted_exact_mf: ExactDKLSurrogate) -> None:
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
    """VariationalDKLSurrogate is compatible with BoTorch acquisitions."""

    _sf_obs = _make_observations([BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0])
    _sf_cands = _make_candidates([BENZENE, ETHANOL])
    _mf_obs = _make_mf_observations(
        [BENZENE, ALANINE, ETHANOL], [1.0, 2.0, 3.0], [1, 2, 3]
    )
    _mf_cands = _make_mf_candidates([BENZENE, ETHANOL], [2, 3])

    def test_ucb_scores_finite(self, fitted_var: VariationalDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            UpperConfidenceBound,
        )

        acq = UpperConfidenceBound(beta=2.0)
        acq.update(fitted_var, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_ei_scores_finite(self, fitted_var: VariationalDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            ExpectedImprovement,
        )

        acq = ExpectedImprovement()
        acq.update(fitted_var, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_log_ei_scores_finite(self, fitted_var: VariationalDKLSurrogate) -> None:
        from activelearning.acquisition.botorch.botorch_analytic import (
            LogExpectedImprovement,
        )

        acq = LogExpectedImprovement()
        acq.update(fitted_var, self._sf_obs)
        _scores_finite(acq.score(self._sf_cands), len(self._sf_cands))

    def test_qmfmes_scores_finite(self, fitted_var_mf: VariationalDKLSurrogate) -> None:
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
        self, fitted_var_mf: VariationalDKLSurrogate
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


class TestEncoderKernelBatchDims:
    """Verify the kernel handles BoTorch's (batch, q, seq_len) input shape."""

    def test_kernel_3d_input_no_fidelity(self):
        """Kernel must work when BoTorch adds a q-dimension."""
        import gpytorch

        encoder = ENCODER_CFG.build()
        base_kernel = gpytorch.kernels.RBFKernel()
        kernel = EncoderKernel(encoder, base_kernel, include_fidelity=False)

        seq_len = encoder.max_tokens
        # BoTorch-style: (batch=2, q=1, seq_len)
        x = torch.zeros(2, 1, seq_len, dtype=torch.float64)
        result = kernel(x, x).evaluate()
        assert result.shape[-1] == 1  # no shape/encoding error

    def test_kernel_3d_input_with_fidelity(self):
        """Kernel must correctly split fidelity from last column with 3-D input."""
        import gpytorch

        encoder = ENCODER_CFG.build()
        gp_input_dim = encoder.latent_dim + 1
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=gp_input_dim)
        kernel = EncoderKernel(encoder, base_kernel, include_fidelity=True)

        seq_len = encoder.max_tokens
        # Last column = encoded fidelity value; shape (batch=2, q=1, seq_len+1)
        x = torch.zeros(2, 1, seq_len + 1, dtype=torch.float64)
        x[..., -1] = 1.0  # encoded target fidelity confidence
        result = kernel(x, x).evaluate()
        assert result.shape[0] == 2  # no shape error

    def test_kernel_diagonal_without_fidelity(self):
        """Kernel diagonal must match the dense covariance diagonal."""
        import gpytorch

        encoder = ENCODER_CFG.build()
        kernel = EncoderKernel(
            encoder,
            gpytorch.kernels.RBFKernel(),
            include_fidelity=False,
        )
        x = torch.zeros(3, encoder.max_tokens, dtype=torch.float64)

        diagonal = kernel(x, x, diag=True).to_dense()
        dense_diagonal = kernel(x, x).to_dense().diagonal()

        torch.testing.assert_close(diagonal, dense_diagonal)

    def test_kernel_diagonal_with_fidelity(self):
        """Kernel diagonal must include the fidelity coordinate."""
        import gpytorch

        encoder = ENCODER_CFG.build()
        kernel = EncoderKernel(
            encoder,
            gpytorch.kernels.RBFKernel(ard_num_dims=encoder.latent_dim + 1),
            include_fidelity=True,
        )
        x = torch.zeros(3, encoder.max_tokens + 1, dtype=torch.float64)
        x[:, -1] = 1.0

        diagonal = kernel(x, x, diag=True).to_dense()
        dense_diagonal = kernel(x, x).to_dense().diagonal()

        torch.testing.assert_close(diagonal, dense_diagonal)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestDKLSurrogateConfigs:
    def test_exact_config_builds(self):
        from activelearning.surrogate.dkl.config import ExactDKLSurrogateConfig

        cfg = ExactDKLSurrogateConfig(encoder=ENCODER_CFG)
        surrogate = cfg.build()
        assert isinstance(surrogate, ExactDKLSurrogate)

    def test_variational_config_builds(self):
        from activelearning.surrogate.dkl.config import (
            VariationalDKLSurrogateConfig,
        )

        cfg = VariationalDKLSurrogateConfig(encoder=ENCODER_CFG, num_inducing=8)
        surrogate = cfg.build()
        assert isinstance(surrogate, VariationalDKLSurrogate)

    def test_standalone_multi_fidelity_config_requires_target_on_build(self):
        """Top-level derivation is unavailable when a DKL config is built alone."""
        from activelearning.surrogate.dkl.config import ExactDKLSurrogateConfig

        cfg = ExactDKLSurrogateConfig(encoder=ENCODER_CFG)
        cfg = cfg.resolve_fidelity_confidences({1: 0.1, 2: 1.0})

        surrogate = cfg.build()

        assert surrogate.is_multi_fidelity


class TestSurrogatePredictionCorrectness:
    """Correctness tests shared by both molecular DKL surrogate variants.

    Each test is run twice: once with the exact GP surrogate and once with the
    variational GP surrogate, via the ``molecule_dkl_surrogate`` fixture.
    """

    def test_predict_is_deterministic(
        self, molecule_dkl_surrogate: DeepKernelSurrogate
    ) -> None:
        """Repeated predictions on the same candidates should be identical."""
        molecule_dkl_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))

        first_result = molecule_dkl_surrogate.predict(_make_candidates([ETHANOL]))
        second_result = molecule_dkl_surrogate.predict(_make_candidates([ETHANOL]))

        assert first_result["mean"] == second_result["mean"]
        assert first_result["std"] == second_result["std"]

    def test_identical_molecules_get_identical_predictions(
        self, molecule_dkl_surrogate: DeepKernelSurrogate
    ) -> None:
        """Identical molecules should receive identical predictive moments."""
        molecule_dkl_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))

        result = molecule_dkl_surrogate.predict(_make_candidates([BENZENE, BENZENE]))

        assert result["mean"][0] == result["mean"][1]
        assert result["std"][0] == result["std"][1]

    def test_predictions_are_permutation_equivariant(
        self, molecule_dkl_surrogate: DeepKernelSurrogate
    ) -> None:
        """Reordering candidates should only reorder the corresponding outputs.

        The Transformer encoder processes each sequence independently (no
        cross-sequence interactions), so the GP posterior for molecule A is
        unaffected by whether molecule B appears before or after it.
        """
        molecule_dkl_surrogate.fit(_make_observations([BENZENE, ALANINE], [1.0, -1.0]))

        forward_result = molecule_dkl_surrogate.predict(
            _make_candidates([BENZENE, ETHANOL])
        )
        reverse_result = molecule_dkl_surrogate.predict(
            _make_candidates([ETHANOL, BENZENE])
        )

        assert forward_result["mean"][0] == reverse_result["mean"][1]
        assert forward_result["mean"][1] == reverse_result["mean"][0]
        assert forward_result["std"][0] == reverse_result["std"][1]
        assert forward_result["std"][1] == reverse_result["std"][0]

    def test_predict_std_strictly_positive_on_training_molecules(
        self, molecule_dkl_surrogate: DeepKernelSurrogate
    ) -> None:
        """Predictive standard deviations should be strictly positive, not zero.

        Both GP variants include observation noise so posterior variance at
        training inputs remains above zero.
        """
        training_selfies = [BENZENE, ALANINE, ETHANOL]
        molecule_dkl_surrogate.fit(
            _make_observations(training_selfies, [1.0, 2.0, 3.0])
        )

        result = molecule_dkl_surrogate.predict(_make_candidates(training_selfies))

        assert all(std > 0.0 for std in result["std"])


class TestExactSurrogatePredictionCorrectness:
    """Correctness tests specific to the exact molecular DKL surrogate.

    These tests rely on properties of the exact GP that do not hold reliably
    for the variational surrogate with tiny training sets.
    """

    ORDERING_TRAINING = DKLTrainingConfig(
        epochs=60, lr=5e-2, mask_ratio=0.15, pretrain_epochs=3
    )

    def test_recovers_training_target_ordering(self) -> None:
        """With sufficient training, the exact GP should rank training molecules.

        An exact GP with enough epochs learns a kernel where the predicted
        posterior mean at each training molecule reflects the relative ordering
        of the observed targets.  This is a fundamental correctness property:
        the model should at minimum learn the sign of the differences between
        training targets.
        """
        torch.manual_seed(0)
        encoder = ENCODER_CFG.build()
        surrogate = ExactDKLSurrogate(
            encoder=encoder,
            training_params=self.ORDERING_TRAINING,
        )
        training_selfies = [BENZENE, ALANINE, ETHANOL]
        surrogate.fit(_make_observations(training_selfies, [-5.0, 0.0, 5.0]))

        result = surrogate.predict(_make_candidates(training_selfies))

        assert result["mean"][0] < result["mean"][1] < result["mean"][2]

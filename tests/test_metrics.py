from functools import partial

import numpy as np
import pytest
import xarray as xr

from pymc_forecast.metrics import (
    DEFAULT_METRICS,
    crps_empirical,
    eval_coverage,
    eval_crps,
    eval_interval_score,
    eval_mae,
    eval_pinball,
    eval_rmse,
    evaluate_forecast,
    make_mase,
)

RNG = np.random.default_rng(7)


@pytest.fixture
def gaussian_pred():
    """1000 standard-normal samples over 20 locations, truth at 0."""
    return RNG.normal(size=(1000, 20)), np.zeros(20)


class TestPointMetrics:
    def test_mae_perfect_forecast_is_zero(self):
        pred = np.zeros((10, 4))
        assert eval_mae(pred, np.zeros(4)) == 0.0

    def test_mae_uses_median(self):
        pred = np.array([[0.0], [0.0], [100.0]])
        assert eval_mae(pred, np.array([0.0])) == 0.0

    def test_rmse_uses_mean(self):
        pred = np.array([[0.0], [2.0]])
        assert eval_rmse(pred, np.array([0.0])) == 1.0


class TestCrps:
    def test_matches_bruteforce(self, gaussian_pred):
        pred, truth = gaussian_pred
        fast = crps_empirical(pred, truth)
        # brute force: E|X - y| - 0.5 E|X - X'|
        abs_err = np.abs(pred - truth).mean(axis=0)
        pairwise = np.abs(pred[:, None, :] - pred[None, :, :]).mean(axis=(0, 1))
        np.testing.assert_allclose(fast, abs_err - 0.5 * pairwise, atol=1e-10)

    def test_needs_two_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            crps_empirical(np.zeros((1, 3)), np.zeros(3))

    def test_known_gaussian_value(self, gaussian_pred):
        # CRPS of N(0,1) forecast at y=0 is (sqrt(2)-1)/sqrt(pi) ~ 0.2337
        pred, truth = gaussian_pred
        assert abs(eval_crps(pred, truth) - 0.2337) < 0.02


class TestIntervalMetrics:
    def test_coverage_calibrated(self, gaussian_pred):
        pred, truth = gaussian_pred
        assert abs(eval_coverage(pred, truth, alpha=0.5) - 1.0) < 0.01  # 0 is central

    def test_coverage_truth_outside(self):
        pred = RNG.normal(size=(500, 10))
        assert eval_coverage(pred, np.full(10, 100.0)) == 0.0

    def test_alpha_validated(self, gaussian_pred):
        pred, truth = gaussian_pred
        with pytest.raises(ValueError, match="alpha"):
            eval_coverage(pred, truth, alpha=1.5)

    def test_pinball_half_mae_at_median(self, gaussian_pred):
        pred, truth = gaussian_pred
        pinball = eval_pinball(pred, truth, quantile=0.5)
        mae = eval_mae(pred, truth)
        np.testing.assert_allclose(pinball, 0.5 * mae, atol=1e-10)

    def test_interval_score_penalizes_misses(self):
        pred = RNG.normal(size=(500, 10))
        inside = eval_interval_score(pred, np.zeros(10))
        outside = eval_interval_score(pred, np.full(10, 10.0))
        assert outside > inside


class TestMase:
    def test_scaling(self):
        train = np.arange(10.0)[:, None]  # naive |diff| scale = 1.0
        mase = make_mase(train)
        pred = np.zeros((5, 3, 1))
        truth = np.full((3, 1), 2.0)
        assert mase(pred, truth) == pytest.approx(2.0)

    def test_dataarray_time_dim(self):
        train = xr.DataArray(np.arange(10.0), dims=("time",))
        assert make_mase(train)(np.zeros((5, 2)), np.full(2, 3.0)) == pytest.approx(3.0)

    def test_constant_series_rejected(self):
        with pytest.raises(ValueError, match="scale is zero"):
            make_mase(np.ones(5))

    def test_seasonality_validated(self):
        with pytest.raises(ValueError, match="seasonality"):
            make_mase(np.arange(10.0), seasonality=0)
        with pytest.raises(ValueError, match="longer than seasonality"):
            make_mase(np.arange(3.0), seasonality=5)


class TestLabeledInputs:
    def test_chain_draw_dims_stacked(self):
        pred = xr.DataArray(
            RNG.normal(size=(2, 50, 4)),
            dims=("chain", "draw", "time_future"),
        )
        truth = xr.DataArray(np.zeros(4), dims=("time_future",))
        labeled = eval_mae(pred, truth)
        raw = eval_mae(pred.values.reshape(100, 4), np.zeros(4))
        assert abs(labeled - raw) < 1e-12

    def test_truth_transposed_by_name(self):
        pred = xr.DataArray(np.zeros((10, 3, 2)), dims=("draw", "time_future", "series"))
        truth_transposed = xr.DataArray(
            np.arange(6.0).reshape(2, 3), dims=("series", "time_future")
        )
        # would be a shape error if alignment were positional
        assert eval_mae(pred, truth_transposed) == pytest.approx(np.arange(6.0).mean())

    def test_missing_sample_dim_rejected(self):
        pred = xr.DataArray(np.zeros((3, 2)), dims=("time_future", "series"))
        with pytest.raises(ValueError, match="sample dimension"):
            eval_mae(pred, np.zeros((3, 2)))


class TestEvaluateForecast:
    def test_default_metrics(self, gaussian_pred):
        pred, truth = gaussian_pred
        result = evaluate_forecast(pred, truth)
        assert set(result) == set(DEFAULT_METRICS)
        assert all(isinstance(v, float) for v in result.values())

    def test_partial_binding(self, gaussian_pred):
        pred, truth = gaussian_pred
        result = evaluate_forecast(
            pred, truth, metrics={"cov80": partial(eval_coverage, alpha=0.8)}
        )
        assert abs(result["cov80"] - 1.0) < 0.05

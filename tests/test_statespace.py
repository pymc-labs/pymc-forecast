"""Tests for the pymc-extras statespace interop (StatespaceForecaster)."""

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from example_models import LocalLevelRegressionStatespace, LocalLevelStatespace

from pymc_forecast.evaluate import backtest
from pymc_forecast.exceptions import AlignmentError, HorizonError, OptionalDependencyError
from pymc_forecast.forecaster import HMCForecaster
from pymc_forecast.statespace import StatespaceForecaster

SEED = 987
FAST = {"draws": 40, "tune": 40, "chains": 2}


def make_local_level_data(t_obs=25, horizon=5, seed=SEED, datetime=False):
    """(data, covariates_full) pair from a simulated local-level process."""
    rng = np.random.default_rng(seed)
    duration = t_obs + horizon
    level = np.cumsum(rng.normal(0.1, 0.15, t_obs))
    index = (
        pd.date_range("2024-01-01", periods=duration, freq="D") if datetime else np.arange(duration)
    )
    data = xr.DataArray(
        level + rng.normal(0, 0.1, t_obs),
        dims=("time",),
        coords={"time": index[:t_obs]},
    )
    covariates = xr.DataArray(
        np.zeros((duration, 0)),
        dims=("time", "covariate"),
        coords={"time": index},
    )
    return data, covariates


@pytest.fixture(scope="module")
def data_and_cov():
    return make_local_level_data()


@pytest.fixture(scope="module")
def forecaster(data_and_cov):
    data, cov = data_and_cov
    return StatespaceForecaster(LocalLevelStatespace(), data, cov, random_seed=SEED, **FAST)


class TestForecast:
    def test_predictions_group_with_future_coords(self, forecaster, data_and_cov):
        _, cov = data_and_cov
        tree = forecaster.forecast(cov, num_samples=30, random_seed=SEED)
        fc = tree["predictions"]["forecast"]
        assert fc.dims == ("chain", "draw", "time_future")
        assert fc.sizes["draw"] == 30
        np.testing.assert_array_equal(fc["time_future"].values, np.arange(25, 30))
        assert np.isfinite(fc.values).all()

    def test_latent_states_included(self, forecaster, data_and_cov):
        _, cov = data_and_cov
        tree = forecaster.forecast(cov, num_samples=10, random_seed=SEED)
        latent = tree["predictions"]["forecast_latent"]
        assert "time_future" in latent.dims and "state" in latent.dims

    def test_horizon_shortcut(self, forecaster):
        tree = forecaster.forecast(horizon=3, num_samples=10, random_seed=SEED)
        np.testing.assert_array_equal(
            tree["predictions"]["forecast"]["time_future"].values, np.arange(25, 28)
        )

    def test_forecast_continues_the_level(self, forecaster, data_and_cov):
        data, cov = data_and_cov
        tree = forecaster.forecast(cov, num_samples=100, random_seed=SEED)
        first_step = tree["predictions"]["forecast"].isel(time_future=0).mean()
        assert abs(float(first_step) - float(data[-1])) < 1.0

    def test_covariates_and_horizon_mutually_exclusive(self, forecaster, data_and_cov):
        _, cov = data_and_cov
        with pytest.raises(ValueError, match="exactly one"):
            forecaster.forecast(cov, horizon=3)
        with pytest.raises(ValueError, match="exactly one"):
            forecaster.forecast()

    def test_no_horizon_raises(self, forecaster, data_and_cov):
        data, cov = data_and_cov
        with pytest.raises(HorizonError, match="no forecast horizon"):
            forecaster.forecast(cov.isel(time=slice(None, data.sizes["time"])))

    def test_var_names_selects_subset(self, forecaster):
        tree = forecaster.forecast(horizon=2, num_samples=10, var_names=["forecast"])
        assert list(tree["predictions"].data_vars) == ["forecast"]

    def test_unknown_var_names_raise(self, forecaster):
        with pytest.raises(KeyError, match="unknown statespace prediction"):
            forecaster.forecast(horizon=2, num_samples=10, var_names=["nope"])

    def test_draw_posterior_protocol(self, forecaster):
        posterior = forecaster.draw_posterior(17, random_seed=SEED)
        assert posterior.sizes["chain"] == 1 and posterior.sizes["draw"] == 17

    def test_is_an_hmc_forecaster(self, forecaster):
        # not just duck-typed: fit and draw_posterior are inherited
        assert isinstance(forecaster, HMCForecaster)


def make_regression_data(t_obs=20, horizon=4, seed=SEED):
    """(data, covariates_full) pair: local level plus one exogenous feature."""
    rng = np.random.default_rng(seed)
    time = np.arange(t_obs + horizon)
    feature = np.linspace(-1.0, 1.0, time.size)
    covariates = xr.DataArray(
        feature[:, None],
        dims=("time", "covariate"),
        coords={"time": time, "covariate": ["x"]},
    )
    data = xr.DataArray(
        np.cumsum(rng.normal(0.0, 0.05, t_obs)) + 1.5 * feature[:t_obs],
        dims=("time",),
        coords={"time": time[:t_obs]},
    )
    return data, covariates


@pytest.fixture(scope="module")
def regression_forecaster():
    data, cov = make_regression_data()
    forecaster = StatespaceForecaster(
        LocalLevelRegressionStatespace(),
        data,
        cov,
        random_seed=SEED,
        # cholesky is fine for the likelihood graph; the forecast covariance is
        # singular (regression states carry no innovations) so it needs svd.
        # Together these exercise the build_kwargs/forecast_kwargs passthroughs.
        build_kwargs={"mvn_method": "cholesky"},
        forecast_kwargs={"mvn_method": "svd"},
        **FAST,
    )
    return forecaster, data, cov


class TestRegressionCovariates:
    """Future covariate values reach the statespace forecast as the scenario."""

    def test_future_covariates_flow_through(self, regression_forecaster):
        forecaster, _, cov = regression_forecaster
        tree = forecaster.forecast(cov, num_samples=20, random_seed=SEED)
        fc = tree["predictions"]["forecast"]
        assert fc.dims == ("chain", "draw", "time_future")
        np.testing.assert_array_equal(fc["time_future"].values, cov["time"].values[20:])
        assert np.isfinite(fc.values).all()

    def test_future_only_covariates_flow_through(self, regression_forecaster):
        forecaster, _, cov = regression_forecaster
        future_covariates = cov.isel(time=slice(20, None))
        tree = forecaster.forecast(
            future_covariates=future_covariates,
            num_samples=20,
            random_seed=SEED,
        )
        fc = tree["predictions"]["forecast"]
        np.testing.assert_array_equal(fc["time_future"], future_covariates["time"])
        assert np.isfinite(fc.values).all()

    def test_horizon_shortcut_rejected_without_future_values(self, regression_forecaster):
        forecaster, _, _ = regression_forecaster
        with pytest.raises(AlignmentError, match="needs future values"):
            forecaster.forecast(horizon=3, num_samples=10)


class TestConstructionValidation:
    """Bad inputs fail before the (expensive) MCMC fit."""

    def test_three_dimensional_data_rejected(self):
        data = xr.DataArray(np.zeros((5, 2, 2)), dims=("time", "a", "b"))
        with pytest.raises(AlignmentError, match="one- or two-dimensional"):
            StatespaceForecaster(LocalLevelStatespace(), data)

    def test_misaligned_covariates_rejected(self, data_and_cov):
        data, cov = data_and_cov
        with pytest.raises(AlignmentError, match="covariates must extend data"):
            StatespaceForecaster(LocalLevelStatespace(), data, cov.isel(time=slice(None, 3)))

    def test_observed_width_mismatch_rejected(self):
        data = xr.DataArray(
            np.zeros((10, 2)),
            dims=("time", "series"),
            coords={"time": np.arange(10), "series": ["a", "b"]},
        )
        with pytest.raises(AlignmentError, match="k_endog=1, data width=2"):
            StatespaceForecaster(LocalLevelStatespace(), data)

    def test_reserved_forecast_kwargs_rejected(self, data_and_cov):
        data, _ = data_and_cov
        with pytest.raises(ValueError, match="adapter-managed"):
            StatespaceForecaster(LocalLevelStatespace(), data, forecast_kwargs={"periods": 3})


class TestPredictInSample:
    def test_obs_with_time_coords(self, forecaster, data_and_cov):
        data, _ = data_and_cov
        tree = forecaster.predict_in_sample(num_samples=20, random_seed=SEED)
        obs = tree["posterior_predictive"]["obs"]
        assert obs.dims == ("chain", "draw", "time")
        np.testing.assert_array_equal(obs["time"].values, data["time"].values)
        # smoothed in-sample predictive should track the observed series
        residual = np.abs(obs.mean(("chain", "draw")).values - data.values).mean()
        assert residual < 0.5


@pytest.fixture(scope="module")
def dt_forecaster():
    data, cov = make_local_level_data(t_obs=20, horizon=4, datetime=True)
    forecaster = StatespaceForecaster(
        LocalLevelStatespace(), data, random_seed=SEED, draws=20, tune=20, chains=2
    )
    return forecaster, data, cov


class TestDatetimeCoords:
    def test_covariate_path_stamps_dates(self, dt_forecaster):
        forecaster, _, cov = dt_forecaster
        tree = forecaster.forecast(cov, num_samples=10, random_seed=SEED)
        np.testing.assert_array_equal(
            tree["predictions"]["forecast"]["time_future"].values, cov["time"].values[20:]
        )

    def test_horizon_path_extends_dates(self, dt_forecaster):
        forecaster, data, _ = dt_forecaster
        tree = forecaster.forecast(horizon=2, num_samples=10, random_seed=SEED)
        expected = pd.date_range(data["time"].values[-1], periods=3, freq="D")[1:]
        np.testing.assert_array_equal(
            tree["predictions"]["forecast"]["time_future"].values, expected.values
        )


class TestBacktestInterop:
    """A statespace model backtests through the same call as a hand-written one."""

    def test_backtest_same_call_and_metrics(self):
        data, _ = make_local_level_data(t_obs=28, horizon=0)
        results = backtest(
            data,
            None,
            LocalLevelStatespace(),
            forecaster_cls=StatespaceForecaster,
            forecaster_options=FAST,
            min_train_window=24,
            test_window=4,
            stride=10,
            num_samples=60,
            eval_train=True,
            keep_predictions=True,
            random_seed=SEED,
        )
        assert [(r.t0, r.t1, r.t2) for r in results] == [(0, 24, 28)]
        result = results[0]
        assert set(result.metrics) == {"mae", "rmse", "crps", "coverage"}
        assert all(np.isfinite(v) for v in result.metrics.values())
        assert result.metrics["mae"] < 2.0
        assert set(result.train_metrics) == {"mae", "rmse", "crps", "coverage"}
        np.testing.assert_array_equal(result.prediction["time_future"].values, np.arange(24, 28))


class TestOptionalDependency:
    def test_missing_pymc_extras(self, monkeypatch, data_and_cov):
        data, _ = data_and_cov
        monkeypatch.setitem(sys.modules, "pymc_extras.statespace", None)
        with pytest.raises(OptionalDependencyError, match="pymc-extras"):
            StatespaceForecaster(LocalLevelStatespace(), data)

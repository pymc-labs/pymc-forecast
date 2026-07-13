"""pymc-extras state-space models satisfy the common forecast/backtest API."""

import builtins

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from pymc_forecast.evaluate import backtest
from pymc_forecast.exceptions import OptionalDependencyError
from pymc_forecast.statespace import StatespaceForecaster

SEED = 818


def local_level_statespace(data: xr.DataArray, covariates: xr.DataArray):
    """Build the pymc-extras equivalent of a noisy local-level model."""
    from pymc_extras.statespace.models import structural as st

    statespace_model = (st.LevelTrend(order=1, innovations_order=1) + st.MeasurementError()).build(
        verbose=False
    )
    with pm.Model(coords=statespace_model.coords) as model:
        pm.Normal(
            "initial_level_trend",
            mu=float(data.isel(time=0)),
            sigma=1.0,
            dims=statespace_model.param_dims["initial_level_trend"],
        )
        pm.HalfNormal(
            "sigma_level_trend",
            sigma=0.5,
            dims=statespace_model.param_dims["sigma_level_trend"],
        )
        pm.HalfNormal("sigma_MeasurementError", sigma=0.5)
        pm.Deterministic(
            "P0",
            pt.eye(statespace_model.k_states),
            dims=statespace_model.param_dims["P0"],
        )
        statespace_model.build_statespace_graph(data.to_pandas(), mvn_method="cholesky")
    return statespace_model, model


def local_level_regression_statespace(data: xr.DataArray, covariates: xr.DataArray):
    """Add a single statespace Regression component fed by package covariates."""
    from pymc_extras.statespace.models import structural as st

    feature_names = covariates["covariate"].values.tolist()
    statespace_model = (
        st.LevelTrend(order=1, innovations_order=1)
        + st.Regression(state_names=feature_names)
        + st.MeasurementError()
    ).build(verbose=False)
    with pm.Model(coords=statespace_model.coords) as model:
        pm.Data("data_regression", covariates.values)
        pm.Normal(
            "initial_level_trend",
            mu=float(data.isel(time=0)),
            sigma=1.0,
            dims=statespace_model.param_dims["initial_level_trend"],
        )
        pm.HalfNormal(
            "sigma_level_trend",
            sigma=0.5,
            dims=statespace_model.param_dims["sigma_level_trend"],
        )
        pm.Normal("beta_regression", dims=statespace_model.param_dims["beta_regression"])
        pm.HalfNormal("sigma_MeasurementError", sigma=0.5)
        pm.Deterministic(
            "P0",
            pt.eye(statespace_model.k_states),
            dims=statespace_model.param_dims["P0"],
        )
        statespace_model.build_statespace_graph(data.to_pandas(), mvn_method="cholesky")
    return statespace_model, model


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data")
@pytest.mark.filterwarnings("ignore:Only .* samples per chain")
def test_statespace_model_uses_backtest_and_metrics():
    rng = np.random.default_rng(SEED)
    latent = np.cumsum(rng.normal(0.1, 0.08, 15))
    data = xr.DataArray(
        latent + rng.normal(0.0, 0.04, latent.size),
        dims="time",
        coords={"time": np.arange(latent.size)},
    )

    results = backtest(
        data,
        None,
        local_level_statespace,
        forecaster_cls=StatespaceForecaster,
        min_train_window=12,
        test_window=3,
        stride=3,
        num_samples=10,
        forecaster_options={
            "draws": 10,
            "tune": 10,
            "chains": 1,
            "sample_kwargs": {"cores": 1, "compute_convergence_checks": False},
            "forecast_kwargs": {"mvn_method": "cholesky"},
        },
        eval_train=True,
        keep_predictions=True,
        random_seed=SEED,
    )

    assert len(results) == 1
    result = results[0]
    assert set(result.metrics) == {"mae", "rmse", "crps", "coverage"}
    assert all(np.isfinite(score) for score in result.metrics.values())
    assert all(np.isfinite(score) for score in result.train_metrics.values())
    assert result.prediction.dims == ("chain", "draw", "time_future")
    np.testing.assert_array_equal(result.prediction["time_future"], np.arange(12, 15))


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data")
@pytest.mark.filterwarnings("ignore:Only .* samples per chain")
def test_statespace_regression_uses_future_covariates():
    rng = np.random.default_rng(SEED)
    time = np.arange(15)
    feature = np.linspace(-1.0, 1.0, time.size)
    covariates = xr.DataArray(
        feature[:, None],
        dims=("time", "covariate"),
        coords={"time": time, "covariate": ["x"]},
    )
    data = xr.DataArray(
        np.cumsum(rng.normal(0.0, 0.05, 12)) + 1.5 * feature[:12],
        dims="time",
        coords={"time": time[:12]},
    )
    forecaster = StatespaceForecaster(
        local_level_regression_statespace,
        data,
        covariates,
        draws=5,
        tune=5,
        chains=1,
        random_seed=SEED,
        sample_kwargs={"cores": 1, "compute_convergence_checks": False},
    )

    prediction = forecaster.forecast(covariates, num_samples=5, random_seed=SEED)["predictions"][
        "forecast"
    ]
    assert prediction.dims == ("chain", "draw", "time_future")
    np.testing.assert_array_equal(prediction["time_future"], time[12:])
    assert np.isfinite(prediction).all()


def test_statespace_dependency_is_lazy(monkeypatch):
    real_import = builtins.__import__

    def missing_extras(name, *args, **kwargs):
        if name == "pymc_extras.statespace.core":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_extras)
    with pytest.raises(OptionalDependencyError, match="pymc-extras"):
        StatespaceForecaster(lambda data, covariates: None, np.arange(5.0))


def test_statespace_builder_contract_is_checked():
    with pytest.raises(TypeError, match="must return"):
        StatespaceForecaster(lambda data, covariates: None, np.arange(5.0))

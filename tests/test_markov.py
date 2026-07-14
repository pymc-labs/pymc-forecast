"""Scan-based Markov latents: logp derivation, replay, forecast continuity."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc_forecast.exceptions import HorizonError
from pymc_forecast.forecaster import HMCForecaster
from pymc_forecast.markov import markov_time_series
from pymc_forecast.model import Horizon, build_model, predict

SEED = 777
T_OBS = 40
HORIZON = 5
DRIFT = 0.15


def rw_model(h: Horizon, covariates: xr.DataArray) -> None:
    """Latent random walk with drift, observed with Normal noise."""
    drift = pm.Normal("drift", 0.0, 0.5)
    sigma = pm.HalfNormal("sigma", 0.2)
    level = markov_time_series(
        h,
        "level",
        init=0.0,
        transition=lambda z, drift, sigma: pm.Normal.dist(z + drift, sigma),
        params=(drift, sigma),
    )
    predict(
        h,
        lambda name, m, dims, observed: pm.Normal(name, m, 0.05, dims=dims, observed=observed),
        level,
    )


@pytest.fixture(scope="module")
def rw_data():
    rng = np.random.default_rng(SEED)
    data = xr.DataArray(
        np.cumsum(rng.normal(DRIFT, 0.05, T_OBS)),
        dims=("time",),
        coords={"time": np.arange(T_OBS)},
    )
    covariates = xr.DataArray(
        np.zeros((T_OBS + HORIZON, 0)),
        dims=("time", "covariate"),
        coords={"time": np.arange(T_OBS + HORIZON)},
    )
    return data, covariates


class TestModelConstruction:
    def test_train_build_shapes_and_logp(self, rw_data):
        data, cov = rw_data
        model = build_model(rw_model, data, cov.isel(time=slice(None, T_OBS)))
        assert model["level"].eval().shape == (T_OBS,)
        point = model.initial_point()
        assert np.isfinite(model.compile_logp()(point))

    def test_forecast_build_has_future_segment(self, rw_data):
        data, cov = rw_data
        model = build_model(rw_model, data, cov)
        names = {rv.name for rv in model.free_RVs}
        assert {"level", "level_future"} <= names
        assert model["level_future"].eval().shape == (HORIZON,)

    def test_forecast_without_data_rejected(self, rw_data):
        h = Horizon(data=None, time=np.arange(T_OBS), time_future=np.arange(T_OBS, T_OBS + 2))
        with pm.Model(coords={"time": h.time, "time_future": h.time_future}):
            with pytest.raises(HorizonError, match="requires observed data"):
                markov_time_series(h, "z", 0.0, lambda z: pm.Normal.dist(z, 1.0))

    def test_xs_must_span_horizon(self, rw_data):
        data, cov = rw_data

        def model_fn(h, covariates):
            markov_time_series(
                h,
                "z",
                0.0,
                lambda z, x: pm.Normal.dist(z + x, 1.0),
                xs=np.zeros(3),  # wrong length
            )

        with pytest.raises(HorizonError, match="cover at least the horizon"):
            build_model(model_fn, data, cov)


class TestForecastContinuity:
    @pytest.fixture(scope="module")
    def forecaster(self, rw_data):
        data, cov = rw_data
        return HMCForecaster(rw_model, data, cov, draws=150, tune=200, chains=2, random_seed=SEED)

    def test_forecast_continues_from_state(self, forecaster, rw_data):
        """The future scan is seeded by the replayed final in-sample state."""
        data, cov = rw_data
        pred = forecaster.forecast(cov, num_samples=200, random_seed=SEED)["predictions"]
        first = float(pred["forecast"].isel(time_future=0).mean())
        last_level = float(data.values[-1])
        # broken conditioning-through-the-carry restarts the walk near 0
        assert abs(first - (last_level + DRIFT)) < 0.3

    def test_drift_recovered(self, forecaster):
        drift_hat = float(forecaster.idata["posterior"]["drift"].mean())
        assert abs(drift_hat - DRIFT) < 0.05

    def test_future_steps_accumulate_drift(self, forecaster, rw_data):
        _, cov = rw_data
        pred = forecaster.forecast(cov, num_samples=300, random_seed=SEED)["predictions"]
        means = pred["forecast"].mean(("chain", "draw")).values
        increments = np.diff(means)
        np.testing.assert_allclose(increments, DRIFT, atol=0.1)


class TestExogenousInputs:
    def test_xs_threaded_through_transition(self, rw_data):
        data, cov = rw_data
        impulses = np.zeros(T_OBS + HORIZON)
        impulses[T_OBS:] = 5.0  # future-only impulse

        def model_fn(h, covariates):
            sigma = pm.HalfNormal("sigma", 0.2)
            level = markov_time_series(
                h,
                "level",
                init=0.0,
                transition=lambda z, x, sigma: pm.Normal.dist(z + x, sigma),
                params=(sigma,),
                xs=impulses,
            )
            predict(
                h,
                lambda name, m, dims, observed: pm.Normal(
                    name, m, 0.05, dims=dims, observed=observed
                ),
                level,
            )

        fc = HMCForecaster(model_fn, data, cov, draws=100, tune=150, chains=2, random_seed=SEED)
        pred = fc.forecast(cov, num_samples=100, random_seed=SEED)["predictions"]
        means = pred["forecast"].mean(("chain", "draw")).values
        np.testing.assert_allclose(np.diff(means), 5.0, atol=0.5)

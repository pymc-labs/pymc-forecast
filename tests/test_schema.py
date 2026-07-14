"""Contract tests for the documented prediction output schema (docs/schema.md).

These lock the public dim/coord/group/variable names; a failure here means a
breaking schema change that needs a changelog entry and a minor release.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from example_models import hierarchical_model, random_walk_model

import pymc_forecast
from pymc_forecast import Forecaster, null_covariates, prediction_samples

SEED = 20260714
T_OBS = 20
HORIZON = 4


def test_public_name_constants():
    assert pymc_forecast.TIME_DIM == "time"
    assert pymc_forecast.FUTURE_DIM == "time_future"
    assert pymc_forecast.OBS_VAR == "obs"
    assert pymc_forecast.FORECAST_VAR == "forecast"
    assert pymc_forecast.MU_VAR == "mu"
    assert pymc_forecast.MU_FORECAST_VAR == "mu_future"
    assert pymc_forecast.CHAIN_DIM == "chain"
    assert pymc_forecast.DRAW_DIM == "draw"
    assert pymc_forecast.SAMPLE_DIMS == ("chain", "draw")


@pytest.fixture(scope="module")
def univariate():
    """Random-walk forecaster on a datetime-indexed series, plus full index."""
    rng = np.random.default_rng(SEED)
    index = pd.date_range("2026-01-05", periods=T_OBS + HORIZON, freq="W")
    data = xr.DataArray(
        np.cumsum(rng.normal(0.1, 0.05, T_OBS)),
        dims=("time",),
        coords={"time": index[:T_OBS]},
    )
    fc = Forecaster(random_walk_model, data, num_steps=300, random_seed=SEED)
    return fc, index


@pytest.fixture(scope="module")
def hierarchical():
    """Hierarchical forecaster on (time, series) data with named series."""
    rng = np.random.default_rng(SEED)
    data = xr.DataArray(
        np.cumsum(rng.normal(0.1, 0.05, (T_OBS, 2)), axis=0),
        dims=("time", "series"),
        coords={"time": np.arange(T_OBS), "series": ["north", "south"]},
    )
    cov = null_covariates(np.arange(T_OBS + HORIZON))
    fc = Forecaster(hierarchical_model, data, cov, num_steps=300, random_seed=SEED)
    return fc, cov


class TestForecastSchema:
    def test_predictions_group_and_dims(self, univariate):
        fc, index = univariate
        result = fc.forecast(null_covariates(index), num_samples=25, random_seed=SEED)
        pred = result["predictions"]
        assert pred["forecast"].dims == ("chain", "draw", "time_future")
        assert pred["forecast"].sizes["draw"] == 25

    def test_time_future_coords_are_real(self, univariate):
        fc, index = univariate
        result = fc.forecast(null_covariates(index), num_samples=10, random_seed=SEED)
        coords = result["predictions"]["time_future"].values
        np.testing.assert_array_equal(coords, index[T_OBS:].values)
        assert np.issubdtype(coords.dtype, np.datetime64)

    def test_future_latents_share_the_schema(self, univariate):
        fc, index = univariate
        pred = fc.forecast(null_covariates(index), num_samples=10, random_seed=SEED)["predictions"]
        assert pred["drift_future"].dims == ("chain", "draw", "time_future")

    def test_batch_dims_follow_time(self, hierarchical):
        fc, cov = hierarchical
        result = fc.forecast(cov, num_samples=10, random_seed=SEED)
        forecast = result["predictions"]["forecast"]
        assert forecast.dims == ("chain", "draw", "time_future", "series")
        assert list(forecast["series"].values) == ["north", "south"]

    def test_noise_free_mu_future(self, univariate):
        fc, index = univariate
        pred = fc.forecast(null_covariates(index), num_samples=10, random_seed=SEED)["predictions"]
        assert pred["mu_future"].dims == ("chain", "draw", "time_future")
        # noise-free: strictly narrower than the predictive draws
        assert float(pred["mu_future"].std("draw").mean()) < float(
            pred["forecast"].std("draw").mean()
        )

    def test_mu_future_carries_batch_dims(self, hierarchical):
        fc, cov = hierarchical
        pred = fc.forecast(cov, num_samples=10, random_seed=SEED)["predictions"]
        assert pred["mu_future"].dims == ("chain", "draw", "time_future", "series")


class TestInSampleSchema:
    def test_posterior_predictive_group_and_dims(self, univariate):
        fc, _ = univariate
        result = fc.predict_in_sample(num_samples=15, random_seed=SEED)
        obs = result["posterior_predictive"]["obs"]
        assert obs.dims == ("chain", "draw", "time")
        assert obs.sizes == {"chain": 1, "draw": 15, "time": T_OBS}

    def test_time_coords_match_training_data(self, univariate):
        fc, index = univariate
        result = fc.predict_in_sample(num_samples=5, random_seed=SEED)
        np.testing.assert_array_equal(
            result["posterior_predictive"]["time"].values, index[:T_OBS].values
        )

    def test_noise_free_mu(self, univariate):
        fc, _ = univariate
        result = fc.predict_in_sample(num_samples=15, random_seed=SEED)
        mu = result["posterior_predictive"]["mu"]
        assert mu.dims == ("chain", "draw", "time")
        assert mu.sizes == {"chain": 1, "draw": 15, "time": T_OBS}

    def test_mu_carries_batch_dims(self, hierarchical):
        fc, _ = hierarchical
        result = fc.predict_in_sample(num_samples=10, random_seed=SEED)
        assert result["posterior_predictive"]["mu"].dims == ("chain", "draw", "time", "series")


class TestPredictionSamplesRemap:
    def test_documented_remap_is_a_one_liner(self, hierarchical):
        fc, cov = hierarchical
        result = fc.forecast(cov, num_samples=10, random_seed=SEED)
        samples = prediction_samples(result)["forecast"]
        remapped = samples.rename({"time_future": "obs_ind", "series": "treated_units"})
        assert remapped.dims == ("chain", "draw", "obs_ind", "treated_units")

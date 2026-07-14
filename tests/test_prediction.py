import numpy as np
import pymc as pm
import pytest
import xarray as xr
from example_models import linear_model, make_trend_data

from pymc_forecast.data import TIME_DIM
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import build_model
from pymc_forecast.prediction import (
    forecast,
    posterior_dataset,
    predict_in_sample,
    thin_draws,
)

SEED = 99


@pytest.fixture(scope="module")
def fitted():
    data, cov = make_trend_data()
    model = build_model(linear_model, data, cov.isel({TIME_DIM: slice(None, 30)}))
    with model:
        idata = pm.sample(draws=150, tune=150, chains=2, progressbar=False, random_seed=SEED)
    return data, cov, idata


class TestThinDraws:
    def test_shape_and_determinism(self, fitted):
        _, _, idata = fitted
        thin = thin_draws(idata, 40, random_seed=7)
        assert thin.sizes["chain"] == 1 and thin.sizes["draw"] == 40
        again = thin_draws(idata, 40, random_seed=7)
        xr.testing.assert_identical(thin, again)

    def test_oversampling_uses_replacement(self, fitted):
        _, _, idata = fitted
        thin = thin_draws(idata, 500, random_seed=7)
        assert thin.sizes["draw"] == 500

    def test_positive_num_samples_required(self, fitted):
        _, _, idata = fitted
        with pytest.raises(ValueError, match="positive"):
            thin_draws(idata, 0)

    def test_accepts_bare_dataset(self, fitted):
        _, _, idata = fitted
        ds = posterior_dataset(idata)
        assert thin_draws(ds, 10, random_seed=0).sizes["draw"] == 10


class TestForecast:
    def test_predictions_group_and_coords(self, fitted):
        data, cov, idata = fitted
        result = forecast(linear_model, idata, data, cov, num_samples=50, random_seed=SEED)
        pred = result["predictions"]
        assert pred["forecast"].sizes["time_future"] == 5
        np.testing.assert_array_equal(pred["time_future"].values, np.arange(30, 35))
        assert pred["forecast"].sizes["draw"] == 50

    def test_full_draws_are_exposed_as_posterior_predictive(self, fitted):
        data, cov, idata = fitted
        result = forecast(linear_model, idata, data, cov, num_samples=50, random_seed=SEED)
        predictions = result["predictions"]
        posterior_predictive = result["posterior_predictive"]
        if hasattr(predictions, "to_dataset"):
            predictions = predictions.to_dataset()
            posterior_predictive = posterior_predictive.to_dataset()
        xr.testing.assert_identical(posterior_predictive, predictions)
        assert posterior_predictive["forecast"].dims == (
            "chain",
            "draw",
            "time_future",
        )
        assert posterior_predictive.sizes["draw"] == 50

    def test_recovers_known_trend(self, fitted):
        # data = 1 + 2 * trend + noise; the forecast must track the truth.
        data, cov, idata = fitted
        result = forecast(linear_model, idata, data, cov, num_samples=200, random_seed=SEED)
        mean = result["predictions"]["forecast"].mean(("chain", "draw")).values
        truth = 1.0 + 2.0 * cov.values[30:, 0]
        np.testing.assert_allclose(mean, truth, atol=0.2)

    def test_no_horizon_raises(self, fitted):
        data, cov, idata = fitted
        with pytest.raises(HorizonError, match="no forecast horizon"):
            forecast(linear_model, idata, data, cov.isel({TIME_DIM: slice(None, 30)}))


class TestPredictInSample:
    def test_obs_group_and_fit(self, fitted):
        data, cov, idata = fitted
        result = predict_in_sample(
            linear_model, idata, data, cov, num_samples=100, random_seed=SEED
        )
        ppc = result["posterior_predictive"]["obs"]
        assert ppc.sizes["time"] == 30
        resid = ppc.mean(("chain", "draw")).values - data.values
        assert np.abs(resid).mean() < 0.15

    def test_full_horizon_covariates_are_truncated(self, fitted):
        data, cov, idata = fitted
        result = predict_in_sample(linear_model, idata, data, cov, num_samples=20)
        assert result["posterior_predictive"]["obs"].sizes["time"] == 30

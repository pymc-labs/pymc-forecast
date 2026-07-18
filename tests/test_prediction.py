import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from example_models import linear_model, make_trend_data, poisson_model

from pymc_forecast.data import TIME_DIM
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import build_model
from pymc_forecast.prediction import (
    forecast,
    posterior_dataset,
    predict_in_sample,
    prediction_samples,
    thin_draws,
)

SEED = 99


def custom_forecast_model(h, covariates):
    """Register obs/forecast directly, without the standard predict() helper."""
    level = pm.Normal("level")
    observed = None if h.data is None else h.data.values
    pm.Normal("obs", level, 1, observed=observed, dims="time")
    if h.future:
        pm.Deterministic("forecast", pt.repeat(level, h.future), dims="time_future")


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

    def test_custom_forecast_model_without_mu_future(self):
        # models that register obs/forecast themselves have no mu_future;
        # the default var_names must not demand it
        data, cov = make_trend_data()
        posterior = xr.Dataset(
            {"level": (("chain", "draw"), np.array([[1.0, 2.0]]))},
            coords={"chain": [0], "draw": [0, 1]},
        )
        result = forecast(custom_forecast_model, posterior, data, cov, random_seed=SEED)
        assert set(result["predictions"].data_vars) == {"forecast"}
        assert result["predictions"]["forecast"].dims == ("chain", "draw", "time_future")


class TestExpectedObservation:
    def test_poisson_log_link_keeps_eta_and_emits_expected_counts(self):
        covariates = xr.DataArray(
            np.linspace(-0.5, 0.5, 5)[:, None],
            dims=("time", "covariate"),
            coords={"time": np.arange(5), "covariate": ["x"]},
        )
        data = xr.DataArray([1, 2, 1], dims="time", coords={"time": np.arange(3)})
        posterior = xr.Dataset(
            {
                "intercept": (("chain", "draw"), [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]),
                "beta": (
                    ("chain", "draw", "covariate"),
                    [[[0.2], [0.3], [0.4]], [[0.5], [0.6], [0.7]]],
                ),
            },
            coords={"chain": [2, 4], "draw": [10, 20, 30], "covariate": ["x"]},
        )

        pre = prediction_samples(
            predict_in_sample(poisson_model, posterior, data, covariates, random_seed=SEED)
        )
        post = prediction_samples(
            forecast(poisson_model, posterior, data, covariates, random_seed=SEED)
        )
        eta = (
            posterior["intercept"].values[..., None]
            + posterior["beta"].values[..., 0, None] * covariates.values[:, 0]
        )

        assert pre["mu"].dims == ("chain", "draw", "time")
        assert pre["expected_observation"].dims == ("chain", "draw", "time")
        assert post["mu_future"].dims == ("chain", "draw", "time_future")
        assert post["expected_observation_future"].dims == (
            "chain",
            "draw",
            "time_future",
        )
        np.testing.assert_allclose(pre["mu"], eta[..., :3])
        np.testing.assert_allclose(pre["expected_observation"], np.exp(eta[..., :3]))
        np.testing.assert_allclose(post["mu_future"], eta[..., 3:])
        np.testing.assert_allclose(post["expected_observation_future"], np.exp(eta[..., 3:]))
        np.testing.assert_array_equal(pre["chain"], posterior["chain"])
        np.testing.assert_array_equal(pre["draw"], posterior["draw"])
        np.testing.assert_array_equal(post["chain"], posterior["chain"])
        np.testing.assert_array_equal(post["draw"], posterior["draw"])
        np.testing.assert_array_equal(pre["time"], data["time"])
        np.testing.assert_array_equal(post["time_future"], covariates["time"][3:])


class TestDrawLevelSamples:
    """The issue #20 contract: full posterior-predictive samples, no reduction."""

    def test_forecast_retains_chain_and_draw(self, fitted):
        data, cov, idata = fitted
        result = forecast(linear_model, idata, data, cov, num_samples=50, random_seed=SEED)
        fc = result["predictions"]["forecast"]
        assert fc.dims == ("chain", "draw", "time_future")
        assert fc.sizes["draw"] == 50
        # genuinely distinct draws, not a broadcast point forecast
        assert float(fc.std(("chain", "draw")).min()) > 0

    def test_in_sample_retains_chain_and_draw(self, fitted):
        data, cov, idata = fitted
        result = predict_in_sample(linear_model, idata, data, cov, num_samples=40)
        obs = result["posterior_predictive"]["obs"]
        assert obs.dims == ("chain", "draw", "time")
        assert obs.sizes["draw"] == 40
        assert float(obs.std(("chain", "draw")).min()) > 0

    def test_prediction_samples_from_forecast_result(self, fitted):
        data, cov, idata = fitted
        result = forecast(linear_model, idata, data, cov, num_samples=30, random_seed=SEED)
        ds = prediction_samples(result)
        assert isinstance(ds, xr.Dataset)
        assert "forecast" in ds
        assert ds["forecast"].sizes["draw"] == 30
        xr.testing.assert_identical(ds["forecast"], result["predictions"]["forecast"])

    def test_prediction_samples_from_in_sample_result(self, fitted):
        data, cov, idata = fitted
        result = predict_in_sample(linear_model, idata, data, cov, num_samples=30)
        ds = prediction_samples(result)
        assert "obs" in ds and set(ds["obs"].dims) == {"chain", "draw", "time"}

    def test_prediction_samples_dataset_passthrough(self, fitted):
        data, cov, idata = fitted
        result = predict_in_sample(linear_model, idata, data, cov, num_samples=10)
        ds = prediction_samples(result)
        assert prediction_samples(ds) is ds

    def test_prediction_samples_rejects_unknown_shapes(self, fitted):
        _, _, idata = fitted
        with pytest.raises(TypeError, match="cannot extract prediction samples"):
            prediction_samples({"posterior": None})
        with pytest.raises(TypeError, match="cannot extract prediction samples"):
            prediction_samples(idata)  # a fit result, not a prediction result


class TestBatchedPrediction:
    """batch_size processes the posterior in draw blocks (numpyro_forecast#65 port)."""

    def test_batched_forecast_shape_and_contiguous_draws(self, fitted):
        data, cov, idata = fitted
        result = forecast(
            linear_model, idata, data, cov, num_samples=50, batch_size=16, random_seed=SEED
        )
        fc = result["predictions"]["forecast"]
        assert fc.dims == ("chain", "draw", "time_future")
        assert fc.sizes["draw"] == 50
        np.testing.assert_array_equal(fc["draw"].values, np.arange(50))
        # genuinely distinct draws in every block, not repeated noise
        assert float(fc.std(("chain", "draw")).min()) > 0

    def test_batched_forecast_is_deterministic(self, fitted):
        data, cov, idata = fitted
        first = forecast(
            linear_model, idata, data, cov, num_samples=30, batch_size=7, random_seed=SEED
        )
        again = forecast(
            linear_model, idata, data, cov, num_samples=30, batch_size=7, random_seed=SEED
        )
        xr.testing.assert_identical(
            first["predictions"]["forecast"], again["predictions"]["forecast"]
        )

    def test_batched_forecast_replays_posterior(self, fitted):
        # the chunks must still condition on the fitted posterior: the batched
        # forecast tracks the known trend exactly like the unbatched one
        data, cov, idata = fitted
        result = forecast(
            linear_model, idata, data, cov, num_samples=200, batch_size=64, random_seed=SEED
        )
        mean = result["predictions"]["forecast"].mean(("chain", "draw")).values
        truth = 1.0 + 2.0 * cov.values[30:, 0]
        np.testing.assert_allclose(mean, truth, atol=0.2)

    def test_batch_size_larger_than_draws_single_pass(self, fitted):
        # oversized batch_size behaves exactly like the unbatched call
        data, cov, idata = fitted
        batched = forecast(
            linear_model, idata, data, cov, num_samples=20, batch_size=500, random_seed=SEED
        )
        plain = forecast(linear_model, idata, data, cov, num_samples=20, random_seed=SEED)
        xr.testing.assert_identical(
            batched["predictions"]["forecast"], plain["predictions"]["forecast"]
        )

    def test_batched_in_sample(self, fitted):
        data, cov, idata = fitted
        result = predict_in_sample(
            linear_model, idata, data, cov, num_samples=30, batch_size=8, random_seed=SEED
        )
        obs = result["posterior_predictive"]["obs"]
        assert obs.sizes["draw"] == 30
        np.testing.assert_array_equal(obs["draw"].values, np.arange(30))
        resid = obs.mean(("chain", "draw")).values - data.values
        assert np.abs(resid).mean() < 0.15

    def test_invalid_batch_size_raises(self, fitted):
        data, cov, idata = fitted
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            forecast(linear_model, idata, data, cov, num_samples=10, batch_size=0)

    def test_batched_forecast_accepts_random_state_seed(self, fitted):
        # a legacy RandomState must work on the predictive batch path just as
        # it does on draw_posterior; SeedSequence cannot wrap one, so the chunk
        # seeds are drawn off the RandomState directly.
        data, cov, idata = fitted
        result = forecast(
            linear_model,
            idata,
            data,
            cov,
            num_samples=30,
            batch_size=7,
            random_seed=np.random.RandomState(SEED),
        )
        fc = result["predictions"]["forecast"]
        assert fc.sizes["draw"] == 30
        np.testing.assert_array_equal(fc["draw"].values, np.arange(30))


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

import numpy as np
import pytest
from example_models import (
    RandomWalkForecastingModel,
    linear_model,
    make_random_walk_data,
    make_trend_data,
    random_walk_model,
)

from pymc_forecast.exceptions import MethodResolutionError
from pymc_forecast.forecaster import Forecaster, HMCForecaster, PathfinderForecaster

SEED = 4242


class TestForecasterVI:
    @pytest.fixture(scope="class")
    def fc(self):
        data, cov = make_trend_data()
        return Forecaster(linear_model, data, cov, num_steps=8_000, random_seed=SEED)

    def test_losses_decrease(self, fc):
        losses = np.asarray(fc.losses)
        assert losses[-100:].mean() < losses[:100].mean()

    def test_forecast_tracks_truth(self, fc):
        _, cov = make_trend_data()
        pred = fc.forecast(cov, num_samples=200, random_seed=SEED)["predictions"]
        truth = 1.0 + 2.0 * cov.values[30:, 0]
        np.testing.assert_allclose(
            pred["forecast"].mean(("chain", "draw")).values, truth, atol=0.25
        )

    def test_predict_in_sample(self, fc):
        data, _ = make_trend_data()
        ppc = fc.predict_in_sample(num_samples=100, random_seed=SEED)
        resid = ppc["posterior_predictive"]["obs"].mean(("chain", "draw")).values - data.values
        assert np.abs(resid).mean() < 0.2

    def test_fullrank_method(self):
        data, cov = make_trend_data()
        fc = Forecaster(
            linear_model,
            data,
            cov,
            method="fullrank_advi",
            num_steps=3_000,
            random_seed=SEED,
        )
        pred = fc.forecast(cov, num_samples=50, random_seed=SEED)
        assert pred["predictions"]["forecast"].sizes["time_future"] == 5

    def test_scalar_learning_rate(self):
        data, cov = make_trend_data()
        fc = Forecaster(linear_model, data, cov, optimizer=0.05, num_steps=2_000, random_seed=SEED)
        assert len(np.asarray(fc.losses)) == 2_000

    def test_bad_optimizer_rejected(self):
        data, cov = make_trend_data()
        with pytest.raises(MethodResolutionError, match="optimizer"):
            Forecaster(linear_model, data, cov, optimizer="adam")

    def test_bad_method_rejected(self):
        data, cov = make_trend_data()
        with pytest.raises(MethodResolutionError, match="unknown VI method"):
            Forecaster(linear_model, data, cov, method="not_a_method", num_steps=10)


class TestForecastByHorizon:
    """The covariate-free horizon= shortcut on a random-walk model."""

    @pytest.fixture(scope="class")
    def fc(self):
        data, _ = make_random_walk_data(t_obs=40, horizon=0)
        return Forecaster(random_walk_model, data, num_steps=4_000, random_seed=SEED)

    def test_horizon_builds_future(self, fc):
        pred = fc.forecast(horizon=5, num_samples=100, random_seed=SEED)["predictions"]
        assert pred["forecast"].sizes["time_future"] == 5
        np.testing.assert_array_equal(pred["time_future"].values, np.arange(40, 45))

    def test_exactly_one_of_covariates_horizon(self, fc):
        _, cov = make_random_walk_data()
        with pytest.raises(ValueError, match="exactly one"):
            fc.forecast(cov, horizon=5)
        with pytest.raises(ValueError, match="exactly one"):
            fc.forecast()


class TestHMCForecaster:
    @pytest.fixture(scope="class")
    def fc(self):
        data, cov = make_random_walk_data()
        return HMCForecaster(
            random_walk_model,
            data,
            cov,
            draws=150,
            tune=200,
            chains=2,
            random_seed=SEED,
        )

    def test_forecast_continues_from_level(self, fc):
        data, cov = make_random_walk_data()
        pred = fc.forecast(cov, num_samples=200, random_seed=SEED)["predictions"]
        first = float(pred["forecast"].isel(time_future=0).mean())
        assert abs(first - float(data.values[-1])) < 0.4

    def test_oop_facade_end_to_end(self):
        data, cov = make_random_walk_data()
        fc = HMCForecaster(
            RandomWalkForecastingModel(),
            data,
            cov,
            draws=100,
            tune=150,
            chains=2,
            random_seed=SEED,
        )
        pred = fc.forecast(cov, num_samples=50, random_seed=SEED)["predictions"]
        assert pred["forecast"].sizes["time_future"] == 5


class TestConsistency:
    def test_advi_and_nuts_agree(self):
        """Upstream's cross-backend consistency check on a conjugate-ish model."""
        data, cov = make_trend_data()
        vi = Forecaster(linear_model, data, cov, num_steps=10_000, random_seed=SEED)
        mcmc = HMCForecaster(
            linear_model, data, cov, draws=200, tune=200, chains=2, random_seed=SEED
        )
        pred_vi = vi.forecast(cov, num_samples=300, random_seed=SEED)
        pred_mcmc = mcmc.forecast(cov, num_samples=300, random_seed=SEED)
        np.testing.assert_allclose(
            pred_vi["predictions"]["forecast"].mean(("chain", "draw")).values,
            pred_mcmc["predictions"]["forecast"].mean(("chain", "draw")).values,
            atol=0.25,
        )


class TestPathfinderForecaster:
    def test_forecast_tracks_truth(self):
        data, cov = make_trend_data()
        fc = PathfinderForecaster(linear_model, data, cov, random_seed=SEED)
        pred = fc.forecast(cov, num_samples=200, random_seed=SEED)["predictions"]
        truth = 1.0 + 2.0 * cov.values[30:, 0]
        np.testing.assert_allclose(
            pred["forecast"].mean(("chain", "draw")).values, truth, atol=0.35
        )

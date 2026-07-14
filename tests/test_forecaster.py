import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from example_models import (
    RandomWalkForecastingModel,
    linear_model,
    make_random_walk_data,
    make_trend_data,
    random_walk_model,
)

from pymc_forecast.exceptions import AlignmentError, MethodResolutionError
from pymc_forecast.forecaster import (
    BaseForecaster,
    Forecaster,
    HMCForecaster,
    PathfinderForecaster,
)
from pymc_forecast.model import predict

SEED = 4242


def deterministic_replay_model(h, covariates):
    """Expose one posterior scalar across every time step without noise."""
    value = pm.Normal("value")
    latent = pt.repeat(value, h.duration)
    predict(
        h,
        lambda name, x, dims, observed: pm.Deterministic(name, x, dims=dims),
        latent,
    )


class StubForecaster(BaseForecaster):
    """No-fit forecaster for testing explicit posterior plumbing."""

    def _fit(self, random_seed):
        self.draw_calls = []

    def draw_posterior(self, num_samples, random_seed=None):
        self.draw_calls.append((num_samples, random_seed))
        return xr.Dataset(
            {"value": (("chain", "draw"), np.zeros((1, num_samples)))},
            coords={"chain": [0], "draw": np.arange(num_samples)},
        )


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


class TestFixedPosterior:
    @pytest.fixture
    def fc(self):
        data = xr.DataArray([0.0, 0.0], dims="time", coords={"time": [0, 1]})
        return StubForecaster(deterministic_replay_model, data)

    @pytest.fixture
    def posterior(self):
        values = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
        return xr.Dataset(
            {"value": (("chain", "draw"), values)},
            coords={"chain": [4, 9], "draw": [10, 20, 30]},
        )

    def test_same_posterior_preserves_draw_alignment(self, fc, posterior):
        inside = fc.predict_in_sample(posterior=posterior, random_seed=SEED)[
            "posterior_predictive"
        ]["obs"]
        future = fc.forecast(horizon=2, posterior=posterior, random_seed=SEED)["predictions"][
            "forecast"
        ]

        assert inside.sizes == {"chain": 2, "draw": 3, "time": 2}
        assert future.sizes == {"chain": 2, "draw": 3, "time_future": 2}
        np.testing.assert_array_equal(inside["chain"], posterior["chain"])
        np.testing.assert_array_equal(inside["draw"], posterior["draw"])
        np.testing.assert_array_equal(future["chain"], posterior["chain"])
        np.testing.assert_array_equal(future["draw"], posterior["draw"])
        np.testing.assert_array_equal(inside.isel(time=0), posterior["value"])
        np.testing.assert_array_equal(future.isel(time_future=0), posterior["value"])
        assert fc.draw_calls == []

    def test_num_samples_is_rejected_with_posterior(self, fc, posterior):
        with pytest.raises(ValueError, match="num_samples cannot be combined"):
            fc.predict_in_sample(num_samples=2, posterior=posterior)
        with pytest.raises(ValueError, match="num_samples cannot be combined"):
            fc.forecast(horizon=2, num_samples=2, posterior=posterior)

    def test_default_still_draws_one_hundred_samples(self, fc):
        result = fc.predict_in_sample(random_seed=SEED)
        assert result["posterior_predictive"]["obs"].sizes["draw"] == 100
        assert fc.draw_calls == [(100, SEED)]


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
            fc.forecast(horizon=5, future_index=[40, 41])
        with pytest.raises(ValueError, match="exactly one"):
            fc.forecast()


class TestForecastByFutureIndex:
    """Horizon-agnostic predict: the future index is supplied at forecast time."""

    @pytest.fixture(scope="class")
    def fc(self):
        data, _ = make_random_walk_data(t_obs=40, horizon=0)
        return Forecaster(random_walk_model, data, num_steps=4_000, random_seed=SEED)

    def test_future_index_builds_future(self, fc):
        pred = fc.forecast(future_index=[40, 41, 42], num_samples=100, random_seed=SEED)[
            "predictions"
        ]
        assert pred["forecast"].sizes["time_future"] == 3
        np.testing.assert_array_equal(pred["time_future"].values, [40, 41, 42])

    def test_arbitrary_later_labels(self, fc):
        pred = fc.forecast(future_index=[50, 60, 70, 80], num_samples=50, random_seed=SEED)[
            "predictions"
        ]
        np.testing.assert_array_equal(pred["time_future"].values, [50, 60, 70, 80])

    def test_index_overlapping_training_rejected(self, fc):
        with pytest.raises(AlignmentError, match="strictly after"):
            fc.forecast(future_index=[39, 40, 41])

    def test_rejected_when_model_has_covariates(self):
        data, cov = make_trend_data()
        fc = Forecaster(linear_model, data, cov, num_steps=100, random_seed=SEED)
        with pytest.raises(AlignmentError, match="fit with covariates"):
            fc.forecast(future_index=[30, 31])
        with pytest.raises(AlignmentError, match="fit with covariates"):
            fc.forecast(horizon=2)


class TestForecastByFutureCovariates:
    """Covariate-conditioned forecasts from a future-only covariate frame."""

    @pytest.fixture(scope="class")
    def fc(self):
        data, cov = make_trend_data()
        return Forecaster(linear_model, data, cov, num_steps=8_000, random_seed=SEED)

    def test_matches_full_covariate_path(self, fc):
        _, cov = make_trend_data()
        future_cov = cov.isel(time=slice(30, None))
        by_future = fc.forecast(future_covariates=future_cov, num_samples=50, random_seed=SEED)
        by_full = fc.forecast(cov, num_samples=50, random_seed=SEED)
        np.testing.assert_allclose(
            by_future["predictions"]["forecast"].values,
            by_full["predictions"]["forecast"].values,
        )

    def test_forecast_is_conditioned_on_covariates(self, fc):
        _, cov = make_trend_data()
        future_cov = cov.isel(time=slice(30, None))
        pred = fc.forecast(future_covariates=future_cov, num_samples=200, random_seed=SEED)
        truth = 1.0 + 2.0 * future_cov.values[:, 0]
        np.testing.assert_allclose(
            pred["predictions"]["forecast"].mean(("chain", "draw")).values, truth, atol=0.25
        )
        np.testing.assert_array_equal(
            pred["predictions"]["time_future"].values, cov["time"].values[30:]
        )

    def test_mismatched_covariate_names_rejected(self, fc):
        _, cov = make_trend_data()
        bad = cov.isel(time=slice(30, None)).assign_coords(covariate=["not_trend"])
        with pytest.raises(AlignmentError, match="coords must match"):
            fc.forecast(future_covariates=bad)

    def test_overlapping_time_rejected(self, fc):
        _, cov = make_trend_data()
        with pytest.raises(AlignmentError, match="strictly after"):
            fc.forecast(future_covariates=cov.isel(time=slice(29, None)))

    def test_exclusive_with_other_horizon_specs(self, fc):
        _, cov = make_trend_data()
        future_cov = cov.isel(time=slice(30, None))
        with pytest.raises(ValueError, match="exactly one"):
            fc.forecast(cov, future_covariates=future_cov)
        with pytest.raises(ValueError, match="exactly one"):
            fc.forecast(future_covariates=future_cov, horizon=5)


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

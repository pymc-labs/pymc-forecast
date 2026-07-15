import inspect
import warnings

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

from pymc_forecast.exceptions import AlignmentError, MethodResolutionError, NotFittedError
from pymc_forecast.forecaster import (
    BaseForecaster,
    Forecaster,
    HMCForecaster,
    PathfinderForecaster,
    _check_vi_convergence,
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
    """No-inference forecaster for exercising the shared base-class plumbing."""

    def _fit(self, random_seed):
        self.fit_seeds = getattr(self, "fit_seeds", [])
        self.fit_seeds.append(random_seed)
        self.draw_calls = []

    def _draw_posterior(self, num_samples, random_seed=None):
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


class TestJAXForecasterVI:
    def test_end_to_end(self):
        pytest.importorskip("jax")
        data, cov = make_trend_data()
        fc = Forecaster(
            linear_model,
            data,
            cov,
            backend="jax",
            optimizer=0.01,
            num_steps=20,
            random_seed=SEED,
        )

        assert len(fc.losses) == 20
        assert np.isfinite(fc.losses).all()
        posterior = fc.draw_posterior(9, random_seed=SEED, batch_size=4)
        assert posterior.sizes["draw"] == 9

    @pytest.mark.parametrize("backend", ["numpyro", "gpu", "JAX"])
    def test_unknown_backend_rejected(self, backend):
        with pytest.raises(MethodResolutionError, match="unknown VI backend"):
            Forecaster(linear_model, backend=backend)

    def test_backend_requires_mean_field_advi(self):
        with pytest.raises(MethodResolutionError, match="method='advi'"):
            Forecaster(linear_model, backend="jax", method="fullrank_advi")

    @pytest.mark.parametrize("optimizer", [0, -0.1, "adam", pm.adam()])
    def test_backend_requires_scalar_learning_rate(self, optimizer):
        with pytest.raises(MethodResolutionError, match="positive learning rate"):
            Forecaster(linear_model, backend="jax", optimizer=optimizer)

    def test_backend_rejects_pymc_fit_kwargs(self):
        with pytest.raises(MethodResolutionError, match="fit_kwargs"):
            Forecaster(linear_model, backend="jax", fit_kwargs={"obj_n_mc": 2})


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


class TestFixedPosterior:
    """The posterior= passthrough: several predictive calls, one set of draws."""

    @pytest.fixture(scope="class")
    def fc(self):
        data, cov = make_trend_data()
        return Forecaster(linear_model, data, cov, num_steps=8_000, random_seed=SEED)

    def test_calls_are_draw_coherent(self, fc):
        # mu / mu_future are deterministic in the parameters, so with a fixed
        # posterior both calls must reproduce it draw-for-draw.
        _, cov = make_trend_data()
        posterior = fc.draw_posterior(50, random_seed=SEED)
        pre = fc.predict_in_sample(posterior=posterior, random_seed=SEED)
        post = fc.forecast(cov, posterior=posterior, random_seed=SEED)

        def expected(cov_slice):
            mu = posterior["intercept"] + xr.dot(posterior["beta"], cov_slice, dim="covariate")
            return mu.transpose("chain", "draw", "time").values

        np.testing.assert_allclose(
            pre["posterior_predictive"]["mu"].values,
            expected(cov.isel(time=slice(None, 30))),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            post["predictions"]["mu_future"].values,
            expected(cov.isel(time=slice(30, None))),
            rtol=1e-5,
        )

    def test_matching_sample_sizes(self, fc):
        _, cov = make_trend_data()
        posterior = fc.draw_posterior(23, random_seed=SEED)
        pre = fc.predict_in_sample(posterior=posterior)
        post = fc.forecast(cov, posterior=posterior)
        assert pre["posterior_predictive"]["obs"].sizes["draw"] == 23
        assert post["predictions"]["forecast"].sizes["draw"] == 23

    def test_vi_posterior_can_be_drawn_in_host_batches(self, fc, monkeypatch):
        calls = []
        draw = fc._draw_posterior

        def record(size, random_seed=None):
            calls.append(size)
            return draw(size, random_seed)

        monkeypatch.setattr(fc, "_draw_posterior", record)
        posterior = fc.draw_posterior(23, random_seed=SEED, batch_size=7)

        assert calls == [7, 7, 7, 2]
        assert posterior.sizes["chain"] == 1
        assert posterior.sizes["draw"] == 23
        np.testing.assert_array_equal(posterior["draw"], np.arange(23))

    def test_accepts_any_posterior_shape(self, fc):
        _, cov = make_trend_data()
        idata = fc.approx.sample(draws=10, random_seed=SEED)  # InferenceData
        pred = fc.forecast(cov, posterior=idata)
        assert pred["predictions"]["forecast"].sizes["draw"] == 10

    def test_num_samples_and_posterior_are_exclusive(self, fc):
        _, cov = make_trend_data()
        posterior = fc.draw_posterior(5, random_seed=SEED)
        with pytest.raises(ValueError, match="not both"):
            fc.forecast(cov, num_samples=5, posterior=posterior)
        with pytest.raises(ValueError, match="not both"):
            fc.predict_in_sample(num_samples=5, posterior=posterior)


class TestFixedPosteriorPlumbing:
    """Coordinate-exact posterior passthrough, checked without any inference."""

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

    def test_same_posterior_preserves_chain_and_draw_coords(self, fc, posterior):
        inside = fc.predict_in_sample(posterior=posterior, random_seed=SEED)[
            "posterior_predictive"
        ]["obs"]
        future = fc.forecast(horizon=2, posterior=posterior, random_seed=SEED)["predictions"][
            "forecast"
        ]
        assert inside.sizes == {"chain": 2, "draw": 3, "time": 2}
        assert future.sizes == {"chain": 2, "draw": 3, "time_future": 2}
        for da in (inside, future):
            np.testing.assert_array_equal(da["chain"], posterior["chain"])
            np.testing.assert_array_equal(da["draw"], posterior["draw"])
        np.testing.assert_array_equal(inside.isel(time=0), posterior["value"])
        np.testing.assert_array_equal(future.isel(time_future=0), posterior["value"])
        assert fc.draw_calls == []  # nothing drawn internally

    def test_default_still_draws_one_hundred_samples(self, fc):
        result = fc.predict_in_sample(random_seed=SEED)
        assert result["posterior_predictive"]["obs"].sizes["draw"] == 100
        assert fc.draw_calls == [(100, SEED)]

    def test_nonpositive_posterior_batch_size_rejected(self, fc):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            fc.draw_posterior(5, batch_size=0)


class TestDeferredFit:
    def test_lifecycle(self):
        fc = StubForecaster(linear_model, random_seed=1)
        assert not fc.is_fitted
        assert fc.model is None
        with pytest.raises(NotFittedError, match="not fitted"):
            fc.predict_in_sample()
        with pytest.raises(NotFittedError, match="not fitted"):
            fc.forecast(horizon=2)
        with pytest.raises(NotFittedError, match="not fitted"):
            fc.draw_posterior(10)

        data, cov = make_trend_data()
        assert fc.fit(data, cov, random_seed=2) is fc
        assert fc.is_fitted
        assert fc.model is not None
        assert fc.fit_seeds == [2]

        fc.fit(data, cov)  # refit; falls back to the constructor seed
        assert fc.fit_seeds == [2, 1]

    def test_fit_after_construction(self):
        data, cov = make_trend_data()
        fc = Forecaster(linear_model, num_steps=2_000, random_seed=SEED)
        assert fc.fit(data, cov) is fc
        pred = fc.forecast(cov, num_samples=20, random_seed=SEED)
        assert pred["predictions"]["forecast"].sizes["time_future"] == 5

    def test_covariates_without_data_rejected(self):
        _, cov = make_trend_data()
        with pytest.raises(ValueError, match="without data"):
            Forecaster(linear_model, covariates=cov)


class TestUniformConstructorSurface:
    def test_progressbar_accepted_by_every_forecaster(self):
        from pymc_forecast.statespace import StatespaceForecaster

        for cls in (Forecaster, HMCForecaster, PathfinderForecaster, StatespaceForecaster):
            assert "progressbar" in inspect.signature(cls.__init__).parameters

    def test_progressbar_kwarg_fits(self):
        data, cov = make_trend_data()
        fc = Forecaster(linear_model, data, cov, num_steps=100, progressbar=False)
        assert len(np.asarray(fc.losses)) == 100
        hmc = HMCForecaster(linear_model, data, cov, draws=20, tune=20, chains=2, progressbar=False)
        assert hmc.idata.posterior.sizes["draw"] == 20

    @pytest.mark.parametrize("cls", [Forecaster, HMCForecaster, PathfinderForecaster])
    def test_progressbar_hoisted_without_fitting(self, cls):
        fc = cls(linear_model, progressbar=True)
        assert fc._progressbar is True
        assert not fc.is_fitted

    def test_legacy_kwargs_progressbar_still_supported(self):
        fc = Forecaster(linear_model, fit_kwargs={"progressbar": True})
        assert fc._progressbar is True
        assert "progressbar" not in fc._fit_kwargs

    def test_duplicate_progressbar_is_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            HMCForecaster(
                linear_model,
                progressbar=False,
                sample_kwargs={"progressbar": True},
            )


class TestConvergenceWarning:
    def test_descending_loss_warns(self):
        # a steady descent much larger than the noise floor
        rng = np.random.default_rng(0)
        losses = np.linspace(1000.0, 100.0, 2_000) + rng.normal(0, 1.0, 2_000)
        with pytest.warns(UserWarning, match="has not converged"):
            _check_vi_convergence(losses, 2_000)

    def test_flat_noisy_loss_does_not_warn(self):
        rng = np.random.default_rng(0)
        losses = 50.0 + rng.normal(0, 5.0, 2_000)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_vi_convergence(losses, 2_000)

    def test_short_history_is_skipped(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_vi_convergence(np.linspace(100.0, 0.0, 10), 10)

    def test_non_finite_losses_warn_unassessable(self):
        losses = np.linspace(1000.0, 100.0, 2_000)
        losses[3] = np.nan
        with pytest.warns(UserWarning, match="could not be assessed"):
            _check_vi_convergence(losses, 2_000)

    def test_underfit_forecaster_warns(self):
        data, cov = make_trend_data()
        with pytest.warns(UserWarning, match="has not converged"):
            Forecaster(linear_model, data, cov, num_steps=300, random_seed=SEED)

    def test_converged_forecaster_does_not_warn(self):
        data, cov = make_trend_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Forecaster(linear_model, data, cov, num_steps=10_000, optimizer=0.05, random_seed=SEED)
        assert not [w for w in caught if "has not converged" in str(w.message)]


class TestPathfinderForecaster:
    def test_forecast_tracks_truth(self):
        data, cov = make_trend_data()
        fc = PathfinderForecaster(linear_model, data, cov, random_seed=SEED)
        pred = fc.forecast(cov, num_samples=200, random_seed=SEED)["predictions"]
        truth = 1.0 + 2.0 * cov.values[30:, 0]
        np.testing.assert_allclose(
            pred["forecast"].mean(("chain", "draw")).values, truth, atol=0.35
        )

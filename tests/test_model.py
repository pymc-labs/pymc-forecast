import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from example_models import (
    RandomWalkForecastingModel,
    hierarchical_model,
    linear_model,
    make_random_walk_data,
    make_trend_data,
    random_walk_model,
)

from pymc_forecast.data import TIME_DIM
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import Horizon, build_model, predict, time_series


class TestHorizon:
    def test_from_arrays_split(self):
        data, cov = make_trend_data(t_obs=30, horizon=5)
        h = Horizon.from_arrays(cov, data)
        assert (h.t_obs, h.future, h.duration) == (30, 5, 35)
        np.testing.assert_array_equal(h.time_future, np.arange(30, 35))

    def test_prior_only(self):
        _, cov = make_trend_data(t_obs=30, horizon=5)
        h = Horizon.from_arrays(cov, None)
        assert (h.t_obs, h.future) == (35, 0)
        assert h.data is None


class TestBuildModel:
    def test_training_model_coords_and_vars(self):
        data, cov = make_trend_data()
        model = build_model(linear_model, data, cov.isel({TIME_DIM: slice(None, 30)}))
        assert "time" in model.coords and "time_future" not in model.coords
        assert list(model.coords["covariate"]) == ["trend"]
        assert "obs" in model.named_vars and "forecast" not in model.named_vars

    def test_forecast_model_has_future(self):
        data, cov = make_trend_data()
        model = build_model(linear_model, data, cov)
        assert len(model.coords["time_future"]) == 5
        assert "forecast" in model.named_vars

    def test_time_series_creates_future_var(self):
        data, cov = make_random_walk_data()
        model = build_model(random_walk_model, data, cov)
        names = {rv.name for rv in model.free_RVs}
        assert {"drift", "drift_future"} <= names
        train = build_model(random_walk_model, data, cov.isel({TIME_DIM: slice(None, 40)}))
        assert "drift_future" not in {rv.name for rv in train.free_RVs}

    def test_hierarchical_dims(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(20, 3))
        cov = np.zeros((25, 0))
        model = build_model(hierarchical_model, data, cov)
        assert len(model.coords["series"]) == 3
        assert model.named_vars["obs"].eval().shape == (20, 3)
        assert model.named_vars["forecast"].eval().shape == (5, 3)

    def test_missing_predict_raises(self):
        def no_predict(h, covariates):
            pm.Normal("x")

        data, cov = make_trend_data()
        with pytest.raises(HorizonError, match="no 'obs' variable"):
            build_model(no_predict, data, cov)

    def test_registers_noise_free_mu(self):
        data, cov = make_trend_data()
        model = build_model(linear_model, data, cov)
        assert {"mu", "mu_future"} <= set(model.named_vars)
        train = build_model(linear_model, data, cov.isel({TIME_DIM: slice(None, 30)}))
        assert "mu" in train.named_vars and "mu_future" not in train.named_vars

    def test_mu_full_shape_for_batch_dims(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(20, 3))
        cov = np.zeros((25, 0))
        model = build_model(hierarchical_model, data, cov)
        assert model.named_vars["mu"].eval().shape == (20, 3)
        assert model.named_vars["mu_future"].eval().shape == (5, 3)

    def test_mu_broadcast_like_the_likelihood(self):
        # a latent with a size-1 series axis broadcasts against the data in
        # the likelihood; mu must be recorded at the same full shape
        def broadcasting(h, covariates):
            drift = time_series(h, "drift", lambda name, dims: pm.Normal(name, 0.0, 0.2, dims=dims))
            sigma = pm.HalfNormal("sigma", 0.5)
            predict(
                h,
                lambda name, m, dims, observed: pm.Normal(
                    name, m, sigma, dims=dims, observed=observed
                ),
                pt.cumsum(drift)[:, None],
            )

        rng = np.random.default_rng(0)
        data = rng.normal(size=(20, 3))
        cov = np.zeros((25, 0))
        model = build_model(broadcasting, data, cov)
        assert model.named_vars["mu"].eval().shape == (20, 3)
        assert model.named_vars["mu_future"].eval().shape == (5, 3)

    def test_reserved_mu_name_collides(self):
        def colliding(h, covariates):
            pm.Normal("mu", 0.0, 1.0)
            linear_model(h, covariates)

        data, cov = make_trend_data()
        with pytest.raises(HorizonError, match="reserves 'mu'"):
            build_model(colliding, data, cov)

    def test_prior_only_build(self):
        _, cov = make_trend_data()
        model = build_model(linear_model, None, cov)
        assert model["obs"] in model.free_RVs  # unobserved in prior-only builds
        with model:
            prior = pm.sample_prior_predictive(draws=10, random_seed=1)
        assert prior["prior"]["obs"].sizes["time"] == 35


class TestForecastingModelFacade:
    def test_builds_same_vars_as_functional(self):
        data, cov = make_random_walk_data()
        oop = build_model(RandomWalkForecastingModel(), data, cov)
        fn = build_model(random_walk_model, data, cov)
        assert {rv.name for rv in oop.free_RVs} == {rv.name for rv in fn.free_RVs}

    def test_horizon_unavailable_outside_build(self):
        instance = RandomWalkForecastingModel()
        with pytest.raises(HorizonError, match="during a model build"):
            instance.time_series("drift", lambda name, dims: None)

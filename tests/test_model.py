import inspect

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
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
from pymc_forecast.model import ForecastingModel, Horizon, build_model, innovations, predict


class TestHorizon:
    def test_from_arrays_split(self):
        data, cov = make_trend_data(t_obs=30, horizon=5)
        h = Horizon.from_data(cov, data)
        assert (h.t_obs, h.future, h.duration) == (30, 5, 35)
        np.testing.assert_array_equal(h.time_future, np.arange(30, 35))

    def test_prior_only(self):
        _, cov = make_trend_data(t_obs=30, horizon=5)
        h = Horizon.from_data(cov, None)
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

    def test_innovations_creates_future_var(self):
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
        def no_predict(covariates, data=None):
            pm.Normal("x")

        data, cov = make_trend_data()
        with pytest.raises(HorizonError, match="no 'obs' variable"):
            build_model(no_predict, data, cov)

    def test_registers_noise_free_mu(self):
        data, cov = make_trend_data()
        model = build_model(linear_model, data, cov)
        assert {"mu", "mu_future"} <= set(model.named_vars)
        assert "expected_observation" not in model.named_vars
        train = build_model(linear_model, data, cov.isel({TIME_DIM: slice(None, 30)}))
        assert "mu" in train.named_vars and "mu_future" not in train.named_vars

    def test_registers_optional_expected_observation(self):
        data, cov = make_random_walk_data()
        model = build_model(random_walk_model, data, cov)
        assert {"expected_observation", "expected_observation_future"} <= set(model.named_vars)
        assert model.named_vars["expected_observation"].eval().shape == (40,)
        assert model.named_vars["expected_observation_future"].eval().shape == (5,)

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
        def broadcasting(covariates, data=None):
            h = Horizon.from_data(covariates, data)
            drift = innovations(h, "drift", pm.Normal.dist(0.0, 0.2))
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
        def colliding(covariates, data=None):
            pm.Normal("mu", 0.0, 1.0)
            linear_model(covariates, data)

        data, cov = make_trend_data()
        with pytest.raises(HorizonError, match=r"already defines \['mu'\]"):
            build_model(colliding, data, cov)

    @pytest.mark.parametrize("name", ["expected_observation", "expected_observation_future"])
    def test_reserved_expected_observation_name_collides_without_the_argument(self, name):
        # the names are reserved unconditionally: a model that never passes
        # expected_observation= must still not be able to define them, or the
        # user variable would be swept into the documented schema slot by
        # _default_var_names / predict_in_sample, which collect by name
        def colliding(covariates, data=None):
            pm.Normal(name, 0.0, 1.0)
            linear_model(covariates, data)

        data, cov = make_trend_data()
        with pytest.raises(HorizonError, match=rf"already defines \['{name}'\]"):
            build_model(colliding, data, cov)

    def test_reserved_expected_observation_name_collides_with_the_argument(self):
        def colliding(covariates, data=None):
            pm.Normal("expected_observation", 0.0, 1.0)
            random_walk_model(covariates, data)

        data, cov = make_random_walk_data()
        with pytest.raises(HorizonError, match=r"already defines \['expected_observation'\]"):
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
        assert {"expected_observation", "expected_observation_future"} <= set(oop.named_vars)

    def test_horizon_unavailable_outside_build(self):
        instance = RandomWalkForecastingModel()
        with pytest.raises(HorizonError, match="during a model build"):
            instance.innovations("drift", pm.Normal.dist(0.0, 1.0))


def test_model_function_receives_covariates_then_data():
    seen = {}

    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        seen["t"] = h.t_obs
        seen["future"] = h.future
        predict(
            h,
            lambda name, m, dims, observed: pm.Normal(name, m, 1.0, dims=dims, observed=observed),
            pt.zeros(h.duration),
        )

    data = xr.DataArray([1.0, 2.0], dims="time", coords={"time": [0, 1]})
    cov = xr.DataArray([0.0, 0.0, 0.0], dims="time", coords={"time": [0, 1, 2]})
    build_model(model, data, cov)
    assert seen == {"t": 2, "future": 1}


def test_from_arrays_absent_and_class_reads_horizon():
    assert not hasattr(Horizon, "from_arrays")
    params = inspect.signature(ForecastingModel.model).parameters
    assert list(params) == ["self", "covariates", "data"]
    assert params["data"].default is None

    seen = {}

    class Probe(ForecastingModel):
        def model(self, covariates, data=None):
            seen["t"] = self.horizon.t_obs
            seen["future"] = self.horizon.future
            pm.Normal("obs", 0.0, 1.0, observed=data.values, dims="time")

    data = xr.DataArray([1.0, 2.0], dims="time", coords={"time": [0, 1]})
    cov = xr.DataArray([0.0, 0.0, 0.0], dims="time", coords={"time": [0, 1, 2]})
    build_model(Probe(), data, cov)
    assert seen == {"t": 2, "future": 1}


def test_time_series_is_not_importable():
    import pymc_forecast

    assert not hasattr(pymc_forecast, "time_series")
    with pytest.raises(ImportError):
        from pymc_forecast import time_series  # noqa: F401


def test_innovations_registers_future_var_and_concatenates_on_axis_0():
    from pymc_forecast.model import innovations

    data, cov = make_random_walk_data(t_obs=8, horizon=3)
    captured = {}

    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        captured["drift"] = innovations(h, "drift", pm.Normal.dist(0.0, 0.25))
        predict(h, lambda mu: pm.Normal.dist(mu, 1.0), captured["drift"])

    built = build_model(model, data, cov)
    assert built.named_vars_to_dims["drift"] == ("time",)
    assert built.named_vars_to_dims["drift_future"] == ("time_future",)
    drift = captured["drift"]
    assert drift.owner.op.axis == 0
    assert tuple(drift.eval().shape) == (11,)


def test_innovations_logp_matches_named_normal():
    from pymc_forecast.model import innovations

    time = np.arange(8)
    sigma = 0.4
    point = np.linspace(-1.0, 1.0, time.size)
    h = Horizon(data=None, time=time)
    with pm.Model(coords={"time": time}) as via_innovations:
        innovations(h, "drift", pm.Normal.dist(0.0, sigma))
    with pm.Model(coords={"time": time}) as via_named:
        pm.Normal("drift", 0.0, sigma, dims="time")
    np.testing.assert_allclose(
        via_innovations.compile_logp()({"drift": point}),
        via_named.compile_logp()({"drift": point}),
    )


def test_innovations_rejects_dist_that_already_has_a_time_dimension():
    from pymc_forecast.model import innovations

    time = np.arange(6)
    h = Horizon(data=None, time=time)
    with pm.Model(coords={"time": time}):
        with pytest.raises(HorizonError, match="time"):
            innovations(h, "drift", pm.Normal.dist(0.0, 1.0, shape=(time.size,)))


def test_predict_callable_registers_obs_and_forecast():
    data, cov = make_random_walk_data(t_obs=8, horizon=3)
    nu = 5.0
    sigma = 0.3

    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        latent = pt.zeros(h.duration)
        predict(h, lambda mu: pm.StudentT.dist(nu, mu, sigma=sigma), latent)

    built = build_model(model, data, cov)
    assert "obs" in built.named_vars
    assert "forecast" in built.named_vars
    assert built.named_vars_to_dims["obs"] == ("time",)
    assert built.named_vars_to_dims["forecast"] == ("time_future",)


def test_predict_studentt_dist_logp_matches_named_observation():
    time = np.arange(7)
    y = np.linspace(-0.5, 0.8, time.size)
    latent_prefix = np.linspace(-0.2, 0.2, time.size)
    nu = 4.0
    sigma = 0.5
    data = xr.DataArray(y, dims="time", coords={"time": time})
    h = Horizon(data=data, time=time)
    with pm.Model(coords={"time": time}) as via_predict:
        predict(h, pm.StudentT.dist(nu, 0.0, sigma=sigma), pt.as_tensor(latent_prefix))
    with pm.Model(coords={"time": time}) as via_named:
        pm.StudentT("obs", nu, latent_prefix, sigma=sigma, observed=y, dims="time")
    np.testing.assert_allclose(via_predict.compile_logp()({}), via_named.compile_logp()({}))


def test_predict_laplace_dist_raises():
    time = np.arange(4)
    data = xr.DataArray(np.zeros(time.size), dims="time", coords={"time": time})
    h = Horizon(data=data, time=time)
    with pm.Model(coords={"time": time}):
        with pytest.raises(HorizonError, match="1-argument callable"):
            predict(h, pm.Laplace.dist(0.0, 1.0), pt.zeros(time.size))


def test_predict_four_argument_factory_still_receives_name_latent_dims_observed():
    time = np.arange(5)
    future = np.arange(5, 8)
    y = np.ones(time.size)
    data = xr.DataArray(y, dims="time", coords={"time": time})
    h = Horizon(data=data, time=time, time_future=future)
    seen = []

    def factory(name, latent, dims, observed):
        seen.append((name, latent, dims, observed))
        return pm.Normal(name, latent, 1.0, dims=dims, observed=observed)

    with pm.Model(coords={"time": time, "time_future": future}):
        predict(h, factory, pt.zeros(time.size + future.size))
    assert [item[0] for item in seen] == ["obs", "forecast"]
    assert seen[0][2] == ("time",)
    assert seen[1][2] == ("time_future",)
    np.testing.assert_array_equal(seen[0][3], y)
    assert seen[1][3] is None
    assert tuple(seen[0][1].shape.eval()) == (time.size,)
    assert tuple(seen[1][1].shape.eval()) == (future.size,)


def test_innovations_broadcasts_dist_parameters_over_the_series_dim():
    time, future, series = np.arange(6), np.arange(6, 10), ["a", "b", "c"]
    sigma = np.array([0.1, 1.0, 5.0])
    h = Horizon(data=None, time=time, time_future=future)
    coords = {"time": time, "time_future": future, "series": series}
    point = {"z": np.ones((6, 3)), "z_future": np.ones((4, 3))}
    with pm.Model(coords=coords) as via_innovations:
        innovations(h, "z", pm.Normal.dist(0.0, sigma), dims=("series",))
    with pm.Model(coords=coords) as via_named:
        pm.Normal("z", 0.0, sigma, dims=("time", "series"))
        pm.Normal("z_future", 0.0, sigma, dims=("time_future", "series"))
    np.testing.assert_allclose(
        via_innovations.compile_logp()(point), via_named.compile_logp()(point)
    )


def test_innovations_accepts_multivariate_dist_when_series_count_equals_horizon():
    time, future, series = np.arange(6), np.arange(6, 8), ["a", "b"]
    h = Horizon(data=None, time=time, time_future=future)
    with pm.Model(coords={"time": time, "time_future": future, "series": series}) as model:
        innovations(h, "z", pm.MvNormal.dist(np.zeros(2), np.eye(2)), dims=("series",))
    assert model.named_vars_to_dims["z"] == ("time", "series")
    assert model.named_vars_to_dims["z_future"] == ("time_future", "series")
    assert tuple(model["z"].eval().shape) == (6, 2)
    assert tuple(model["z_future"].eval().shape) == (2, 2)

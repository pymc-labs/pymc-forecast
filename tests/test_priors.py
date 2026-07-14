"""Tests for pymc-extras Prior interop (user-injectable priors, issue #23)."""

from typing import ClassVar

import numpy as np
import pytensor.tensor as pt
import pytest
import xarray as xr
from example_models import make_random_walk_data

from pymc_forecast.exceptions import HorizonError
from pymc_forecast.forecaster import Forecaster
from pymc_forecast.model import ForecastingModel, build_model, predict, time_series

Prior = pytest.importorskip("pymc_extras.prior").Prior

SEED = 20260714


def prior_model(h, covariates):
    """Random-walk model written entirely with Prior objects."""
    drift = time_series(
        h, "drift", Prior("Normal", mu=Prior("Normal", mu=0, sigma=1), sigma=0.1)
    )
    predict(h, Prior("Normal", sigma=Prior("HalfNormal", sigma=0.5)), pt.cumsum(drift))


class PriorRandomWalk(ForecastingModel):
    default_priors: ClassVar = {
        "drift": Prior("Normal", mu=0, sigma=0.1),
        "noise": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.5)),
    }

    def model(self, h, covariates):
        drift = self.time_series("drift", self.priors["drift"])
        self.predict(self.priors["noise"], pt.cumsum(drift))


class TestPriorPrimitives:
    def test_training_build_registers_prior_variables(self):
        data, cov = make_random_walk_data(t_obs=15, horizon=0)
        model = build_model(prior_model, data, cov)
        names = {rv.name for rv in model.free_RVs}
        assert {"drift", "drift_mu", "obs_sigma"} <= names

    def test_hyperpriors_shared_across_the_split(self):
        """The replay contract: future segments reuse the in-sample hyper-priors."""
        data, cov = make_random_walk_data(t_obs=15, horizon=5)
        model = build_model(prior_model, data, cov)
        names = {rv.name for rv in model.free_RVs}
        assert {"drift_future", "forecast"} <= names
        # one shared hyper-prior per base name, none re-created for the future
        assert "drift_mu" in names and "drift_future_mu" not in names
        assert "obs_sigma" in names and "forecast_sigma" not in names

    def test_time_dimmed_hyperprior_rejected(self):
        data, cov = make_random_walk_data(t_obs=15, horizon=5)
        bad_hyper = Prior("HalfNormal", sigma=1)
        bad_hyper.dims = ("time",)

        def model_fn(h, covariates):
            drift = time_series(h, "drift", Prior("Normal", mu=0, sigma=bad_hyper))
            predict(
                h,
                Prior("Normal", sigma=0.5),
                pt.cumsum(drift),
            )

        with pytest.raises(HorizonError, match="cannot be shared"):
            build_model(model_fn, data, cov)

    def test_observation_prior_with_mu_rejected(self):
        data, cov = make_random_walk_data(t_obs=10, horizon=2)

        def model_fn(h, covariates):
            drift = time_series(h, "drift", Prior("Normal", mu=0, sigma=0.1))
            predict(h, Prior("Normal", mu=1.0, sigma=0.5), pt.cumsum(drift))

        with pytest.raises(ValueError, match="leave 'mu' unset"):
            build_model(model_fn, data, cov)

    def test_end_to_end_forecast(self):
        data, cov = make_random_walk_data(t_obs=30, horizon=5)
        fc = Forecaster(prior_model, data, num_steps=2_000, random_seed=SEED)
        pred = fc.forecast(cov, num_samples=100, random_seed=SEED)["predictions"]
        assert pred["forecast"].dims == ("chain", "draw", "time_future")
        assert np.isfinite(pred["forecast"].values).all()
        # the forecast continues from the fitted level, i.e. the posterior is
        # actually replayed through the Prior-built variables
        first = float(pred["forecast"].isel(time_future=0).mean())
        assert abs(first - float(data.values[-1])) < 0.5


class TestForecastingModelPriors:
    def test_defaults_apply_without_overrides(self):
        model = PriorRandomWalk()
        assert model.priors["drift"] == PriorRandomWalk.default_priors["drift"]

    def test_overrides_merge_over_defaults(self):
        override = Prior("StudentT", nu=4, mu=0, sigma=0.3)
        model = PriorRandomWalk(priors={"drift": override})
        assert model.priors["drift"] == override
        assert model.priors["noise"] == PriorRandomWalk.default_priors["noise"]

    def test_priors_fall_back_without_super_init(self):
        class NoInit(PriorRandomWalk):
            def __init__(self):
                pass

        assert NoInit().priors == dict(PriorRandomWalk.default_priors)

    def test_overridden_prior_reaches_the_model_graph(self):
        data, cov = make_random_walk_data(t_obs=15, horizon=0)
        model = build_model(
            PriorRandomWalk(priors={"drift": Prior("StudentT", nu=4, mu=0, sigma=0.3)}),
            data,
            cov,
        )
        (drift_rv,) = [rv for rv in model.free_RVs if rv.name == "drift"]
        assert drift_rv.owner.op.name == "t"  # the StudentT RV op

    def test_end_to_end_with_overrides(self):
        data, cov = make_random_walk_data(t_obs=30, horizon=5)
        fc = Forecaster(
            PriorRandomWalk(priors={"drift": Prior("Normal", mu=0.1, sigma=0.1)}),
            data,
            num_steps=2_000,
            random_seed=SEED,
        )
        pred = fc.forecast(cov, num_samples=50, random_seed=SEED)["predictions"]
        assert np.isfinite(pred["forecast"].values).all()


class TestBatchDims:
    def test_prior_latent_with_series_dim(self):
        rng = np.random.default_rng(SEED)
        data = xr.DataArray(
            rng.normal(size=(12, 2)),
            dims=("time", "series"),
            coords={"time": np.arange(12), "series": ["a", "b"]},
        )
        cov = xr.DataArray(
            np.zeros((15, 0)),
            dims=("time", "covariate"),
            coords={"time": np.arange(15)},
        )

        def model_fn(h, covariates):
            drift = time_series(h, "drift", Prior("Normal", mu=0, sigma=0.2), dims=("series",))
            predict(h, Prior("Normal", sigma=0.5), pt.cumsum(drift, axis=0))

        model = build_model(model_fn, data, cov)
        assert model.named_vars_to_dims["drift"] == ("time", "series")
        assert model.named_vars_to_dims["drift_future"] == ("time_future", "series")

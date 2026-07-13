"""Shared example models used across the test suite."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pymc_forecast.model import ForecastingModel, Horizon, predict, time_series
from pymc_forecast.statespace import StatespaceModel

SEED = 20260709


def linear_model(h: Horizon, covariates: xr.DataArray) -> None:
    """Static regression: intercept + covariates @ beta, Normal noise."""
    intercept = pm.Normal("intercept", 0.0, 2.0)
    beta = pm.Normal("beta", 0.0, 1.0, dims="covariate")
    sigma = pm.HalfNormal("sigma", 1.0)
    mu = intercept + pt.dot(covariates.values, beta)
    predict(
        h,
        lambda name, m, dims, observed: pm.Normal(name, m, sigma, dims=dims, observed=observed),
        mu,
    )


def random_walk_model(h: Horizon, covariates: xr.DataArray) -> None:
    """Level = cumsum of per-step drift latents; the replay workhorse."""
    drift_loc = pm.Normal("drift_loc", 0.0, 1.0)
    drift = time_series(h, "drift", lambda name, dims: pm.Normal(name, drift_loc, 0.1, dims=dims))
    level = pt.cumsum(drift)
    sigma = pm.HalfNormal("sigma", 0.5)
    predict(
        h,
        lambda name, m, dims, observed: pm.Normal(name, m, sigma, dims=dims, observed=observed),
        level,
    )


class RandomWalkForecastingModel(ForecastingModel):
    """OOP facade version of :func:`random_walk_model`."""

    def model(self, h: Horizon, covariates: xr.DataArray) -> None:
        drift_loc = pm.Normal("drift_loc", 0.0, 1.0)
        drift = self.time_series(
            "drift", lambda name, dims: pm.Normal(name, drift_loc, 0.1, dims=dims)
        )
        sigma = pm.HalfNormal("sigma", 0.5)
        self.predict(
            lambda name, m, dims, observed: pm.Normal(name, m, sigma, dims=dims, observed=observed),
            pt.cumsum(drift),
        )


def hierarchical_model(h: Horizon, covariates: xr.DataArray) -> None:
    """Per-series intercept + shared per-step drift; data dims (time, series)."""
    intercept = pm.Normal("intercept", 0.0, 2.0, dims="series")
    drift = time_series(h, "drift", lambda name, dims: pm.Normal(name, 0.0, 0.2, dims=dims))
    mu = intercept + pt.cumsum(drift)[:, None]
    sigma = pm.HalfNormal("sigma", 0.5)
    predict(
        h,
        lambda name, m, dims, observed: pm.Normal(name, m, sigma, dims=dims, observed=observed),
        mu,
    )


def poisson_model(h: Horizon, covariates: xr.DataArray) -> None:
    """GLM-style count model: log-link on intercept + covariate effect."""
    intercept = pm.Normal("intercept", 0.0, 1.0)
    beta = pm.Normal("beta", 0.0, 1.0, dims="covariate")
    eta = intercept + pt.dot(covariates.values, beta)
    predict(
        h,
        lambda name, e, dims, observed: pm.Poisson(name, pt.exp(e), dims=dims, observed=observed),
        eta,
    )


class LocalLevelStatespace(StatespaceModel):
    """pymc-extras local linear trend + measurement error.

    Priors are derived generically from ``param_info`` (constraint-driven), so
    the model stays valid across pymc-extras component/parameter renames.
    """

    def statespace(self, covariates):
        from pymc_extras.statespace import structural as st

        trend = st.LevelTrend(order=2, innovations_order=[0, 1])
        return (trend + st.MeasurementError()).build(verbose=False)

    def priors(self, ss_mod, covariates):
        for name, info in ss_mod.param_info.items():
            dims = info["dims"]
            size = {"dims": dims} if dims else {"shape": info["shape"]}
            if info["constraints"] == "Positive semi-definite":
                diag = pm.Gamma(f"{name}_diag", alpha=2, beta=5, dims=dims[0])
                pm.Deterministic(name, pt.diag(diag), dims=dims)
            elif info["constraints"] == "Positive":
                pm.Gamma(name, alpha=2, beta=10, **size)
            else:
                pm.Normal(name, 0.0, 1.0, **size)


def make_trend_data(t_obs: int = 30, horizon: int = 5, seed: int = SEED):
    """(data, covariates_full) pair: noisy linear trend + linear covariate."""
    rng = np.random.default_rng(seed)
    duration = t_obs + horizon
    trend = np.linspace(0.0, 3.0, duration)
    covariates = xr.DataArray(
        trend[:, None],
        dims=("time", "covariate"),
        coords={"time": np.arange(duration), "covariate": ["trend"]},
    )
    data = xr.DataArray(
        1.0 + 2.0 * trend[:t_obs] + rng.normal(0, 0.1, t_obs),
        dims=("time",),
        coords={"time": np.arange(t_obs)},
    )
    return data, covariates


def make_random_walk_data(t_obs: int = 40, horizon: int = 5, seed: int = SEED):
    """(data, covariates_full) pair for the random-walk models."""
    rng = np.random.default_rng(seed)
    duration = t_obs + horizon
    data = xr.DataArray(
        np.cumsum(rng.normal(0.125, 0.02, t_obs)),
        dims=("time",),
        coords={"time": np.arange(t_obs)},
    )
    covariates = xr.DataArray(
        np.zeros((duration, 0)),
        dims=("time", "covariate"),
        coords={"time": np.arange(duration)},
    )
    return data, covariates

"""Functional fitters share one code path with the forecaster classes."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from example_models import LocalLevelStatespace

from pymc_forecast.exceptions import MethodResolutionError, OptionalDependencyError
from pymc_forecast.fit import draw_posterior, fit_mcmc, fit_vi
from pymc_forecast.forecaster import Forecaster, HMCForecaster
from pymc_forecast.model import Horizon, innovations, predict
from pymc_forecast.statespace import StatespaceForecaster


def local_level(covariates, data=None):
    """Local level: cumulative drift innovations, observed with noise."""
    h = Horizon.from_data(covariates, data)
    sigma = pm.HalfNormal("sigma", 1.0)
    drift = innovations(h, "drift", pm.Normal.dist(0.0, 0.5))
    predict(
        h,
        lambda name, mu, dims, observed: pm.Normal(name, mu, sigma, dims=dims, observed=observed),
        pt.cumsum(drift),
    )


def conjugate_normal(covariates, data=None):
    """One shared mean, unit observation noise — a tiny conjugate model."""
    h = Horizon.from_data(covariates, data)
    theta = pm.Normal("theta", 0.0, 1.0)
    predict(
        h,
        lambda name, mu, dims, observed: pm.Normal(name, mu, 1.0, dims=dims, observed=observed),
        pt.ones(h.duration) * theta,
    )


def _series(n=8, value=1.0):
    data = xr.DataArray(np.full(n, value), dims="time", coords={"time": np.arange(n)})
    covariates = xr.DataArray(
        np.zeros((n, 0)),
        dims=("time", "covariate"),
        coords={"time": np.arange(n)},
    )
    return data, covariates


def test_fit_vi_returns_approx_and_losses_without_idata():
    from pymc_forecast.fit import FitResult, fit_vi

    data, cov = _series()
    result = fit_vi(local_level, data, cov, num_steps=5, random_seed=0, progressbar=False)
    assert isinstance(result, FitResult)
    assert result.approx is not None
    assert result.idata is None
    assert len(np.asarray(result.losses)) == 5


def test_draw_posterior_has_chain_and_draw_dims():
    from pymc_forecast.fit import draw_posterior, fit_vi

    data, cov = _series()
    result = fit_vi(local_level, data, cov, num_steps=5, random_seed=0, progressbar=False)
    posterior = draw_posterior(result, 20, random_seed=0)
    assert "chain" in posterior.dims
    assert "draw" in posterior.dims
    assert posterior.sizes["draw"] == 20


def test_fit_mcmc_matches_hmc_forecaster_free_rvs():
    from pymc_forecast.fit import fit_mcmc

    data, cov = _series(n=6, value=0.5)
    kwargs = dict(draws=10, tune=10, chains=1, random_seed=0, progressbar=False)
    result = fit_mcmc(conjugate_normal, data, cov, **kwargs)
    forecaster = HMCForecaster(conjugate_normal, data, cov, **kwargs)
    model_names = {rv.name for rv in forecaster.model.free_RVs}
    assert model_names <= set(result.idata.posterior.data_vars)
    assert model_names <= set(forecaster.idata.posterior.data_vars)
    assert {name for name in result.idata.posterior.data_vars if name in model_names} == {
        name for name in forecaster.idata.posterior.data_vars if name in model_names
    }


def test_forecaster_losses_match_fit_vi_for_the_same_seed():
    from pymc_forecast.fit import fit_vi

    data, cov = _series(n=4, value=0.3)
    kwargs = dict(num_steps=20, random_seed=0, progressbar=False)
    forecaster = Forecaster(conjugate_normal, data, cov, **kwargs)
    result = fit_vi(conjugate_normal, data, cov, **kwargs)
    np.testing.assert_array_equal(np.asarray(forecaster.losses), np.asarray(result.losses))


def test_jax_backend_still_requires_advi():
    from pymc_forecast.fit import fit_vi

    data, cov = _series(n=4)
    with pytest.raises(MethodResolutionError, match="method='advi'"):
        fit_vi(
            conjugate_normal,
            data,
            cov,
            method="fullrank_advi",
            backend="jax",
            num_steps=5,
        )


def test_fit_pathfinder_raises_when_extra_is_missing(monkeypatch):
    import builtins

    from pymc_forecast.fit import fit_pathfinder

    real_import = builtins.__import__

    def deny_extras(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pymc_extras" or name.startswith("pymc_extras."):
            raise ImportError("pymc_extras missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", deny_extras)
    data, cov = _series(n=4)
    with pytest.raises(OptionalDependencyError, match="pymc-extras"):
        fit_pathfinder(conjugate_normal, data, cov, random_seed=0)


def test_statespace_fit_passes_the_kalman_model(monkeypatch):
    from pymc_forecast.fit import FitResult, fit_mcmc

    captured = {}

    def fake_fit_mcmc(*args, model=None, **kwargs):
        captured["model"] = model
        captured["called_build_model"] = "build_model" in fit_mcmc.__code__.co_names
        return FitResult(idata=None, approx=None, losses=None, method="mcmc")

    monkeypatch.setattr("pymc_forecast.forecaster.fit_mcmc", fake_fit_mcmc)
    data, cov = _series(n=6)
    forecaster = StatespaceForecaster(
        LocalLevelStatespace(),
        data,
        cov,
        draws=1,
        tune=1,
        chains=1,
        random_seed=0,
        progressbar=False,
    )
    assert "_fit" not in StatespaceForecaster.__dict__
    assert StatespaceForecaster._fit is HMCForecaster._fit
    assert captured["model"] is forecaster.model
    assert forecaster.model is not None


@pytest.mark.parametrize(
    "fit",
    [
        lambda data, cov: fit_vi(local_level, data, cov, num_steps=5, random_seed=0),
        lambda data, cov: fit_mcmc(
            local_level, data, cov, draws=5, tune=5, chains=1, random_seed=0
        ),
    ],
    ids=["vi", "mcmc"],
)
def test_fitters_drop_covariate_rows_past_the_training_window(fit):
    """Horizon covariates must not put future latents into the posterior."""
    data, _ = _series(n=6)
    full = xr.DataArray(np.zeros((9, 0)), dims=("time", "covariate"), coords={"time": np.arange(9)})
    posterior = draw_posterior(fit(data, full), 4, random_seed=0)
    assert "drift_future" not in posterior.data_vars
    assert "forecast" not in posterior.data_vars

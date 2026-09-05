"""Regression cases for coordinate mistakes that otherwise produce plausible forecasts."""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from pymc_forecast import build_model, predict
from pymc_forecast.exceptions import AlignmentError
from pymc_forecast.forecaster import BaseForecaster


def regression(h, covariates):
    beta = pm.Normal("beta", 0, 1, dims="covariate")
    predict(
        h,
        lambda name, mu, dims, obs: pm.Normal(name, mu, 1, dims=dims, observed=obs),
        covariates.values @ beta,
    )


class FixedForecaster(BaseForecaster):
    def _fit(self, random_seed):
        pass

    def _draw_posterior(self, num_samples, random_seed=None):
        raise AssertionError("invalid coordinates must fail before posterior sampling")


@pytest.fixture
def covariates():
    return xr.DataArray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dims=("time", "covariate"),
        coords={"time": [0, 1, 2], "covariate": ["a", "b"]},
    )


@pytest.mark.parametrize("change", ["reorder", "rename", "width", "unlabeled", "dims"])
def test_full_covariate_structure_rejected_before_sampling(covariates, change):
    fc = FixedForecaster(regression, np.array([101.0, 202.0]), covariates)
    bad = {
        "reorder": covariates.sel(covariate=["b", "a"]),
        "rename": covariates.assign_coords(covariate=["c", "d"]),
        "width": covariates.isel(covariate=[0]),
        "unlabeled": covariates.drop_vars("covariate"),
        "dims": covariates.rename(covariate="feature"),
    }[change]
    with pytest.raises(
        AlignmentError, match=r"coords must match|size mismatch|unlabeled|same dims"
    ):
        fc.forecast(bad)


def test_full_covariates_replay_fixed_regression_coefficients(covariates):
    fc = FixedForecaster(regression, np.array([101.0, 202.0]), covariates)
    posterior = xr.Dataset(
        {"beta": (("chain", "draw", "covariate"), np.tile([1.0, 10.0], (1, 3, 1)))},
        coords={"chain": [0], "draw": [0, 1, 2], "covariate": ["a", "b"]},
    )
    for kwargs in ({"covariates": covariates}, {"future_covariates": covariates.isel(time=[2])}):
        result = fc.forecast(**kwargs, posterior=posterior, random_seed=1)
        np.testing.assert_array_equal(result["predictions"]["mu_future"], 303.0)


@pytest.mark.parametrize("kind", ["reorder", "size", "unlabeled"])
def test_shared_panel_coordinates_rejected_before_model_body(kind):
    data = xr.DataArray(
        [[1.0, 10.0], [2.0, 20.0]],
        dims=("time", "series"),
        coords={"time": [0, 1], "series": ["a", "b"]},
    )
    cov = {
        "reorder": data.sel(series=["b", "a"]),
        "size": data.isel(series=[0]),
        "unlabeled": data.drop_vars("series"),
    }[kind]

    def model(h, covariates):
        raise AssertionError("coordinate errors must fail before the model is built")

    with pytest.raises(AlignmentError, match=r"coords must match|size mismatch|unlabeled"):
        build_model(model, data, cov)


@pytest.mark.parametrize(
    "index",
    [
        pd.period_range("2024-01", periods=2, freq="M"),
        pd.date_range("2024-01-01", periods=2, freq="D"),
    ],
)
def test_forecast_preserves_short_index_frequency(index):
    def model(h, covariates):
        theta = pm.Normal("theta")
        predict(
            h,
            lambda name, mu, dims, obs: pm.Normal(name, mu, 1, dims=dims, observed=obs),
            pt.ones(h.duration) * theta,
        )

    fc = FixedForecaster(model, pd.Series([0.0, 0.0], index=index))
    posterior = xr.Dataset({"theta": (("chain", "draw"), [[0.0, 0.0]])})
    result = fc.forecast(horizon=1, posterior=posterior, random_seed=1)
    assert result["predictions"].get_index("time_future")[0] == index[-1] + index.freq

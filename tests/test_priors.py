from typing import ClassVar

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from pymc_forecast.model import ForecastingModel, build_model
from pymc_forecast.priors import PriorConfig
from pymc_forecast.statespace import StatespaceModel


class ConfigurableRegression(ForecastingModel):
    def model(self, h, covariates):
        intercept = self.create_prior("intercept")
        beta = self.create_prior("beta")
        sigma = self.create_prior("sigma")
        mu = intercept + pt.dot(covariates.values, beta)
        self.predict(
            lambda name, value, dims, observed: pm.Normal(
                name, value, sigma, dims=dims, observed=observed
            ),
            mu,
        )


def _configured_model():
    from pymc_extras.prior import Prior

    ConfigurableRegression.default_priors = {
        "intercept": Prior("Normal", mu=0, sigma=2),
        "beta": Prior("Normal", mu=0, sigma=1, dims="covariate"),
        "sigma": Prior("HalfNormal", sigma=1),
    }
    return ConfigurableRegression()


def _data_and_covariates():
    data = xr.DataArray(np.arange(3.0), dims="time", coords={"time": [0, 1, 2]})
    covariates = xr.DataArray(
        np.arange(5.0)[:, None],
        dims=("time", "covariate"),
        coords={"time": [0, 1, 2, 10, 20], "covariate": ["trend"]},
    )
    return data, covariates


def test_pymc_extras_prior_defaults_can_be_overridden():
    from pymc_extras.prior import Prior

    model = _configured_model()
    override = Prior("Normal", mu=5, sigma=0.1, dims="covariate")
    model = ConfigurableRegression(priors={"beta": override})
    assert model.prior_config["beta"] is override
    assert ConfigurableRegression.default_priors["beta"] is not override

    data, covariates = _data_and_covariates()
    training = build_model(model, data, covariates.isel(time=slice(None, 3)))
    forecasting = build_model(model, data, covariates)
    assert {"intercept", "beta", "sigma", "obs"} <= set(training.named_vars)
    assert "forecast" in forecasting.named_vars


def test_callable_prior_protocol():
    class CallableModel(PriorConfig):
        default_priors: ClassVar = {"offset": lambda name: pm.Normal(name, 0, 1)}

    configured = CallableModel()
    with pm.Model() as model:
        variable = configured.create_prior("offset")
    assert variable.name == "offset"
    assert "offset" in model.named_vars


def test_missing_and_invalid_prior_errors_are_actionable():
    configured = PriorConfig(priors={"bad": object()})
    with pytest.raises(KeyError, match="available priors"):
        configured.create_prior("missing")
    with pytest.raises(TypeError, match="create_variable"):
        configured.create_prior("bad")


def test_statespace_models_share_prior_configuration():
    class ConfiguredStatespace(StatespaceModel):
        default_priors: ClassVar = {"scale": lambda name: pm.HalfNormal(name, 1)}

        def statespace(self, data, covariates):
            raise NotImplementedError

        def priors(self, ss_mod, data, covariates):
            self.create_prior("scale")

    def override(name):
        return pm.HalfNormal(name, 0.25)

    model = ConfiguredStatespace(priors={"scale": override})
    assert model.prior_config["scale"] is override

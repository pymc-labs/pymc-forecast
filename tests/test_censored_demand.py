"""The censored-demand notebook's filter, likelihood, and forecast registration.

The percent file is the source the tests execute. ``docs/examples/censored_demand.ipynb``
is the executed gallery copy.
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy.stats import norm

from pymc_forecast import ForecastingModel, build_model, forecast, fourier_features, ssoe

NOTEBOOK = Path(__file__).resolve().parents[1] / "docs" / "examples" / "censored_demand.py"


def model_namespace():
    """Execute the notebook cell that defines the censored AR(2)."""
    source = NOTEBOOK.read_text()
    cells = []
    current = []
    kind = None
    for line in source.splitlines():
        if line.startswith("# %%"):
            if kind == "code" and current:
                cells.append("\n".join(current))
            current = []
            kind = "markdown" if "markdown" in line else "code"
            continue
        if kind == "code":
            current.append(line)
    if kind == "code" and current:
        cells.append("\n".join(current))
    cell = next(text for text in cells if "class CensoredAR2" in text)
    namespace = {
        "np": np,
        "pm": pm,
        "pt": pt,
        "ForecastingModel": ForecastingModel,
        "ssoe": ssoe,
    }
    exec(cell, namespace)
    return namespace


def covariates(n, available=None, censored=None):
    time = np.arange(n)
    if available is None:
        available = np.ones(n)
    if censored is None:
        censored = np.zeros(n)
    fourier = fourier_features(time, period=7.0, num_terms=2).rename(fourier="input")

    def channel(name, values):
        return xr.DataArray(
            np.asarray(values, dtype=float)[:, None],
            dims=("time", "input"),
            coords={"time": time, "input": [name]},
        )

    return xr.concat(
        [channel("availability", available), channel("censored", censored), fourier],
        dim="input",
    )


def series(values):
    values = np.asarray(values, dtype=float)
    return xr.DataArray(values, dims="time", coords={"time": np.arange(len(values))})


@pytest.fixture(scope="module")
def ns():
    return model_namespace()


def test_filter_keeps_clean_days_and_replaces_corrupted_lags(ns):
    carried = ns["filtered_lag"](
        pt.as_tensor_variable([1.5, 2.2, 2.2, 0.0, -0.4]),
        pt.as_tensor_variable([0.4, 3.0, 1.0, 1.7, 0.2]),
        pt.as_tensor_variable([1.0, 1.0, 1.0, 0.0, 1.0]),
        pt.as_tensor_variable([0.0, 1.0, 1.0, 0.0, 0.0]),
    ).eval()
    np.testing.assert_allclose(carried, [1.5, 3.0, 2.2, 1.7, 0.0])


def test_likelihood_is_density_survival_or_zero(ns):
    value = pt.as_tensor_variable([0.0, 2.2, 9.0])
    mu = pt.as_tensor_variable([0.0, 1.0, 1.0])
    sigma = pt.as_tensor_variable([1.0, 0.5, 1.0])
    censored = pt.as_tensor_variable([0.0, 1.0, 1.0])
    valid = pt.as_tensor_variable([1.0, 1.0, 0.0])
    got = ns["censored_logp"](value, mu, sigma, censored, valid).eval()
    expected = [
        norm.logpdf(0.0, 0.0, 1.0),
        norm.logsf(2.2, 1.0, 0.5),
        0.0,
    ]
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("mu_value", [0.0, 2.2, 4.4])
def test_censored_likelihood_and_gradients_remain_finite_in_tails(ns, mu_value):
    value, sigma_value = 2.2, 0.05
    mu, sigma = pt.dscalars("mu", "sigma")
    logp = ns["censored_logp"](value, mu, sigma, 1, 1)
    evaluate = pytensor.function([mu, sigma], [logp, *pt.grad(logp, [mu, sigma])])
    got = evaluate(mu_value, sigma_value)

    z = (value - mu_value) / sigma_value
    log_survival = norm.logsf(z)
    hazard = np.exp(norm.logpdf(z) - log_survival)
    expected = [log_survival, hazard / sigma_value, z * hazard / sigma_value]
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-7)


def test_swapped_input_labels_are_rejected(ns):
    data = series([1.0, 1.1, 1.2])
    labels = ["censored", "availability", "sin_1", "sin_2", "cos_1", "cos_2"]
    swapped = covariates(3).reindex(input=labels)
    with pytest.raises(ValueError, match="expected inputs"):
        build_model(ns["CensoredAR2"](), data, swapped)


def test_stockout_value_does_not_change_the_likelihood(ns):
    available = [1.0, 1.0, 0.0, 1.0]
    censored = [0.0, 0.0, 0.0, 1.0]
    first = build_model(
        ns["CensoredAR2"](),
        series([1.0, 1.2, 0.0, 2.2]),
        covariates(4, available, censored),
    )
    second = build_model(
        ns["CensoredAR2"](),
        series([1.0, 1.2, 9.0, 2.2]),
        covariates(4, available, censored),
    )
    point = first.initial_point()
    assert first.compile_logp()(point) == pytest.approx(second.compile_logp()(point))


def test_cap_flag_switches_the_last_day_from_density_to_survival(ns):
    data = series([1.0, 1.2, 2.2])
    censored = build_model(ns["CensoredAR2"](), data, covariates(3, censored=[0.0, 0.0, 1.0]))
    naive = build_model(ns["CensoredAR2"](), data, covariates(3, censored=[0.0, 0.0, 0.0]))
    point = censored.initial_point()
    censored_logp = censored.compile_logp()(point)
    naive_logp = naive.compile_logp()(point)
    assert censored_logp != pytest.approx(naive_logp)
    mu, sigma = censored.replace_rvs_by_values([censored["mu"], censored["sigma"]])
    mu_val, sigma_val = censored.compile_fn(
        [mu, sigma],
        inputs=censored.value_vars,
        on_unused_input="ignore",
    )(point)
    expected_gap = norm.logsf(2.2, mu_val[-1], sigma_val) - norm.logpdf(2.2, mu_val[-1], sigma_val)
    assert censored_logp - naive_logp == pytest.approx(expected_gap)


def test_training_build_has_no_future_error_and_forecast_is_clipped(ns):
    data = series([1.0, 1.2, 0.8, 2.2, 1.1])
    training = build_model(ns["CensoredAR2"](), data, covariates(5))
    assert "eps_future" not in {rv.name for rv in training.free_RVs}
    assert "forecast" not in training.named_vars
    training.compile_dlogp()(training.initial_point())

    forecasting = build_model(ns["CensoredAR2"](), data, covariates(8))
    assert "eps_future" in {rv.name for rv in forecasting.free_RVs}
    labels = ["sin_1", "sin_2", "cos_1", "cos_2"]
    posterior = xr.Dataset(
        {
            "intercept": (("chain", "draw"), [[0.0]]),
            "phi_1": (("chain", "draw"), [[0.0]]),
            "phi_2": (("chain", "draw"), [[0.0]]),
            "sigma": (("chain", "draw"), [[4.0]]),
            "beta_seasonal": (("chain", "draw", "fourier"), np.zeros((1, 1, 4))),
        },
        coords={"fourier": labels},
    )
    draws = forecast(
        ns["CensoredAR2"](),
        posterior,
        data,
        covariates(8),
        random_seed=3,
    ).predictions
    raw = draws.mu_future + draws.eps_future
    assert np.any(raw.values < 0)
    np.testing.assert_allclose(draws.forecast, np.maximum(raw, 0))


def test_future_availability_is_read_by_the_filter(ns):
    data = series([1.5, 1.4, 1.6, 1.3, 1.5])
    posterior = xr.Dataset(
        {
            "intercept": (("chain", "draw"), [[2.0]]),
            "phi_1": (("chain", "draw"), [[0.5]]),
            "phi_2": (("chain", "draw"), [[0.0]]),
            "sigma": (("chain", "draw"), [[1.0]]),
            "beta_seasonal": (("chain", "draw", "fourier"), np.zeros((1, 1, 4))),
        },
        coords={"fourier": ["sin_1", "sin_2", "cos_1", "cos_2"]},
    )
    on_shelf = covariates(7)
    off_shelf = covariates(7, available=[1, 1, 1, 1, 1, 0, 1])
    on_draws = forecast(ns["CensoredAR2"](), posterior, data, on_shelf, random_seed=3).predictions
    off_draws = forecast(ns["CensoredAR2"](), posterior, data, off_shelf, random_seed=3).predictions
    np.testing.assert_allclose(
        on_draws.forecast.isel(time_future=0), off_draws.forecast.isel(time_future=0)
    )
    mu0 = on_draws.mu_future.isel(time_future=0).values
    y0 = mu0 + on_draws.eps_future.isel(time_future=0).values
    expected = 0.5 * (np.maximum(y0, 0.0) - np.maximum(mu0, 0.0))
    got = (on_draws.mu_future.isel(time_future=1) - off_draws.mu_future.isel(time_future=1)).values
    np.testing.assert_allclose(got, expected)
    assert float(np.max(np.abs(expected))) > 1e-3


def test_prior_predictive_draws_are_not_clipped_at_the_cap(ns):
    model = build_model(
        ns["CensoredAR2"](),
        series([1.0, 1.2, 2.2, 0.4]),
        covariates(4, censored=[0.0, 0.0, 1.0, 0.0]),
    )
    with model:
        draws = pm.sample_prior_predictive(150, random_seed=1, var_names=["obs"])
    assert float(draws.prior_predictive.obs.max()) > 2.2

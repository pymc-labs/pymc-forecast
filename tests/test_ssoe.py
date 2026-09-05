"""Analytical recursion, posterior replay and labeled-input checks for SSOE."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from pymc_forecast import (
    AlignmentError,
    HMCForecaster,
    Horizon,
    HorizonError,
    build_model,
    forecast,
    null_covariates,
    predict_in_sample,
    ssoe,
)


def history(values=(1.0, 2.0, 4.0)):
    return xr.DataArray(np.asarray(values), dims="time", coords={"time": np.arange(len(values))})


def arma(h, covariates):
    phi = pm.Normal("phi", 0, 0.5)
    theta = pm.Normal("theta", 0, 0.5)
    sigma = pm.HalfNormal("sigma", 1)
    result = ssoe(
        h,
        "eps",
        (0.0, 0.0),
        mean=lambda state, x, phi, theta: phi * state[0] + theta * state[1],
        update=lambda state, y, eps, x, phi, theta: (y, eps),
        noise_fn=lambda name, dims: pm.Normal(name, 0, sigma, dims=dims),
        params=(phi, theta),
    )
    pm.Normal("obs", result.mu, sigma, observed=h.data.values, dims="time")
    pm.Deterministic("mu", result.mu, dims="time")
    if h.future:
        pm.Deterministic("mu_future", result.mu_future, dims="time_future")
        pm.Deterministic("forecast", result.y_future, dims="time_future")


def fixed_posterior(draws=2000):
    return xr.Dataset(
        {
            name: (("chain", "draw"), np.full((1, draws), value))
            for name, value in {"phi": 0.6, "theta": 0.3, "sigma": 0.5}.items()
        }
    )


def test_filter_logp_gradient_and_empty_future():
    data = history()
    model = build_model(arma, data, null_covariates(data.time))
    assert {rv.name for rv in model.free_RVs} == {"phi", "theta", "sigma"}
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.isfinite(model.compile_dlogp()(model.initial_point())).all()
    result = predict_in_sample(arma, fixed_posterior(3), data, random_seed=8)
    # Independently calculate the ARMA one-step innovations.
    expected, prev_y, prev_eps = [], 0.0, 0.0
    for y in data.values:
        mu = 0.6 * prev_y + 0.3 * prev_eps
        expected.append(mu)
        prev_y, prev_eps = y, y - mu
    np.testing.assert_allclose(result.posterior_predictive.mu.values, [[expected] * 3])


def test_forecast_replays_parameters_and_propagates_errors():
    data = history()
    covariates = null_covariates(np.arange(7))
    draws = forecast(arma, fixed_posterior(), data, covariates, random_seed=71).predictions
    np.testing.assert_array_equal(draws.time_future, [3, 4, 5, 6])
    assert draws.forecast.dims == ("chain", "draw", "time_future")
    # In-sample errors are 1, 1.1, 2.47; the final one seeds the forecast.
    first_mean = 0.6 * 4 + 0.3 * 2.47
    np.testing.assert_allclose(draws.mu_future.isel(time_future=0), first_mean)
    np.testing.assert_allclose(draws.forecast - draws.mu_future, draws.eps_future)
    np.testing.assert_allclose(
        draws.mu_future.values[..., 1:],
        0.6 * draws.forecast.values[..., :-1] + 0.3 * draws.eps_future.values[..., :-1],
    )
    # ARMA impulse coefficients: psi_0=1, psi_j=(phi+theta)*phi**(j-1).
    psi = np.r_[1.0, 0.9 * 0.6 ** np.arange(3)]
    expected_variances = 0.5**2 * np.cumsum(psi**2)
    np.testing.assert_allclose(
        draws.forecast.var(("chain", "draw")),
        expected_variances,
        rtol=0.12,
    )


def test_future_inputs_and_permuted_panel_dims():
    y = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("series", "time"),
        coords={"series": ["a", "b"], "time": [0, 1]},
    )
    xs = xr.DataArray(
        [[0, 0, 10, 20], [0, 0, 30, 40]],
        dims=("series", "time"),
        coords={"series": ["a", "b"], "time": [0, 1, 2, 3]},
    )

    def model_fn(h, cov):
        r = ssoe(
            h,
            "eps",
            np.zeros(2),
            mean=lambda state, x: state + x,
            update=lambda state, y, eps, x: y,
            noise_fn=lambda n, d: pm.MvNormal(n, np.zeros(2), np.eye(2), dims=d),
            xs=xs,
        )
        pm.Deterministic("means", r.mu, dims=("time", "series"))
        pm.Normal("obs", r.mu, 1, observed=h.data.values, dims=("time", "series"))
        if h.future:
            pm.Deterministic("forecast", r.y_future, dims=("time_future", "series"))

    model = build_model(model_fn, y, null_covariates(np.arange(4)))
    fn = pytensor.function([model["eps_future"]], model["forecast"])
    np.testing.assert_allclose(fn(np.zeros((2, 2))), [[12, 34], [32, 74]])
    np.testing.assert_allclose(model["means"].eval(), [[0, 0], [1, 3]])


def test_zero_horizon_and_single_observation():
    h = Horizon(history([2.0]), np.arange(1))
    with pm.Model(coords={"time": h.time}):
        r = ssoe(
            h,
            "eps",
            0.0,
            lambda state, x: state,
            lambda state, y, eps, x: y,
            lambda n, d: pytest.fail("noise factory must not be called during training"),
        )
        assert r.dims == ()
        assert r.mu.eval().shape == (1,)
        assert r.mu_future.eval().shape == r.y_future.eval().shape == (0,)


def test_gate_freezes_state_and_no_future_data_leakage():
    h = Horizon(history([2.0, 999.0, 4.0]), np.arange(3), np.arange(3, 5))
    xs = xr.DataArray([1, 0, 1, 0, 0], dims="time", coords={"time": np.arange(5)})
    with pm.Model(coords={"time": h.time, "time_future": h.time_future}) as model:
        r = ssoe(
            h,
            "eps",
            0.0,
            lambda state, x: state,
            lambda state, y, eps, x: pt.where(x, y, state),
            lambda n, d: pm.Normal(n, 0, 1, dims=d),
            xs=xs,
        )
        fn = pytensor.function([model["eps_future"]], [r.mu, r.mu_future, r.y_future])
        mu, future_mu, future_y = fn([1.0, 2.0])
    np.testing.assert_allclose(mu, [0, 2, 2])
    np.testing.assert_allclose(future_mu, [4, 4])
    np.testing.assert_allclose(future_y, [5, 6])


@pytest.mark.parametrize("bad", [None, history([])])
def test_missing_history_rejected(bad):
    h = Horizon(bad, np.arange(0 if bad is not None else 3))
    with pm.Model(coords={"time": h.time}), pytest.raises(HorizonError, match="history"):
        ssoe(h, "eps", 0, lambda s, x: s, lambda s, y, e, x: y, None)


@pytest.mark.parametrize("what", ["y_time", "xs_time", "xs_short", "batch", "raw_xs", "nan"])
def test_invalid_labeled_inputs_rejected(what):
    y = history()
    xs = xr.DataArray(np.ones(5), dims="time", coords={"time": np.arange(5)})
    h = Horizon(y, np.arange(3), np.arange(3, 5))
    if what == "y_time":
        y = y.assign_coords(time=[1, 2, 3])
    elif what == "xs_time":
        xs = xs.assign_coords(time=[1, 0, 2, 3, 4])
    elif what == "xs_short":
        xs = xs.isel(time=slice(3))
    elif what == "batch":
        y = y.expand_dims(series=["wrong"])
    elif what == "raw_xs":
        xs = np.ones(5)
    else:
        y = y.copy(data=[1.0, np.nan, 3.0])
    with pm.Model(coords={"time": h.time, "time_future": h.time_future, "series": ["a"]}):
        with pytest.raises((AlignmentError, ValueError)):
            ssoe(h, "eps", 0, lambda s, x: s, lambda s, y, e, x: y, None, y=y, xs=xs)


@pytest.mark.parametrize("bad", ["mean_shape", "state_shape", "state_structure", "random"])
def test_callback_contract(bad):
    def mean(state, x):
        if bad == "mean_shape":
            return pt.zeros(3)
        if bad == "random":
            return pm.Normal.dist()
        return state

    def update(state, y, eps, x):
        if bad == "state_shape":
            return pt.zeros(2)
        if bad == "state_structure":
            return (y,)
        return y

    h = Horizon(history(), np.arange(3))
    with pm.Model(coords={"time": h.time}), pytest.raises(ValueError):
        ssoe(h, "eps", 0.0, mean, update, None)


def test_hmc_forecaster_uses_scan_gradients_and_shared_protocol():
    rng = np.random.default_rng(5)
    y = np.zeros(30)
    for i in range(1, len(y)):
        y[i] = 0.6 * y[i - 1] + rng.normal(0, 0.3)
    fc = HMCForecaster(
        arma, history(y), draws=30, tune=30, chains=1, progressbar=False, random_seed=55
    )
    posterior = fc.draw_posterior(20, random_seed=3)
    result = fc.forecast(horizon=2, posterior=posterior, random_seed=9)
    assert result.predictions.forecast.shape == (1, 20, 2)
    assert np.isfinite(result.predictions.forecast).all()
    # Real fitted posteriors also contain deterministic training means. Those
    # must not break replay of the parameters into the final filtered state.
    previous_y = previous_error = 0.0
    for observed in y:
        mean = posterior.phi.values * previous_y + posterior.theta.values * previous_error
        previous_y, previous_error = observed, observed - mean
    expected = posterior.phi.values * previous_y + posterior.theta.values * previous_error
    np.testing.assert_allclose(result.predictions.mu_future.isel(time_future=0), expected)


def test_holt_winters_matches_reference_with_mixed_state_shapes():
    """One shared update reproduces the original two-scan Holt-Winters model."""
    data = history([2.0, 1.0, 3.0, 2.5])
    h = Horizon(data, np.arange(4), np.arange(4, 7))
    initial = (1.0, 0.2, np.array([0.3, -0.3]))
    alpha, beta, gamma, phi = 0.4, 0.1, 0.2, 0.8
    future_errors = np.array([0.5, -0.2, 0.0])
    state = initial
    reference_means, reference_y = [], []
    for i in range(h.duration):
        level, trend, seasons = state
        mu = level + phi * trend + seasons[0]
        error = data.values[i] - mu if i < h.t_obs else future_errors[i - h.t_obs]
        reference_means.append(mu)
        reference_y.append(mu + error)
        state = (
            level + phi * trend + alpha * error,
            phi * trend + beta * error,
            np.r_[seasons[1:], seasons[0] + gamma * error],
        )

    def update(state, y, error, x):
        level, trend, seasons = state
        return (
            level + phi * trend + alpha * error,
            phi * trend + beta * error,
            pt.concatenate([seasons[1:], (seasons[0] + gamma * error)[None]]),
        )

    with pm.Model(coords={"time": h.time, "time_future": h.time_future}) as model:
        r = ssoe(
            h,
            "eps",
            initial,
            lambda s, x: s[0] + phi * s[1] + s[2][0],
            update,
            lambda n, d: pm.Normal(n, 0, 1, dims=d),
        )
        fn = pytensor.function([model["eps_future"]], [r.mu, r.mu_future, r.y_future])
        mu, future_mu, future_y = fn(future_errors)
    np.testing.assert_allclose(mu, reference_means[: h.t_obs])
    np.testing.assert_allclose(future_mu, reference_means[h.t_obs :])
    np.testing.assert_allclose(future_y, reference_y[h.t_obs :])


def test_future_error_factory_must_preserve_named_dims():
    h = Horizon(history(), np.arange(3), np.arange(3, 5))
    with pm.Model(coords={"time": h.time, "time_future": h.time_future}):
        with pytest.raises(AlignmentError, match="supplied dimensions"):
            ssoe(
                h,
                "eps",
                0.0,
                lambda s, x: s,
                lambda s, y, e, x: y,
                lambda n, d: pm.Normal(n, 0, 1, shape=2),
            )

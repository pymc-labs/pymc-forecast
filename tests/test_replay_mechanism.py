"""Spike: validate the posterior-replay forecasting mechanism (issue #2).

The package's central invariant, inherited from pyro/numpyro_forecast: one model
definition both trains and forecasts. In-sample time latents keep fixed names;
the forecast horizon uses separate ``{name}_future`` variables. Because those
future variables are absent from the posterior trace,
``pm.sample_posterior_predictive`` must

1. replay the in-sample latents from the trace (NOT resample them), and
2. draw the ``*_future`` variables (and everything downstream) fresh.

These tests pin that behavior down for both NUTS and ADVI posteriors, on a
random-walk model where breakage is unmistakable: the data trends from 0 to ~5,
so a replayed forecast starts near 5 while a from-the-prior forecast starts
near 0.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

SEED = 1234
T_OBS = 40
HORIZON = 5
DRIFT_PER_STEP = 5.0 / T_OBS


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.cumsum(rng.normal(DRIFT_PER_STEP, 0.02, size=T_OBS))


def build_model(data: np.ndarray, future: int) -> pm.Model:
    """Random-walk model: level = cumsum(drift); obs prefix, forecast suffix."""
    t_obs = len(data)
    coords: dict = {"time": np.arange(t_obs)}
    if future:
        coords["time_future"] = np.arange(t_obs, t_obs + future)
    with pm.Model(coords=coords) as model:
        drift_loc = pm.Normal("drift_loc", 0.0, 1.0)
        drift = pm.Normal("drift", drift_loc, 0.1, dims="time")
        pieces = [drift]
        if future:
            drift_future = pm.Normal("drift_future", drift_loc, 0.1, dims="time_future")
            pieces.append(drift_future)
        level = pt.cumsum(pt.concatenate(pieces))
        pm.Normal("obs", level[:t_obs], 0.05, observed=data, dims="time")
        if future:
            pm.Normal("forecast", level[t_obs:], 0.05, dims="time_future")
    return model


@pytest.fixture(scope="module")
def nuts_posterior(data):
    with build_model(data, future=0):
        return pm.sample(
            draws=200, tune=300, chains=2, progressbar=False, random_seed=SEED
        )


@pytest.fixture(scope="module")
def advi_posterior(data):
    approx = pm.fit(
        n=20_000, method="advi", model=build_model(data, future=0), random_seed=SEED,
        progressbar=False,
    )
    return approx.sample(400, random_seed=SEED)


def _forecast(posterior, data):
    model = build_model(data, future=HORIZON)
    return pm.sample_posterior_predictive(
        posterior,
        model=model,
        var_names=["forecast", "drift_future"],
        predictions=True,
        progressbar=False,
        random_seed=SEED,
    )


@pytest.mark.parametrize("kind", ["nuts", "advi"])
def test_forecast_continues_from_replayed_state(kind, data, request):
    """In-sample latents replayed: the forecast starts at the last level (~5).

    The tolerance is loose (mean-field ADVI accumulates per-step variational
    noise through the cumsum) but decisive: a broken replay resamples the
    in-sample drift from its prior and the forecast starts near 0, not ~5.
    """
    posterior = request.getfixturevalue(f"{kind}_posterior")
    pred = _forecast(posterior, data)["predictions"]
    first_step = pred["forecast"].isel(time_future=0)
    assert abs(float(first_step.mean()) - data[-1]) < 0.75, (
        "forecast does not continue from the fitted level; in-sample latents "
        "were probably resampled from the prior instead of replayed"
    )
    assert set(pred["forecast"].dims) >= {"time_future"}
    np.testing.assert_array_equal(
        pred["time_future"].values, np.arange(T_OBS, T_OBS + HORIZON)
    )


def test_future_latents_are_prior_draws_not_replay(nuts_posterior, data):
    """drift_future is drawn fresh (from prior given replayed drift_loc)."""
    pred = _forecast(nuts_posterior, data)["predictions"]
    drift_future = pred["drift_future"]
    # Fresh noise: per-step variance across draws must reflect the 0.1 prior
    # sigma (posterior over in-sample drift steps is far tighter, ~0.02).
    assert float(drift_future.std()) > 0.05
    # Centered at the *posterior* drift_loc, not the prior's 0: mean per step
    # should be near the true per-step drift.
    assert abs(float(drift_future.mean()) - DRIFT_PER_STEP) < 0.05


def test_nuts_and_advi_forecasts_agree(nuts_posterior, advi_posterior, data):
    """Upstream's consistency check: both inference paths give the same forecast."""
    pred_nuts = _forecast(nuts_posterior, data)["predictions"]["forecast"]
    pred_advi = _forecast(advi_posterior, data)["predictions"]["forecast"]
    np.testing.assert_allclose(
        pred_nuts.mean(("chain", "draw")).values,
        pred_advi.mean(("chain", "draw")).values,
        atol=0.75,
    )


def test_in_sample_posterior_predictive(nuts_posterior, data):
    """Rebuilding with future=0 and sampling 'obs' gives in-sample fits."""
    model = build_model(data, future=0)
    ppc = pm.sample_posterior_predictive(
        nuts_posterior, model=model, progressbar=False, random_seed=SEED
    )["posterior_predictive"]
    resid = ppc["obs"].mean(("chain", "draw")).values - data
    assert np.abs(resid).mean() < 0.1

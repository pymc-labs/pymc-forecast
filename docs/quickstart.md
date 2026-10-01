# Quickstart

One model definition serves fitting and forecasting: {func}`~pymc_forecast.innovations`
creates separate `{name}_future` latents that are absent from the fitted posterior, so
posterior predictive sampling replays the fitted parameters while drawing the horizon
forward. Write a model with one {func}`~pymc_forecast.predict` call, fit it with
{func}`~pymc_forecast.fit_vi`, and forecast:

```python
import numpy as np, pandas as pd, pymc as pm, pytensor.tensor as pt
from pymc_forecast import (
    ForecastingModel,
    Horizon,
    backtest,
    draw_posterior,
    evaluate_forecast,
    fit_mcmc,
    fit_vi,
    forecast,
    innovations,
    null_covariates,
    predict,
)

# a trending weekly series; hold out the last 8 weeks
dates = pd.date_range("2024-01-07", periods=60, freq="W")
y = pd.Series(np.cumsum(np.random.default_rng(0).normal(0.2, 1.0, 60)) + 10, index=dates)
train, test = y.iloc[:52], y.iloc[52:]


def model(covariates, data=None):
    h = Horizon.from_data(covariates, data)
    # a per-step drift latent; innovations adds the matching `_future` latent
    drift = innovations(h, "drift", pm.Normal.dist(0.0, 0.5))
    predict(h, pm.StudentT.dist(nu=3, sigma=1.0), pt.cumsum(drift))


result = fit_vi(model, train, num_steps=5_000, random_seed=0)  # ADVI
posterior = draw_posterior(result, 500, random_seed=0)
idata = forecast(model, posterior, train, null_covariates(dates), random_seed=0)
forecast_draws = idata["predictions"]["forecast"]  # dims: (chain, draw, time_future)

# score against the held-out weeks (aligned by dim name, not axis position)
truth = test.to_xarray().rename({"index": "time_future"})
print(evaluate_forecast(forecast_draws, truth))  # {'mae': ..., 'rmse': ..., 'crps': ..., 'coverage': ...}


# rolling-origin backtest over the whole series
results = backtest(
    y,
    None,
    model,
    min_train_window=48,
    test_window=4,
    stride=4,
    num_samples=200,
    forecaster_options={"num_steps": 3_000},
    random_seed=0,
)
```

`ForecastingModel` is the object-oriented wrapper around the same primitives,
not a second model definition:

```python
class LocalLevel(ForecastingModel):
    def model(self, covariates, data=None):
        drift = self.innovations("drift", pm.Normal.dist(0.0, 0.5))
        self.predict(pm.StudentT.dist(nu=3, sigma=1.0), pt.cumsum(drift))
```

For a non-identity link, keep the link-scale latent and provide the
outcome-scale expectation explicitly. A 1-argument callable receives the
windowed latent and returns a `.dist()`:

```python
h = Horizon.from_data(covariates, data)
eta = intercept + pt.dot(covariates, beta)
predict(
    h,
    # the callable receives the *windowed* latent, so it applies the inverse
    # link itself — it cannot reuse the full-horizon `pt.exp(eta)` below
    lambda eta_window: pm.Poisson.dist(pt.exp(eta_window)),
    eta,
    expected_observation=pt.exp(eta),
)
```

Predictions then include `mu` / `mu_future` (the supplied `eta`) and
`expected_observation` / `expected_observation_future` (expected counts);
all retain the same chain, draw, and time coordinates. See the
[prediction output schema](schema.md) for the full contract.

## Check VI convergence

{func}`~pymc_forecast.fit_vi` uses mean-field ADVI, which can underconverge
silently and hand back confidently wrong forecasts. A post-fit heuristic warns
when the ELBO loss is still clearly descending, but its absence is not proof
of convergence — inspect `result.losses` and confirm it has plateaued before
trusting results. Increase `num_steps`, raise the learning rate
(`optimizer=0.05`), or switch to {func}`~pymc_forecast.fit_mcmc` when accuracy
matters more than speed. A VI {class}`~pymc_forecast.FitResult` has `idata is
None` until {func}`~pymc_forecast.draw_posterior` samples the approximation.

## Other inference backends

Swap {func}`~pymc_forecast.fit_vi` for {func}`~pymc_forecast.fit_mcmc` (NUTS,
with `nuts_sampler="nutpie"/"numpyro"/...`) or
{func}`~pymc_forecast.fit_pathfinder` (pymc-extras). Each fitter takes an
optional already-built `model=`; when it is omitted, the fitter builds the
model once from `data` and `covariates`. {class}`~pymc_forecast.Forecaster`,
{class}`~pymc_forecast.HMCForecaster`, and
{class}`~pymc_forecast.PathfinderForecaster` call those functions. Their
constructors and attributes did not change, and each accepts `progressbar=`
directly:

```python
result = fit_mcmc(model, train, draws=1_000, progressbar=True)
```

## Covariates and richer latents

For models with real covariates, pass full-horizon `covariates` to
{func}`~pymc_forecast.forecast` — see the
[electricity example](examples/victoria_electricity.ipynb). See
{func}`~pymc_forecast.markov_series` for state-space latents and
{func}`~pymc_forecast.predict_mvn` for observation noise correlated across time.

## Statespace models

[pymc-extras statespace](https://github.com/pymc-devs/pymc-extras) structural models
(level/trend, seasonality, SARIMAX, ...) are first-class citizens too: define one as a
{class}`~pymc_forecast.StatespaceModel` and fit it with
{class}`~pymc_forecast.StatespaceForecaster` — the same `forecast` (including
exogenous-regression covariates), `predict_in_sample`, `backtest`, and metrics calls
apply, with the Kalman filter marginalizing the latent states instead of sampling them.
See the [scan-vs-statespace comparison](examples/scan_vs_statespace_local_level.ipynb).

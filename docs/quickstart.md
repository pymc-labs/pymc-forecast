# Quickstart

One model definition serves fitting and forecasting: {func}`~pymc_forecast.time_series`
creates separate `{name}_future` latents that are absent from the fitted posterior, so
posterior predictive sampling replays the fitted parameters while drawing the horizon
forward. Write a model with one {func}`~pymc_forecast.predict` call, fit it, and
forecast:

```python
import numpy as np, pandas as pd, pymc as pm, pytensor.tensor as pt
from pymc_forecast import Forecaster, backtest, evaluate_forecast, predict, time_series

# a trending weekly series; hold out the last 8 weeks
dates = pd.date_range("2024-01-07", periods=60, freq="W")
y = pd.Series(np.cumsum(np.random.default_rng(0).normal(0.2, 1.0, 60)) + 10, index=dates)
train, test = y.iloc[:52], y.iloc[52:]

def model(h, covariates):
    # a per-step drift latent; time_series adds the matching `_future` latent
    drift = time_series(h, "drift", lambda name, dims: pm.Normal(name, 0.0, 0.5, dims=dims))
    sigma = pm.HalfNormal("sigma", 1.0)
    predict(
        h,
        lambda name, mu, dims, obs: pm.Normal(name, mu, sigma, dims=dims, observed=obs),
        pt.cumsum(drift),                       # local-linear trend
    )

fc = Forecaster(model, train, num_steps=5_000, random_seed=0)   # ADVI
idata = fc.forecast(horizon=8, num_samples=500, random_seed=0)
forecast = idata["predictions"]["forecast"]     # dims: (chain, draw, time_future)

# score against the held-out weeks (aligned by dim name, not axis position)
truth = test.to_xarray().rename({"index": "time_future"})
print(evaluate_forecast(forecast, truth))       # {'mae': ..., 'rmse': ..., 'crps': ..., 'coverage': ...}

# rolling-origin backtest over the whole series
results = backtest(y, None, model, min_train_window=48, test_window=4, stride=4,
                   num_samples=200, forecaster_options={"num_steps": 3_000}, random_seed=0)
```

## Check VI convergence

{class}`~pymc_forecast.Forecaster` uses mean-field ADVI, which can
underconverge silently and hand back confidently wrong forecasts. A post-fit
heuristic warns when the ELBO loss is still clearly descending, but its
absence is not proof of convergence — inspect `fc.losses` and confirm it has
plateaued before trusting results. Increase `num_steps`, raise the learning
rate (`optimizer=0.05`), or switch to {class}`~pymc_forecast.HMCForecaster`
when accuracy matters more than speed.

## Other inference backends

Swap {class}`~pymc_forecast.Forecaster` for {class}`~pymc_forecast.HMCForecaster`
(NUTS, with `nuts_sampler="nutpie"/"numpyro"/...`) or
{class}`~pymc_forecast.PathfinderForecaster` (pymc-extras) — the fit/forecast
interface is identical. Every forecaster (including
{class}`~pymc_forecast.StatespaceForecaster`) accepts `progressbar=` directly,
so backends can be swapped without moving that option into `fit_kwargs`,
`sample_kwargs`, or `pathfinder_kwargs`:

```python
fc = HMCForecaster(model, train, draws=1_000, progressbar=True)
```

For configure-now / fit-later lifecycles (sklearn-style adapters), omit the
data at construction and call `fit()` explicitly — passing data to the
constructor remains the equivalent one-step path:

```python
fc = Forecaster(model, num_steps=5_000, progressbar=False)
# store or pass `fc` as a configured object, then fit when data is available
fc.fit(train, random_seed=0)
idata = fc.forecast(horizon=8, num_samples=500)
```

The same object can be refit; its backend configuration is reused. Predictive
methods raise {class}`~pymc_forecast.NotFittedError` until `fit()` has
completed, and `fc.is_fitted` reports the state.

## Covariates and richer latents

For models with real covariates, pass full-horizon `covariates` to `.forecast()`
instead of `horizon=` — see the
[electricity example](examples/victoria_electricity.ipynb). See
{func}`~pymc_forecast.markov_time_series` for state-space latents and
{func}`~pymc_forecast.predict_mvn` for observation noise correlated across time.

## Statespace models

[pymc-extras statespace](https://github.com/pymc-devs/pymc-extras) structural models
(level/trend, seasonality, SARIMAX, ...) are first-class citizens too: define one as a
{class}`~pymc_forecast.StatespaceModel` and fit it with
{class}`~pymc_forecast.StatespaceForecaster` — the same `forecast` (including
exogenous-regression covariates), `predict_in_sample`, `backtest`, and metrics calls
apply, with the Kalman filter marginalizing the latent states instead of sampling them.
See the [scan-vs-statespace comparison](examples/scan_vs_statespace_local_level.ipynb).

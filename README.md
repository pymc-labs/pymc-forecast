# PyMC-Forecast

[![CI](https://github.com/pymc-labs/pymc-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/pymc-labs/pymc-forecast/actions/workflows/ci.yml)
[![Docs](https://github.com/pymc-labs/pymc-forecast/actions/workflows/docs.yml/badge.svg)](https://pymc-labs.github.io/pymc-forecast/)

Bayesian time-series forecasting with [PyMC](https://www.pymc.io): you write the
generative model; the package handles the train/forecast plumbing, inference,
backtesting, and evaluation.

A PyMC port of [numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast)
(itself a port of Pyro's `pyro.contrib.forecast`) — redesigned around PyMC idioms
rather than a 1:1 translation.

> **Status: early development.** The design and roadmap live in
> [PLAN.md](PLAN.md) and the
> [issue tracker](https://github.com/pymc-labs/pymc-forecast/issues).

**Documentation:** <https://pymc-labs.github.io/pymc-forecast/> — API reference and
executed example notebooks (univariate, hierarchical, covariates, state-space).

## Quickstart

Write a model with one `predict()` call, fit it, and forecast:

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
# Outputs stay draw-level (never reduced to means/quantiles); prediction_samples(idata)
# extracts the samples Dataset from a forecast or in-sample result alike.

# score against the held-out weeks (aligned by dim name, not axis position)
truth = test.to_xarray().rename({"index": "time_future"})
print(
    evaluate_forecast(forecast_draws, truth)
)  # {'mae': ..., 'rmse': ..., 'crps': ..., 'coverage': ...}


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

The same model body works with `fit_mcmc` (NUTS, with `nuts_sampler="nutpie"/"numpyro"/...`)
or `fit_pathfinder` (pymc-extras). `ForecastingModel` is the object-oriented wrapper
around the same primitives — not a second model definition:

```python
class LocalLevel(ForecastingModel):
    def model(self, covariates, data=None):
        drift = self.innovations("drift", pm.Normal.dist(0.0, 0.5))
        self.predict(pm.StudentT.dist(nu=3, sigma=1.0), pt.cumsum(drift))
```

For models with real covariates, pass full-horizon `covariates` to `forecast()`
instead of a null time index. Covariate-free models can also forecast over an exact —
even irregular — later time index; the horizon length is the surplus covariate
steps. See `ssoe` for observation/error-driven recursions
such as ARMA and Holt-Winters, `markov_series` for state-space latents, and
`predict_mvn` for observation noise correlated across time.

[pymc-extras statespace](https://github.com/pymc-devs/pymc-extras) structural models
(level/trend, seasonality, SARIMAX, ...) are first-class citizens too: define one as a
`StatespaceModel` and fit it with `StatespaceForecaster` — the same `forecast`
(including exogenous-regression covariates), `predict_in_sample`, `backtest`, and
metrics calls apply, with the Kalman filter marginalizing the latent states instead
of sampling them (see `docs/examples/scan_vs_statespace_local_level.ipynb`).
The pymc-extras integrations (`fit_pathfinder` / `PathfinderForecaster`, `StatespaceForecaster`) are an
optional extra: install with `pip install 'pymc-forecast[extras]'`.

Priors stay user-injectable: pass `pymc_extras.prior.Prior`
objects directly to `innovations`/`predict`, or declare named
`default_priors` on a `ForecastingModel`/`StatespaceModel` subclass and let
callers override any subset with `Model(priors={...})` — nested hyper-priors
are shared across the train/forecast split so replay semantics hold.

## Design principles

- **One model trains and forecasts.** In-sample time latents are fitted; the
  forecast horizon uses separate `{name}_future` variables that
  `pm.sample_posterior_predictive` draws from the prior while replaying the
  posterior for everything else.
- **Dims and coords everywhere.** No positional axis conventions: variables carry
  named dims (`"time"`, `"time_future"`, `"obs"`, batch dims), results are
  `arviz.InferenceData` / `xarray` objects with real coordinates, and metrics are
  dim-aware. The [prediction output schema](docs/schema.md) is documented,
  test-locked public API.
- **Not AutoML.** No model zoo, no automatic feature pipelines — a clean path from
  a hand-written PyMC model to probabilistic forecasts and scores.
- **Leverage the ecosystem.** ADVI/NUTS from PyMC core, Pathfinder and state-space
  models from [pymc-extras](https://github.com/pymc-devs/pymc-extras), diagnostics
  and storage from [ArviZ](https://python.arviz.org).

## Development

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
```

To build the documentation (the example notebooks are committed fully executed;
CI re-executes them with reduced sampling settings):

```bash
uv sync --all-extras --group docs
uv run sphinx-build -b html docs docs/_build/html
```

## License

Apache-2.0. Portions derived from
[numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast) (Apache-2.0).

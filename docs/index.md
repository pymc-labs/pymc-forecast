# PyMC-Forecast

Bayesian time-series forecasting with [PyMC](https://www.pymc.io): you write the
generative model; the package handles the train/forecast plumbing, inference,
backtesting, and evaluation.

A PyMC port of [numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast)
(itself a port of Pyro's `pyro.contrib.forecast`) — redesigned around PyMC idioms
rather than a 1:1 translation.

```{note}
**Status: early development.** The design and roadmap live in
[PLAN.md](https://github.com/pymc-labs/pymc-forecast/blob/main/PLAN.md) and the
[issue tracker](https://github.com/pymc-labs/pymc-forecast/issues).
```

## Installation

```bash
pip install pymc-forecast            # core: PyMC + ArviZ
pip install 'pymc-forecast[extras]'  # + pymc-extras (Pathfinder, statespace)
```

## At a glance

```python
import pymc as pm, pytensor.tensor as pt
from pymc_forecast import Forecaster, predict, time_series

def local_level(h, covariates):
    drift = time_series(h, "drift", lambda name, dims: pm.Normal(name, 0.0, 0.5, dims=dims))
    sigma = pm.HalfNormal("sigma", 1.0)
    predict(
        h,
        lambda name, mu, dims, obs: pm.Normal(name, mu, sigma, dims=dims, observed=obs),
        pt.cumsum(drift),
    )

fc = Forecaster(local_level, train, num_steps=5_000)      # ADVI
idata = fc.forecast(horizon=8, num_samples=500)
forecast = idata["predictions"]["forecast"]               # dims: (chain, draw, time_future)
```

**[Start with the full workflow →](quickstart.md)** ·
**[Browse the examples →](examples/index.md)** ·
**[Prediction schema →](prediction-schema.md)** ·
**[API reference →](api/index.md)**

## Design principles

- **One model trains and forecasts.** In-sample time latents are fitted; the
  forecast horizon uses separate `{name}_future` variables that
  `pm.sample_posterior_predictive` draws from the prior while replaying the
  posterior for everything else.
- **Dims and coords everywhere.** No positional axis conventions: variables carry
  named dims (`"time"`, `"time_future"`, `"obs"`, batch dims), results are
  `arviz.InferenceData` / `xarray` objects with real coordinates, and metrics are
  dim-aware.
- **Not AutoML.** No model zoo, no automatic feature pipelines — a clean path from
  a hand-written PyMC model to probabilistic forecasts and scores.
- **Leverage the ecosystem.** ADVI/NUTS from PyMC core, Pathfinder and state-space
  models from [pymc-extras](https://github.com/pymc-devs/pymc-extras), diagnostics
  and storage from [ArviZ](https://python.arviz.org).

```{toctree}
:hidden:
:maxdepth: 2

quickstart
prediction-schema
examples/index
api/index
```

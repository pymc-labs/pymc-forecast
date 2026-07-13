# Bayesian forecasting, PyMC-style

`pymc_forecast` supplies the train/forecast plumbing around a model you write
in PyMC. Fit with ADVI or NUTS, generate labeled probabilistic forecasts, and
score rolling backtests without giving up direct control of the generative
model.

The core design is dims-first: model variables use named dimensions, forecasts
carry real `time_future` coordinates, and metrics align xarray objects by name
instead of relying on positional axes.

```python
import pymc as pm
import pytensor.tensor as pt

from pymc_forecast import Forecaster, predict, time_series


def local_level(h, covariates):
    drift = time_series(
        h,
        "drift",
        lambda name, dims: pm.Normal(name, 0, 0.2, dims=dims),
    )
    sigma = pm.HalfNormal("sigma", 1)
    predict(
        h,
        lambda name, mu, dims, obs: pm.Normal(
            name, mu, sigma, dims=dims, observed=obs
        ),
        pt.cumsum(drift),
    )


forecaster = Forecaster(local_level, train, num_steps=5_000)
idata = forecaster.forecast(horizon=12, num_samples=500)
samples = idata["predictions"]["forecast"]
```

[Start with the full workflow](quickstart.md){ .md-button .md-button--primary }
[Browse examples](examples/index.md){ .md-button }

!!! note

    This project is in early development. APIs may change while the initial
    roadmap is completed.

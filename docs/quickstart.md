# Quickstart

The same model definition is used for fitting and forecasting. The
`time_series` helper creates separate future latents that are absent from the
posterior trace; posterior predictive sampling therefore replays fitted
parameters while drawing the horizon forward.

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc_forecast import Forecaster, evaluate_forecast, predict, time_series

dates = pd.date_range("2024-01-07", periods=60, freq="W")
rng = np.random.default_rng(0)
y = pd.Series(np.cumsum(rng.normal(0.2, 1, 60)) + 10, index=dates)
train, test = y.iloc[:52], y.iloc[52:]


def model(h, covariates):
    drift = time_series(
        h,
        "drift",
        lambda name, dims: pm.Normal(name, 0, 0.5, dims=dims),
    )
    sigma = pm.HalfNormal("sigma", 1)
    predict(
        h,
        lambda name, mu, dims, obs: pm.Normal(
            name, mu, sigma, dims=dims, observed=obs
        ),
        pt.cumsum(drift),
    )


fc = Forecaster(model, train, num_steps=5_000, random_seed=0)
idata = fc.forecast(horizon=8, num_samples=500, random_seed=0)
forecast = idata["predictions"]["forecast"]
truth = test.to_xarray().rename({"index": "time_future"})
print(evaluate_forecast(forecast, truth))
```

Use `HMCForecaster` for NUTS or `PathfinderForecaster` from the optional
`extras` dependency; their forecast interface is the same. If a model has
known future regressors, pass covariates spanning both training and forecast
time instead of `horizon=`.

For repeated out-of-sample evaluation, pass the same model to `backtest`:

```python
from pymc_forecast import backtest

folds = backtest(
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

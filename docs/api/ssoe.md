# Observation-driven recursions

```{automodule} pymc_forecast.ssoe
:members:
:undoc-members:
:show-inheritance:
```

## A minimal ARMA model

The state contains the previous observation and its one-step prediction error.
The same callbacks filter training observations and simulate future observations.

```python
import pymc as pm
from pymc_forecast import ssoe


def arma(h, covariates):
    phi = pm.Uniform("phi", -0.95, 0.95)
    theta = pm.Uniform("theta", -0.95, 0.95)
    sigma = pm.HalfNormal("sigma", 1)

    result = ssoe(
        h,
        "eps",
        init=(0.0, 0.0),
        mean=lambda state, x, phi, theta: phi * state[0] + theta * state[1],
        update=lambda state, y, error, x, phi, theta: (y, error),
        noise_fn=lambda name, dims: pm.Normal(name, 0, sigma, dims=dims),
        params=(phi, theta),
    )
    pm.Normal("obs", result.mu, sigma, observed=h.data.values, dims="time")
    pm.Deterministic("mu", result.mu, dims="time")
    if h.future:
        pm.Deterministic("mu_future", result.mu_future, dims="time_future")
        pm.Deterministic("forecast", result.y_future, dims="time_future")
```

The helper registers only `eps_future`, and only during forecasting. The caller
owns the observation distribution and can compose several recursion channels.
**Do not pass `y_future` to `predict` with another noisy observation distribution:**
`y_future` already contains the observation error used to update the future state.
The explicit Deterministics above use the usual prediction schema and are returned
by `forecast` and `predict_in_sample`.

`mu` is a one-step fitted mean conditioned on the actual observed history. In-sample
posterior prediction samples observations around these fitted means; it does not
simulate a new training history. `mu_future` conditions on previously simulated
future observations and excludes only the current step's error. It is not an
unconditional multi-step mean with all future errors integrated out.

## Labeled inputs and update gates

`y` defaults to `h.data`; a transformed `DataArray` can be passed explicitly.
Its `time` coordinates must match the observed window exactly. Non-time dimensions
are inferred by name, checked against model coordinates, and preserved in the
result. `xs` is a `DataArray` covering the full horizon; its `time` dimension can
appear anywhere. Each callback receives one time slice in its other dimension order.
Pass model parameters via `params` and keep both callbacks deterministic.

For unavailable or missing observations, supply a finite placeholder and use an
explicit gate in `xs` to prevent state updates. The **likelihood must also exclude
those observations**; an update gate alone does not mask a likelihood. Future gates
must come from known inputs or an explicit scenario, never held-out observations.
The helper does not infer a missing-data or censoring mechanism.

For sampled hidden states, use `markov_time_series`. For linear-Gaussian hidden
states, the `pymc-extras` statespace backend can marginalize the state path. See the
[ARMA](../examples/arma.ipynb) and [Holt-Winters](../examples/exponential_smoothing_state_space.ipynb)
examples for observation-driven recursions.

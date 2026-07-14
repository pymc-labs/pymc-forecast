# Prediction output schema

The dimension, coordinate, group, and variable names on prediction outputs
are **public API**. Downstream code (plotting, metrics, adapters such as
CausalPy) can rely on the names below; any change to them is a breaking
change, made only in a minor release and called out in the
[changelog](https://github.com/pymc-labs/pymc_forecast/blob/main/CHANGELOG.md).

## Groups

Prediction results are ArviZ trees (`DataTree` / `InferenceData`) with exactly
one predictive group:

| Producer | Group | Contents |
|---|---|---|
| `forecast(...)` | `predictions` | out-of-sample forecasts |
| `predict_in_sample(...)` | `posterior_predictive` | in-sample predictive of the observed variable |

The group names follow the ArviZ convention (out-of-sample predictions live in
`predictions`). {func}`pymc_forecast.prediction_samples` extracts the samples
`Dataset` from either group, so adapters don't need to branch on the name.

## Variables

| Name | Constant | Where | Meaning |
|---|---|---|---|
| `obs` | {data}`pymc_forecast.OBS_VAR` | `posterior_predictive` | the observed (in-sample) variable |
| `forecast` | {data}`pymc_forecast.FORECAST_VAR` | `predictions` | the forecast-horizon variable |
| `{name}_future` | — | `predictions` | forecast-horizon slice of each per-step latent registered with `time_series` |

The statespace adapter additionally exposes the latent state trajectories as
`forecast_latent` in its `predictions` group.

## Dimensions and coordinates

Every predictive variable carries, in order:

1. **Sample dims** `("chain", "draw")` — always both, always leading, holding
   the full draw-level posterior-predictive samples (no reduction to means or
   quantiles happens on the default path).
2. **A time dim** — `"time"` ({data}`pymc_forecast.TIME_DIM`) on in-sample
   variables, `"time_future"` ({data}`pymc_forecast.FUTURE_DIM`) on forecast
   variables. Its coordinate values are real: whatever time index the inputs
   carried (a `DatetimeIndex`, periods, or the integer-range fallback).
   `time_future` carries the horizon index supplied at forecast time (the
   surplus covariate steps, or the index extended by `horizon=`).
3. **Batch dims** carried over from the data — e.g. `"series"` for 2-d data
   normalized by {func}`pymc_forecast.as_dataarray`, or any named dims of an
   `xarray` input.

So a univariate forecast is `(chain, draw, time_future)` and a hierarchical
one `(chain, draw, time_future, series)`, with real coordinate values on the
time and series axes.

## Mapping onto downstream coordinates

Because the names are fixed and the coordinates are real, remapping is a
one-liner, e.g.:

```python
samples = pymc_forecast.prediction_samples(result)["forecast"]
samples = samples.rename({"time_future": "obs_ind", "series": "treated_units"})
```

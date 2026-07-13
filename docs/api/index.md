# API reference

Everything below is importable from the top-level `pymc_forecast` namespace;
the modules group the reference by responsibility.

- [`pymc_forecast.model`](model.md) — Model-building core: `Horizon`, `time_series`, `predict`, `build_model`, and the `ForecastingModel` facade.
- [`pymc_forecast.forecaster`](forecaster.md) — Forecaster classes: `Forecaster` (ADVI), `HMCForecaster` (NUTS), `PathfinderForecaster`.
- [`pymc_forecast.prediction`](prediction.md) — Posterior-predictive plumbing: `forecast`, `predict_in_sample`, `thin_draws`.
- [`pymc_forecast.evaluate`](evaluate.md) — Rolling/expanding-window backtesting: `backtest`, `BacktestResult`, `results_to_dataframe`.
- [`pymc_forecast.metrics`](metrics.md) — Dim-aware forecast metrics: CRPS, pinball, interval score, coverage, MASE, `evaluate_forecast`.
- [`pymc_forecast.features`](features.md) — Feature builders: Fourier design matrices and periodic tiling.
- [`pymc_forecast.datasets`](datasets.md) — Example datasets: BART ridership and Victoria electricity demand.
- [`pymc_forecast.data`](data.md) — Labeled-array helpers: normalization to `DataArray`, time-index extension, alignment checks.
- [`pymc_forecast.gaussian`](gaussian.md) — Time-correlated Gaussian observation noise: `predict_mvn` and the explicit Gaussian conditional.
- [`pymc_forecast.markov`](markov.md) — Scan-based Markov latents: `markov_time_series`.
- [`pymc_forecast.statespace`](statespace.md) — pymc-extras statespace interop: `StatespaceModel` and `StatespaceForecaster`.
- [`pymc_forecast.exceptions`](exceptions.md) — The package's error taxonomy.

```{toctree}
:hidden:
:maxdepth: 1

model
forecaster
prediction
evaluate
metrics
features
datasets
data
gaussian
markov
statespace
exceptions
```

# Examples

Executed end-to-end and re-run in CI with reduced sampling settings.

- [Univariate forecasting](forecasting_univariate.ipynb) — weekly BART ridership with a
  random-walk local level, annual Fourier seasonality, and a Student-T likelihood;
  ADVI, CRPS evaluation, and a rolling-origin backtest.
- [Hierarchical forecasting](hierarchical_forecasting.ipynb) — hourly arrivals to one
  BART station from all 50 origins at once: batch dims, per-series levels and weekly
  seasonality, shared scales.
- [Electricity demand with covariates](victoria_electricity.ipynb) — Victoria hourly
  demand with daily/weekly seasonality and a quadratic temperature response;
  forecasting with full-horizon covariates.
- [Exponential smoothing in state-space form](exponential_smoothing_state_space.ipynb)
  — a damped Holt-Winters single-source-of-error recursion written with
  `pytensor.scan`, fit with NUTS.
- [Local level two ways](scan_vs_statespace_local_level.ipynb) — scan-based Markov
  latents vs. the `pymc-extras` statespace backend on the same model: posterior
  quality, runtime, and a shared backtest.
- [Retail demand under stockouts](fresh_retail_stockout.ipynb) — a 50,000-series
  FreshRetailNet panel fit with JAX ADVI on GPU, memory-bounded posterior draws,
  holdout evaluation, and a full-availability demand scenario.

```{toctree}
:hidden:
:maxdepth: 1

forecasting_univariate
hierarchical_forecasting
victoria_electricity
exponential_smoothing_state_space
scan_vs_statespace_local_level
fresh_retail_stockout
```

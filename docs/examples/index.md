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
  the shared `ssoe` recursion helper, fit with NUTS.
- [Local level two ways](scan_vs_statespace_local_level.ipynb) — scan-based Markov
  latents vs. the `pymc-extras` statespace backend on the same model: posterior
  quality, runtime, and a shared backtest.
- [Retail demand under stockouts](retail_stockouts.ipynb) — censored demand on the
  FreshRetailNet-50K panel: a hierarchical damped-trend model with a floored
  saturating availability factor, batched predictive sampling, and a counterfactual
  demand forecast at full availability.
- [ARMA forecasting](arma.ipynb) — one observation/error recursion for filtering and
  forecasting, parameter recovery and expanding-window evaluation.
- [VAR forecasting with Impulso](var.ipynb) — a VAR(2) on quarterly US growth rates,
  fit by Impulso's Minnesota prior and scored with an expanding-window backtest.
- [Intermittent demand](intermittent_demand.ipynb) — Bernoulli occurrence and positive
  Gamma quantities, stockout zeros, a naive baseline and full-availability scenarios.
- [Censored demand](censored_demand.ipynb) — an AR(2) with a right-censored normal
  likelihood for a known shelf cap; stockout days are gated out of the lag filter,
  and the same model trained as if the cap were an exact sale is the baseline.
- [Comparing inference methods](inference_methods_comparison.ipynb) — the same ARMA
  model fit with NUTS, ADVI, full-rank ADVI and Pathfinder; diagnostics, timing and scores.
- [M5 forecasting](m5_forecasting.ipynb) — top-down, bottom-up, and middle-out
  reconciliation of a Walmart hierarchy: official-weight check on the full panel, then
  the three starter-kit models fit, inspected, backtested with `backtest`, and scored with
  the competition's weighted scaled CRPS on an 84-series panel with all 12 levels.

```{toctree}
:hidden:
:maxdepth: 1

forecasting_univariate
arma
var
intermittent_demand
censored_demand
inference_methods_comparison
hierarchical_forecasting
victoria_electricity
exponential_smoothing_state_space
scan_vs_statespace_local_level
retail_stockouts
m5_forecasting
```

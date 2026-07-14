# Changelog

Notable changes to pymc_forecast are recorded here, most recent first.

The dim/coord/group/variable names on prediction outputs (documented in
[docs/schema.md](docs/schema.md)) are public API: any change to them is a
breaking change, made only in a minor release and called out here.

## Unreleased

- The prediction output schema (dims `chain`/`draw`/`time`/`time_future`,
  groups `predictions`/`posterior_predictive`, variables `obs`/`forecast`/
  `{name}_future`) is now documented and covered by contract tests; the
  variable-name constants `OBS_VAR` and `FORECAST_VAR` are exported at the
  package level ([#21](https://github.com/pymc-labs/pymc_forecast/issues/21)).

## 0.0.1 (2026-07-10)

- Initial release: train/forecast plumbing (`Forecaster`, `HMCForecaster`,
  `PathfinderForecaster`, `StatespaceForecaster`), model-building primitives
  (`time_series`, `predict`, `predict_mvn`, `markov_time_series`),
  backtesting, and dim-aware metrics.

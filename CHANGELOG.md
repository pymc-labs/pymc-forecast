# Changelog

Notable changes to pymc_forecast are recorded here, most recent first.

The dim/coord/group/variable names on prediction outputs (documented in
[docs/schema.md](docs/schema.md)) are public API: any change to them is a
breaking change, made only in a minor release and called out here.

## Unreleased

- `BaseForecaster.forecast()` and `predict_in_sample()` now accept an explicit
  `posterior=` so multiple predictive calls can reuse identical parameter
  draws and remain aligned by `chain`/`draw`
  ([#37](https://github.com/pymc-labs/pymc-forecast/issues/37)).

## 0.1.0 (2026-07-14)

Five features aimed at making the package cleanly wrappable as a model
provider (e.g. by CausalPy):

- Horizon-agnostic predict: `forecast(future_index=...)` samples over an
  arbitrary — even irregular — later time index supplied at forecast time;
  the horizon length is derived from it, never fixed at fit
  ([#22](https://github.com/pymc-labs/pymc-forecast/issues/22)).
- Covariate-conditioned forecasts from a future-only frame:
  `forecast(future_covariates=...)` appends predict-time covariate rows to
  the training covariates after strict structural validation (matching dims,
  covariate names and order, time index strictly after training)
  ([#19](https://github.com/pymc-labs/pymc-forecast/issues/19)).
- Draw-level output contract: variables in the `predictions` and
  `posterior_predictive` groups always retain full `chain`/`draw` samples,
  and `prediction_samples()` extracts the samples `Dataset` from any result
  shape ([#20](https://github.com/pymc-labs/pymc-forecast/issues/20)).
- The prediction output schema (dims `chain`/`draw`/`time`/`time_future`,
  groups `predictions`/`posterior_predictive`, variables `obs`/`forecast`/
  `{name}_future`) is documented ([docs/schema.md](docs/schema.md)) and
  covered by contract tests; the name constants `OBS_VAR`, `FORECAST_VAR`,
  `CHAIN_DIM`, `DRAW_DIM`, and `SAMPLE_DIMS` are exported at the package
  level ([#21](https://github.com/pymc-labs/pymc-forecast/issues/21)).
- User-injectable priors: pymc-extras `Prior` objects are accepted by
  `time_series`/`predict` (nested hyper-priors are shared across the
  train/forecast split so replay semantics hold), and the `PriorConfig`
  mixin gives `ForecastingModel` and `StatespaceModel` overridable
  `default_priors` + a `priors=` constructor argument
  ([#23](https://github.com/pymc-labs/pymc-forecast/issues/23)).

## 0.0.1 (2026-07-10)

- Initial release: train/forecast plumbing (`Forecaster`, `HMCForecaster`,
  `PathfinderForecaster`, `StatespaceForecaster`), model-building primitives
  (`time_series`, `predict`, `predict_mvn`, `markov_time_series`),
  backtesting, and dim-aware metrics.

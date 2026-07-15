# Changelog

Notable changes to pymc_forecast are recorded here, most recent first.

The dim/coord/group/variable names on prediction outputs (documented in
[docs/schema.md](docs/schema.md)) are public API: any change to them is a
breaking change, made only in a minor release and called out here.

## Unreleased

- GPU variational inference: `Forecaster(..., backend="jax")` optimizes PyMC's
  mean-field ADVI objective with a JAX-native `lax.scan` and returns the usual
  PyMC approximation. `draw_posterior(..., batch_size=N)` bounds posterior
  allocations for wide panels. The new FreshRetailNet-50K example combines
  these paths to forecast censored retail sales and full-availability demand
  ([#47](https://github.com/pymc-labs/pymc-forecast/issues/47)).

## 0.2.0 (2026-07-14)

- Schema addition — noise-free latent predictor: prediction outputs of models
  registered through `predict()` now carry the draw-level latent before
  observation noise, as `mu` in `posterior_predictive` and `mu_future` in
  `predictions` (constants `MU_VAR` / `MU_FORECAST_VAR`). The names `mu` and
  `mu_future` are now reserved; a model body defining them raises a clear
  error ([#36](https://github.com/pymc-labs/pymc-forecast/issues/36)).
- Draw-coherent predictions: `forecast(...)` and `predict_in_sample(...)` on
  every forecaster accept `posterior=` (typically from `draw_posterior()`) to
  condition several predictive calls on the same posterior draws; mutually
  exclusive with `num_samples`
  ([#37](https://github.com/pymc-labs/pymc-forecast/issues/37)).
- `Forecaster` warns (`UserWarning`) when the ELBO loss is still clearly
  descending at the end of the fit — VI results should be convergence-checked
  (`fc.losses`) before use
  ([#38](https://github.com/pymc-labs/pymc-forecast/issues/38)).
- Uniform constructor surface: every forecaster (including
  `StatespaceForecaster`) accepts `progressbar=` directly (the escape-hatch
  kwargs still accept it for compatibility; passing both raises), and all
  support a deferred fit — construct without data, then
  `fc.fit(data, covariates)` (returns `self`; refitting reuses the backend
  configuration). Predictive calls on an unfitted forecaster raise the new
  `NotFittedError`, and `is_fitted` reports the state
  ([#39](https://github.com/pymc-labs/pymc-forecast/issues/39)).

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

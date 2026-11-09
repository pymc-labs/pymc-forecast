# Changelog

Notable changes to pymc_forecast are recorded here, most recent first.

The dim/coord/group/variable names on prediction outputs (documented in
[docs/schema.md](docs/schema.md)) are public API: any change to them is a
breaking change, made only in a minor release and called out here.

## Unreleased
- ``predict`` left-aligns a length-``duration`` latent across batch dims. A
  1-d time effect on a square panel is no longer broadcast as a series effect,
  so both ``mu`` and the observation log-probability follow the time axis.
  The same holds for any time-major latent with fewer axes than the panel,
  such as ``(duration, 1)`` or ``(duration, n_region)`` with
  ``dims=("region", "store")``. A 0-d latent now raises ``HorizonError``
  instead of an ``IndexError``.

- Fix ``fourier_features`` on NumPy and Python scalar phases: a 0-d position
  such as ``np.int64(10)`` or ``10.0`` is one phase, not an indexing error.
  A built-in ``int`` is still a duration. A boolean ``time`` (``True``,
  ``np.bool_(True)``) now raises ``TypeError`` instead of being read as a
  duration of one or a phase of one.
- Nested ``Prior`` parameters are name-aligned to each segment's dims. A
  hyper-prior dimmed on ``"a"`` no longer trailing-broadcasts onto the last
  dim of ``("time", "a", "b")`` or of the forecast segment, and the same RV
  is shared. Observation likelihoods use the same alignment. A
  ``centered=False`` nested prior is included: it name-aligns instead of
  raising when the parameter is not a raw tensor.

- `make_mase` accumulates the seasonal-naive scale in float64. Integer and
  low-precision training series no longer wrap into a negative or hugely
  inflated scale. A boolean training series now yields a scale instead of a
  ``TypeError``. A NaN or infinity in ``train_data`` now raises ``ValueError``
  instead of returning NaN for every score, and the factory exposes the scale
  as ``mase.scale``.

- `as_dataarray` ignores scalar pandas axis names. A named index or columns
  axis is no longer treated as an xarray dimension, so `set_index("date")`
  and a square frame whose index is named `covariate` keep the `time` /
  `series` / `covariate` dims. Timezone and frequency on a named
  `DatetimeIndex` are preserved. A Series whose own name equals a dim name
  (`"time"`, `"series"`, ...) is returned unnamed, so `to_dataset()` works.

- `predict_in_sample` and `forecast` raise `AlignmentError` when a posterior's
  non-sample coordinates differ in labels or order from the model, instead of
  replaying that posterior positionally and stamping the model's coordinates
  on the result. A posterior that has no coordinate for a labeled model dim is
  rejected too, as is a posterior labeled by a model coordinate for a variable
  the model declares without dims (`shape=`). A model dim registered with a
  length but no labels is checked by length only. A tz-aware `DatetimeIndex`
  is compared as a pandas index, so a matching zone is not rejected as naive
  UTC. An unlabeled `shape=` variable cannot be checked, so reordering the
  data behind one still replays positionally.

  Breaking for the functional `forecast`: the posterior's in-sample `time`
  must equal the `data` window. Models built with `predict()` record `mu`
  (and `expected_observation`) over that window, so forecasting from a later
  origin with a posterior fitted on an earlier window now raises instead of
  running; refit on the new window. `Forecaster.forecast` always uses its
  training data and is unaffected.

- Keep a single-zone time index's timezone and clock. Time helpers, model
  coords, and forecast outputs no longer rebuild a timezone-aware
  ``DatetimeIndex`` through ``np.asarray`` or ``.values``, which shifted the
  clock to naive UTC and made later comparisons raise ``AlignmentError``.
  Two aware indexes are compared as instants: a future index or covariates in
  another zone (or the same zone from another tz library, e.g. ``zoneinfo``
  versus ``pytz``) are converted to the training data's zone instead of
  becoming an object index. A future index built on another zone's calendar
  can land off the training grid (UTC midnights on New York daily data become
  ``19:00-05:00``); localize it in the training zone to keep the grid.
  ``forecast(future_covariates=)`` uses that converted index, so a future
  covariate frame in another zone no longer crashes inside
  ``xarray.concat``. A naive index beside an aware one used
  to match after the zone was stripped to naive UTC; it now raises
  ``AlignmentError`` telling you to localize both sides, instead of being
  reported as a generic misalignment. Time coords are compared independent
  of datetime resolution, so data at ``datetime64[s]`` aligns with
  covariates at ``datetime64[ns]`` on pandas 2 too. Prediction outputs and
  posterior draws (every ``draw_posterior`` result) carry
  ``time``/``time_future`` as a tz-aware ``DatetimeIndex``, so ``.sel`` by
  date string and ``.resample`` work. ``FitResult.idata`` keeps the object
  index PyMC returns, because ArviZ summaries and plots reject a tz-aware
  ``DatetimeIndex`` coord.
  Statespace ``forecast`` stamps ``time_future`` without treating the
  training index name ``"time"`` as that dimension.
- Breaking: ``Horizon.from_data`` stores ``time``/``time_future`` as a
  ``pandas.Index`` (timezone included) instead of a ``numpy.ndarray``. Model
  code that applies ndarray-only operations to them (e.g. ``h.time[:, None]``)
  should call ``np.asarray`` first, which drops the timezone.
- Docs: notebook figures no longer render vertically stretched. Retina outputs
  carry `width`/`height` attributes; the theme capped the width but kept the
  fixed height. `docs/_static/custom.css` sets `height: auto` on cell-output
  images.
- Docs: every public function, class, method, and property now has a complete
  NumPy-style docstring: typed `Parameters` (the API reference shows no type
  hints, so these are the only types it renders), `Returns`, `Raises`, and
  `Warns`. Incorrect statements were fixed, for example: `backtest` windows
  are positional slices, `load_m5(cache_dir=...)` downloads into that directory,
  `make_mase` pools one scale across all series, and `periodic_repeat` needs
  `period=` for every PyTensor input.

## 0.3.0 (2026-10-05)

- Refresh the example-notebook gallery: shared ArviZ darkgrid style, 94% HDI as the ArviZ interval default, retina figures, and cycle colors (`C0`, `C1`, ...) instead of named series colors. `black` reference series are unchanged. Notebooks were re-executed at full sampling settings; CI still smoke-tests.

- Add an executed VAR example. Impulso fits the model; `backtest` scores the
  density forecasts per fold and per series, with expanding-window forecast
  plots. Impulso is a `docs` and `notebooks` dependency, not a core dependency.
- Add `datasets.load_us_macro`: the quarterly US real GDP, consumption and
  investment levels (statsmodels `macrodata`, public domain), bundled as a CSV.
- Docs: enable the MyST `dollarmath` and `amsmath` extensions, so `$...$` and
  `$$...$$` math in the example notebooks renders instead of showing as text.

- New example notebook — *Demand forecasting with a censored likelihood*: an
  AR(2) whose lag filter and likelihood distinguish stockouts from a known
  capacity cap, ported from the
  [upstream censored-demand example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/censored_demand.html).
  CI executes the reduced window.
- Add `load_m5` and an M5 forecasting example. `load_m5` downloads the
  competition files once and returns labeled `(time, series)` sales and
  price panels plus the identifier, calendar, and official-weight tables.
  The notebook defines the 12-level hierarchy, the competition scores
  (weighted scaled CRPS and WSPL), and the three starter-kit reconciliation
  models; the formulas follow the Pyro kit as ported by numpyro_forecast.
  Inference is mean-field ADVI through `Forecaster` (JAX backend), not the
  kit's clipped minibatch SVI, so posterior draws are not expected to match
  a NumPyro run. The notebook checks the official weights on the full panel,
  then fits an 84-series subset on which all 12 levels are distinct sets of
  series (three best sellers per department in four stores), inspects the
  posteriors with ArviZ, backtests the three models on the kit's three
  windows with `backtest`, and scores the holdout. CI reruns the notebook
  on a synthetic panel and does not download the competition files.
- Fix `StatespaceForecaster.forecast` and `predict_in_sample` on current
  pymc-extras (verified on 0.15.1): the thinned posterior now carries the
  fit's `observed_data` and `constant_data` groups, which pymc-extras reads
  to recover the fit coords, observed data, and exogenous inputs.
- **Breaking** ([#57](https://github.com/pymc-labs/pymc-forecast/issues/57)).
  This cutover ships in `0.3.0`; `0.2.0` does not include it. Prediction
  schema names did not change.
  - Model signature `(h, covariates)` to `(covariates, data=None)`.
  - `ForecastingModel.model(self, h, covariates)` to `model(self, covariates, data=None)`.
  - `time_series` removed. Use `innovations`. A `.dist()` or a pymc-extras `Prior`, not an `RVFactory`. A `.dist()`'s parameters broadcast against `("time", *dims)`, so per-series scales and multivariate dists work. A model variable passed where a `.dist()` is expected (e.g. `pm.Normal("raw", ...)`), or a `.dist()` whose parameters depend on random variables that are not in the model, is rejected at registration with the offending name.
  - `markov_time_series` removed. Use `markov_series`. No `advance`.
  - `Horizon.from_arrays` removed. Use `Horizon.from_data`.
  - `ssoe(h, name, init, mean, update, noise_fn, *, y=..., params=...)` becomes `ssoe(h, name, y, init, mean, update, noise, xs=None, *, params=..., dims=...)`. `y=None` still means `h.data`. `noise` is an unnamed, zero-centered `.dist()` (the location of `Normal` / `StudentT` noise is checked) or a `Prior` with constant parameters. A `Prior` with `Prior`-valued parameters is rejected: its hyper-priors would exist only on the forecast horizon and never be fitted. Create the scale in the model body, use it in the observation, and pass `pm.Normal.dist(0, sigma)`. Its parameters broadcast against `("time_future", *dims)`.
  - `predict`'s second argument is no longer only a 4-argument factory. A callable with four positional parameters `(name, latent, dims, observed)` is the factory, whatever their names or defaults. Other callables take the 1-argument `segment_latent -> .dist()` path: the latent is their first positional argument, so a bare PyMC `.dist` classmethod is accepted only when that argument is `mu` (e.g. `pm.Poisson.dist`; `pm.StudentT.dist` is rejected because it takes `nu` first). The callable is called once per segment and must not create model variables. A zero-centered `.dist()` is accepted for `Normal` and `StudentT`.
  - Functional fitters: `fit_vi`, `fit_mcmc`, `fit_pathfinder`, `FitResult`, `draw_posterior`. The classes call those functions. Like the classes, the fitters drop covariate rows past the training window. Constructors and attributes did not change. A `model=` passed to a fitter must be a training-window model; one with `forecast` or `*_future` free variables is rejected before sampling. A VI `FitResult` keeps `idata=None`; pass the `Dataset` returned by `draw_posterior` to `forecast`.

- Add `ssoe` and `SSOEResult` for observation-driven recursions with named inputs, shared training/forecast updates, and fresh future errors under posterior replay.
- Add executed ARMA, intermittent-demand and inference-comparison notebooks;
  refactor Holt-Winters to use `ssoe`, expose conditional future means, and
  center initial seasonality to distinguish it from the initial level.

- Schema addition — conditional expected observation: models can pass
  `expected_observation=` to `predict(...)` to emit `expected_observation` /
  `expected_observation_future` (exported as
  `pymc_forecast.EXPECTED_OBSERVATION_VAR` /
  `EXPECTED_OBSERVATION_FORECAST_VAR`) alongside the existing `mu` /
  `mu_future`. These carry `E[Y | parameters, latent state, covariates]` in
  observed outcome units, so GLM-style models can expose outcome-scale
  expectations while `mu` stays the link-scale latent predictor. Both new
  names are reserved unconditionally: a model body that defines its own
  variable under either name now raises `HorizonError`, whether or not it uses
  the new argument ([#52](https://github.com/pymc-labs/pymc-forecast/issues/52)).
- Preserve supplied posterior draw coordinates when `batch_size=` splits
  predictive sampling into blocks, including the expected-observation outputs.
  Unlabeled posterior datasets receive continuous default draw indices.
- Broadcast recorded predictors over the full panel before splitting the
  horizon, avoiding a PyTensor 3.3 shape error for singleton panel axes.
- Validate non-time dimensions and coordinates on every forecast covariate
  input path and shared panel dimensions during model construction. Reordered
  or renamed features now raise before posterior sampling instead of silently
  resampling fitted coefficients; invalid future time coordinates also raise.
- Align labeled forecast metrics on matching time/series coordinates, rejecting
  different or duplicate labels and incompatible shapes. CRPS now calculates
  in float64 to avoid overflow for integer counts and low-precision samples.
- Support `horizon=` with pandas `PeriodIndex` and preserve stored datetime
  frequencies for short training series. Reject empty/non-increasing indices
  and invalid forecast/backtest window sizes with actionable errors.
- Require `xarray>=2024.10` for `DataTree` support. CI tests the installed wheel
  without optional extras, against both current and minimum direct dependencies.
  Add a conjugate-posterior accuracy regression test for JAX ADVI.

- GPU variational inference: `Forecaster(..., backend="jax")` optimizes PyMC's
  mean-field ADVI objective with a JAX-native `lax.scan` (on GPU when a CUDA
  JAX is installed) and returns the usual PyMC approximation; requires the new
  `jax` extra ([#47](https://github.com/pymc-labs/pymc-forecast/issues/47)).
- Batched predictive sampling: `forecast(...)` and `predict_in_sample(...)`
  (module functions and forecaster methods) accept `batch_size=` to process
  the posterior in consecutive draw blocks, and `draw_posterior(...,
  batch_size=N)` bounds the peak allocation of posterior sampling on
  VI backends — together the port of upstream's chunk-and-offload prediction
  ([numpyro_forecast#65](https://github.com/juanitorduz/numpyro_forecast/pull/65),
  [#47](https://github.com/pymc-labs/pymc-forecast/issues/47)).
- New example notebook — *Forecasting retail demand under stockouts*: the
  FreshRetailNet-50K panel with a hierarchical damped-trend model and a
  floored saturating availability factor for censored demand, ported from
  [the upstream blog post](https://juanitorduz.github.io/fresh_retail_stockout/)
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

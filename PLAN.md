# Porting `numpyro_forecast` to PyMC

Design and porting plan for building PyMC-Forecast (`pymc_forecast`) from
[juanitorduz/numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast)
(itself a JAX/NumPyro port of Pyro's `pyro.contrib.forecast`).

**This is a redesign, not a translation**: we adopt PyMC idioms throughout —
named dims/coords instead of positional axis conventions, `InferenceData`/xarray
results instead of raw arrays, and pymc-extras for Pathfinder and state-space
models. No API or layout compatibility with the NumPyro version is kept.

The actionable roadmap lives in the
[issue tracker](https://github.com/pymc-labs/pymc-forecast/issues); this document
records the source-package analysis and the concept mapping behind those issues.

## What the source package is

A small, focused toolkit for Bayesian time-series forecasting. The user writes the
generative model; the package handles train/forecast plumbing, inference, and
evaluation. ~3.3k lines in the main package + ~2k in `functional/` and `contrib/`,
plus ~40 test modules.

Core design invariants of the source:

1. **One model both trains and forecasts.** In-sample time latents use a fixed site
   name (e.g. `drift`); the forecast horizon uses a separate `{name}_future` site.
   The guide/posterior never sees the future site, so `Predictive` replays the
   posterior for in-sample latents and draws the forecast suffix from the prior.
2. **Horizon inferred from shapes**: `future = covariates.shape[-2] - data.shape[-2]`.
3. **Array layout**: time at axis `-2`, observation dim at `-1`, batch dims left
   (dropped in this port in favor of dims/coords).

Module inventory of the source:

| Module | Lines | Role |
|---|---|---|
| `functional/models.py` | 417 | `Horizon`, `time_series`, `markov_time_series` (scan), `predict`, `predict_glm`, `forecasting_model` |
| `surgery.py` | 368 | Distribution surgery: `shift_loc`, `slice_time`, `prefix_condition` (singledispatch; elementwise families + MVN Gaussian conditional) |
| `functional/svi.py` | 278 | Guide/optimizer resolution, `fit_svi` (AutoNormal default) |
| `functional/mcmc.py` | 249 | Kernel resolution, `fit_mcmc` (NUTS default) |
| `functional/prediction.py` | 279 | `forecast`, `predict_in_sample` via `Predictive`, chunked/vmapped sampling |
| `functional/posterior.py` | 135 | `draw_posterior` dispatch over fit types |
| `forecaster.py` | 556 | OOP shims: `ForecastingModel`, `Forecaster` (SVI), `HMCForecaster`, `PathfinderForecaster` |
| `contrib/blackjax.py` | 581 | Pathfinder VI via BlackJAX |
| `evaluate.py` | 1090 | `backtest` (expanding/rolling windows), `backtest_vectorized` (vmapped SVI), `evaluate_forecast`, result dataclasses |
| `metrics.py` | 220 | CRPS, pinball, interval score, MASE factory |
| `convert.py` | 380 | Fits → ArviZ-schema `xarray.DataTree`, forecast groups |
| `features.py` | 80 | Fourier design matrices, periodic tiling |
| `datasets.py` | 137 | BART ridership, Victoria electricity |
| `arrays.py`, `typing.py`, `exceptions.py`, `optional.py` | ~390 | Helpers, jaxtyping/beartype runtime typing, error taxonomy |

## Concept mapping: NumPyro → PyMC

| numpyro_forecast concept | pymc_forecast equivalent |
|---|---|
| Model = pure callable `(covariates, data=None)` | Model-builder callable returning a `pm.Model` (built fresh per fit/forecast call with different horizons) |
| `Horizon` from array shapes, plates `time` / `time_future` | `Horizon` from coords: dims `"time"` / `"time_future"` with real coordinate values (dates, periods) |
| `time_series`: site + `{name}_future` site, concat on axis -2 | Same trick: an RV with `dims="time"` plus a fresh `{name}_future` RV with `dims="time_future"`, `pt.concatenate` — `sample_posterior_predictive` replays trace vars and **samples vars missing from the trace from the prior**, which is exactly the `_future`-site mechanism |
| `markov_time_series` (numpyro `scan`) | `pytensor.scan` inside the model (with `collect_default_updates`) or `CustomDist` with a scan-based dist; forecast scan seeded from the final in-sample state. For linear-Gaussian cases: `pymc_extras.statespace` |
| `predict(noise_dist, prediction)` + `shift_loc` | Unnecessary as surgery — PyMC likelihoods take `mu` directly. `predict()` becomes API sugar registering `obs` (observed, dims `("time", ...)`) + `obs_future` (unobserved, dims `("time_future", ...)`) + a `forecast` deterministic |
| `slice_time` / `prefix_condition` (elementwise) | Free: index the latent/params on the future coords when building the future likelihood; conditional = marginal for elementwise noise |
| `prefix_condition` (MultivariateNormal over time) | Must be ported: explicit Gaussian conditional (Cholesky/solve) in PyTensor, mirroring `_mvn_prefix_condition` |
| `fit_svi` + `AutoNormal` guide | `pm.fit(method="advi")` (mean-field ≙ AutoNormal); `fullrank_advi` ≙ AutoMultivariateNormal; optimizer via `obj_optimizer=pm.adam(...)` |
| `fit_mcmc` + NUTS, kernel resolution | `pm.sample()`; backend choice via `nuts_sampler=` (`"pymc"`, `"nutpie"`, `"numpyro"`, `"blackjax"`) replaces kernel resolution |
| Pathfinder via BlackJAX (581 lines) | Collapses to a thin wrapper over `pymc_extras.fit_pathfinder` |
| `draw_posterior` (fit-type dispatch) | `approx.sample(n)` for VI; thin/subsample `idata.posterior` for MCMC — normalize both to `InferenceData` |
| `Predictive` → `forecast` site | `pm.sample_posterior_predictive(idata, var_names=[...], predictions=True)` on the extended-horizon model — the predictions group comes labeled with forecast-time coords |
| `backtest` (per-window refit) | Same structure, framework-agnostic loop; PRNG keys → integer `random_seed` streams |
| `backtest_vectorized` (vmapped SVI over windows) | **No vmap equivalent.** Mitigation: rolling windows have fixed shapes → build the window model once with `pm.Data`, iterate via `pm.set_data` to avoid recompiles; nutpie compiled-model reuse for MCMC. Statistical parity, not wall-clock parity |
| `convert.py` → ArviZ DataTree | Mostly free: PyMC returns `InferenceData` natively and `predictions=True` creates the predictions groups. Keep a small helper for forecast-time coord labeling |
| `metrics.py` / `evaluate_forecast` (jitted JAX, axis conventions) | Dim-aware xarray/NumPy implementations: metrics take forecast draws and truth as labeled arrays, reduce over named dims |
| `features.py`, `datasets.py` | Near-verbatim port to NumPy; feature builders return labeled outputs (coords for the feature dim) |
| jaxtyping/beartype runtime typing | Drop; plain annotations + dim/coord validation at API boundaries |
| Exceptions taxonomy | Port with renames (guide/kernel resolution errors → VI-method/sampler resolution errors) |

## Where pymc-extras fits

- **`pymc_extras.statespace`**: structural time series (level/trend, seasonality,
  cycles, AR, regression components), SARIMAX, VARMAX — with Kalman filtering and
  built-in `.forecast()`. This covers the linear-Gaussian slice of what
  `markov_time_series` is used for, with exact marginalization instead of sampling
  per-step latents (usually better posteriors and faster). The port should
  interoperate: statespace models as first-class citizens in `backtest`/metrics.
- **`fit_pathfinder`**: replaces the entire BlackJAX contrib module.
- Scan-based `markov_time_series` remains valuable for arbitrary nonlinear /
  non-Gaussian transitions that statespace can't express.

## Key design decisions

1. **Dims/coords everywhere (decided).** No positional axis conventions anywhere in
   the API: model variables carry named dims, forecast outputs are
   `InferenceData`/xarray with real time coordinates, metrics reduce over named
   dims. Coords (e.g. `pandas.DatetimeIndex`) flow from input data through to
   forecast outputs.
2. **Model API shape.** Keep the two-level design — a functional core where the
   user writes a model body against a `Horizon` (primitives: `time_series`,
   `markov_time_series`, `predict`, `predict_glm`) executed inside a managed
   `pm.Model` context, plus an OOP `ForecastingModel` facade. The builder
   constructs a fresh model per call (train: no future coords; forecast: extended
   coords), which is idiomatic PyMC and keeps the "one model definition" invariant.
3. **Randomness.** `rng_key` threading → `random_seed` integers / numpy Generators;
   derive per-window seeds deterministically in `backtest`.
4. **Performance posture.** Accept that per-window refits recompile unless shapes
   are fixed; document `nuts_sampler="nutpie"` / `"numpyro"` for speed; use
   `pm.Data` + `set_data` wherever shapes allow.

## What gets dropped or shrinks

- `surgery.py` singledispatch machinery — dissolves into direct likelihood
  construction (only the MVN conditional survives).
- `contrib/blackjax.py` (581 lines) → thin wrapper around pymc-extras.
- `convert.py` (380 lines) → small coords/labeling helper on native `InferenceData`.
- `arrays.py` axis-layout helpers → replaced by coord handling.
- jaxtyping/beartype import hook, JAX-specific vmap invariant tests.

## Risks / open questions

- **Posterior replay semantics**: upstream relies on "guide never sees `_future`
  sites". PyMC's `sample_posterior_predictive` prior-samples missing vars, but
  deterministic replay vs. resampling of *dependent* vars needs careful
  `var_names` handling — cover with the upstream consistency tests (SVI vs. MCMC
  forecasts agree on conjugate examples). Prototype first.
- **Scan + posterior replay**: replaying scan-internal latents from a trace while
  continuing the state forward is the trickiest mechanism; prototype early.
- **Wall-clock**: no vmapped multi-window SVI; decide how much of
  `backtest_vectorized`'s speed promise to keep.
- **Multivariate/hierarchical batch semantics**: upstream leans on NumPyro plates +
  broadcasting; dims/coords cover it, but the shape-heavy tests should be ported
  wholesale (translated to dims).

# PyMC-Forecast

Bayesian time-series forecasting toolkit for PyMC (import name: `pymc_forecast`). A port of
[numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast), redesigned
around PyMC idioms — see PLAN.md for the full design and the issue tracker for
the roadmap.

## Conventions

- **Dims/coords everywhere.** No positional axis conventions (the upstream
  "time at axis -2" rule is deliberately NOT carried over). Model variables carry
  named dims (`"time"`, `"time_future"`, `"obs"`, batch dims); results are
  `arviz.InferenceData` / `xarray` with real coordinates; metrics are dim-aware.
- **Forecasting mechanism.** One model definition trains and forecasts: forecast
  horizon latents live on separate `{name}_future` variables absent from the
  trace, so `pm.sample_posterior_predictive` replays the posterior in-sample and
  draws the future from the prior.
- Inference: ADVI/NUTS from PyMC core; Pathfinder and state-space models from
  pymc-extras.

## Workflow

- `uv sync --all-extras` to set up; `uv run pytest` to test; `uv run ruff check .`
  and `uv run ruff format .` for lint/format.
- Upstream reference source (for porting): https://github.com/juanitorduz/numpyro_forecast

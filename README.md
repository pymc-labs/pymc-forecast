# pymc_forecast

Bayesian time-series forecasting with [PyMC](https://www.pymc.io): you write the
generative model; the package handles the train/forecast plumbing, inference,
backtesting, and evaluation.

A PyMC port of [numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast)
(itself a port of Pyro's `pyro.contrib.forecast`) — redesigned around PyMC idioms
rather than a 1:1 translation.

> **Status: early development.** The design and roadmap live in
> [PLAN.md](PLAN.md) and the
> [issue tracker](https://github.com/pymc-labs/pymc_forecast/issues).

## Design principles

- **One model trains and forecasts.** In-sample time latents are fitted; the
  forecast horizon uses separate `{name}_future` variables that
  `pm.sample_posterior_predictive` draws from the prior while replaying the
  posterior for everything else.
- **Dims and coords everywhere.** No positional axis conventions: variables carry
  named dims (`"time"`, `"time_future"`, `"obs"`, batch dims), results are
  `arviz.InferenceData` / `xarray` objects with real coordinates, and metrics are
  dim-aware.
- **Not AutoML.** No model zoo, no automatic feature pipelines — a clean path from
  a hand-written PyMC model to probabilistic forecasts and scores.
- **Leverage the ecosystem.** ADVI/NUTS from PyMC core, Pathfinder and state-space
  models from [pymc-extras](https://github.com/pymc-devs/pymc-extras), diagnostics
  and storage from [ArviZ](https://python.arviz.org).

## Development

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
```

## License

Apache-2.0. Portions derived from
[numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast) (Apache-2.0).

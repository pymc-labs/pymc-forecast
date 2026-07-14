# Prediction output schema

Prediction dimension and coordinate names are part of the public API. Adapters
can consume forecast draws without inspecting positional axes or renaming
package-defined dimensions.

## Dimension contract

| Output | Variable | Dimensions |
|---|---|---|
| `forecast(...)` | `predictions/forecast` | `(chain, draw, time_future, ...)` |
| `predict_in_sample(...)` | `posterior_predictive/obs` | `(chain, draw, time, ...)` |

The names are also exported as `CHAIN_DIM`, `DRAW_DIM`, `SAMPLE_DIMS`,
`TIME_DIM`, and `FUTURE_DIM` from `pymc_forecast`.

- `chain` and `draw` are always the first two dimensions and are never reduced
  by the prediction API. `num_samples` selects draws; it does not summarize
  them.
- `time` carries the exact observed training coordinate.
- `time_future` carries the exact forecast coordinate supplied at prediction
  time, or the coordinate inferred by `horizon=`.
- Every non-time data dimension follows the time dimension in its original
  xarray order. Its coordinate values are preserved. A pandas `DataFrame`
  target uses the default name `series`; an explicitly labeled xarray target
  can use a domain name such as `unit`, `treated_units`, or `region`.
- Forecast latent variables requested through `var_names` follow the same
  `(chain, draw, time_future, ...)` ordering and retain their named model dims.

For example, a target with dims `(time, treated_units)` produces forecast draws
with dims `(chain, draw, time_future, treated_units)`. A downstream causal
adapter only needs to rename `time_future` to its own observation-index name;
the unit coordinate already aligns by label.

Changes to this contract are breaking API changes and will be called out in the
changelog.

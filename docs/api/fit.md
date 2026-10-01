# `pymc_forecast.fit`

Functional fitters shared with the forecaster classes. Each takes an optional
already-built `model=`. When `model` is omitted, the fitter drops covariate
rows past the training window, calls `build_model(model_fn, data, covariates)`
once, and samples that model. Class `_fit` methods pass `model=self.model` and do not build again.

```{eval-rst}
.. automodule:: pymc_forecast.fit
   :members:
   :show-inheritance:
```

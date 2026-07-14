# `pymc_forecast.priors`

Model objects can keep their prior choices visible and user-overridable with
the `pymc-extras` `Prior` API:

```python
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.prior import Prior
from pymc_forecast import ForecastingModel

class Regression(ForecastingModel):
    default_priors = {
        "intercept": Prior("Normal", mu=0, sigma=2),
        "beta": Prior("Normal", mu=0, sigma=1, dims="covariate"),
        "sigma": Prior("HalfNormal", sigma=1),
    }

    def model(self, h, covariates):
        intercept = self.create_prior("intercept")
        beta = self.create_prior("beta")
        sigma = self.create_prior("sigma")
        mu = intercept + pt.dot(covariates.values, beta)
        self.predict(
            lambda name, value, dims, observed: pm.Normal(
                name, value, sigma, dims=dims, observed=observed
            ),
            mu,
        )

model = Regression(
    priors={"beta": Prior("Normal", mu=0, sigma=0.25, dims="covariate")}
)
```

`model.prior_config` contains the effective mapping after defaults and user
overrides are combined. The same mechanism is available to `StatespaceModel`;
call `create_prior` from its `priors` method.

The integration uses a small `create_variable(name)` protocol instead of
importing pymc-extras in core, so the optional dependency is only required
when its `Prior` objects are actually used. A callable such as
`lambda name: pm.Normal(name, 0, 1)` is also accepted.

```{eval-rst}
.. automodule:: pymc_forecast.priors
   :members:
   :show-inheritance:
```

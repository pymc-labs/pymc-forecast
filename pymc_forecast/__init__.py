"""pymc_forecast: Bayesian time-series forecasting with PyMC.

A PyMC port of `numpyro_forecast <https://github.com/juanitorduz/numpyro_forecast>`_,
redesigned around PyMC idioms: named dims/coords everywhere, ``InferenceData``
results, and inference via PyMC core and pymc-extras.
"""

from pymc_forecast.data import (
    FUTURE_DIM,
    TIME_DIM,
    as_dataarray,
    null_covariates,
)
from pymc_forecast.exceptions import (
    AlignmentError,
    BacktestWindowError,
    HorizonError,
    MethodResolutionError,
    OptionalDependencyError,
    PymcForecastError,
)
from pymc_forecast.forecaster import (
    Forecaster,
    HMCForecaster,
    PathfinderForecaster,
)
from pymc_forecast.model import (
    ForecastingModel,
    Horizon,
    build_model,
    predict,
    time_series,
)
from pymc_forecast.prediction import (
    forecast,
    predict_in_sample,
    thin_draws,
)

__version__ = "0.0.1.dev0"

__all__ = [
    "FUTURE_DIM",
    "TIME_DIM",
    "AlignmentError",
    "BacktestWindowError",
    "Forecaster",
    "ForecastingModel",
    "HMCForecaster",
    "Horizon",
    "HorizonError",
    "MethodResolutionError",
    "OptionalDependencyError",
    "PathfinderForecaster",
    "PymcForecastError",
    "__version__",
    "as_dataarray",
    "build_model",
    "forecast",
    "null_covariates",
    "predict",
    "predict_in_sample",
    "thin_draws",
    "time_series",
]

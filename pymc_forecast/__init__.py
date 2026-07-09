"""pymc_forecast: Bayesian time-series forecasting with PyMC.

A PyMC port of `numpyro_forecast <https://github.com/juanitorduz/numpyro_forecast>`_,
redesigned around PyMC idioms: named dims/coords everywhere, ``InferenceData``
results, and inference via PyMC core and pymc-extras.
"""

from pymc_forecast.data import (
    FUTURE_DIM,
    TIME_DIM,
    as_dataarray,
    extend_time_index,
    null_covariates,
)
from pymc_forecast.evaluate import (
    BacktestResult,
    backtest,
    results_to_dataframe,
)
from pymc_forecast.exceptions import (
    AlignmentError,
    BacktestWindowError,
    HorizonError,
    MethodResolutionError,
    OptionalDependencyError,
    PymcForecastError,
)
from pymc_forecast.features import fourier_features, periodic_repeat
from pymc_forecast.forecaster import (
    Forecaster,
    HMCForecaster,
    PathfinderForecaster,
)
from pymc_forecast.gaussian import conditional_mvn, predict_mvn
from pymc_forecast.markov import markov_time_series
from pymc_forecast.metrics import (
    DEFAULT_METRICS,
    crps_empirical,
    eval_coverage,
    eval_crps,
    eval_interval_score,
    eval_mae,
    eval_pinball,
    eval_rmse,
    evaluate_forecast,
    make_mase,
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
    "DEFAULT_METRICS",
    "FUTURE_DIM",
    "TIME_DIM",
    "AlignmentError",
    "BacktestResult",
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
    "backtest",
    "build_model",
    "conditional_mvn",
    "crps_empirical",
    "eval_coverage",
    "eval_crps",
    "eval_interval_score",
    "eval_mae",
    "eval_pinball",
    "eval_rmse",
    "evaluate_forecast",
    "extend_time_index",
    "forecast",
    "fourier_features",
    "make_mase",
    "markov_time_series",
    "null_covariates",
    "periodic_repeat",
    "predict",
    "predict_in_sample",
    "predict_mvn",
    "results_to_dataframe",
    "thin_draws",
    "time_series",
]

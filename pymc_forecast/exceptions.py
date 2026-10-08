"""Exception taxonomy for pymc_forecast.

Package-specific errors derive from :class:`PymcForecastError`, so callers
can catch that family with one clause. Specific subclasses exist where a
caller might plausibly branch on the failure mode. Plain argument validation
raises builtin exceptions instead: mostly ``ValueError``/``TypeError`` (e.g.
in metrics and features), plus ``KeyError`` for unknown names (prior names,
statespace ``var_names``) and ``NotImplementedError`` for unsupported
statespace configurations.
"""

__all__ = [
    "AlignmentError",
    "BacktestWindowError",
    "HorizonError",
    "MethodResolutionError",
    "NotFittedError",
    "OptionalDependencyError",
    "PymcForecastError",
]


class PymcForecastError(Exception):
    """Base class for all pymc_forecast errors."""


class HorizonError(PymcForecastError, ValueError):
    """Forecast-horizon or model-structure misuse.

    Raised when the train/forecast horizon cannot be derived or is
    inconsistent, and for related model-building misuse, e.g. missing model
    coords, horizon helpers used outside a model build, misuse of ``.dist()``
    or ``Prior`` observation/noise specs, reserved predictive-output names
    (``mu``, ``expected_observation``, ...) already defined in the model,
    multivariate data passed to ``predict_mvn``, or a model without the
    expected ``obs``/future variables.
    """


class AlignmentError(PymcForecastError, ValueError):
    """Inputs are malformed or do not align along the time dimension.

    Covers data/covariate misalignment as well as invalid input shapes (e.g.
    no ``"time"`` dim), invalid horizons or non-extendable time indices,
    misordered future indices, and covariate structure mismatches.
    """


class MethodResolutionError(PymcForecastError, ValueError):
    """A VI method, optimizer, or VI backend specification could not be resolved."""


class BacktestWindowError(PymcForecastError, ValueError):
    """Backtest windowing parameters are invalid or admit no valid windows."""


class NotFittedError(PymcForecastError, RuntimeError):
    """A predictive method was called on a forecaster that has not been fit."""


class OptionalDependencyError(PymcForecastError, ImportError):
    """An optional dependency is required for the requested feature.

    Parameters
    ----------
    package : str
        Distribution name of the missing dependency (used in the install
        hint).
    extra : str
        Name of the ``pymc-forecast`` extra that provides it.
    feature : str
        Human-readable name of the feature that needs it.

    Attributes
    ----------
    package : str
        Distribution name of the missing dependency.
    """

    def __init__(self, package: str, extra: str, feature: str) -> None:
        self.package = package
        super().__init__(
            f"{feature} requires the optional dependency '{package}'. "
            f"Install it with: pip install 'pymc-forecast[{extra}]' "
            f"or: pip install {package}"
        )

"""Exception taxonomy for pymc_forecast.

Every package-raised error derives from :class:`PymcForecastError`, so callers
can catch the whole family with one clause. Specific subclasses exist where a
caller might plausibly branch on the failure mode.
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
    """The train/forecast horizon could not be derived or is inconsistent."""


class AlignmentError(PymcForecastError, ValueError):
    """Data and covariates do not align along the time dimension."""


class MethodResolutionError(PymcForecastError, ValueError):
    """A VI-method, optimizer, or sampler specification could not be resolved."""


class BacktestWindowError(PymcForecastError, ValueError):
    """Backtest windowing parameters admit no valid windows."""


class NotFittedError(PymcForecastError, RuntimeError):
    """A predictive method was called on a forecaster that has not been fit."""


class OptionalDependencyError(PymcForecastError, ImportError):
    """An optional dependency is required for the requested feature."""

    def __init__(self, package: str, extra: str, feature: str) -> None:
        self.package = package
        super().__init__(
            f"{feature} requires the optional dependency '{package}'. "
            f"Install it with: pip install 'pymc-forecast[{extra}]' "
            f"or: pip install {package}"
        )

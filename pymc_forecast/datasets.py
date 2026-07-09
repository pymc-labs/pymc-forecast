"""Dataset helpers for the examples and docs.

:func:`load_victoria_electricity` reads a small CSV bundled with the package
and returns labeled arrays with real hourly time coords.

The upstream BART origin-destination loaders are not ported yet: they depend on
numpyro's example-dataset mirror, and pulling in a JAX stack for data plumbing
is not acceptable for a PyMC package. See the tracking issue for the
data-source decision.
"""

import importlib.resources

import numpy as np
import pandas as pd
import xarray as xr

from pymc_forecast.data import TIME_DIM

__all__ = ["load_victoria_electricity"]

_VICTORIA_START = "2014-01-01"


def load_victoria_electricity() -> tuple[xr.DataArray, xr.DataArray]:
    """Load hourly Victoria (Australia) electricity demand and temperature.

    The series covers the first eight weeks of 2014, sampled hourly — the
    Victoria electricity demand data used in the TensorFlow Probability
    structural-time-series case study and in Hyndman & Athanasopoulos'
    *Forecasting: Principles and Practice* (original half-hourly data
    downsampled to hourly). Bundled as a small CSV.

    Returns
    -------
    demand : xarray.DataArray
        Hourly electricity demand (GW), dims ``("time",)`` with an hourly
        ``DatetimeIndex`` coord.
    temperature : xarray.DataArray
        Hourly temperature (°C), aligned with ``demand``.
    """
    source = importlib.resources.files("pymc_forecast").joinpath("data", "victoria_electricity.csv")
    with source.open("r", encoding="utf-8") as handle:
        table = np.loadtxt(handle, delimiter=",", skiprows=1, dtype=np.float64)
    index = pd.date_range(_VICTORIA_START, periods=table.shape[0], freq="h")
    demand = xr.DataArray(table[:, 0], dims=(TIME_DIM,), coords={TIME_DIM: index}, name="demand")
    temperature = xr.DataArray(
        table[:, 1], dims=(TIME_DIM,), coords={TIME_DIM: index}, name="temperature"
    )
    return demand, temperature

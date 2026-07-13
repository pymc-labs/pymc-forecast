"""Dataset helpers for the examples and docs.

:func:`load_bart_od` downloads and caches the complete hourly BART
origin-destination panel. :func:`load_bart_weekly` and
:func:`load_bart_weekly_by_origin` derive compact weekly examples from that
source, while :func:`load_victoria_electricity` reads a small CSV bundled with
the package. All loaders return labeled arrays.
"""

import importlib.resources
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import xarray as xr

from pymc_forecast.data import TIME_DIM

__all__ = [
    "load_bart_od",
    "load_bart_weekly",
    "load_bart_weekly_by_origin",
    "load_victoria_electricity",
]

_HOURS_PER_WEEK = 24 * 7
_VICTORIA_START = "2014-01-01"
_BART_DATA = pooch.create(
    path=pooch.os_cache("pymc_forecast"),
    base_url="https://raw.githubusercontent.com/pyro-ppl/datasets/master/bart/",
    registry={
        "bart_0.npz": "sha256:9900a4956849c095f2fa9484a9dc12b48349865ea2cdff10a5a8fd16c7fb6170",
        "bart_1.npz": "sha256:0318d47c6b7ffc163ca54b1cbb95de207d9398956de37fcd6d7ea0f10588d4c4",
        "bart_2.npz": "sha256:f34a4787d2a85c500dfad0ec3f83438ff5b055fc0977a91c85c2015192096523",
        "bart_3.npz": "sha256:f0bf98d8876b3a2ebf7c57716edf2d06556bb45cb7dea0329a01df3ed515d52a",
    },
)


def _bart_file_paths() -> list[Path]:
    return [Path(_BART_DATA.fetch(name, progressbar=False)) for name in _BART_DATA.registry]


def load_bart_od() -> xr.DataArray:
    """Load complete hourly BART origin-destination ridership counts.

    The four source shards are downloaded from the public Pyro dataset mirror,
    verified by SHA-256, and cached in the operating system's user cache.

    Returns
    -------
    xarray.DataArray
        Integer counts with dims ``("time", "origin", "destination")``.
        The time coordinate is hourly from 2011-01-01, and station names label
        both origin and destination.
    """
    counts = []
    stations = None
    start_date = None
    for path in _bart_file_paths():
        with np.load(path, allow_pickle=True) as shard:
            if stations is None:
                stations = np.asarray(shard["stations"], dtype=str)
                start_date = shard["start_date"].item()
            counts.append(np.asarray(shard["counts"]))

    values = np.concatenate(counts, axis=0)
    time = np.datetime64(start_date, "h") + np.arange(values.shape[0])
    return xr.DataArray(
        values,
        dims=(TIME_DIM, "origin", "destination"),
        coords={TIME_DIM: time, "origin": stations, "destination": stations},
        name="rides",
    )


def load_bart_weekly() -> xr.DataArray:
    """Load total weekly BART ridership on the log scale.

    The series is derived at load time from the complete public BART
    origin-destination dataset used by the Pyro and NumPyro forecasting
    examples. Hourly counts are summed over all origin-destination pairs,
    aggregated into non-overlapping weeks, and log-transformed.

    Returns
    -------
    xarray.DataArray
        Log weekly totals with dims ``("time",)`` and integer week coords.
    """
    hourly_totals = []
    for path in _bart_file_paths():
        with np.load(path, allow_pickle=True) as shard:
            hourly_totals.append(shard["counts"].sum(axis=(1, 2), dtype=np.int64))
    hourly = np.concatenate(hourly_totals)
    num_weeks = hourly.size // _HOURS_PER_WEEK
    weekly = hourly[: num_weeks * _HOURS_PER_WEEK]
    weekly = weekly.reshape(num_weeks, _HOURS_PER_WEEK).sum(axis=1)
    values = np.log(weekly)
    return xr.DataArray(
        values,
        dims=(TIME_DIM,),
        coords={TIME_DIM: np.arange(values.size)},
        name="log_rides",
    )


def load_bart_weekly_by_origin(num_series: int | None = 8) -> xr.DataArray:
    """Load a weekly BART ridership panel grouped by origin station.

    Counts are summed over destination stations and aggregated into
    non-overlapping weeks before applying ``log1p``. Aggregation happens a
    shard at a time, avoiding materializing the much larger full
    origin-destination panel. By default only the eight busiest origins are
    returned, which keeps hierarchical examples quick; pass ``None`` for all
    stations.

    Parameters
    ----------
    num_series
        Number of busiest origin stations to retain, or ``None`` for all.

    Returns
    -------
    xarray.DataArray
        Log weekly counts with dims ``("time", "series")`` and station names
        on the ``"series"`` coordinate.
    """
    hourly_shards = []
    stations = None
    for path in _bart_file_paths():
        with np.load(path, allow_pickle=True) as shard:
            if stations is None:
                stations = np.asarray(shard["stations"], dtype=str)
            hourly_shards.append(shard["counts"].sum(axis=2, dtype=np.int64))

    hourly = np.concatenate(hourly_shards, axis=0)
    num_weeks = hourly.shape[0] // _HOURS_PER_WEEK
    weekly = hourly[: num_weeks * _HOURS_PER_WEEK]
    values = weekly.reshape(num_weeks, _HOURS_PER_WEEK, -1).sum(axis=1)
    if num_series is not None:
        if num_series < 1:
            msg = f"num_series must be positive or None, got {num_series}"
            raise ValueError(msg)
        order = np.argsort(values.sum(axis=0))[::-1][:num_series]
        values = values[:, order]
        stations = stations[order]
    return xr.DataArray(
        np.log1p(values),
        dims=(TIME_DIM, "series"),
        coords={TIME_DIM: np.arange(values.shape[0]), "series": stations},
        name="log_rides",
    )


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

"""Dataset helpers for the examples and docs.

:func:`load_bart_od` downloads and caches the complete hourly BART
origin-destination panel. :func:`load_bart_weekly` and
:func:`load_bart_weekly_by_origin` derive compact weekly examples from that
source, :func:`load_victoria_electricity` reads a small CSV bundled with the
package, and :func:`load_m5` downloads the M5 competition files once and
returns dense sales and price arrays. All loaders return labeled arrays or
tables.
"""

import importlib.resources
import re
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pooch
import xarray as xr

from pymc_forecast.data import TIME_DIM

__all__ = [
    "M5Data",
    "load_bart_od",
    "load_bart_weekly",
    "load_bart_weekly_by_origin",
    "load_m5",
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


_M5_KEYS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
_M5_FILES = (
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sales_test_evaluation.csv",
    "sell_prices.csv",
    "weights_evaluation.csv",
)
_M5_BASE_URL = (
    "https://github.com/Nixtla/m5-forecasts/raw/72b8e7fd3b565b3c538adcb1d1a05117d8562d7e/datasets/"
)
_M5_SHA256 = "sha256:cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
_DAY_COLUMN = re.compile(r"d_(\d+)")


class M5Data(NamedTuple):
    """M5 evaluation data as dense arrays plus the identifier and calendar tables.

    ``sales`` and ``price`` are float32 arrays shaped ``(days, series)``, series
    in sales-file order. ``price`` is the weekly shelf price repeated over the
    days of ``wm_yr_wk``, and is NaN where the item was not listed. ``keys`` has
    one row per series (``id`` is ``item_id`` and ``store_id`` joined by ``_``).
    ``calendar`` has one row per day. ``weights`` is the official evaluation
    weight table.
    """

    sales: np.ndarray
    price: np.ndarray
    keys: pd.DataFrame
    calendar: pd.DataFrame
    weights: pd.DataFrame


def _m5_directory(cache_dir: str | Path | None) -> Path:
    directory = (
        Path(pooch.os_cache("pymc_forecast")) / "m5" if cache_dir is None else Path(cache_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    if all((directory / name).is_file() for name in _M5_FILES):
        return directory
    fetcher = pooch.create(path=directory, base_url=_M5_BASE_URL, registry={"m5.zip": _M5_SHA256})
    archive = Path(fetcher.fetch("m5.zip", progressbar=False))
    with zipfile.ZipFile(archive) as archive_file:
        archive_file.extractall(directory, members=list(_M5_FILES))
    missing = [name for name in _M5_FILES if not (directory / name).is_file()]
    if missing:
        msg = f"M5 archive did not contain {missing}"
        raise ValueError(msg)
    return directory


def _day_columns(columns) -> list[str]:
    numbered = []
    for name in columns:
        match = _DAY_COLUMN.fullmatch(str(name))
        if match:
            numbered.append((int(match.group(1)), str(name)))
    numbered.sort()
    return [name for _, name in numbered]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"{source} is missing {missing}"
        raise ValueError(msg)


def _sales_frame(directory: Path) -> tuple[pd.DataFrame, list[str]]:
    train = pd.read_csv(directory / "sales_train_evaluation.csv")
    test = pd.read_csv(directory / "sales_test_evaluation.csv")
    _require_columns(train, _M5_KEYS, source="sales_train_evaluation.csv")
    _require_columns(test, _M5_KEYS, source="sales_test_evaluation.csv")
    if train.duplicated(list(_M5_KEYS)).any() or test.duplicated(list(_M5_KEYS)).any():
        msg = "sales files have duplicate item-store keys"
        raise ValueError(msg)
    train_days = _day_columns(train.columns)
    test_days = _day_columns(test.columns)
    overlap = sorted(set(train_days) & set(test_days), key=lambda name: int(name.split("_")[1]))
    if overlap:
        msg = f"sales day columns overlap: {overlap[:3]}"
        raise ValueError(msg)
    day_columns = _day_columns([*train_days, *test_days])
    numbers = [int(name.split("_")[1]) for name in day_columns]
    if not numbers or numbers[0] != 1 or numbers != list(range(1, numbers[-1] + 1)):
        msg = "sales day columns must be the contiguous range d_1, d_2, ..."
        raise ValueError(msg)
    merged = train.merge(test[list(_M5_KEYS) + test_days], on=list(_M5_KEYS), how="left")
    keys = merged[list(_M5_KEYS)].copy()
    keys.insert(0, "id", keys["item_id"].astype(str) + "_" + keys["store_id"].astype(str))
    if keys["id"].duplicated().any():
        msg = "item_id and store_id do not identify a unique series"
        raise ValueError(msg)
    sales = merged[day_columns].to_numpy(dtype=np.float32).T
    return pd.concat([keys, pd.DataFrame(sales.T, columns=day_columns)], axis=1), day_columns


def _expand_prices(directory: Path, keys: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    prices = pd.read_csv(directory / "sell_prices.csv")
    _require_columns(
        prices,
        ("store_id", "item_id", "wm_yr_wk", "sell_price"),
        source="sell_prices.csv",
    )
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        msg = "sell_prices.csv has duplicate store-item-week rows"
        raise ValueError(msg)
    prices = prices.copy()
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype(np.int64)
    wide = prices.pivot(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
    order = pd.MultiIndex.from_frame(keys[["store_id", "item_id"]])
    wide = wide.reindex(order)
    weeks = calendar["wm_yr_wk"].astype(np.int64).to_numpy()
    positions = wide.columns.get_indexer(weeks)
    values = wide.to_numpy(dtype=np.float64)
    price = np.full((len(keys), len(weeks)), np.nan, dtype=np.float32)
    valid = positions >= 0
    if valid.any():
        price[:, valid] = values[:, positions[valid]]
    return price.T


def load_m5(cache_dir: str | Path | None = None) -> M5Data:
    """Load the M5 evaluation panel, downloading and caching the archive once.

    The files come from Nixtla's mirror of the competition data, pinned to
    commit ``72b8e7fd`` and checked against a SHA-256 digest. The archive is
    about 50 MB; the dense sales and price arrays need roughly 500 MB. Pass
    ``cache_dir`` to read an already extracted directory instead of the
    default cache (``pooch.os_cache("pymc_forecast") / "m5"``). When every
    competition file is already in that directory, nothing is downloaded.

    Series order follows ``sales_train_evaluation.csv``. The test file is
    left-joined onto it, so a test-only row is ignored and a missing test row
    leaves NaN sales on the evaluation days. Weekly prices are mapped by
    ``wm_yr_wk``; a calendar week absent from ``sell_prices.csv`` is NaN and
    does not shift the other weeks.

    Returns
    -------
    M5Data
        Sales, prices, identifiers, calendar, and official evaluation weights.
    """
    directory = _m5_directory(cache_dir)
    frame, day_columns = _sales_frame(directory)
    keys = frame[["id", *_M5_KEYS]].reset_index(drop=True)
    sales = frame[day_columns].to_numpy(dtype=np.float32).T
    calendar = pd.read_csv(directory / "calendar.csv", parse_dates=["date"])
    if len(calendar) < sales.shape[0]:
        msg = f"calendar has {len(calendar)} rows for {sales.shape[0]} sales days"
        raise ValueError(msg)
    calendar = calendar.iloc[: sales.shape[0]].reset_index(drop=True)
    _require_columns(
        calendar,
        ("date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"),
        source="calendar.csv",
    )
    price = _expand_prices(directory, keys, calendar)
    weights = pd.read_csv(directory / "weights_evaluation.csv")
    _require_columns(
        weights,
        ("Level_id", "Agg_Level_1", "Agg_Level_2", "Dollar_Sales", "weight"),
        source="weights_evaluation.csv",
    )
    return M5Data(sales=sales, price=price, keys=keys, calendar=calendar, weights=weights)

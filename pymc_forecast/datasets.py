"""Dataset helpers for the examples and docs.

:func:`load_bart_od` downloads and caches the complete hourly BART
origin-destination panel. :func:`load_bart_weekly` and
:func:`load_bart_weekly_by_origin` derive compact weekly examples from that
source, :func:`load_victoria_electricity` and :func:`load_us_macro` read small
CSVs bundled with the package, and :func:`load_m5` downloads the M5 competition
files once and returns labeled sales and price panels. Every loader returns
labeled arrays; :func:`load_m5` adds the identifier, calendar, and weight tables.
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
    "load_us_macro",
    "load_victoria_electricity",
]

_HOURS_PER_WEEK = 24 * 7
_VICTORIA_START = "2014-01-01"
_US_MACRO_START = "1959-01-01"
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


def load_us_macro() -> xr.DataArray:
    """Load quarterly US real GDP, consumption and investment levels.

    The ``statsmodels`` ``macrodata`` panel (public domain; FRED, accessed
    December 2009), restricted to the three real-activity series: 1959Q1 to
    2009Q3, seasonally adjusted annual rates in billions of chained 2005 US$.
    Bundled as a small CSV.

    Returns
    -------
    xarray.DataArray
        Levels with dims ``("time", "series")``; ``time`` is a quarter-start
        ``DatetimeIndex`` and ``series`` is ``["realgdp", "realcons",
        "realinv"]``.
    """
    source = importlib.resources.files("pymc_forecast").joinpath("data", "us_macro.csv")
    with source.open("r", encoding="utf-8") as handle:
        names = handle.readline().strip().split(",")
        table = np.loadtxt(handle, delimiter=",", dtype=np.float64)
    index = pd.date_range(_US_MACRO_START, periods=table.shape[0], freq="QS")
    return xr.DataArray(
        table,
        dims=(TIME_DIM, "series"),
        coords={TIME_DIM: index, "series": names},
        name="levels",
    )


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
    """M5 evaluation data: labeled sales and price panels plus three tables.

    ``sales`` and ``price`` are float32 :class:`xarray.DataArray` panels with
    dims ``("time", "series")``: the ``"time"`` coord is the calendar date of
    each day and the ``"series"`` coord is the series ``id`` (``item_id`` and
    ``store_id`` joined by ``_``), in sales-file order. ``price`` is the weekly
    shelf price repeated over the days of ``wm_yr_wk``, NaN where the item
    was not listed. ``keys`` has one row per series with the ``id`` and the
    five hierarchy columns; ``calendar`` has one row per day; ``weights`` is
    the official evaluation weight table.
    """

    sales: xr.DataArray
    price: xr.DataArray
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


def _read_sales(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a sales file with the day columns as float32 (half the default int64)."""
    header = pd.read_csv(path, nrows=0)
    _require_columns(header, _M5_KEYS, source=path.name)
    days = _day_columns(header.columns)
    frame = pd.read_csv(path, usecols=[*_M5_KEYS, *days], dtype=dict.fromkeys(days, np.float32))
    if frame.duplicated(list(_M5_KEYS)).any():
        msg = f"{path.name} has duplicate item-store keys"
        raise ValueError(msg)
    return frame, days


def _sales_panel(directory: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Series keys and the ``(days, series)`` float32 sales, train days then test days."""
    train, train_days = _read_sales(directory / "sales_train_evaluation.csv")
    test, test_days = _read_sales(directory / "sales_test_evaluation.csv")
    overlap = sorted(set(train_days) & set(test_days), key=lambda name: int(name.split("_")[1]))
    if overlap:
        msg = f"sales day columns overlap: {overlap[:3]}"
        raise ValueError(msg)
    numbers = [int(name.split("_")[1]) for name in _day_columns([*train_days, *test_days])]
    if not numbers or numbers[0] != 1 or numbers != list(range(1, numbers[-1] + 1)):
        msg = "sales day columns must be the contiguous range d_1, d_2, ..."
        raise ValueError(msg)
    keys = train[list(_M5_KEYS)].reset_index(drop=True)
    keys.insert(0, "id", keys["item_id"].astype(str) + "_" + keys["store_id"].astype(str))
    if keys["id"].duplicated().any():
        msg = "item_id and store_id do not identify a unique series"
        raise ValueError(msg)
    # Align the test rows on the train keys: a test-only row drops out, a
    # missing test row is NaN on the evaluation days.
    test_values = (
        test.set_index(list(_M5_KEYS))[test_days]
        .reindex(pd.MultiIndex.from_frame(keys[list(_M5_KEYS)]))
        .to_numpy(dtype=np.float32)
    )
    sales = np.empty((len(train_days) + len(test_days), len(keys)), dtype=np.float32)
    sales[: len(train_days)] = train[train_days].to_numpy(dtype=np.float32).T
    sales[len(train_days) :] = test_values.T
    return keys, sales


def _level_codes(values: pd.Series, levels: pd.Index) -> np.ndarray:
    """Position of each categorical value in ``levels``; -1 when absent or missing."""
    categories = values.cat.categories.astype(str)
    mapped = np.append(levels.get_indexer(categories), -1)  # trailing -1 catches NaN codes
    return mapped[values.cat.codes.to_numpy()]


def _expand_prices(directory: Path, keys: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """Weekly shelf prices repeated over the days, as a ``(days, series)`` float32 array.

    The 6.8 million price rows are mapped to integer series and week codes
    and scattered into a ``(weeks, series)`` table; a pivot on the string keys
    would hold the same table behind object-dtype indexes several times over.
    """
    header = pd.read_csv(directory / "sell_prices.csv", nrows=0)
    _require_columns(
        header, ("store_id", "item_id", "wm_yr_wk", "sell_price"), source="sell_prices.csv"
    )
    prices = pd.read_csv(
        directory / "sell_prices.csv",
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": np.int64,
            "sell_price": np.float32,
        },
    )
    stores = pd.Index(keys["store_id"].astype(str).unique())
    items = pd.Index(keys["item_id"].astype(str).unique())
    key_pair = stores.get_indexer(keys["store_id"].astype(str)) * len(items) + items.get_indexer(
        keys["item_id"].astype(str)
    )
    store = _level_codes(prices["store_id"], stores)
    item = _level_codes(prices["item_id"], items)
    pair = np.where((store >= 0) & (item >= 0), store * len(items) + item, -1)
    series = pd.Index(key_pair).get_indexer(pair)
    calendar_weeks = pd.Index(calendar["wm_yr_wk"].astype(np.int64).unique())
    week = calendar_weeks.get_indexer(prices["wm_yr_wk"])
    # Rows for a series or a week outside the panel carry no information.
    keep = (series >= 0) & (week >= 0)
    cell = week[keep].astype(np.int64) * len(keys) + series[keep]
    if np.unique(cell).size != cell.size:
        msg = "sell_prices.csv has duplicate store-item-week rows"
        raise ValueError(msg)
    weekly = np.full((len(calendar_weeks), len(keys)), np.nan, dtype=np.float32)
    weekly[week[keep], series[keep]] = prices["sell_price"].to_numpy()[keep]
    return weekly[calendar_weeks.get_indexer(calendar["wm_yr_wk"].astype(np.int64))]


def load_m5(cache_dir: str | Path | None = None) -> M5Data:
    """Load the M5 evaluation panel, downloading and caching the archive once.

    The files come from Nixtla's mirror of the competition data, pinned to
    commit ``72b8e7fd`` and checked against a SHA-256 digest. The archive is
    about 50 MB and is extracted into ``pooch.os_cache("pymc_forecast") /
    "m5"`` (``~/.cache/pymc_forecast/m5`` on Linux,
    ``~/Library/Caches/pymc_forecast/m5`` on macOS). Pass ``cache_dir`` to
    read an already extracted directory instead; when every competition file
    is in that directory, nothing is downloaded. The two returned panels hold
    about 480 MB; parsing the CSVs peaks at about 2 GB.

    Series order follows ``sales_train_evaluation.csv``. The test file is
    aligned on the train keys, so a test-only row is ignored and a missing
    test row leaves NaN sales on the evaluation days. Weekly prices are mapped
    by ``wm_yr_wk``; a calendar week absent from ``sell_prices.csv`` is NaN
    and does not shift the other weeks.

    Returns
    -------
    M5Data
        Labeled sales and price panels, identifiers, calendar, and official
        evaluation weights.
    """
    directory = _m5_directory(cache_dir)
    keys, sales = _sales_panel(directory)
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
    if calendar["date"].duplicated().any():
        msg = "calendar.csv has duplicate dates"
        raise ValueError(msg)
    price = _expand_prices(directory, keys, calendar)
    weights = pd.read_csv(directory / "weights_evaluation.csv")
    _require_columns(
        weights,
        ("Level_id", "Agg_Level_1", "Agg_Level_2", "Dollar_Sales", "weight"),
        source="weights_evaluation.csv",
    )
    coords = {TIME_DIM: calendar["date"].to_numpy(), "series": keys["id"].to_numpy(dtype=str)}
    return M5Data(
        sales=xr.DataArray(sales, dims=(TIME_DIM, "series"), coords=coords, name="sales"),
        price=xr.DataArray(price, dims=(TIME_DIM, "series"), coords=coords, name="price"),
        keys=keys,
        calendar=calendar,
        weights=weights,
    )

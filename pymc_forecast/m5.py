"""M5 hierarchy, scoring, and the three starter-kit reconciliation models.

The formulas follow the Pyro M5 starter kit as ported by numpyro_forecast:
12-level aggregation, dollar-share weights, lag-1 scales, a Poisson share
split, and the three model predictors. Inference is ordinary mean-field ADVI
through :class:`~pymc_forecast.forecaster.Forecaster`; it is not the kit's
clipped, decaying, minibatch SVI, so posterior draws are not expected to
match a NumPyro run.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp
import xarray as xr

from pymc_forecast.data import FUTURE_DIM, TIME_DIM
from pymc_forecast.features import fourier_features
from pymc_forecast.metrics import crps_empirical
from pymc_forecast.model import FORECAST_VAR, ForecastingModel
from pymc_forecast.prediction import prediction_samples

__all__ = [
    "BEXP_BOUND",
    "BOTTOM_CHANNELS",
    "GAMMA_FLOOR",
    "HEADS",
    "HORIZON",
    "LAGS",
    "LEVELS",
    "M5_QUANTILES",
    "MA_LAG",
    "MA_WINDOWS",
    "N_DAYS",
    "N_DAYS_TRAIN",
    "RATE_CAP",
    "SHARE_WINDOW",
    "T0",
    "TAIL",
    "WEEKDAYS",
    "Aggregation",
    "BottomUpModel",
    "MiddleOutModel",
    "TopDownModel",
    "aggregate",
    "bottom_covariates",
    "bounded_exp",
    "calendar_features",
    "disaggregate",
    "evaluation_origins",
    "forecast_draws",
    "lagged_log_moving_average",
    "last_28_day_shares",
    "level_panel",
    "log_total",
    "m5_scales",
    "m5_weights",
    "mean_level_score",
    "middle_covariates",
    "reconcile_middle_out",
    "reconcile_top_down",
    "require_history",
    "score_bottom",
    "select_series",
    "top_covariates",
    "weight_gap",
    "ws_crps_levels",
    "ws_pinball",
]

LEVELS: dict[str, tuple[str, ...]] = {
    "Level1": (),
    "Level2": ("state_id",),
    "Level3": ("store_id",),
    "Level4": ("cat_id",),
    "Level5": ("dept_id",),
    "Level6": ("state_id", "cat_id"),
    "Level7": ("state_id", "dept_id"),
    "Level8": ("store_id", "cat_id"),
    "Level9": ("store_id", "dept_id"),
    "Level10": ("item_id",),
    "Level11": ("state_id", "item_id"),
    "Level12": ("item_id", "store_id"),
}
"""The 12 M5 aggregation levels, in competition order."""

M5_INTERVALS = np.array([0.5, 0.67, 0.95, 0.99])
M5_QUANTILES = np.sort(
    np.concatenate([[0.5], (1.0 - M5_INTERVALS) / 2.0, (1.0 + M5_INTERVALS) / 2.0])
)
"""The nine pinball quantiles of the M5 uncertainty competition."""

T0 = 121
"""First day whose three lagged moving averages are defined (``37 + 28 * 3``)."""
MA_WINDOWS = (28, 56, 84)
MA_LAG = 28
HORIZON = 28
N_DAYS_TRAIN = 1941
N_DAYS = N_DAYS_TRAIN + HORIZON
SHARE_WINDOW = 28
GAMMA_FLOOR = 1e-3
RATE_CAP = 1e9
BEXP_BOUND = 1e3
TAIL = 7
BOTTOM_CHANNELS = ("dow", "snap", "saled", "ma28", "ma56", "ma84")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
HEADS = ("mean", "scale")
LAGS = ("ma28", "ma56", "ma84")
_SNAP = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


class Aggregation:
    """Sparse sum from bottom series to the 12 M5 levels.

    ``group_index[level]`` maps each bottom series to a column of that level,
    in sorted label order. ``labels`` are ``Level/Agg_Level_1/Agg_Level_2``,
    the official weight-file key.
    """

    def __init__(
        self,
        matrix,
        labels: list[str],
        slices: dict[str, slice],
        group_index: dict[str, np.ndarray],
    ):
        self.matrix = matrix
        self.labels = labels
        self.slices = slices
        self.group_index = group_index

    def labels_for(self, level: str) -> list[str]:
        """Return the ``Agg_Level_1/Agg_Level_2`` labels of one level, sorted."""
        sl = self.slices[level]
        return [label.split("/", 1)[1] for label in self.labels[sl]]


def _pair_label(keys: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if not columns:
        return pd.Series(["Total/X"] * len(keys), index=keys.index)
    first = keys[columns[0]].astype(str)
    second = pd.Series("X", index=keys.index) if len(columns) == 1 else keys[columns[1]].astype(str)
    return first + "/" + second


def build_aggregation(keys: pd.DataFrame) -> Aggregation:
    """Build the hierarchy of ``keys``, with group ids in sorted label order."""
    missing = [
        column for columns in LEVELS.values() for column in columns if column not in keys.columns
    ]
    if missing:
        msg = f"keys are missing {sorted(set(missing))}"
        raise ValueError(msg)
    n_series = len(keys)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    labels: list[str] = []
    slices: dict[str, slice] = {}
    group_index: dict[str, np.ndarray] = {}
    offset = 0
    for level, columns in LEVELS.items():
        pairs = _pair_label(keys, columns)
        ordered = sorted(pairs.unique())
        index = {label: i for i, label in enumerate(ordered)}
        groups = pairs.map(index).to_numpy(dtype=np.int64)
        group_index[level] = groups
        rows.append(np.arange(n_series))
        cols.append(offset + groups)
        labels.extend(f"{level}/{label}" for label in ordered)
        slices[level] = slice(offset, offset + len(ordered))
        offset += len(ordered)
    matrix = sp.csr_matrix(
        (
            np.ones(n_series * len(LEVELS), dtype=np.float64),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_series, offset),
    )
    return Aggregation(matrix, labels, slices, group_index)


def aggregate(values, matrix) -> np.ndarray:
    """Sum the last axis of ``values`` through a ``(n_series, n_aggregates)`` matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != matrix.shape[0]:
        msg = f"last axis is {values.shape[-1]}, aggregation expects {matrix.shape[0]} series"
        raise ValueError(msg)
    flat = values.reshape(-1, matrix.shape[0])
    summed = matrix.T.dot(flat.T).T
    return summed.reshape(*values.shape[:-1], matrix.shape[1])


def m5_scales(y) -> np.ndarray:
    """Mean absolute lag-1 difference over each column's active period.

    The active period starts at the first nonzero value. The jump from zero on
    that day is excluded, the numerator is clamped at one, and the denominator
    is the number of active days minus one. An all-zero column uses the clamp.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 2:
        msg = f"scales expect a 2-d array, got ndim {values.ndim}"
        raise ValueError(msg)
    duration, n_columns = values.shape
    if duration < 2:
        msg = f"scales need at least 2 days, got {duration}"
        raise ValueError(msg)
    active = np.maximum((np.cumsum(values, axis=0) != 0).sum(0), 2)
    start_value = values[duration - active, np.arange(n_columns)]
    lag1_norm = np.abs(np.diff(values, axis=0, prepend=0.0)).sum(0) - np.abs(start_value)
    return np.maximum(lag1_norm, 1.0) / (active - 1)


def m5_weights(
    sales,
    price,
    aggregation: Aggregation,
    t1: int,
    *,
    window: int = SHARE_WINDOW,
) -> np.ndarray:
    """Dollar-sales share of every aggregate over the ``window`` days before ``t1``."""
    if t1 < window:
        msg = f"weights need {window} days before t1, got t1={t1}"
        raise ValueError(msg)
    sales = np.asarray(sales, dtype=np.float64)
    price = np.nan_to_num(np.asarray(price, dtype=np.float64), nan=0.0)
    dollars = (sales[t1 - window : t1] * price[t1 - window : t1]).sum(0)
    aggregated = aggregate(dollars, aggregation.matrix)
    weights = np.empty(aggregated.shape, dtype=np.float64)
    for sl in aggregation.slices.values():
        total = aggregated[sl].sum()
        width = sl.stop - sl.start
        weights[sl] = aggregated[sl] / total if total > 0.0 else np.full(width, 1.0 / width)
    return weights


def official_label(weights: pd.DataFrame) -> pd.Series:
    """Join key used by ``weights_evaluation.csv``."""
    return (
        weights["Level_id"].astype(str)
        + "/"
        + weights["Agg_Level_1"].astype(str)
        + "/"
        + weights["Agg_Level_2"].astype(str)
    )


def weight_gap(official: pd.DataFrame, labels: Sequence[str], ours: np.ndarray) -> float:
    """Largest absolute gap between official weights and ``ours``, matched on label."""
    table = official.assign(label=official_label(official), weight=official["weight"].astype(float))
    ours_frame = pd.DataFrame(
        {"label": list(labels), "weight_ours": np.asarray(ours, dtype=np.float64)}
    )
    joined = table.merge(ours_frame, on="label", how="inner")
    if len(joined) != len(labels):
        msg = f"{len(labels) - len(joined)} aggregates did not match official weights"
        raise ValueError(msg)
    return float((joined["weight"] - joined["weight_ours"]).abs().max())


def last_28_day_shares(train_sales, group_index, *, window: int = SHARE_WINDOW) -> np.ndarray:
    """Share of each series in its group's sales over the last ``window`` training days.

    A group whose total is zero is split uniformly across its members.
    """
    sales = np.asarray(train_sales, dtype=np.float64)
    groups = np.asarray(group_index, dtype=np.int64)
    if sales.ndim != 2:
        msg = "train_sales must be (days, series)"
        raise ValueError(msg)
    if sales.shape[0] < window:
        msg = f"shares need {window} training days, got {sales.shape[0]}"
        raise ValueError(msg)
    if groups.shape != (sales.shape[1],):
        msg = f"group_index length {groups.shape} does not match {sales.shape[1]} series"
        raise ValueError(msg)
    if groups.size and groups.min() < 0:
        msg = "group_index contains a negative id"
        raise ValueError(msg)
    totals = sales[-window:].sum(0)
    n_groups = int(groups.max()) + 1 if groups.size else 0
    group_totals = np.bincount(groups, weights=totals, minlength=n_groups)
    group_sizes = np.bincount(groups, minlength=n_groups)
    positive = group_totals[groups] > 0.0
    shares = np.empty(groups.shape, dtype=np.float64)
    shares[positive] = totals[positive] / group_totals[groups][positive]
    shares[~positive] = 1.0 / group_sizes[groups][~positive]
    return shares


def disaggregate(
    seed: int,
    group_draws,
    shares,
    group_index,
    *,
    chunk: int = 50,
    rate_cap: float = RATE_CAP,
) -> np.ndarray:
    """Poisson bottom-level draws with rate ``group draw * share``.

    Non-finite rates are clipped to ``[0, rate_cap]``. Draws follow the C-order
    ravel of the rate, so ``chunk`` only bounds the temporary and does not
    change the result. The return is float32 counts shaped like the rate
    broadcast to the bottom series.
    """
    draws = np.asarray(group_draws)
    share = np.asarray(shares, dtype=np.float64)
    groups = np.asarray(group_index, dtype=np.int64)
    if groups.shape != share.shape:
        msg = f"group_index shape {groups.shape} does not match shares {share.shape}"
        raise ValueError(msg)
    if groups.size and (int(groups.min()) < 0 or int(groups.max()) >= draws.shape[-1]):
        msg = "group_index is outside the last axis of group_draws"
        raise ValueError(msg)
    rate = np.asarray(draws[..., groups], dtype=np.float64) * share
    rate = np.clip(np.nan_to_num(rate, nan=0.0, posinf=rate_cap, neginf=0.0), 0.0, rate_cap)
    flat = np.ascontiguousarray(rate).ravel()
    rng = np.random.default_rng(seed)
    out = np.empty(flat.shape, dtype=np.float32)
    trailing = int(np.prod(rate.shape[1:], dtype=int)) if rate.ndim > 1 else flat.size or 1
    block = flat.size if chunk is None else max(int(chunk), 1) * trailing
    cursor = 0
    while cursor < flat.size:
        stop = min(cursor + block, flat.size)
        out[cursor:stop] = rng.poisson(flat[cursor:stop])
        cursor = stop
    return out.reshape(rate.shape)


def lagged_log_moving_average(
    y, window: int, lag: int, *, floor: float = GAMMA_FLOOR
) -> np.ndarray:
    """Log mean of ``y`` over ``window`` days ending ``lag`` days before each day.

    Days without a full window are ``log(floor)``. The cumulative sum is float64.
    """
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 2:
        msg = "moving averages expect sales shaped (days, series)"
        raise ValueError(msg)
    if window < 1 or lag < 0:
        msg = f"window must be >= 1 and lag >= 0, got window={window}, lag={lag}"
        raise ValueError(msg)
    padded = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    t = np.arange(values.shape[0])
    stop = np.clip(t - lag + 1, 0, None)
    start = np.clip(stop - window, 0, None)
    mean = (padded[stop] - padded[start]) / window
    mean[t - lag + 1 - window < 0] = 0.0
    return np.log(np.maximum(mean, floor))


def bounded_exp(x, bound: float = BEXP_BOUND) -> np.ndarray:
    """Exponential capped at ``bound`` so early steps cannot overflow."""
    z = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-(z - np.log(bound))))) * bound


def _bounded_exp(x, bound: float = BEXP_BOUND):
    return pt.sigmoid(x - float(np.log(bound))) * bound


def calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Weekday, years-since-start, Christmas, day-of-month dummies, and SNAP flags.

    Weekday is Monday = 0, taken from the calendar date. ``years`` is the row
    number divided by 365, matching the starter kit rather than a 365.25 year.
    """
    if "date" not in calendar.columns:
        msg = "calendar is missing date"
        raise ValueError(msg)
    dates = pd.to_datetime(calendar["date"])
    out = pd.DataFrame(
        {
            "dow": dates.dt.dayofweek.to_numpy(dtype=np.int64),
            "years": np.arange(len(dates), dtype=np.float64) / 365.0,
            "christmas": ((dates.dt.month == 12) & (dates.dt.day == 25)).to_numpy(dtype=np.float64),
        }
    )
    for day in range(1, 32):
        out[f"dom_{day}"] = (dates.dt.day == day).to_numpy(dtype=np.float64)
    missing = [column for column in _SNAP.values() if column not in calendar.columns]
    if missing:
        msg = f"calendar is missing {missing}"
        raise ValueError(msg)
    for column in _SNAP.values():
        out[column] = calendar[column].to_numpy(dtype=np.float64)
    return out


def _time_index(calendar: pd.DataFrame, n_days: int) -> np.ndarray:
    if len(calendar) != n_days:
        msg = f"calendar has {len(calendar)} rows, expected {n_days}"
        raise ValueError(msg)
    if "date" not in calendar.columns:
        return np.arange(n_days)
    dates = pd.to_datetime(calendar["date"])
    if dates.duplicated().any():
        msg = "calendar dates are not unique"
        raise ValueError(msg)
    return dates.to_numpy()


def top_covariates(calendar: pd.DataFrame) -> xr.DataArray:
    """Covariates for the top-down model: years, weekday, and 31 day-of-month dummies."""
    features = calendar_features(calendar)
    columns = ["years", "dow", *[f"dom_{day}" for day in range(1, 32)]]
    return xr.DataArray(
        features[columns].to_numpy(dtype=np.float64),
        dims=(TIME_DIM, "feature"),
        coords={TIME_DIM: _time_index(calendar, len(features)), "feature": columns},
        name="covariates",
    )


def middle_covariates(calendar: pd.DataFrame) -> xr.DataArray:
    """Covariates for the middle-out model: years, weekday, and 52 yearly Fourier pairs."""
    features = calendar_features(calendar)
    fourier = fourier_features(len(features), period=365.25, num_terms=52)
    names = ["years", "dow", *fourier.coords["fourier"].values.tolist()]
    values = np.column_stack(
        [features["years"].to_numpy(), features["dow"].to_numpy(), fourier.values]
    )
    return xr.DataArray(
        values,
        dims=(TIME_DIM, "feature"),
        coords={TIME_DIM: _time_index(calendar, len(features)), "feature": names},
        name="covariates",
    )


def bottom_covariates(sales, price, calendar: pd.DataFrame, keys: pd.DataFrame) -> xr.DataArray:
    """Six named channels: weekday, SNAP, listed-and-open, and three lagged log moving averages.

    Moving averages are computed on ``sales`` before any later slice, so a
    training window that starts at day ``T0`` still sees the earlier history.
    """
    sales = np.asarray(sales, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64)
    if sales.shape != price.shape:
        msg = f"sales shape {sales.shape} does not match price {price.shape}"
        raise ValueError(msg)
    if len(keys) != sales.shape[1]:
        msg = f"keys has {len(keys)} rows for {sales.shape[1]} series"
        raise ValueError(msg)
    features = calendar_features(calendar)
    if len(features) != sales.shape[0]:
        msg = f"calendar has {len(features)} rows for {sales.shape[0]} sales days"
        raise ValueError(msg)
    christmas = features["christmas"].to_numpy()
    saled = (~np.isnan(price)).astype(np.float64) * (1.0 - christmas[:, None])
    try:
        state_index = np.array([list(_SNAP).index(state) for state in keys["state_id"].astype(str)])
    except ValueError as exc:
        msg = f"unknown state_id in {sorted(set(keys['state_id'].astype(str)) - set(_SNAP))}"
        raise ValueError(msg) from exc
    snap_table = features[list(_SNAP.values())].to_numpy(dtype=np.float64)
    channels = {
        "dow": np.broadcast_to(features["dow"].to_numpy()[:, None], sales.shape).copy(),
        "snap": snap_table[:, state_index],
        "saled": saled,
    }
    for window, name in zip(MA_WINDOWS, LAGS, strict=True):
        channels[name] = lagged_log_moving_average(sales, window, MA_LAG)
    values = np.stack([channels[name] for name in BOTTOM_CHANNELS], axis=-1)
    return xr.DataArray(
        values,
        dims=(TIME_DIM, "series", "channel"),
        coords={
            TIME_DIM: _time_index(calendar, sales.shape[0]),
            "series": keys["id"].astype(str).to_numpy(),
            "channel": list(BOTTOM_CHANNELS),
        },
        name="covariates",
    )


def select_series(keys: pd.DataFrame, *, stores: Sequence[str], items_per_dept: int) -> np.ndarray:
    """Positional indices of the first ``items_per_dept`` items of each department.

    Selection keeps sales-file order. Item identity is the first occurrence of
    ``item_id``; the selected items are then crossed with ``stores``.
    """
    if items_per_dept < 1:
        msg = f"items_per_dept must be >= 1, got {items_per_dept}"
        raise ValueError(msg)
    frame = keys.reset_index(drop=True)
    unknown = [store for store in stores if store not in set(frame["store_id"].astype(str))]
    if unknown:
        msg = f"unknown store_id {unknown}"
        raise ValueError(msg)
    chosen: set[str] = set()
    for _, group in frame.drop_duplicates("item_id").groupby("dept_id", sort=False):
        chosen.update(group["item_id"].astype(str).head(items_per_dept))
    mask = frame["store_id"].astype(str).isin(list(stores)) & frame["item_id"].astype(str).isin(
        chosen
    )
    indices = np.flatnonzero(mask.to_numpy())
    if indices.size == 0:
        msg = "selection is empty"
        raise ValueError(msg)
    return indices


def require_history(n_days: int, *, minimum: int = T0 + HORIZON) -> None:
    """Reject a day count that cannot host day ``T0`` and a 28-day horizon."""
    if n_days < minimum:
        msg = (
            f"need at least {minimum} days so day {T0} and a {HORIZON}-day horizon "
            f"exist, got {n_days}"
        )
        raise ValueError(msg)


def evaluation_origins(
    duration: int,
    *,
    test_window: int = HORIZON,
    stride: int = 35,
    n_windows: int = 3,
) -> list[int]:
    """Last ``n_windows`` expanding origins, ``stride`` apart, each holding out ``test_window``.

    ``evaluation_origins(1941)`` is ``[1843, 1878, 1913]``, the three windows of
    the upstream M5 backtest.
    """
    if test_window < 1 or stride < 1 or n_windows < 1:
        msg = "test_window, stride, and n_windows must be positive"
        raise ValueError(msg)
    last = duration - test_window
    origins = [last - stride * k for k in range(n_windows - 1, -1, -1)]
    if origins[0] < 1:
        msg = (
            f"duration {duration} cannot host {n_windows} windows of {test_window} "
            f"strided by {stride}"
        )
        raise ValueError(msg)
    return origins


def log_total(sales, *, floor: float = GAMMA_FLOOR) -> np.ndarray:
    """Log of the panel total, with non-positive totals floored before the log."""
    total = np.asarray(sales, dtype=np.float64).sum(-1)
    return np.log(np.maximum(total, floor))


def level_panel(sales, aggregation: Aggregation, level: str) -> np.ndarray:
    """Aggregate ``sales`` and return one level's columns, in sorted label order."""
    sl = aggregation.slices[level]
    return aggregate(sales, aggregation.matrix)[..., sl]


def _feature(covariates: xr.DataArray, name: str) -> np.ndarray:
    return np.asarray(covariates.sel(feature=name).values)


def _dom(covariates: xr.DataArray) -> tuple[list[str], np.ndarray]:
    names = [
        str(name) for name in covariates.coords["feature"].values if str(name).startswith("dom_")
    ]
    if not names:
        msg = "top-down covariates have no dom_* features"
        raise ValueError(msg)
    return names, np.asarray(covariates.sel(feature=names).values, dtype=np.float64)


def _fourier(covariates: xr.DataArray) -> tuple[list[str], np.ndarray]:
    names = [
        str(name)
        for name in covariates.coords["feature"].values
        if str(name).startswith(("sin_", "cos_"))
    ]
    if not names:
        msg = "middle-out covariates have no Fourier features"
        raise ValueError(msg)
    return names, np.asarray(covariates.sel(feature=names).values, dtype=np.float64)


def _channel(covariates: xr.DataArray, name: str) -> np.ndarray:
    return np.asarray(covariates.sel(channel=name).values)


class TopDownModel(ForecastingModel):
    """Kit model 1: trend, weekday, and day-of-month effects on the log total."""

    def model(self, covariates, data=None) -> None:
        years = _feature(covariates, "years")
        dow = _feature(covariates, "dow").astype(np.int32)
        dom_names, dom = _dom(covariates)
        model = pm.modelcontext(None)
        model.add_coord("day_of_week", list(WEEKDAYS))
        model.add_coord("dom", dom_names)
        bias = pm.Normal("bias", 0.0, 10.0)
        trend = pm.LogNormal("trend", -2.0, 1.0)
        weight = pm.Normal("weight", 0.0, 1.0, dims="dom")
        seasonal = pm.Normal("seasonal", 0.0, 5.0, dims="day_of_week")
        prediction = bias + trend * years + seasonal[dow] + pt.dot(dom, weight)
        dof = pm.Uniform("dof", 1.0, 10.0)
        noise_scale = pm.LogNormal("noise_scale", -2.0, 1.0)
        self.predict(pm.StudentT.dist(nu=dof, mu=0.0, sigma=noise_scale), prediction[:, None])


class BottomUpModel(ForecastingModel):
    """Kit model 2: department-level mean and scale for every store, Gamma observations."""

    def __init__(self, store_index, dept_index, store_ids: Sequence[str], dept_ids: Sequence[str]):
        self.store_index = np.asarray(store_index, dtype=np.int32)
        self.dept_index = np.asarray(dept_index, dtype=np.int32)
        self.store_ids = tuple(store_ids)
        self.dept_ids = tuple(dept_ids)
        if self.store_index.shape != self.dept_index.shape or self.store_index.ndim != 1:
            msg = "store_index and dept_index must be 1-d and the same length"
            raise ValueError(msg)
        self._check_index(self.store_index, len(self.store_ids), "store")
        self._check_index(self.dept_index, len(self.dept_ids), "dept")

    @staticmethod
    def _check_index(index: np.ndarray, n_ids: int, name: str) -> None:
        if index.size and (int(index.min()) < 0 or int(index.max()) >= n_ids):
            msg = f"{name}_index is outside 0..{n_ids - 1}"
            raise ValueError(msg)

    def model(self, covariates, data=None) -> None:
        model = pm.modelcontext(None)
        model.add_coord("store", list(self.store_ids))
        model.add_coord("dept", list(self.dept_ids))
        model.add_coord("head", list(HEADS))
        model.add_coord("lag", list(LAGS))
        model.add_coord("day_of_week", list(WEEKDAYS))
        ma_weight = pm.Normal("ma_weight", 0.0, 1.0, dims=("store", "head", "lag", "dept"))
        snap_weight = pm.Normal("snap_weight", 0.0, 1.0, dims=("store", "head", "dept"))
        seasonal = pm.Normal("seasonal", 0.0, 1.0, dims=("store", "day_of_week", "head", "dept"))
        dow = _channel(covariates, "dow")[:, 0].astype(np.int32)
        snap = _channel(covariates, "snap")
        saled = _channel(covariates, "saled")
        log_ma = np.stack([_channel(covariates, name) for name in LAGS], axis=-1)
        store, dept = self.store_index, self.dept_index
        moving = (ma_weight[store, :, :, dept][None] * log_ma[:, :, None, :]).sum(-1)
        snap_effect = snap_weight[store, :, dept][None] * snap[:, :, None]
        seasonal_effect = seasonal[store, :, :, dept][:, dow, :].transpose(1, 0, 2)
        combined = moving + snap_effect + seasonal_effect
        mean = _bounded_exp(combined[..., 0]) * saled + GAMMA_FLOOR
        scale = _bounded_exp(combined[..., 1]) * saled + GAMMA_FLOOR
        t_obs = self.horizon.t_obs

        def gamma(name, latent, dims, observed):
            scale_seg = scale[t_obs:] if name == FORECAST_VAR else scale[:t_obs]
            return pm.Gamma(
                name,
                alpha=latent / scale_seg,
                beta=1.0 / scale_seg,
                dims=dims,
                observed=observed,
            )

        self.predict(gamma, mean)


class MiddleOutModel(ForecastingModel):
    """Kit model 3: one scaled regression per series, shared Student-T degrees of freedom."""

    def model(self, covariates, data=None) -> None:
        years = _feature(covariates, "years")[:, None]
        dow = _feature(covariates, "dow").astype(np.int32)
        fourier_names, feature = _fourier(covariates)
        model = pm.modelcontext(None)
        model.add_coord("day_of_week", list(WEEKDAYS))
        model.add_coord("fourier", fourier_names)
        bias = pm.Normal("bias", 0.0, 10.0, dims="series")
        trend = pm.LogNormal("trend", -1.0, 1.0, dims="series")
        weight = pm.Normal("weight", 0.0, 1.0, dims=("series", "fourier"))
        seasonal = pm.Normal("seasonal", 0.0, 1.0, dims=("day_of_week", "series"))
        noise_scale = pm.LogNormal("noise_scale", -1.0, 1.0, dims="series")
        dof = pm.Uniform("dof", 1.0, 10.0)
        prediction = bias + trend * years + seasonal[dow] + pt.dot(feature, weight.T)
        self.predict(pm.StudentT.dist(nu=dof, mu=0.0, sigma=noise_scale), prediction)


def forecast_draws(result) -> xr.DataArray:
    """Stack chain and draw, with the sample axis first and the horizon second."""
    samples = prediction_samples(result)["forecast"]
    if "chain" in samples.dims and "draw" in samples.dims:
        samples = samples.stack(sample=("chain", "draw"))
    elif "draw" in samples.dims:
        samples = samples.rename(draw="sample")
    time_dim = FUTURE_DIM if FUTURE_DIM in samples.dims else TIME_DIM
    others = [dim for dim in samples.dims if dim not in ("sample", time_dim)]
    return samples.transpose("sample", time_dim, *others)


def reconcile_top_down(forecast, train_sales, *, seed: int) -> np.ndarray:
    """Exponentiate a log-total forecast and split it by the last 28-day shares."""
    units = np.exp(np.asarray(forecast_draws(forecast).values, dtype=np.float64))
    if units.ndim == 2:
        units = units[..., None]
    groups = np.zeros(np.asarray(train_sales).shape[1], dtype=np.int64)
    return disaggregate(seed, units, last_28_day_shares(train_sales, groups), groups)


def reconcile_middle_out(forecast, train_sales, group_index, scale, *, seed: int) -> np.ndarray:
    """Clip a scaled level-9 forecast at zero, restore units, and split within each group."""
    units = np.clip(np.asarray(forecast_draws(forecast).values, dtype=np.float64), 0.0, None)
    units = units * np.asarray(scale, dtype=np.float64)
    return disaggregate(seed, units, last_28_day_shares(train_sales, group_index), group_index)


def _levels(pred, truth, aggregation: Aggregation) -> tuple[np.ndarray, np.ndarray]:
    return aggregate(pred, aggregation.matrix), aggregate(truth, aggregation.matrix)


def ws_crps_levels(
    pred_levels,
    truth_levels,
    weights,
    scales,
    slices: Mapping[str, slice],
) -> dict[str, float]:
    """Weighted scaled CRPS of each level. ``pred_levels`` has the sample axis first."""
    pred = np.asarray(pred_levels, dtype=np.float64)
    truth = np.asarray(truth_levels, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    scores = {}
    for level, sl in slices.items():
        crps = crps_empirical(pred[..., sl], truth[..., sl]).mean(axis=0)
        scores[level] = float(np.sum(weight[sl] * crps / scale[sl]))
    return scores


def ws_pinball(pred_levels, truth_levels, weights, scales, slices: Mapping[str, slice]) -> float:
    """M5 WSPL: pinball loss at the nine competition quantiles, mean over levels."""
    pred = np.asarray(pred_levels, dtype=np.float64)
    truth = np.asarray(truth_levels, dtype=np.float64)
    quantiles = np.quantile(pred, M5_QUANTILES, axis=0)
    error = quantiles - truth[None]
    u = M5_QUANTILES[:, None, None]
    per_series = (np.where(error <= 0.0, -u, 1.0 - u) * error).mean(axis=(0, 1))
    weight = np.asarray(weights, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    level_scores = [np.sum(weight[sl] * per_series[sl] / scale[sl]) for sl in slices.values()]
    return float(np.mean(level_scores))


def mean_level_score(scores: Mapping[str, float]) -> float:
    """Mean of the 12 level scores, in ``LEVELS`` order."""
    return float(np.mean([scores[level] for level in LEVELS]))


def score_bottom(
    bottom_draws,
    truth,
    aggregation: Aggregation,
    weights,
    scales,
) -> tuple[dict[str, float], float]:
    """Aggregate bottom draws and return per-level WS-CRPS plus WSPL."""
    pred_levels, truth_levels = _levels(bottom_draws, truth, aggregation)
    scores = ws_crps_levels(pred_levels, truth_levels, weights, scales, aggregation.slices)
    return scores, ws_pinball(pred_levels, truth_levels, weights, scales, aggregation.slices)

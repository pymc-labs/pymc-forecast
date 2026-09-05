"""Input normalization: everything becomes an ``xarray.DataArray`` with a leading
``"time"`` dim.

The package is dims/coords-first: models, forecasts, and metrics all speak
named dimensions. Users may still pass pandas or numpy objects at the API
boundary; this module converts them once, attaching real time coordinates
(a ``DatetimeIndex``, periods, or a fallback integer range).
"""

from numbers import Integral
from typing import Literal

import numpy as np
import xarray as xr

from pymc_forecast.exceptions import AlignmentError

__all__ = [
    "CHAIN_DIM",
    "DRAW_DIM",
    "FUTURE_DIM",
    "SAMPLE_DIMS",
    "TIME_DIM",
    "as_dataarray",
    "concat_covariates",
    "concat_time_index",
    "extend_time_index",
    "null_covariates",
    "validate_alignment",
]

CHAIN_DIM = "chain"
"""First posterior-sample dim on every prediction output."""

DRAW_DIM = "draw"
"""Second posterior-sample dim on every prediction output."""

SAMPLE_DIMS = (CHAIN_DIM, DRAW_DIM)
"""The ordered sample dims guaranteed to lead every prediction output."""

TIME_DIM = "time"
"""Dim name of the observed (in-sample) time dimension."""

FUTURE_DIM = "time_future"
"""Dim name of the forecast-horizon time dimension."""

_DEFAULT_SECOND_DIM: dict[str, str] = {"data": "series", "covariates": "covariate"}

Role = Literal["data", "covariates"]


def as_dataarray(obj, *, role: Role = "data") -> xr.DataArray:
    """Normalize ``obj`` to a ``DataArray`` with ``"time"`` as the leading dim.

    Accepted inputs:

    - ``xarray.DataArray`` with a ``"time"`` dim (transposed time-first);
    - ``pandas.Series`` (index becomes the time coord) or ``pandas.DataFrame``
      (index → time coord, columns → ``"series"``/``"covariate"`` coord);
    - 1-d/2-d ``numpy`` arrays (integer-range time coord is attached).

    Parameters
    ----------
    obj
        The object to normalize.
    role
        ``"data"`` or ``"covariates"``; sets the default name of the second dim
        for 2-d pandas/numpy inputs (``"series"`` / ``"covariate"``).
    """
    second = _DEFAULT_SECOND_DIM[role]
    if isinstance(obj, xr.DataArray):
        if TIME_DIM not in obj.dims:
            msg = f"DataArray {role} must have a '{TIME_DIM}' dim, got dims {obj.dims}"
            raise AlignmentError(msg)
        da = obj.transpose(TIME_DIM, ...)
    else:
        # Lazy import: pandas is an optional path but a hard dependency of both
        # pymc and xarray in practice, so this never fails in a working env.
        import pandas as pd

        if isinstance(obj, pd.Series):
            da = xr.DataArray(
                obj.to_numpy(), dims=(TIME_DIM,), coords={TIME_DIM: obj.index}, name=obj.name
            )
        elif isinstance(obj, pd.DataFrame):
            da = xr.DataArray(
                obj.to_numpy(),
                dims=(TIME_DIM, second),
                coords={TIME_DIM: obj.index, second: obj.columns},
            )
        else:
            arr = np.asarray(obj)
            if arr.ndim == 1:
                da = xr.DataArray(arr, dims=(TIME_DIM,))
            elif arr.ndim == 2:
                da = xr.DataArray(arr, dims=(TIME_DIM, second))
            else:
                msg = (
                    f"bare numpy {role} must be 1-d or 2-d (got ndim={arr.ndim}); "
                    "pass an xarray.DataArray with named dims for higher-dimensional data"
                )
                raise AlignmentError(msg)
        if isinstance(obj, pd.Series | pd.DataFrame) and isinstance(obj.index, pd.DatetimeIndex):
            # Older xarray converts pandas' microsecond datetimes to ns and
            # drops freq in the process. Restore the known frequency on the
            # converted index, without forcing a datetime unit on newer xarray.
            time_index = da.get_index(TIME_DIM)
            if obj.index.freq is not None and time_index.freq is None:
                da = da.assign_coords({TIME_DIM: pd.DatetimeIndex(time_index, freq=obj.index.freq)})
    if TIME_DIM not in da.coords:
        da = da.assign_coords({TIME_DIM: np.arange(da.sizes[TIME_DIM])})
    return da


def null_covariates(index) -> xr.DataArray:
    """Zero-width covariates carrying only the time coord.

    Covariates are the horizon carrier of the whole API (their time coord spans
    train + forecast). Models without real covariates use this helper:
    ``null_covariates(full_time_index)``.

    Parameters
    ----------
    index
        Time coordinate values spanning the full horizon (observed + future),
        e.g. a ``pandas.DatetimeIndex`` or an integer range.
    """
    index = np.asarray(index)
    return xr.DataArray(
        np.zeros((len(index), 0)),
        dims=(TIME_DIM, _DEFAULT_SECOND_DIM["covariates"]),
        coords={TIME_DIM: index},
    )


def extend_time_index(index, horizon: int):
    """Extend a time index by ``horizon`` steps, inferring the spacing.

    Used to build the forecast horizon for covariate-free models: a
    ``DatetimeIndex`` is extended at its stored or inferred frequency, a
    ``PeriodIndex`` at its stored frequency, and a numeric index by its
    constant step. Returns the full ``observed + horizon`` index.

    Raises
    ------
    AlignmentError
        If a datetime frequency cannot be inferred, or the numeric spacing is
        not constant.
    """
    import pandas as pd

    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 0:
        msg = f"horizon must be a non-negative integer, got {horizon}"
        raise AlignmentError(msg)
    idx = pd.Index(index)
    if horizon == 0:
        return idx
    if len(idx) == 0 or not (idx.is_monotonic_increasing and idx.is_unique):
        msg = "training time index must be non-empty and strictly increasing"
        raise AlignmentError(msg)
    if isinstance(idx, pd.PeriodIndex):
        return idx.append(pd.period_range(idx[-1] + 1, periods=horizon, freq=idx.freq))
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or (pd.infer_freq(idx) if len(idx) >= 3 else None)
        if freq is None:
            msg = (
                "cannot infer a frequency from the datetime index to build the "
                "forecast horizon; pass future_index= or explicit covariates instead of horizon="
            )
            raise AlignmentError(msg)
        future = pd.date_range(idx[-1], periods=horizon + 1, freq=freq)[1:]
        return idx.append(future)
    values = np.asarray(idx)
    if not np.issubdtype(values.dtype, np.number):
        msg = "horizon= requires a numeric, datetime, or period time index; pass future_index="
        raise AlignmentError(msg)
    if len(values) < 2:
        step = 1
    else:
        steps = np.diff(values)
        if not np.allclose(steps, steps[0]):
            msg = (
                "numeric time index is not evenly spaced; cannot extend by "
                "horizon=, pass explicit covariates instead"
            )
            raise AlignmentError(msg)
        step = steps[0]
    future = values[-1] + step * np.arange(1, horizon + 1)
    return pd.Index(np.concatenate([values, future]))


def concat_time_index(index, future_index):
    """Concatenate a training time index with a predict-time future index.

    The future index supplies the forecast horizon at predict time (its length
    need not be known when the model is fit). Its values must be strictly
    increasing and lie strictly after the last training value; gaps are
    allowed — forecast steps are labeled with the supplied coordinates.
    Returns the full ``observed + future`` index.

    Parameters
    ----------
    index
        Time coordinate values of the training window.
    future_index
        Time coordinate values of the forecast horizon.

    Raises
    ------
    AlignmentError
        If the future index is empty, not strictly increasing, does not sort
        after the training index, or cannot be compared to it.
    """
    import pandas as pd

    idx = pd.Index(np.asarray(index))
    fut = pd.Index(np.asarray(future_index))
    if len(fut) == 0:
        msg = "future time index is empty; supply at least one forecast step"
        raise AlignmentError(msg)
    if not (fut.is_monotonic_increasing and fut.is_unique):
        msg = "future time index must be strictly increasing"
        raise AlignmentError(msg)
    try:
        starts_after = len(idx) == 0 or fut[0] > idx[-1]
    except TypeError as err:
        msg = (
            "future time index is not comparable to the training time index "
            f"({fut.dtype} vs {idx.dtype})"
        )
        raise AlignmentError(msg) from err
    if not starts_after:
        msg = (
            "future time index must lie strictly after the training window: "
            f"got first future value {fut[0]!r} <= last training value {idx[-1]!r}"
        )
        raise AlignmentError(msg)
    return idx.append(fut)


def concat_covariates(covariates, future_covariates) -> xr.DataArray:
    """Append future covariate rows to training covariates along ``"time"``.

    ``future_covariates`` covers only the forecast horizon (both inputs are
    normalized via :func:`as_dataarray` first); its time index must lie
    strictly after the training window and its non-time structure — dims, and
    covariate names in order — must match the training covariates, since
    models consume covariate columns positionally. Every mismatch is rejected
    explicitly, so xarray never silently aligns, reorders, or fills feature
    columns. Returns the full-horizon covariates.

    Raises
    ------
    AlignmentError
        On a time index that does not extend the training window, or on any
        dim/coord mismatch.
    """
    covariates = as_dataarray(covariates, role="covariates")
    fut = as_dataarray(future_covariates, role="covariates")
    _validate_covariate_structure(covariates, fut)
    concat_time_index(covariates[TIME_DIM].values, fut[TIME_DIM].values)
    return xr.concat([covariates, fut], dim=TIME_DIM, join="exact")


def _validate_covariate_structure(covariates: xr.DataArray, fut: xr.DataArray) -> None:
    """Require identical non-time structure for every predict-time input path."""
    if covariates.dims != fut.dims:
        msg = (
            "prediction covariates must have the same dims as the training "
            f"covariates: got {fut.dims}, expected {covariates.dims}"
        )
        raise AlignmentError(msg)
    _validate_shared_coords(covariates, fut, context="prediction covariates")


def _validate_shared_coords(left: xr.DataArray, right: xr.DataArray, *, context: str) -> None:
    """Reject positional ambiguity on shared non-time dimensions."""
    for dim in left.dims:
        if dim == TIME_DIM or dim not in right.dims:
            continue
        if left.sizes[dim] != right.sizes[dim]:
            msg = (
                f"{context} size mismatch along '{dim}': "
                f"got {right.sizes[dim]}, expected {left.sizes[dim]}"
            )
            raise AlignmentError(msg)
        if (dim in left.coords) != (dim in right.coords):
            msg = f"{context} must carry matching '{dim}' coordinates; one side is unlabeled"
            raise AlignmentError(msg)
        if dim in left.coords and not left.get_index(dim).equals(right.get_index(dim)):
            msg = (
                f"{context} '{dim}' coords must match (same names, same order): "
                f"got {right[dim].values!r}, expected {left[dim].values!r}"
            )
            raise AlignmentError(msg)


def validate_alignment(data: xr.DataArray, covariates: xr.DataArray) -> None:
    """Require the covariate time coord to extend the data time coord.

    ``covariates`` must be at least as long as ``data`` along ``"time"``, and
    the first ``len(data.time)`` coordinate values must match exactly — the
    surplus is the forecast horizon and must be strictly increasing after
    training. Shared non-time dimensions must have identical sizes and
    coordinates, so panel covariates cannot silently change series order.
    """
    t_obs = data.sizes[TIME_DIM]
    if covariates.sizes[TIME_DIM] < t_obs:
        msg = (
            f"covariates must extend data along '{TIME_DIM}': got "
            f"{covariates.sizes[TIME_DIM]} covariate steps < {t_obs} data steps"
        )
        raise AlignmentError(msg)
    cov_prefix = covariates[TIME_DIM].values[:t_obs]
    if not np.array_equal(cov_prefix, data[TIME_DIM].values):
        msg = (
            f"the first {t_obs} covariate '{TIME_DIM}' coords must equal the data "
            f"'{TIME_DIM}' coords; the arrays are misaligned"
        )
        raise AlignmentError(msg)
    _validate_shared_coords(data, covariates, context="data and covariates")
    if covariates.sizes[TIME_DIM] > t_obs:
        concat_time_index(data[TIME_DIM].values, covariates[TIME_DIM].values[t_obs:])

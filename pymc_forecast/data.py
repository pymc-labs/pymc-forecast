"""Input normalization: everything becomes an ``xarray.DataArray`` with a leading
``"time"`` dim.

The package is dims/coords-first: models, forecasts, and metrics all speak
named dimensions. Users may still pass pandas or numpy objects at the API
boundary; this module converts them once, attaching real time coordinates
(a ``DatetimeIndex``, periods, or a fallback integer range).
"""

from typing import Literal

import numpy as np
import xarray as xr

from pymc_forecast.exceptions import AlignmentError

__all__ = [
    "FUTURE_DIM",
    "TIME_DIM",
    "as_dataarray",
    "extend_time_index",
    "null_covariates",
    "validate_alignment",
]

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
    ``DatetimeIndex`` is extended at its inferred frequency, a numeric index by
    its constant step. Returns the full ``observed + horizon`` index.

    Raises
    ------
    AlignmentError
        If a datetime frequency cannot be inferred, or the numeric spacing is
        not constant.
    """
    import pandas as pd

    if horizon < 0:
        msg = f"horizon must be non-negative, got {horizon}"
        raise AlignmentError(msg)
    idx = pd.Index(index)
    if horizon == 0:
        return idx
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is None:
            msg = (
                "cannot infer a frequency from the datetime index to build the "
                "forecast horizon; pass explicit covariates instead of horizon="
            )
            raise AlignmentError(msg)
        future = pd.date_range(idx[-1], periods=horizon + 1, freq=freq)[1:]
        return idx.append(future)
    values = np.asarray(idx)
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


def validate_alignment(data: xr.DataArray, covariates: xr.DataArray) -> None:
    """Require the covariate time coord to extend the data time coord.

    ``covariates`` must be at least as long as ``data`` along ``"time"``, and
    the first ``len(data.time)`` coordinate values must match exactly — the
    surplus is the forecast horizon.
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

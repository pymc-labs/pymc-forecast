"""Probabilistic forecast metrics, dim-aware.

Each ``eval_*`` metric takes forecast samples and ground truth and reduces them
to a single float; :func:`crps_empirical` returns elementwise scores and
:func:`make_mase` builds a metric from training data.

Inputs may be labeled (``xarray.DataArray``) or raw numpy:

- **DataArray predictions** carry their sample dimensions by name — ``chain`` /
  ``draw`` (as produced by the forecast drivers) or an already-stacked
  ``sample`` dim. Labeled truth is transposed to the prediction's dim order
  and reordered to matching coordinates. Different or duplicate labels are
  rejected. Unlabeled inputs are positional and must match the value shape
  exactly; implicit broadcasting is rejected.
- **numpy predictions** follow the classical convention: sample axis first.

Ported from numpyro_forecast (itself porting ``pyro.ops.stats.crps_empirical``)
with the JAX kernels replaced by NumPy.
"""

from collections.abc import Callable, Mapping

import numpy as np
import xarray as xr

from pymc_forecast.data import SAMPLE_DIMS

__all__ = [
    "DEFAULT_METRICS",
    "crps_empirical",
    "eval_coverage",
    "eval_crps",
    "eval_interval_score",
    "eval_mae",
    "eval_pinball",
    "eval_rmse",
    "evaluate_forecast",
    "make_mase",
]

_SAMPLE_DIMS = (*SAMPLE_DIMS, "sample")
"""Dim names recognized as sample dimensions of labeled predictions."""

Metric = Callable[..., float]
"""A metric: ``(pred, truth) -> float`` with the sample axis first (numpy) or
labeled sample dims (DataArray)."""


def _as_sample_first(pred, truth) -> tuple[np.ndarray, np.ndarray]:
    """Normalize (pred, truth) to numpy with a flattened sample axis first."""
    if isinstance(pred, xr.DataArray):
        sample_dims = [d for d in pred.dims if d in _SAMPLE_DIMS]
        if not sample_dims:
            msg = (
                "labeled predictions need a sample dimension (one of "
                f"{_SAMPLE_DIMS}); got dims {pred.dims}"
            )
            raise ValueError(msg)
        value_dims = [d for d in pred.dims if d not in _SAMPLE_DIMS]
        pred = pred.transpose(*sample_dims, *value_dims)
        if isinstance(truth, xr.DataArray):
            missing = [d for d in value_dims if d not in truth.dims]
            if missing:
                msg = f"truth is missing prediction dims {missing}"
                raise ValueError(msg)
            truth = truth.transpose(*value_dims)
            for dim in value_dims:
                if pred.sizes[dim] != truth.sizes[dim]:
                    msg = f"prediction and truth sizes must match along '{dim}'"
                    raise ValueError(msg)
                if dim in pred.coords and dim in truth.coords:
                    pred_index, truth_index = pred.get_index(dim), truth.get_index(dim)
                    if not (pred_index.is_unique and truth_index.is_unique):
                        msg = f"prediction and truth coordinates must be unique along '{dim}'"
                        raise ValueError(msg)
                    if pred_index.equals(truth_index):
                        continue
                    indexer = truth_index.get_indexer(pred_index)
                    if (indexer < 0).any():
                        msg = f"prediction and truth coordinates must match along '{dim}'"
                        raise ValueError(msg)
                    truth = truth.isel({dim: indexer})
        pred_np = pred.values.reshape(-1, *pred.shape[len(sample_dims) :])
    else:
        pred_np = np.asarray(pred)
    truth_np = np.asarray(truth.values if isinstance(truth, xr.DataArray) else truth)
    if pred_np.ndim < 1 or pred_np.shape[1:] != truth_np.shape:
        msg = "truth shape must equal prediction shape without the sample axis"
        raise ValueError(msg)
    if pred_np.shape[0] == 0:
        msg = "predictions must contain at least one sample"
        raise ValueError(msg)
    return pred_np, truth_np


def crps_empirical(pred, truth) -> np.ndarray:
    r"""Elementwise empirical Continuous Ranked Probability Score.

    .. math::

        \mathrm{CRPS}(F, y) = \mathbb{E}|X - y| - \tfrac{1}{2}\,\mathbb{E}|X - X'|

    estimated from the forecast samples with the sorted-sample
    :math:`O(n \log n)` identity.

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples: a DataArray with sample dims named ``chain``, ``draw``
        or ``sample``, or an array with the sample axis first. At least 2
        samples are required.
    truth : xarray.DataArray or array_like
        Ground-truth values with the prediction's shape without the sample
        axis. When ``pred`` is labeled, labeled truth is aligned by dim name
        and coordinates; otherwise it is used positionally.

    Returns
    -------
    numpy.ndarray
        Elementwise CRPS as an unlabeled float64 array, one value per data
        location (value dims in the prediction's dim order).

    Raises
    ------
    ValueError
        If there are fewer than 2 samples, a labeled ``pred`` has no sample
        dim, or ``pred`` and ``truth`` disagree on dims, sizes, coordinates or
        shape.
    """
    pred, truth = _as_sample_first(pred, truth)
    # Integer counts and low-precision forecasts must not overflow in pairwise
    # differences or the quadratic sample weights.
    pred = pred.astype(np.float64, copy=False)
    truth = truth.astype(np.float64, copy=False)
    num_samples = pred.shape[0]
    if num_samples < 2:
        msg = f"crps_empirical needs at least 2 samples, got {num_samples}"
        raise ValueError(msg)
    pred_sorted = np.sort(pred, axis=0)
    diff = pred_sorted[1:] - pred_sorted[:-1]
    lower = np.arange(1, num_samples, dtype=pred.dtype)
    upper = np.arange(num_samples - 1, 0, -1, dtype=pred.dtype)
    weight = (lower * upper).reshape((num_samples - 1,) + (1,) * (diff.ndim - 1))
    absolute_error = np.abs(pred - truth).mean(axis=0)
    return absolute_error - (diff * weight).sum(axis=0) / float(num_samples) ** 2


def eval_mae(pred, truth) -> float:
    """Mean absolute error of the forecast sample median.

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples: a DataArray with sample dims named ``chain``, ``draw``
        or ``sample``, or an array with the sample axis first.
    truth : xarray.DataArray or array_like
        Ground-truth values with the prediction's shape without the sample
        axis. When ``pred`` is labeled, labeled truth is aligned by dim name
        and coordinates; otherwise it is used positionally.

    Returns
    -------
    float
        The absolute error averaged over every value element.

    Raises
    ------
    ValueError
        If ``pred`` has no samples, a labeled ``pred`` has no sample dim, or
        ``pred`` and ``truth`` disagree on dims, sizes, coordinates or shape.
    """
    pred, truth = _as_sample_first(pred, truth)
    return float(np.abs(np.median(pred, axis=0) - truth).mean())


def eval_rmse(pred, truth) -> float:
    """Root mean squared error of the forecast sample mean.

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples: a DataArray with sample dims named ``chain``, ``draw``
        or ``sample``, or an array with the sample axis first.
    truth : xarray.DataArray or array_like
        Ground-truth values with the prediction's shape without the sample
        axis. When ``pred`` is labeled, labeled truth is aligned by dim name
        and coordinates; otherwise it is used positionally.

    Returns
    -------
    float
        The root of the squared error averaged over every value element.

    Raises
    ------
    ValueError
        If ``pred`` has no samples, a labeled ``pred`` has no sample dim, or
        ``pred`` and ``truth`` disagree on dims, sizes, coordinates or shape.
    """
    pred, truth = _as_sample_first(pred, truth)
    return float(np.sqrt(np.square(pred.mean(axis=0) - truth).mean()))


def eval_crps(pred, truth) -> float:
    """Mean empirical CRPS over all data elements (see :func:`crps_empirical`).

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples: a DataArray with sample dims named ``chain``, ``draw``
        or ``sample``, or an array with the sample axis first. At least 2
        samples are required.
    truth : xarray.DataArray or array_like
        Ground-truth values with the prediction's shape without the sample
        axis. When ``pred`` is labeled, labeled truth is aligned by dim name
        and coordinates; otherwise it is used positionally.

    Returns
    -------
    float
        The elementwise CRPS averaged over every value element.

    Raises
    ------
    ValueError
        If there are fewer than 2 samples, a labeled ``pred`` has no sample
        dim, or ``pred`` and ``truth`` disagree on dims, sizes, coordinates or
        shape.
    """
    return float(crps_empirical(pred, truth).mean())


def eval_coverage(pred, truth, *, alpha: float = 0.9) -> float:
    """Empirical coverage of the central ``alpha`` prediction interval.

    A well-calibrated forecast has coverage close to ``alpha``. Bind a
    non-default level with ``functools.partial(eval_coverage, alpha=0.8)``.

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples: a DataArray with sample dims named ``chain``, ``draw``
        or ``sample``, or an array with the sample axis first.
    truth : xarray.DataArray or array_like
        Ground-truth values with the prediction's shape without the sample
        axis. When ``pred`` is labeled, labeled truth is aligned by dim name
        and coordinates; otherwise it is used positionally.
    alpha : float, default 0.9
        Nominal interval probability, in the open interval (0, 1). The bounds
        are the ``(1 - alpha) / 2`` and ``1 - (1 - alpha) / 2`` sample
        quantiles and are inclusive.

    Returns
    -------
    float
        Fraction of value elements whose truth lies inside the interval.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1), ``pred`` has no samples, a labeled
        ``pred`` has no sample dim, or ``pred`` and ``truth`` disagree on dims,
        sizes, coordinates or shape.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    pred, truth = _as_sample_first(pred, truth)
    tail = (1.0 - alpha) / 2.0
    lo = np.quantile(pred, tail, axis=0)
    hi = np.quantile(pred, 1.0 - tail, axis=0)
    return float(((truth >= lo) & (truth <= hi)).mean())


def eval_pinball(pred, truth, *, quantile: float = 0.5) -> float:
    """Mean pinball (quantile) loss of the forecast ``quantile``.

    At ``quantile=0.5`` this is half the mean absolute error.

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples: a DataArray with sample dims named ``chain``, ``draw``
        or ``sample``, or an array with the sample axis first.
    truth : xarray.DataArray or array_like
        Ground-truth values with the prediction's shape without the sample
        axis. When ``pred`` is labeled, labeled truth is aligned by dim name
        and coordinates; otherwise it is used positionally.
    quantile : float, default 0.5
        Quantile level, in the open interval (0, 1); the point estimate is the
        corresponding sample quantile.

    Returns
    -------
    float
        The pinball loss averaged over every value element.

    Raises
    ------
    ValueError
        If ``quantile`` is not in (0, 1), ``pred`` has no samples, a labeled
        ``pred`` has no sample dim, or ``pred`` and ``truth`` disagree on dims,
        sizes, coordinates or shape.
    """
    if not 0.0 < quantile < 1.0:
        msg = f"quantile must be in (0, 1), got {quantile}"
        raise ValueError(msg)
    pred, truth = _as_sample_first(pred, truth)
    estimate = np.quantile(pred, quantile, axis=0)
    diff = truth - estimate
    return float(np.maximum(quantile * diff, (quantile - 1.0) * diff).mean())


def eval_interval_score(pred, truth, *, alpha: float = 0.9) -> float:
    """Mean Winkler interval score of the central ``alpha`` interval.

    Rewards narrow intervals, penalizes truth falling outside; lower is better.

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples: a DataArray with sample dims named ``chain``, ``draw``
        or ``sample``, or an array with the sample axis first.
    truth : xarray.DataArray or array_like
        Ground-truth values with the prediction's shape without the sample
        axis. When ``pred`` is labeled, labeled truth is aligned by dim name
        and coordinates; otherwise it is used positionally.
    alpha : float, default 0.9
        Nominal interval probability, in the open interval (0, 1). The bounds
        are the ``(1 - alpha) / 2`` and ``1 - (1 - alpha) / 2`` sample
        quantiles.

    Returns
    -------
    float
        The interval score averaged over every value element.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1), ``pred`` has no samples, a labeled
        ``pred`` has no sample dim, or ``pred`` and ``truth`` disagree on dims,
        sizes, coordinates or shape.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    pred, truth = _as_sample_first(pred, truth)
    tail = (1.0 - alpha) / 2.0
    lo = np.quantile(pred, tail, axis=0)
    hi = np.quantile(pred, 1.0 - tail, axis=0)
    penalty = 2.0 / (1.0 - alpha)
    below = penalty * (lo - truth) * (truth < lo)
    above = penalty * (truth - hi) * (truth > hi)
    return float((hi - lo + below + above).mean())


def make_mase(train_data, *, seasonality: int = 1) -> Metric:
    """Build a Mean Absolute Scaled Error metric scaled by ``train_data``.

    MASE divides the forecast MAE (sample-median point estimate) by the
    in-sample MAE of the seasonal-naive forecast on ``train_data``. The scale
    is computed once at factory time as a single scalar pooled over every
    element of ``train_data`` (not per series).

    Parameters
    ----------
    train_data : xarray.DataArray or array_like
        Training data — a DataArray with a ``"time"`` dim, or an array with
        time on axis 0.
    seasonality : int, default 1
        Seasonal period (``>= 1``); ``1`` is the random-walk naive baseline.

    Returns
    -------
    callable
        A metric ``mase(pred, truth) -> float`` equal to
        :func:`eval_mae` divided by the training scale; it accepts the same
        ``pred``/``truth`` inputs and raises the same errors as
        :func:`eval_mae`.

    Raises
    ------
    ValueError
        If ``seasonality < 1``, ``train_data`` is not longer than
        ``seasonality`` along time, or the seasonal-naive scale is zero
        (constant training series).
    """
    if seasonality < 1:
        msg = f"seasonality must be >= 1, got {seasonality}"
        raise ValueError(msg)
    if isinstance(train_data, xr.DataArray):
        train_data = train_data.transpose("time", ...).values
    train_data = np.asarray(train_data)
    if train_data.shape[0] <= seasonality:
        msg = (
            "train_data must be longer than seasonality along time "
            f"(got length {train_data.shape[0]}, seasonality {seasonality})"
        )
        raise ValueError(msg)
    scale = float(np.abs(train_data[seasonality:] - train_data[:-seasonality]).mean())
    if scale == 0.0:
        msg = (
            "seasonal-naive scale is zero (constant training series); MASE is "
            "undefined. Use a different metric or seasonality."
        )
        raise ValueError(msg)

    def mase(pred, truth) -> float:
        return eval_mae(pred, truth) / scale

    return mase


DEFAULT_METRICS: dict[str, Metric] = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
    "coverage": eval_coverage,
}
"""Default metrics used by :func:`evaluate_forecast` and ``backtest``."""


def evaluate_forecast(pred, truth, *, metrics: Mapping[str, Metric] | None = None) -> dict:
    """Apply several metrics to the same forecast samples and ground truth.

    Parameters
    ----------
    pred : xarray.DataArray or array_like
        Forecast samples, passed unchanged to every metric (labeled with
        sample dims, or an array with the sample axis first).
    truth : xarray.DataArray or array_like
        Ground-truth values, passed unchanged to every metric.
    metrics : mapping, optional
        Mapping of name to metric ``fn(pred, truth)``; ``None`` uses
        :data:`DEFAULT_METRICS`. Bind metric parameters with
        ``functools.partial``.

    Returns
    -------
    dict
        Maps each metric name (``str``) to its value (``float``, cast with
        ``float``), in the
        mapping's iteration order.

    Raises
    ------
    ValueError
        If a metric rejects the inputs (see the individual metrics).
    """
    metrics = DEFAULT_METRICS if metrics is None else metrics
    return {name: float(fn(pred, truth)) for name, fn in metrics.items()}

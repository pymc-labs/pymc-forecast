"""Probabilistic forecast metrics, dim-aware.

Every metric takes forecast samples and ground truth and reduces to a float.
Inputs may be labeled (``xarray.DataArray``) or raw numpy:

- **DataArray predictions** carry their sample dimensions by name — ``chain`` /
  ``draw`` (as produced by the forecast drivers) or an already-stacked
  ``sample`` dim. Labeled truth is transposed to the prediction's dim order, so
  callers never think about axis positions.
- **numpy predictions** follow the classical convention: sample axis first.

Ported from numpyro_forecast (itself porting ``pyro.ops.stats.crps_empirical``)
with the JAX kernels replaced by NumPy.
"""

from collections.abc import Callable, Mapping

import numpy as np
import xarray as xr

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

_SAMPLE_DIMS = ("chain", "draw", "sample")
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
        pred_np = pred.values.reshape(-1, *pred.shape[len(sample_dims) :])
    else:
        pred_np = np.asarray(pred)
    truth_np = np.asarray(truth.values if isinstance(truth, xr.DataArray) else truth)
    return pred_np, truth_np


def crps_empirical(pred, truth) -> np.ndarray:
    r"""Elementwise empirical Continuous Ranked Probability Score.

    .. math::

        \mathrm{CRPS}(F, y) = \mathbb{E}|X - y| - \tfrac{1}{2}\,\mathbb{E}|X - X'|

    estimated from the forecast samples with the sorted-sample
    :math:`O(n \log n)` identity.

    Parameters
    ----------
    pred
        Forecast samples (labeled, or numpy with the sample axis first). At
        least 2 samples are required.
    truth
        Ground-truth values (prediction shape without the sample axis).

    Returns
    -------
    numpy.ndarray
        Elementwise CRPS, one value per data location.
    """
    pred, truth = _as_sample_first(pred, truth)
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
    """Mean absolute error of the forecast sample median."""
    pred, truth = _as_sample_first(pred, truth)
    return float(np.abs(np.median(pred, axis=0) - truth).mean())


def eval_rmse(pred, truth) -> float:
    """Root mean squared error of the forecast sample mean."""
    pred, truth = _as_sample_first(pred, truth)
    return float(np.sqrt(np.square(pred.mean(axis=0) - truth).mean()))


def eval_crps(pred, truth) -> float:
    """Mean empirical CRPS over all data elements (see :func:`crps_empirical`)."""
    return float(crps_empirical(pred, truth).mean())


def eval_coverage(pred, truth, *, alpha: float = 0.9) -> float:
    """Empirical coverage of the central ``alpha`` prediction interval.

    A well-calibrated forecast has coverage close to ``alpha``. Bind a
    non-default level with ``functools.partial(eval_coverage, alpha=0.8)``.
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
    is computed once at factory time.

    Parameters
    ----------
    train_data
        Training data — a DataArray with a ``"time"`` dim, or numpy with time
        on axis 0.
    seasonality
        Seasonal period (``>= 1``); ``1`` is the random-walk naive baseline.
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
    pred, truth
        As accepted by the individual metrics (labeled or numpy).
    metrics
        Mapping of name to metric; defaults to :data:`DEFAULT_METRICS`. Bind
        metric parameters with ``functools.partial``.

    Returns
    -------
    dict[str, float]
        Each metric name mapped to its value.
    """
    metrics = DEFAULT_METRICS if metrics is None else metrics
    return {name: float(fn(pred, truth)) for name, fn in metrics.items()}

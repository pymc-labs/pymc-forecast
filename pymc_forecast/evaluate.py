"""Backtesting: score a forecasting model over moving train/test windows.

Each window is a pure refit: the forecaster is constructed on the window's
training slice and asked to forecast the test slice, and metrics are computed
on the sampled predictions. Two windowing strategies are supported —
``"expanding"`` (train from the start of history) and ``"rolling"`` (fixed
training length) — with window bookkeeping ported from numpyro_forecast /
Pyro. Windows are positional (integer) slices along ``"time"``: results carry
integer split points and the predictions keep their real time coords.
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from time import perf_counter
from typing import Any, Literal

import numpy as np
import xarray as xr

from pymc_forecast.data import FUTURE_DIM, TIME_DIM, as_dataarray, null_covariates
from pymc_forecast.exceptions import BacktestWindowError
from pymc_forecast.forecaster import Forecaster
from pymc_forecast.metrics import DEFAULT_METRICS, Metric, evaluate_forecast

__all__ = ["BacktestResult", "WindowType", "backtest", "results_to_dataframe"]

WindowType = Literal["expanding", "rolling"]
"""Allowed ``window_type`` values of :func:`backtest`, ``"expanding"`` or ``"rolling"``."""


@dataclass(frozen=True)
class BacktestResult:
    """Per-window result of a :func:`backtest` run.

    Each parameter is available as an attribute of the same name.

    Parameters
    ----------
    t0 : int
        Inclusive start of the training slice (positional index into the
        data's ``"time"`` dim); ``0`` for expanding windows.
    t1 : int
        Train/test split: exclusive end of training and inclusive start of
        the test slice.
    t2 : int
        Exclusive end of the test slice.
    num_samples : int
        Requested number of forecast samples (the ``backtest`` argument).
    train_walltime : float
        Wall-clock seconds to construct (build and fit) the forecaster.
    test_walltime : float
        Wall-clock seconds for the forecast call.
    metrics : dict
        Metric name → value (``float``) on the test window.
    train_metrics : dict, optional
        Metric name → in-sample value (``float``); empty unless
        ``eval_train=True``. Defaults to an empty dict.
    prediction : xarray.DataArray, optional
        Forecast samples for the window, after ``transform`` if one was given
        (``None`` unless ``keep_predictions=True``).
    """

    t0: int
    t1: int
    t2: int
    num_samples: int
    train_walltime: float
    test_walltime: float
    metrics: dict[str, float]
    train_metrics: dict[str, float] = field(default_factory=dict)
    prediction: xr.DataArray | None = None


def _resolve_window_type(window_type: WindowType | None, train_window: int | None) -> WindowType:
    if window_type is None:
        return "expanding" if train_window is None else "rolling"
    if window_type == "expanding" and train_window is not None:
        msg = "window_type='expanding' and train_window are mutually exclusive"
        raise BacktestWindowError(msg)
    if window_type == "rolling" and train_window is None:
        msg = "window_type='rolling' requires train_window"
        raise BacktestWindowError(msg)
    if window_type not in ("expanding", "rolling"):
        msg = f"unknown window_type {window_type!r}"
        raise BacktestWindowError(msg)
    return window_type


def _windows(
    duration: int,
    *,
    window_type: WindowType,
    train_window: int | None,
    min_train_window: int,
    test_window: int | None,
    min_test_window: int,
    stride: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(t0, t1, t2)`` split points (bookkeeping ported from Pyro)."""
    for name, value in {
        "train_window": train_window,
        "min_train_window": min_train_window,
        "test_window": test_window,
        "min_test_window": min_test_window,
        "stride": stride,
    }.items():
        if value is None and name in ("train_window", "test_window"):
            continue
        if isinstance(value, bool) or not isinstance(value, Integral):
            msg = f"{name} must be an integer, got {value!r}"
            raise BacktestWindowError(msg)
    if min_train_window < 1:
        msg = f"min_train_window must be >= 1, got {min_train_window}"
        raise BacktestWindowError(msg)
    if min_test_window < 1:
        msg = f"min_test_window must be >= 1, got {min_test_window}"
        raise BacktestWindowError(msg)
    if test_window is not None and test_window < min_test_window:
        msg = f"test_window ({test_window}) must be >= min_test_window ({min_test_window})"
        raise BacktestWindowError(msg)
    if stride < 1:
        msg = f"stride must be >= 1, got {stride}"
        raise BacktestWindowError(msg)
    if train_window is not None and train_window < min_train_window:
        msg = f"train_window ({train_window}) must be >= min_train_window ({min_train_window})"
        raise BacktestWindowError(msg)
    first_t1 = min_train_window if train_window is None else train_window
    if first_t1 > duration - min_test_window:
        msg = (
            f"no valid windows: need at least {first_t1} training and "
            f"{min_test_window} test steps but the series has {duration}"
        )
        raise BacktestWindowError(msg)
    for t1 in range(first_t1, duration - min_test_window + 1, stride):
        t0 = 0 if window_type == "expanding" else t1 - train_window
        t2 = duration if test_window is None else min(duration, t1 + test_window)
        yield t0, t1, t2


def backtest(
    data,
    covariates,
    model_fn,
    *,
    forecaster_cls: type = Forecaster,
    metrics: Mapping[str, Metric] | None = None,
    per_window_metrics: Callable[[int, int, int], Mapping[str, Metric]] | None = None,
    transform: Callable | None = None,
    window_type: WindowType | None = None,
    train_window: int | None = None,
    min_train_window: int = 1,
    test_window: int | None = None,
    min_test_window: int = 1,
    stride: int = 1,
    num_samples: int = 100,
    forecaster_options: Mapping[str, Any]
    | Callable[[int, int, int], Mapping[str, Any]]
    | None = None,
    eval_train: bool = False,
    keep_predictions: bool = False,
    random_seed=None,
) -> list[BacktestResult]:
    """Backtest a forecasting model on moving ``(train, test)`` windows.

    Parameters
    ----------
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like
        The full series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim).
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like or None
        Covariates over the same span as ``data`` (normalized with
        ``as_dataarray``; 2-D input gets a ``"covariate"`` dim), sliced by
        position alongside ``data``. ``None`` for models without covariates.
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body or :class:`~pymc_forecast.model.ForecastingModel`.
    forecaster_cls : type, default pymc_forecast.forecaster.Forecaster
        Forecaster class constructed per window as
        ``forecaster_cls(model_fn, train_data, window_covariates,
        random_seed=..., **options)``, where ``window_covariates`` are the
        covariate rows ``t0:t2`` (training plus test span; zero-width
        covariates when ``covariates`` is ``None``).
    metrics : mapping, optional
        Metric name → function; defaults to
        :data:`~pymc_forecast.metrics.DEFAULT_METRICS`. Bind metric parameters
        with ``functools.partial``.
    per_window_metrics : callable, optional
        Optional ``(t0, t1, t2) -> Mapping`` producing extra metrics per
        window (e.g. a MASE scaled by that window's training slice via
        :func:`~pymc_forecast.metrics.make_mase`); entries override
        same-named ``metrics``.
    transform : callable, optional
        Optional ``(pred, truth) -> (pred, truth)`` applied before metrics
        (both labeled: ``pred`` has ``chain``/``draw`` sample dims, ``truth``
        is renamed to ``time_future`` so names align). With ``eval_train``,
        it is also applied to the in-sample ``obs`` samples and the training
        slice, which both use the ``"time"`` dim.
    window_type : {"expanding", "rolling"}, optional
        ``"expanding"`` trains from the start of the series; ``"rolling"``
        uses a fixed ``train_window``. ``None`` infers ``"rolling"`` when
        ``train_window`` is set and ``"expanding"`` otherwise.
    train_window : int, optional
        Fixed training length for rolling windows; required for
        ``"rolling"`` and not allowed with ``"expanding"``.
    min_train_window : int, default 1
        Minimum training length; the first split for expanding windows.
    test_window : int, optional
        Maximum test length; ``None`` tests to the end of the series.
    min_test_window : int, default 1
        Minimum test length; splits stop once fewer than this many test steps
        would remain, so every window has at least ``min_test_window`` test
        steps.
    stride : int, default 1
        Step between successive splits.
    num_samples : int, default 100
        Forecast samples per window (also used for in-sample predictions).
    forecaster_options : mapping or callable, optional
        Extra constructor kwargs for ``forecaster_cls`` — a mapping, or a
        callable ``(t0, t1, t2) -> mapping`` for per-window options.
    eval_train : bool, default False
        Also score the in-sample posterior predictive of each window.
    keep_predictions : bool, default False
        Keep each window's (transformed) forecast samples on the result.
    random_seed : int or sequence of int, optional
        Base seed (``numpy.random.SeedSequence`` entropy); per-window
        fit/forecast seeds are derived deterministically.

    Returns
    -------
    list of BacktestResult
        One result per window, in time order.

    Raises
    ------
    pymc_forecast.exceptions.BacktestWindowError
        If ``window_type`` is unknown or conflicts with ``train_window``, a
        window parameter is not an integer (or is a bool), a minimum is below
        1, ``test_window < min_test_window``, ``stride < 1``,
        ``train_window < min_train_window``, or the series is too short for
        any window.
    pymc_forecast.exceptions.AlignmentError
        If ``data`` or ``covariates`` cannot be normalized; errors raised by
        ``forecaster_cls`` (fitting/forecasting) and by the metrics propagate.

    Warns
    -----
    UserWarning
        From the VI convergence check of the default ``forecaster_cls``.
    """
    data_da = as_dataarray(data, role="data")
    if covariates is None:
        cov_da = null_covariates(data_da[TIME_DIM].values)
    else:
        cov_da = as_dataarray(covariates, role="covariates")
    duration = data_da.sizes[TIME_DIM]
    resolved_type = _resolve_window_type(window_type, train_window)
    metrics = dict(DEFAULT_METRICS if metrics is None else metrics)

    splits = list(
        _windows(
            duration,
            window_type=resolved_type,
            train_window=train_window,
            min_train_window=min_train_window,
            test_window=test_window,
            min_test_window=min_test_window,
            stride=stride,
        )
    )
    seed_children = np.random.SeedSequence(random_seed).spawn(len(splits))

    results: list[BacktestResult] = []
    for (t0, t1, t2), seed_seq in zip(splits, seed_children, strict=True):
        fit_seed, forecast_seed = (int(s) for s in seed_seq.generate_state(2))
        window_metrics = dict(metrics)
        if per_window_metrics is not None:
            window_metrics.update(per_window_metrics(t0, t1, t2))
        options = (
            forecaster_options(t0, t1, t2)
            if callable(forecaster_options)
            else (forecaster_options or {})
        )

        train_data = data_da.isel({TIME_DIM: slice(t0, t1)})
        window_cov = cov_da.isel({TIME_DIM: slice(t0, t2)})

        start = perf_counter()
        forecaster = forecaster_cls(
            model_fn, train_data, window_cov, random_seed=fit_seed, **options
        )
        train_walltime = perf_counter() - start

        start = perf_counter()
        pred_tree = forecaster.forecast(
            window_cov, num_samples=num_samples, random_seed=forecast_seed
        )
        test_walltime = perf_counter() - start
        pred = pred_tree["predictions"]["forecast"]

        truth = data_da.isel({TIME_DIM: slice(t1, t2)}).rename({TIME_DIM: FUTURE_DIM})
        if transform is not None:
            pred, truth = transform(pred, truth)
        scores = evaluate_forecast(pred, truth, metrics=window_metrics)

        train_scores: dict[str, float] = {}
        if eval_train:
            ppc = forecaster.predict_in_sample(num_samples=num_samples, random_seed=forecast_seed)
            train_pred = ppc["posterior_predictive"]["obs"]
            train_truth = train_data
            if transform is not None:
                train_pred, train_truth = transform(train_pred, train_truth)
            train_scores = evaluate_forecast(train_pred, train_truth, metrics=window_metrics)

        results.append(
            BacktestResult(
                t0=t0,
                t1=t1,
                t2=t2,
                num_samples=num_samples,
                train_walltime=train_walltime,
                test_walltime=test_walltime,
                metrics=scores,
                train_metrics=train_scores,
                prediction=pred if keep_predictions else None,
            )
        )
    return results


def results_to_dataframe(results: list[BacktestResult]):
    """Flatten backtest results into a ``pandas.DataFrame``.

    One row per window: split points, walltimes, one column per metric, and
    ``train_``-prefixed columns for in-sample metrics when present.

    Parameters
    ----------
    results : list of BacktestResult
        Results of :func:`backtest`.

    Returns
    -------
    pandas.DataFrame
        Columns ``t0``, ``t1``, ``t2``, ``num_samples``, ``train_walltime``,
        ``test_walltime``, then one per metric and one ``train_<name>`` per
        in-sample metric; ``prediction`` is not included.
    """
    import pandas as pd

    rows = []
    for r in results:
        row: dict[str, Any] = {
            "t0": r.t0,
            "t1": r.t1,
            "t2": r.t2,
            "num_samples": r.num_samples,
            "train_walltime": r.train_walltime,
            "test_walltime": r.test_walltime,
        }
        row.update(r.metrics)
        row.update({f"train_{k}": v for k, v in r.train_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)

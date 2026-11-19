"""Backtesting: score a forecasting model over moving train/test windows.

Each window is a pure refit: the model is fitted on the window's training
slice and forecasts the test slice, and metrics are computed on the sampled
predictions. A forecaster class (``forecaster_cls``) or a closure
(``forecast_fn``) does the fit and the forecast. Two windowing strategies are
supported — ``"expanding"`` (train from the start of history) and
``"rolling"`` (fixed training length) — with window bookkeeping ported from
numpyro_forecast / Pyro. Windows are positional (integer) slices along
``"time"``: results carry integer split points and the predictions keep their
real time coords.
"""

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from time import perf_counter
from typing import Any, Literal

import numpy as np
import xarray as xr

from pymc_forecast.data import (
    CHAIN_DIM,
    DRAW_DIM,
    FUTURE_DIM,
    SAMPLE_DIMS,
    TIME_DIM,
    as_dataarray,
    as_time_index,
    null_covariates,
    validate_alignment,
)
from pymc_forecast.exceptions import AlignmentError, BacktestWindowError
from pymc_forecast.forecaster import Forecaster
from pymc_forecast.metrics import DEFAULT_METRICS, Metric, evaluate_forecast
from pymc_forecast.model import FORECAST_VAR, OBS_VAR

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
        Forecast samples per window: the requested ``num_samples`` with
        ``forecaster_cls``; with ``forecast_fn``, the ``chain`` size times the
        ``draw`` size of the draws it returned, which may differ from the
        request.
    train_walltime : float
        Wall-clock seconds to construct (build and fit) the forecaster;
        ``nan`` with ``forecast_fn``, whose fit is timed in ``test_walltime``.
    test_walltime : float
        Wall-clock seconds for the forecast call: ``forecaster.forecast``, or
        the whole ``forecast_fn`` call, fit included.
    metrics : dict
        Metric name → value (``float``) on the test window.
    train_metrics : dict, optional
        Metric name → in-sample value (``float``); empty unless
        ``eval_train=True``. Defaults to an empty dict.
    prediction : xarray.DataArray, optional
        Forecast samples for the window, after ``transform`` if one was given
        (``None`` unless ``keep_predictions=True``). With a ``transform`` its
        dims are the ones ``transform`` returns.
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


def _window_seeds(random_seed, num_windows: int) -> list[np.random.SeedSequence]:
    """One child ``SeedSequence`` per window.

    A ``Generator`` spawns the children itself (``Generator.spawn``, which
    advances its spawn counter); any other seed goes through
    ``SeedSequence(random_seed)``.
    """
    if isinstance(random_seed, np.random.Generator):
        return [child.bit_generator.seed_seq for child in random_seed.spawn(num_windows)]
    return np.random.SeedSequence(random_seed).spawn(num_windows)


def _check_drivers(
    data,
    *,
    forecaster_cls,
    forecaster_options,
    forecast_fn,
    in_sample_fn,
    var_name: str,
    transform,
    eval_train: bool,
) -> None:
    """Reject argument combinations that ``backtest`` cannot run, before any fit."""
    if forecast_fn is None:
        if in_sample_fn is not None:
            msg = (
                "in_sample_fn requires forecast_fn; forecaster_cls draws the in-sample "
                "predictive with its predict_in_sample method"
            )
            raise ValueError(msg)
        if isinstance(data, xr.Dataset):
            msg = "Dataset data requires forecast_fn: the forecaster classes take one DataArray"
            raise TypeError(msg)
    else:
        excluded = [
            name
            for name, conflicts in (
                ("forecaster_cls", forecaster_cls is not None),
                ("forecaster_options", forecaster_options is not None),
                ("var_name", var_name != FORECAST_VAR),
            )
            if conflicts
        ]
        if excluded:
            msg = (
                f"forecast_fn cannot be combined with {', '.join(excluded)}: the closure "
                "fits each window and returns the draws that are scored"
            )
            raise ValueError(msg)
        if eval_train and in_sample_fn is None:
            msg = "eval_train=True with forecast_fn requires in_sample_fn"
            raise ValueError(msg)
    if eval_train and var_name != FORECAST_VAR:
        msg = (
            f"var_name={var_name!r} cannot be combined with eval_train=True: the "
            f"in-sample score is of {OBS_VAR!r}, which is not the in-sample "
            f"counterpart of {var_name!r}"
        )
        raise ValueError(msg)
    if isinstance(data, xr.Dataset) and transform is None:
        msg = (
            "Dataset data requires transform, which returns the truth of the target as a DataArray"
        )
        raise ValueError(msg)


def _as_backtest_data(data) -> xr.DataArray | xr.Dataset:
    """Normalize ``data`` with ``as_dataarray``, or check a ``Dataset`` of tables."""
    if not isinstance(data, xr.Dataset):
        return as_dataarray(data, role="data")
    if TIME_DIM not in data.dims:
        msg = f"Dataset data must have a '{TIME_DIM}' dim, got dims {tuple(data.dims)}"
        raise AlignmentError(msg)
    untimed = [name for name, var in data.data_vars.items() if TIME_DIM not in var.dims]
    if untimed:
        msg = (
            f"Dataset data variables without a '{TIME_DIM}' dim would reach the model "
            f"whole, past the forecast origin: {untimed}"
        )
        raise AlignmentError(msg)
    if TIME_DIM not in data.coords:
        data = data.assign_coords({TIME_DIM: np.arange(data.sizes[TIME_DIM])})
    return data


def _num_draws(draws, source: str) -> int:
    """Check the draws a closure returned and count them (``chain`` times ``draw``)."""
    if not isinstance(draws, xr.DataArray):
        msg = f"{source} must return an xarray.DataArray of draws, got {type(draws).__name__}"
        raise TypeError(msg)
    if not set(SAMPLE_DIMS) <= set(draws.dims):
        msg = f"{source} must return draws with dims {SAMPLE_DIMS}, got dims {draws.dims}"
        raise ValueError(msg)
    return draws.sizes[CHAIN_DIM] * draws.sizes[DRAW_DIM]


@dataclass(frozen=True)
class _WindowForecast:
    """One window's forecast draws and timings, and its in-sample sampler."""

    pred: xr.DataArray
    num_samples: int
    train_walltime: float
    test_walltime: float
    in_sample: Callable[[], xr.DataArray]


def _forecaster_window(
    forecaster_cls,
    model_fn,
    train_data,
    window_cov,
    seed_seq: np.random.SeedSequence,
    *,
    options: Mapping[str, Any],
    num_samples: int,
    var_name: str,
    batch_size: int | None,
) -> _WindowForecast:
    fit_seed, forecast_seed = (int(s) for s in seed_seq.generate_state(2))
    # Adapters written before ``batch_size`` existed take no such keyword.
    predict_kwargs = {} if batch_size is None else {"batch_size": batch_size}
    start = perf_counter()
    forecaster = forecaster_cls(model_fn, train_data, window_cov, random_seed=fit_seed, **options)
    train_walltime = perf_counter() - start
    start = perf_counter()
    tree = forecaster.forecast(
        window_cov, num_samples=num_samples, random_seed=forecast_seed, **predict_kwargs
    )
    test_walltime = perf_counter() - start

    def in_sample() -> xr.DataArray:
        ppc = forecaster.predict_in_sample(
            num_samples=num_samples, random_seed=forecast_seed, **predict_kwargs
        )
        return ppc["posterior_predictive"][OBS_VAR]

    pred = tree["predictions"][var_name]
    return _WindowForecast(pred, num_samples, train_walltime, test_walltime, in_sample)


def _closure_window(
    forecast_fn,
    in_sample_fn,
    model_fn,
    train_data,
    window_cov,
    seed_seq: np.random.SeedSequence,
    *,
    num_samples: int,
    batch_size: int | None,
) -> _WindowForecast:
    # Separate streams, so that turning on eval_train leaves the forecasts as they are.
    forecast_seq, in_sample_seq = seed_seq.spawn(2)
    train_cov = window_cov.isel({TIME_DIM: slice(0, train_data.sizes[TIME_DIM])})
    start = perf_counter()
    pred = forecast_fn(
        np.random.default_rng(forecast_seq),
        model_fn,
        train_data,
        train_cov,
        window_cov,
        num_samples,
        batch_size=batch_size,
    )
    test_walltime = perf_counter() - start
    drawn = _num_draws(pred, "forecast_fn")

    def in_sample() -> xr.DataArray:
        train_pred = in_sample_fn(
            np.random.default_rng(in_sample_seq),
            model_fn,
            train_data,
            train_cov,
            num_samples,
            batch_size=batch_size,
        )
        _num_draws(train_pred, "in_sample_fn")
        return train_pred

    return _WindowForecast(pred, drawn, math.nan, test_walltime, in_sample)


def backtest(
    data,
    covariates,
    model_fn,
    *,
    forecaster_cls: type | None = None,
    forecast_fn: Callable[..., xr.DataArray] | None = None,
    in_sample_fn: Callable[..., xr.DataArray] | None = None,
    var_name: str = FORECAST_VAR,
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
    batch_size: int | None = None,
    forecaster_options: Mapping[str, Any]
    | Callable[[int, int, int], Mapping[str, Any]]
    | None = None,
    eval_train: bool = False,
    keep_predictions: bool = False,
    random_seed=None,
) -> list[BacktestResult]:
    """Backtest a forecasting model on moving ``(train, test)`` windows.

    Each window is fitted and forecast in one of two ways: by a forecaster
    class (``forecaster_cls``, default
    :class:`~pymc_forecast.forecaster.Forecaster`) or by a closure
    (``forecast_fn``) that fits the window and returns the forecast draws
    itself. The closure suits models outside the forecaster classes, several
    observed tables (``data`` as a ``Dataset``), and targets other than the
    observed series.

    **Forecaster protocol.** ``backtest`` uses ``forecaster_cls`` through the
    calls below and nothing else, so any class that has them works:

    - ``forecaster_cls(model_fn, train_data, window_covariates,
      random_seed=fit_seed, **options)`` fits the window. ``train_data`` is
      rows ``t0:t1`` of ``data``, ``window_covariates`` the covariate rows
      ``t0:t2`` (training plus test span; zero-width covariates when
      ``covariates`` is ``None``) and ``fit_seed`` an ``int``.
    - ``.forecast(window_covariates, num_samples=num_samples,
      random_seed=forecast_seed)``, where ``forecast_seed`` is an ``int``.
      ``backtest`` scores ``result["predictions"][var_name]``.
    - ``.predict_in_sample(num_samples=num_samples,
      random_seed=forecast_seed)``, only with ``eval_train=True``.
      ``backtest`` scores ``result["posterior_predictive"]["obs"]`` against
      ``train_data``.

    When ``batch_size`` is not ``None``, both methods also get
    ``batch_size=batch_size``;
    :class:`~pymc_forecast.statespace.StatespaceForecaster` does not accept
    it.

    **Closure protocol.** ``forecast_fn`` and ``in_sample_fn`` get positional
    arguments and a keyword ``batch_size``::

        forecast_fn(rng, model_fn, train_data, train_covariates,
                    full_covariates, num_samples, *, batch_size=None)
        in_sample_fn(rng, model_fn, train_data, train_covariates,
                     num_samples, *, batch_size=None)

    ``rng`` is a ``numpy.random.Generator``, a separate stream per window and
    per closure. ``model_fn`` is passed through as given. ``train_covariates``
    are the covariate rows ``t0:t1`` and ``full_covariates`` the rows
    ``t0:t2``. ``forecast_fn`` returns the forecast draws of the test rows and
    ``in_sample_fn`` the draws of the training rows, each an
    ``xarray.DataArray`` with the sample dims ``chain`` and ``draw``. Without
    a ``transform`` the other dims must align with the truth (``time_future``
    or ``time``, and the data dims); with one, any dims are allowed. The
    closure may return another number of draws than ``num_samples``; the
    result records what it returned.

    Each closure call must fit its own window: the functional
    :func:`~pymc_forecast.prediction.forecast` rejects a posterior fitted on
    another window. A closure that builds and fits the model also owns any
    reuse of a compiled model across windows. A ``forecast_fn`` on the
    functional API::

        def forecast_fn(rng, model_fn, train_data, train_covariates,
                        full_covariates, num_samples, /, *, batch_size=None):
            fit = fit_vi(model_fn, train_data, train_covariates,
                         random_seed=int(rng.integers(2**31)))
            posterior = draw_posterior(fit, num_samples, rng)
            tree = forecast(model_fn, posterior, train_data, full_covariates,
                            var_names=["forecast"], batch_size=batch_size,
                            random_seed=rng)
            return tree["predictions"]["forecast"]

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset, pandas.Series, pandas.DataFrame or array_like
        The full series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim). An ``xarray.Dataset`` holds several observed
        tables, cut at each window like an array, with the dims and dtype of
        each table kept. It requires ``forecast_fn`` and ``transform``, and
        every data variable must have a ``"time"`` dim so that no table
        reaches the closure past the forecast origin. A ``Dataset`` without
        ``"time"`` coords gets an integer range.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like or None
        Covariates over the same span as ``data`` (normalized with
        ``as_dataarray``; 2-D input gets a ``"covariate"`` dim), sliced by
        position alongside ``data``. ``None`` for models without covariates.
        With ``forecast_fn``, the covariate ``"time"`` coords must start with
        the ones of ``data``, as the forecasters require.
    model_fn : callable, pymc_forecast.model.ForecastingModel or None
        The model body or :class:`~pymc_forecast.model.ForecastingModel`,
        passed to ``forecaster_cls`` or to the closures. May be ``None`` with
        ``forecast_fn``.
    forecaster_cls : type, optional
        Forecaster class constructed per window (see the forecaster protocol
        above). ``None`` means :class:`~pymc_forecast.forecaster.Forecaster`
        when ``forecast_fn`` is not given. Cannot be combined with
        ``forecast_fn``.
    forecast_fn : callable, optional
        Closure that fits each window and returns its forecast draws (see the
        closure protocol above).
    in_sample_fn : callable, optional
        Closure that fits each window and returns its in-sample draws (see
        the closure protocol above). Requires ``forecast_fn``; ``eval_train``
        with ``forecast_fn`` requires it.
    var_name : str, default "forecast"
        Variable of the ``predictions`` group of ``forecaster_cls.forecast``
        that is scored and kept, e.g. ``"mu_future"`` or another
        ``*_future`` variable. Another name than ``"forecast"`` cannot be
        combined with ``forecast_fn``, which returns its target directly, or
        with ``eval_train``, whose in-sample score is of ``"obs"``.
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
        is the test rows of ``data`` renamed to ``time_future`` so names
        align). With ``eval_train``, it is also applied to the in-sample
        samples and the training slice, which both use the ``"time"`` dim.
        Required for ``Dataset`` data: ``truth`` is then a ``Dataset``, and
        ``transform`` returns the truth of the target as a ``DataArray``.
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
        Forecast samples requested per window (also used for in-sample
        predictions).
    batch_size : int, optional
        Maximum posterior draws per predictive pass, passed to the closures,
        and to the ``forecast`` and ``predict_in_sample`` methods of
        ``forecaster_cls`` when not ``None``.
    forecaster_options : mapping or callable, optional
        Extra constructor kwargs for ``forecaster_cls`` — a mapping, or a
        callable ``(t0, t1, t2) -> mapping`` for per-window options. Cannot be
        combined with ``forecast_fn``.
    eval_train : bool, default False
        Also score the in-sample posterior predictive of each window.
    keep_predictions : bool, default False
        Keep each window's (transformed) forecast samples on the result.
    random_seed : int, sequence of int, numpy.random.Generator or None, optional
        Base seed. Each window gets a child ``numpy.random.SeedSequence``,
        spawned from ``SeedSequence(random_seed)``, or by
        ``Generator.spawn`` for a ``Generator``. ``forecaster_cls`` gets two
        ``int`` seeds per window (fit and forecast) from the child. The
        closures get one ``Generator`` each, from two children of the
        window's ``SeedSequence``, so ``eval_train`` does not change the
        forecasts. ``Generator.spawn`` advances the generator's spawn counter:
        two calls with the same ``Generator`` object give different results,
        and two generators created with the same seed give the same results.

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
        If ``data`` or ``covariates`` cannot be normalized, a ``Dataset`` data
        variable has no ``"time"`` dim, or (with ``forecast_fn``) the
        covariate ``"time"`` coords do not start with the ones of ``data``.
        Errors raised by ``forecaster_cls`` or the closures
        (fitting/forecasting) and by the metrics propagate.
    ValueError
        Before any fit, if ``forecast_fn`` is combined with
        ``forecaster_cls``, ``forecaster_options`` or a ``var_name`` other
        than ``"forecast"``; ``in_sample_fn`` is given without
        ``forecast_fn``; ``eval_train`` is set with ``forecast_fn`` and no
        ``in_sample_fn``, or with a ``var_name`` other than ``"forecast"``;
        or ``data`` is a ``Dataset`` and ``transform`` is not given. After a
        closure call, if its draws lack the ``chain`` or ``draw`` dim.
    TypeError
        If ``data`` is a ``Dataset`` and ``forecast_fn`` is not given, or a
        closure returns something other than an ``xarray.DataArray``.

    Warns
    -----
    UserWarning
        From the VI convergence check of the default ``forecaster_cls``.
    """
    _check_drivers(
        data,
        forecaster_cls=forecaster_cls,
        forecaster_options=forecaster_options,
        forecast_fn=forecast_fn,
        in_sample_fn=in_sample_fn,
        var_name=var_name,
        transform=transform,
        eval_train=eval_train,
    )
    data_obj = _as_backtest_data(data)
    time_index = as_time_index(data_obj)
    if covariates is None:
        cov_da = null_covariates(time_index)
    else:
        cov_da = as_dataarray(covariates, role="covariates")
        if forecast_fn is not None:
            # The forecasters check this per window; a closure gets no check.
            time_only = xr.DataArray(
                np.zeros(len(time_index)), dims=(TIME_DIM,), coords={TIME_DIM: time_index}
            )
            validate_alignment(time_only, cov_da)
    if forecast_fn is None and forecaster_cls is None:
        forecaster_cls = Forecaster
    duration = data_obj.sizes[TIME_DIM]
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
    seeds = _window_seeds(random_seed, len(splits))

    results: list[BacktestResult] = []
    for (t0, t1, t2), seed_seq in zip(splits, seeds, strict=True):
        window_metrics = dict(metrics)
        if per_window_metrics is not None:
            window_metrics.update(per_window_metrics(t0, t1, t2))

        train_data = data_obj.isel({TIME_DIM: slice(t0, t1)})
        window_cov = cov_da.isel({TIME_DIM: slice(t0, t2)})
        if forecast_fn is None:
            options = (
                forecaster_options(t0, t1, t2)
                if callable(forecaster_options)
                else (forecaster_options or {})
            )
            window = _forecaster_window(
                forecaster_cls,
                model_fn,
                train_data,
                window_cov,
                seed_seq,
                options=options,
                num_samples=num_samples,
                var_name=var_name,
                batch_size=batch_size,
            )
        else:
            window = _closure_window(
                forecast_fn,
                in_sample_fn,
                model_fn,
                train_data,
                window_cov,
                seed_seq,
                num_samples=num_samples,
                batch_size=batch_size,
            )

        pred = window.pred
        truth = data_obj.isel({TIME_DIM: slice(t1, t2)}).rename({TIME_DIM: FUTURE_DIM})
        if transform is not None:
            pred, truth = transform(pred, truth)
        scores = evaluate_forecast(pred, truth, metrics=window_metrics)

        train_scores: dict[str, float] = {}
        if eval_train:
            train_pred = window.in_sample()
            train_truth = train_data
            if transform is not None:
                train_pred, train_truth = transform(train_pred, train_truth)
            train_scores = evaluate_forecast(train_pred, train_truth, metrics=window_metrics)

        results.append(
            BacktestResult(
                t0=t0,
                t1=t1,
                t2=t2,
                num_samples=window.num_samples,
                train_walltime=window.train_walltime,
                test_walltime=window.test_walltime,
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

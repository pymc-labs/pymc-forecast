"""Observation-driven recursions with a single source of error.

The deterministic training filter and the generative forecast share the same
mean and update functions. Only future errors are random variables; they are
absent during fitting and drawn fresh during posterior replay. Inspired by
``numpyro_forecast.models.ssoe`` (Apache-2.0), using PyTensor and named coords.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import signature

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from pymc.pytensorf import collect_default_updates
from pytensor.raise_op import Assert

from pymc_forecast.data import FUTURE_DIM, TIME_DIM
from pymc_forecast.exceptions import AlignmentError, HorizonError
from pymc_forecast.model import Horizon, RVFactory

__all__ = ["SSOEResult", "ssoe"]

# PyTensor 2.26 (the PyMC 5.20 floor) always returns an updates dictionary.
_SCAN_HAS_RETURN_UPDATES = "return_updates" in signature(pytensor.scan).parameters


@dataclass(frozen=True)
class SSOEResult:
    """Outputs of :func:`ssoe`, before registering an observation likelihood.

    ``mu`` contains in-sample one-step-ahead means. ``mu_future`` contains
    forecast means conditional on the preceding simulated observations;
    ``y_future`` contains those means plus their current observation errors.
    Both future tensors have length zero during training. Time is first in
    these symbolic tensors; ``dims`` records the remaining named dimensions.
    Use ``mu_future`` to exclude the *current* observation error, remembering
    that earlier future errors still affect the state.
    """

    mu: pt.TensorVariable
    mu_future: pt.TensorVariable
    y_future: pt.TensorVariable
    dims: tuple[str, ...]


def _history(h: Horizon, y: xr.DataArray | None, dims: tuple[str, ...] | None):
    y = h.data if y is None else y
    if y is None or h.t_obs == 0:
        raise HorizonError("ssoe requires a nonempty observed history")
    if not isinstance(y, xr.DataArray) or TIME_DIM not in y.dims:
        raise AlignmentError("ssoe y must be a DataArray with a 'time' dimension")
    if not np.array_equal(y[TIME_DIM].values, h.time):
        raise AlignmentError("ssoe y coordinates must match the observed time window exactly")
    inferred = tuple(d for d in y.dims if d != TIME_DIM)
    dims = inferred if dims is None else tuple(dims)
    if len(set(dims)) != len(dims) or set(dims) != set(inferred):
        raise AlignmentError("ssoe dims must name each non-time dimension of y exactly once")
    if FUTURE_DIM in dims:
        raise AlignmentError("time_future cannot be a batch dimension")
    model = pm.modelcontext(None)
    for dim in dims:
        if dim not in model.coords or not np.array_equal(y[dim].values, model.coords[dim]):
            raise AlignmentError(f"ssoe y coordinates for {dim!r} must match the model")
    values = np.asarray(y.transpose(TIME_DIM, *dims).values, dtype=pytensor.config.floatX)
    if not np.isfinite(values).all():
        raise ValueError(
            "ssoe y must be finite; use explicit update gates for missing observations"
        )
    return values, dims


def _inputs(h: Horizon, xs: xr.DataArray | None):
    if xs is None:
        return None
    if not isinstance(xs, xr.DataArray) or TIME_DIM not in xs.dims:
        raise AlignmentError("ssoe xs must be a DataArray with a 'time' dimension")
    time = np.concatenate([h.time, h.time_future]) if h.future else h.time
    if xs.sizes[TIME_DIM] < h.duration or not np.array_equal(
        xs[TIME_DIM].values[: h.duration], time
    ):
        raise AlignmentError("ssoe xs must cover the horizon with matching time coordinates")
    for dim in xs.dims:
        if dim == TIME_DIM:
            continue
        coords = pm.modelcontext(None).coords
        if dim in coords and not np.array_equal(xs[dim].values, coords[dim]):
            raise AlignmentError(f"ssoe xs coordinates for {dim!r} must match the model")
    values = np.asarray(xs.transpose(TIME_DIM, ...).values[: h.duration])
    if not np.isfinite(values).all():
        raise ValueError("ssoe xs must be finite")
    return pt.as_tensor_variable(values)


def ssoe(
    h: Horizon,
    name: str,
    init,
    mean: Callable,
    update: Callable,
    noise_fn: RVFactory,
    *,
    y: xr.DataArray | None = None,
    params: Sequence = (),
    xs: xr.DataArray | None = None,
    dims: tuple[str, ...] | None = None,
) -> SSOEResult:
    """Filter observed values, then simulate a recursive forecast.

    Parameters
    ----------
    h, name
        Model horizon and base name of the future error variable. Only
        ``f"{name}_future"`` is registered, and only when forecasting.
    init
        Initial state: one tensor or a nonempty tuple of tensors. States may
        have different shapes (e.g. scalar level and vector seasonality).
    mean
        ``(state, x_t, *params) -> mu_t``. Returns the one-step-ahead mean,
        shaped like one row of ``y``. ``x_t`` is ``None`` without ``xs``.
    update
        ``(state, y_t, eps_t, x_t, *params) -> state``. Returns the next state
        with the same structure and shapes as ``init``. In-sample,
        ``eps_t = y_t - mu_t``; in the future, ``eps_t`` is freshly drawn and
        ``y_t = mu_t + eps_t``. Both callbacks must be deterministic.
    noise_fn
        ``(name, dims) -> RV`` factory for independent, zero-centered
        per-step errors, e.g. ``lambda n, d: pm.Normal(n, 0, sigma, dims=d)``.
        Errors may be correlated across the observation dimensions, using
        e.g. ``pm.MvNormal``.
    y
        Labeled driving history, defaults to ``h.data``. Must cover exactly
        the training window. A transformed history can be supplied to compose
        multiple recursion channels. Prior-only builds need an explicit
        driving history; this helper does not generate an in-sample history.
    params
        Tensor parameters passed explicitly to both callbacks. Pass random
        parameters here so PyTensor can differentiate the training recursion.
    xs
        Optional labeled inputs spanning the full horizon. The time dimension
        is selected by name, and coordinates are checked. Future inputs must
        be known covariates or explicit scenarios: never derive future update
        gates from held-out observations. Extra rows are ignored during a
        shorter training build.
    dims
        Non-time observation dimensions, inferred from ``y`` by default.
        Labeled data are transposed into this order before entering the scan.

    Returns
    -------
    SSOEResult
        In-sample means, future means and future samples. The caller registers
        ``obs`` against ``result.mu`` and ``forecast`` as a Deterministic of
        ``result.y_future``. Register ``mu`` and ``mu_future`` Deterministics
        to include the means in the standard prediction outputs. Do not add
        another observation draw to ``y_future``: it already includes noise.

    Notes
    -----
    This is an observation-driven filter, not a latent Markov process. For
    sampled hidden states use :func:`~pymc_forecast.markov.markov_time_series`;
    for linear-Gaussian hidden states consider the statespace backend.
    """
    values, dims = _history(h, y, dims)
    inputs = _inputs(h, xs)
    tuple_state = isinstance(init, tuple)
    initial = list(init) if tuple_state else [init]
    if not initial:
        raise ValueError("ssoe init must contain at least one state tensor")
    initial = [pt.as_tensor_variable(v).astype(pytensor.config.floatX) for v in initial]
    parameters = [pt.as_tensor_variable(p) for p in params]
    allowed_rngs = set(collect_default_updates([*initial, *parameters]))
    n_states = len(initial)
    row_shape = values.shape[1:]
    model = pm.modelcontext(None)
    registered = set(model.named_vars)

    def step(future):
        def body(value_t, *args):
            x_t, *rest = args if inputs is not None else (None, *args)
            states, parameters_t = rest[:n_states], rest[n_states:]
            state = tuple(states) if tuple_state else states[0]
            mu = pt.as_tensor_variable(mean(state, x_t, *parameters_t))
            if mu.ndim != len(row_shape):
                raise ValueError("ssoe mean must have the same dimensions as a row of y")
            mu = pt.specify_shape(mu, row_shape)
            eps = value_t if future else value_t - mu
            observed = mu + eps if future else value_t
            next_state = update(state, observed, eps, x_t, *parameters_t)
            if isinstance(next_state, tuple) != tuple_state:
                raise ValueError("ssoe update must preserve the initial state structure")
            next_states = list(next_state) if tuple_state else [next_state]
            if len(next_states) != n_states:
                raise ValueError("ssoe update must preserve the number of state tensors")
            checked = []
            for previous, new in zip(states, next_states, strict=True):
                new = pt.as_tensor_variable(new).astype(previous.dtype)
                if new.ndim != previous.ndim:
                    raise ValueError("ssoe update must preserve each state tensor's shape")
                new = Assert("ssoe update must preserve each state tensor's shape")(
                    new, pt.all(pt.eq(new.shape, previous.shape))
                )
                checked.append(new)
            result = [*checked, mu, observed]
            if set(collect_default_updates(result)) - allowed_rngs:
                raise ValueError("ssoe mean and update must be deterministic")
            return result

        return body

    def scan(driving, states, inputs_slice, *, future):
        sequences = [driving] + ([] if inputs_slice is None else [inputs_slice])
        result = pytensor.scan(
            step(future),
            sequences=sequences,
            outputs_info=[*states, None, None],
            non_sequences=parameters,
            strict=True,
            **({"return_updates": False} if _SCAN_HAS_RETURN_UPDATES else {}),
        )
        outputs, updates = (result, {}) if _SCAN_HAS_RETURN_UPDATES else result
        if updates or set(model.named_vars) != registered:
            raise ValueError(
                "ssoe mean and update must be deterministic; pass parameters via params"
            )
        return outputs

    outputs = scan(
        pt.as_tensor_variable(values),
        initial,
        None if inputs is None else inputs[: h.t_obs],
        future=False,
    )
    mu = outputs[-2]
    empty = pt.zeros((0, *row_shape), dtype=mu.dtype)
    if h.future == 0:
        return SSOEResult(mu, empty, empty, dims)
    errors = noise_fn(f"{name}_future", (FUTURE_DIM, *dims))
    if model.named_vars_to_dims.get(errors.name) != (FUTURE_DIM, *dims):
        raise AlignmentError("ssoe noise_fn must register the supplied dimensions")
    errors = pt.specify_shape(errors, (h.future, *row_shape))
    registered = set(model.named_vars)
    outputs = scan(
        errors,
        [state[-1] for state in outputs[:n_states]],
        None if inputs is None else inputs[h.t_obs :],
        future=True,
    )
    return SSOEResult(mu, outputs[-2], outputs[-1], dims)

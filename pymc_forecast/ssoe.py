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
from pytensor.tensor.random.basic import NormalRV, StudentTRV

from pymc_forecast._dist import check_unnamed_dist, expand_dist, is_dist, is_zero_constant
from pymc_forecast.data import FUTURE_DIM, TIME_DIM
from pymc_forecast.exceptions import AlignmentError, HorizonError
from pymc_forecast.model import Horizon
from pymc_forecast.priors import is_prior_like, prior_rv_factory

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

    Each parameter is available as an attribute of the same name.

    Parameters
    ----------
    mu : pytensor.tensor.TensorVariable
        In-sample one-step-ahead means, shape ``(t_obs, *row)``.
    mu_future : pytensor.tensor.TensorVariable
        Future means, shape ``(future, *row)`` (length zero during training).
    y_future : pytensor.tensor.TensorVariable
        Future simulated observations ``mu_future`` plus the current errors
        (length zero during training).
    dims : tuple of str
        Non-time dims of the rows, in the order used by the tensors.
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


_NOISE_PATTERN = (
    "create sigma in the model body (sigma = pm.HalfNormal('sigma', 1)), use it "
    "in the observation, and pass pm.Normal.dist(0, sigma) as noise"
)


def _accept_noise(noise) -> None:
    """Reject noise that cannot forecast the errors the fitted model implies.

    ``noise`` is a ``.dist()`` or a ``Prior``; an ``RVFactory`` is rejected.
    A ``Prior`` must have constant parameters: a nested hyper-prior would be
    created only on a forecasting build, so it is absent from the fitted
    posterior and drawn from its prior. A Normal/StudentT ``Prior`` must leave
    ``mu`` unset or zero. A ``.dist()`` must be unnamed and depend only on
    model variables; a ``pm.Normal.dist`` / ``pm.StudentT.dist`` must have a
    constant zero location. Other dists are not location-checked. Runs on
    every build, so training builds fail fast too.
    """
    if is_prior_like(noise):
        _accept_prior_noise(noise)
        return
    if not is_dist(noise):
        raise HorizonError("ssoe noise must be a .dist() or a Prior, not an RVFactory")
    check_unnamed_dist(noise, owner="ssoe noise")
    op = noise.owner.op
    params = op.dist_params(noise.owner)
    mu = params[0] if isinstance(op, NormalRV) else params[1] if isinstance(op, StudentTRV) else 0
    if not is_zero_constant(mu):
        msg = (
            "ssoe noise must be zero-centered: its location is not a constant zero. "
            "pm.Normal.dist(sigma) binds sigma as mu; write pm.Normal.dist(0, sigma) "
            "(or pass mu=0 to pm.StudentT.dist)"
        )
        raise HorizonError(msg)


def _accept_prior_noise(noise) -> None:
    nested = sorted(
        key
        for key, value in noise.parameters.items()
        if is_prior_like(value) or (isinstance(value, pt.Variable) and value.owner is not None)
    )
    if nested:
        msg = (
            f"ssoe noise Prior has non-constant parameters {nested}: a Prior is copied "
            "when the forecast-only error is created, so hyper-priors and model "
            "variables in it are never tied to the fitted posterior and the forecast "
            f"draws them from the prior; {_NOISE_PATTERN}"
        )
        raise HorizonError(msg)
    mu = noise.parameters.get("mu", 0)
    zero = (
        is_zero_constant(mu) if isinstance(mu, pt.Variable) else bool(np.all(np.asarray(mu) == 0))
    )
    if noise.distribution in ("Normal", "StudentT") and not zero:
        msg = f"ssoe noise must be zero-centered: leave the Prior's 'mu' unset or 0, got {noise}"
        raise HorizonError(msg)


def _future_noise(name: str, noise, dims: tuple[str, ...]) -> pt.TensorVariable:
    """Expand ``noise`` and register ``{name}_future`` only."""
    future_dims = (FUTURE_DIM, *dims)
    if is_prior_like(noise):
        return prior_rv_factory(noise, name)(f"{name}_future", future_dims)
    model = pm.modelcontext(None)
    try:
        shape = tuple(model.dim_lengths[dim] for dim in future_dims)
    except KeyError as exc:
        msg = f"ssoe requires model coord {exc.args[0]!r}"
        raise HorizonError(msg) from exc
    return model.register_rv(
        expand_dist(noise, shape, owner="ssoe"), f"{name}_future", dims=future_dims
    )


def ssoe(
    h: Horizon,
    name: str,
    y: xr.DataArray | None,
    init,
    mean: Callable,
    update: Callable,
    noise,
    xs: xr.DataArray | None = None,
    *,
    params: Sequence = (),
    dims: tuple[str, ...] | None = None,
) -> SSOEResult:
    """Filter observed values, then simulate a recursive forecast.

    ``mean(state, x, *params)`` and ``update(state, y, error, x, *params)``
    take ``params`` because PyTensor scan cannot close over random variables.
    ``noise`` is not scanned: it is a ``.dist()`` or a ``Prior`` registered
    only as ``{name}_future``. Pass random coefficients in ``params``. This
    helper does not auto-detect closed-over RVs.

    Parameters
    ----------
    h : pymc_forecast.model.Horizon
        The horizon of the current model build.
    name : str
        Base name of the future error variable. Only ``f"{name}_future"`` is
        registered, and only when forecasting.
    y : xarray.DataArray or None
        Labeled driving history. ``None`` uses ``h.data``. Must cover exactly
        the training window (time coords equal to ``h.time``), with finite
        values and non-time coords matching the model coords. A transformed
        history can be supplied to compose multiple recursion channels.
        Prior-only builds need an explicit driving history; this helper does
        not generate an in-sample history.
    init : pytensor.tensor.TensorVariable or tuple of pytensor.tensor.TensorVariable
        Initial state: one tensor-like or a nonempty ``tuple`` of them (only a
        ``tuple`` means multiple states; each is cast to ``floatX``). States may
        have different shapes (e.g. scalar level and vector seasonality).
    mean : callable
        ``(state, x_t, *params) -> mu_t``. Returns the one-step-ahead mean,
        shaped like one row of ``y``. ``x_t`` is ``None`` without ``xs``.
    update : callable
        ``(state, y_t, eps_t, x_t, *params) -> state``. Returns the next state
        with the same structure and shapes as ``init``. In-sample,
        ``eps_t = y_t - mu_t``; in the future, ``eps_t`` is freshly drawn and
        ``y_t = mu_t + eps_t``. Both callbacks must be deterministic.
    noise : pymc_extras.prior.Prior or pytensor.tensor.TensorVariable
        Unnamed ``.dist()`` or pymc-extras ``Prior`` for independent,
        zero-centered per-step future errors, expanded and registered as
        ``f"{name}_future"`` only. An ``RVFactory`` or other callable is
        rejected. Parameters broadcast against ``("time_future", *dims)``, so a
        per-series scale works. Errors may be correlated across the
        observation dimensions, using e.g. ``pm.MvNormal.dist``. In-sample
        errors are residuals, not random variables.

        A learned scale must be a model variable shared with the observation:
        create ``sigma = pm.HalfNormal("sigma", 1)`` in the model body, use it
        in the observation, and pass ``pm.Normal.dist(0, sigma)``. Hence a
        ``Prior`` with a hyper-prior parameter (which would exist only on
        forecasting builds and be drawn from its prior), a model variable
        instead of a ``.dist()``, and a ``.dist()`` depending on unnamed random
        variables are rejected, on training builds too. A Normal/StudentT
        ``Prior`` must leave ``mu`` unset or zero, and a ``pm.Normal.dist`` /
        ``pm.StudentT.dist`` must have a constant zero location
        (``pm.Normal.dist(sigma)`` binds ``sigma`` as ``mu``). The locations of
        other dists are not checked.
    xs : xarray.DataArray, optional
        Labeled inputs spanning the full horizon (``None`` means no inputs).
        The time dimension is selected by name and its coordinates are
        checked; non-time dims are checked only when they are model coords.
        Values must be finite. Future inputs must
        be known covariates or explicit scenarios: never derive future update
        gates from held-out observations. Extra rows are ignored during a
        shorter training build.
    params : sequence, default ``()``
        Tensor parameters passed explicitly to both callbacks. Required for
        random coefficients: scan cannot close over RVs, and closed-over RVs
        are not detected.
    dims : tuple of str, optional
        Non-time observation dimensions; ``None`` infers them from ``y``.
        Labeled data are transposed into this order before entering the scan.

    Returns
    -------
    SSOEResult
        In-sample means, future means and future samples. The caller registers
        ``obs`` against ``result.mu`` and ``forecast`` as a Deterministic of
        ``result.y_future``. Register ``mu`` and ``mu_future`` Deterministics
        to include the means in the standard prediction outputs. Do not add
        another observation draw to ``y_future``: it already includes noise.

    Raises
    ------
    pymc_forecast.exceptions.HorizonError
        If there is no observed history (``y`` and ``h.data`` are both
        ``None``, or ``h.t_obs == 0``); if ``noise`` is invalid (see above;
        checked on every build); or, when forecasting, if a dim in
        ``("time_future", *dims)`` is not a model coord or a ``.dist()`` noise
        has more axes than ``dims``.
    pymc_forecast.exceptions.AlignmentError
        If ``y`` or ``xs`` is not a DataArray with a ``"time"`` dim; if the time
        coords of ``y`` differ from ``h.time``, or ``xs`` is shorter than the
        horizon or its time coords differ; if ``dims`` does not name each
        non-time dim of ``y`` exactly once or contains ``"time_future"``; or if
        a dim's coords do not match the model coords.
    ValueError
        If ``y`` or ``xs`` has non-finite values; if ``init`` is an empty tuple;
        or if ``mean``/``update`` violate their contract (``mean`` with the
        wrong number of dims, ``update`` changing the state structure, count or
        number of dims, or the callbacks creating random updates or model
        variables).

    Notes
    -----
    This is an observation-driven filter, not a latent Markov process. For
    sampled hidden states use :func:`~pymc_forecast.markov.markov_series`;
    for linear-Gaussian hidden states consider the statespace backend
    (:class:`~pymc_forecast.statespace.StatespaceForecaster`).
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
    _accept_noise(noise)
    if h.future == 0:
        return SSOEResult(mu, empty, empty, dims)
    errors = _future_noise(name, noise, dims)
    errors = pt.specify_shape(errors, (h.future, *row_shape))
    registered = set(model.named_vars)
    outputs = scan(
        errors,
        [state[-1] for state in outputs[:n_states]],
        None if inputs is None else inputs[h.t_obs :],
        future=True,
    )
    return SSOEResult(mu, outputs[-2], outputs[-1], dims)

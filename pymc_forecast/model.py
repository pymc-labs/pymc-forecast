"""Model-building core: the train/forecast :class:`Horizon` and the primitives
that register time-series latents and observation variables against it.

The package's central invariant (inherited from pyro / numpyro_forecast): **one
model definition both trains and forecasts**. In-sample time latents live on
variables dimmed ``"time"``; the forecast horizon lives on separate
``{name}_future`` variables dimmed ``"time_future"``. Those future variables
are absent from the fitted posterior, so ``pm.sample_posterior_predictive``
replays the posterior in-sample and draws the future from the prior —
conditioned on the replayed parents (see ``tests/test_replay_mechanism.py``).

A model is a callable ``(covariates, data) -> None`` executed inside a
managed ``pm.Model`` whose coords carry real time coordinates. The horizon is
derived from the *coords*: ``future = len(covariates.time) - len(data.time)``.
"""

import abc
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pymc_forecast.data import (
    FUTURE_DIM,
    TIME_DIM,
    as_dataarray,
    validate_alignment,
)
from pymc_forecast.exceptions import HorizonError
from pymc_forecast._dist import expand_dist
from pymc_forecast.priors import (
    PriorConfig,
    is_prior_like,
    prior_obs_factory,
    prior_rv_factory,
)

__all__ = [
    "EXPECTED_OBSERVATION_FORECAST_VAR",
    "EXPECTED_OBSERVATION_VAR",
    "FORECAST_VAR",
    "MU_FORECAST_VAR",
    "MU_VAR",
    "OBS_VAR",
    "ForecastingModel",
    "Horizon",
    "build_model",
    "innovations",
    "predict",
]

OBS_VAR = "obs"
"""Reserved name of the observed (in-sample) variable registered by :func:`predict`."""

FORECAST_VAR = "forecast"
"""Reserved name of the forecast-horizon variable registered by :func:`predict`."""

EXPECTED_OBSERVATION_VAR = "expected_observation"
"""Reserved name of the in-sample conditional expected observation registered
by :func:`predict`, in observed outcome units and excluding observation noise."""

EXPECTED_OBSERVATION_FORECAST_VAR = "expected_observation_future"
"""Reserved name of the forecast-horizon conditional expected observation
registered by :func:`predict` (see :data:`EXPECTED_OBSERVATION_VAR`)."""

MU_VAR = "mu"
"""Reserved name of the in-sample noise-free latent predictor registered by
:func:`predict` — the latent passed to it, before observation noise (for
GLM-style models this is the linear predictor, not the distribution mean)."""

MU_FORECAST_VAR = "mu_future"
"""Reserved name of the forecast-horizon noise-free latent predictor
registered by :func:`predict` (see :data:`MU_VAR`)."""

RVFactory = Callable[[str, tuple[str, ...]], pt.TensorVariable]
"""``(name, dims) -> RV``: creates a named model variable with exactly these dims."""

ObsFactory = Callable[..., pt.TensorVariable]
"""``(name, latent, dims, observed) -> RV``: creates the observation variable.

``latent`` is the time slice of the full-horizon predictor for this variable
(time on axis 0), ``dims`` the variable's dims, and ``observed`` the observed
values (``None`` for the forecast suffix and during prior-only builds).
"""



@dataclass(frozen=True)
class Horizon:
    """The train/forecast split of a single model build, derived from coords.

    Attributes
    ----------
    data
        Observed data (time-first ``DataArray``), or ``None`` for prior-only
        builds.
    time
        Coordinate values of the observed window (length ``t_obs``).
    time_future
        Coordinate values of the forecast horizon (empty while training).
    """

    data: xr.DataArray | None
    time: np.ndarray
    time_future: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def t_obs(self) -> int:
        """Number of observed (in-sample) time steps."""
        return len(self.time)

    @property
    def future(self) -> int:
        """Number of forecast time steps (``0`` while training)."""
        return len(self.time_future)

    @property
    def duration(self) -> int:
        """Total horizon length ``t_obs + future``."""
        return self.t_obs + self.future

    @classmethod
    def from_data(cls, covariates: xr.DataArray, data: xr.DataArray | None) -> "Horizon":
        """Derive the horizon from normalized data/covariate time coords.

        ``covariates`` span the full horizon; ``data`` (if given) covers the
        observed prefix. With ``data=None`` (prior-only builds) the whole
        covariate span counts as observed time.
        """
        cov_time = np.asarray(covariates[TIME_DIM].values)
        if data is None:
            return cls(data=None, time=cov_time)
        validate_alignment(data, covariates)
        t_obs = data.sizes[TIME_DIM]
        return cls(data=data, time=cov_time[:t_obs], time_future=cov_time[t_obs:])


def _segment_shape(dims: tuple[str, ...]) -> tuple:
    """Lengths of ``dims`` from the active model's coords."""
    model = pm.modelcontext(None)
    try:
        return tuple(model.dim_lengths[dim] for dim in dims)
    except KeyError as exc:
        msg = f"innovations requires model coord {exc.args[0]!r}"
        raise HorizonError(msg) from exc


def _assert_dist_has_no_time_axis(dist: pt.TensorVariable) -> None:
    """Reject a ``.dist()`` that already carries the time axis this helper owns."""
    model = pm.modelcontext(None)
    time_lengths = [
        len(model.coords[dim])
        for dim in (TIME_DIM, FUTURE_DIM)
        if model.coords.get(dim) is not None
    ]
    if any(isinstance(length, int) and length in time_lengths for length in dist.type.shape):
        msg = (
            "innovations owns the time axis; a .dist() that already has a "
            "time dimension is not accepted"
        )
        raise HorizonError(msg)


def innovations(
    h: Horizon,
    name: str,
    dist,
    *,
    dims: tuple[str, ...] = (),
) -> pt.TensorVariable:
    """Sample a per-step latent over the full horizon.

    ``dist`` is a pymc-extras ``Prior`` or an unnamed ``.dist()`` tensor.
    A ``Prior`` follows :func:`~pymc_forecast.priors.prior_rv_factory`,
    including one shared draw of nested hyper-priors. A ``.dist()`` is
    expanded to the segment shape and registered as ``name`` with dims
    ``("time", *dims)`` and, when forecasting, ``{name}_future`` with dims
    ``("time_future", *dims)``.

    Returns the latent over the full horizon, concatenated on axis 0. When
    ``h.future == 0`` the forecast suffix is omitted.
    """
    if is_prior_like(dist):
        rv_fn = prior_rv_factory(dist, name)
        prefix = rv_fn(name, (TIME_DIM, *dims))
        if h.future == 0:
            return prefix
        suffix = rv_fn(f"{name}_future", (FUTURE_DIM, *dims))
        return pt.concatenate([prefix, suffix], axis=0)

    if not isinstance(dist, pt.TensorVariable) or dist.owner is None:
        msg = "innovations expects a Prior or an unnamed .dist() tensor"
        raise HorizonError(msg)
    _assert_dist_has_no_time_axis(dist)
    model = pm.modelcontext(None)
    prefix = model.register_rv(
        expand_dist(dist, _segment_shape((TIME_DIM, *dims))),
        name,
        dims=(TIME_DIM, *dims),
    )
    if h.future == 0:
        return prefix
    suffix = model.register_rv(
        expand_dist(dist, _segment_shape((FUTURE_DIM, *dims))),
        f"{name}_future",
        dims=(FUTURE_DIM, *dims),
    )
    return pt.concatenate([prefix, suffix], axis=0)


def _join_names(names: Sequence[str]) -> str:
    """Render names as a comma-separated list with a final "and"."""
    quoted = [repr(name) for name in names]
    if len(quoted) < 3:
        return " and ".join(quoted)
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


_DIST_LOC_ERROR = (
    "only a zero-centered Normal or StudentT .dist() can be shifted onto the "
    "latent; pass a 1-argument callable (segment_latent) -> unnamed .dist() instead"
)


def _has_observed_parameter(fn) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "observed" in signature.parameters


def _is_zero_constant(var) -> bool:
    from pytensor.graph.basic import Constant

    if not isinstance(var, Constant):
        return False
    return bool(np.all(np.asarray(var.data) == 0))


def _rebuild_zero_centered(dist, latent) -> pt.TensorVariable:
    if not isinstance(dist, pt.TensorVariable) or dist.owner is None:
        raise HorizonError(_DIST_LOC_ERROR)
    op_name = type(dist.owner.op).__name__
    inputs = dist.owner.inputs
    if op_name == "NormalRV":
        if not _is_zero_constant(inputs[2]):
            raise HorizonError(_DIST_LOC_ERROR)
        return pm.Normal.dist(latent, inputs[3])
    if op_name == "StudentTRV":
        if not _is_zero_constant(inputs[3]):
            raise HorizonError(_DIST_LOC_ERROR)
        return pm.StudentT.dist(inputs[2], latent, sigma=inputs[4])
    raise HorizonError(_DIST_LOC_ERROR)


def _register_unnamed(name: str, dist, dims: tuple[str, ...], observed) -> pt.TensorVariable:
    if not isinstance(dist, pt.TensorVariable) or dist.owner is None:
        msg = "observation callable must return an unnamed .dist()"
        raise HorizonError(msg)
    return pm.modelcontext(None).register_rv(dist, name, dims=dims, observed=observed)


def _emit_observation(obs, name: str, segment, dims: tuple[str, ...], observed):
    """Dispatch one observation segment. The 4-argument factory path is unchanged."""
    if callable(obs) and _has_observed_parameter(obs):
        return obs(name, segment, dims, observed)
    if callable(obs):
        return _register_unnamed(name, obs(segment), dims, observed)
    return _register_unnamed(name, _rebuild_zero_centered(obs, segment), dims, observed)


def predict(
    h: Horizon,
    obs,
    latent: pt.TensorVariable,
    *,
    expected_observation: pt.TensorVariable | None = None,
    dims: tuple[str, ...] | None = None,
) -> None:
    """Register the observation and forecast variables of the model.

    ``latent`` is the deterministic full-horizon predictor (time on axis 0).
    The observed prefix becomes the likelihood (``"obs"``, dims
    ``("time", *dims)``); when forecasting, the suffix becomes the unobserved
    ``"forecast"`` variable (dims ``("time_future", *dims)``) that
    ``pm.sample_posterior_predictive`` draws.

    The latent itself is also recorded, noise-free, as ``"mu"`` (in-sample)
    and ``"mu_future"`` (forecast horizon) Deterministics — the documented
    way to separate parameter/latent uncertainty from observation noise (see
    ``docs/schema.md``). For GLM-style models this is the linear predictor
    passed to ``predict``, not the distribution mean. The names ``"mu"`` and
    ``"mu_future"`` are therefore reserved: a model body must not define
    variables with these names.

    Model authors may additionally pass ``expected_observation`` to record the
    conditional expected observation in observed outcome units, excluding
    observation noise. It is emitted as ``"expected_observation"`` and
    ``"expected_observation_future"`` without changing the meaning of
    ``"mu"`` / ``"mu_future"``. This value is explicit because ``predict``
    does not infer inverse links or distribution means from ``obs``.

    This single primitive covers both upstream ``predict`` (location-family
    noise: pass ``lambda name, mu, dims, observed: pm.Normal(name, mu, sigma,
    dims=dims, observed=observed)``) and upstream ``predict_glm`` (any link,
    e.g. ``lambda name, eta, dims, observed: pm.Poisson(name, pt.exp(eta),
    dims=dims, observed=observed)``) — PyMC likelihoods take their parameters
    directly, so no distribution surgery is needed.

    Parameters
    ----------
    h
        The horizon of the current model build.
    obs
        Observation specification, dispatched in order: a pymc-extras
        ``Prior`` (``mu`` left unset; nested hyper-priors shared across
        segments), a callable whose signature has an ``observed`` parameter
        (``(name, latent, dims, observed) -> RV``), any other callable
        (``segment_latent ->`` unnamed ``.dist()``, called once per segment),
        or a zero-centered ``pm.Normal.dist`` / ``pm.StudentT.dist`` whose
        location is replaced by the segment latent. Any other dist, or a
        non-zero location, raises :class:`~pymc_forecast.exceptions.HorizonError`.
    latent
        Full-horizon predictor with time on axis 0.
    expected_observation
        Optional full-horizon conditional expected observation in observed
        outcome units, with time on axis 0. For a Poisson log-link model whose
        ``latent`` is ``eta``, pass ``pt.exp(eta)``. Its generated variable
        names are reserved whether or not this argument is provided.
    dims
        Extra (non-time) dims of the observation. Default: inferred from the
        data's non-time dims (``()`` for prior-only builds).
    """
    if is_prior_like(obs):
        obs = prior_obs_factory(obs, OBS_VAR)
    if dims is None:
        dims = () if h.data is None else tuple(d for d in h.data.dims if d != TIME_DIM)
    observed = None if h.data is None else h.data.transpose(TIME_DIM, ...).values
    model = pm.modelcontext(None)
    reserved = {
        MU_VAR,
        MU_FORECAST_VAR,
        EXPECTED_OBSERVATION_VAR,
        EXPECTED_OBSERVATION_FORECAST_VAR,
    }
    taken = reserved.intersection(model.named_vars)
    if taken:
        msg = (
            f"the model already defines {sorted(taken)}; predict() reserves "
            f"{_join_names(sorted(reserved))} for "
            "generated predictive outputs — rename the model variable"
        )
        raise HorizonError(msg)
    # Broadcast the recorded predictors over the full panel before slicing.
    # Broadcasting an already-sliced singleton panel axis can be rewritten
    # incorrectly by PyTensor 3.3, producing a shape error for mu_future.
    shape = (h.duration, *(model.dim_lengths[d] for d in dims))
    latent_output = pt.broadcast_to(latent, shape)
    expected_output = (
        None if expected_observation is None else pt.broadcast_to(expected_observation, shape)
    )
    _emit_observation(obs, OBS_VAR, latent[: h.t_obs], (TIME_DIM, *dims), observed)
    pm.Deterministic(MU_VAR, latent_output[: h.t_obs], dims=(TIME_DIM, *dims))
    if expected_output is not None:
        pm.Deterministic(
            EXPECTED_OBSERVATION_VAR,
            expected_output[: h.t_obs],
            dims=(TIME_DIM, *dims),
        )
    if h.future > 0:
        _emit_observation(obs, FORECAST_VAR, latent[h.t_obs :], (FUTURE_DIM, *dims), None)
        pm.Deterministic(MU_FORECAST_VAR, latent_output[h.t_obs :], dims=(FUTURE_DIM, *dims))
        if expected_output is not None:
            pm.Deterministic(
                EXPECTED_OBSERVATION_FORECAST_VAR,
                expected_output[h.t_obs :],
                dims=(FUTURE_DIM, *dims),
            )


ModelFunction = Callable[[xr.DataArray, xr.DataArray | None], None]
"""A model body: ``(covariates, data) -> None``, called inside a ``pm.Model``."""


class ForecastingModel(PriorConfig, abc.ABC):
    """Object-oriented facade over the functional primitives.

    Subclasses implement :meth:`model` and use the bound helpers
    :meth:`innovations` / :meth:`predict`, which thread the current
    :class:`Horizon` automatically. An instance is a valid model function for
    :func:`build_model` and the forecaster classes.

    Priors are user-injectable (see :class:`~pymc_forecast.priors.PriorConfig`):
    a subclass declares its overridable defaults in
    :attr:`~pymc_forecast.priors.PriorConfig.default_priors` and reads
    ``self.prior_config[...]`` in the model body; callers override any subset
    at construction time::

        from pymc_extras.prior import Prior

        class LocalLevel(ForecastingModel):
            default_priors = {
                "drift": Prior("Normal", mu=0, sigma=0.1),
                "noise": Prior("Normal", sigma=Prior("HalfNormal", sigma=1)),
            }

            def model(self, covariates, data=None):
                drift = self.innovations("drift", self.prior_config["drift"])
                self.predict(self.prior_config["noise"], pt.cumsum(drift))

        LocalLevel(priors={"drift": Prior("StudentT", nu=4, mu=0, sigma=0.2)})
    """

    _horizon: Horizon | None = None

    @abc.abstractmethod
    def model(self, covariates, data=None) -> None:
        """Define the generative model; call :meth:`predict` exactly once."""

    @property
    def horizon(self) -> Horizon:
        """The :class:`Horizon` of the model build currently in progress."""
        return self._require_horizon()

    def _require_horizon(self) -> Horizon:
        if self._horizon is None:
            msg = "horizon is only available during a model build"
            raise HorizonError(msg)
        return self._horizon

    def innovations(self, name, dist, *, dims=()) -> pt.TensorVariable:
        """Bound :func:`innovations` using the current build's horizon."""
        return innovations(self._require_horizon(), name, dist, dims=dims)

    def predict(
        self,
        obs,
        latent: pt.TensorVariable,
        *,
        expected_observation: pt.TensorVariable | None = None,
        dims: tuple[str, ...] | None = None,
    ) -> None:
        """Bound :func:`predict` using the current build's horizon."""
        predict(
            self._require_horizon(),
            obs,
            latent,
            expected_observation=expected_observation,
            dims=dims,
        )

    def __call__(self, covariates, data=None) -> None:
        """Run the model body with the horizon bound (used by :func:`build_model`)."""
        self._horizon = Horizon.from_data(covariates, data)
        try:
            self.model(covariates, data)
        finally:
            self._horizon = None


def _coords_from(arrays: Iterable[xr.DataArray | None]) -> dict[str, np.ndarray]:
    """Collect coords of every non-time dim of the given arrays."""
    coords: dict[str, np.ndarray] = {}
    for da in arrays:
        if da is None:
            continue
        for dim in da.dims:
            if dim == TIME_DIM or dim in coords:
                continue
            values = da[dim].values if dim in da.coords else np.arange(da.sizes[dim])
            coords[str(dim)] = values
    return coords


def build_model(
    model_fn: ModelFunction | ForecastingModel,
    data,
    covariates,
    *,
    coords: Mapping[str, object] | None = None,
) -> pm.Model:
    """Build a ``pm.Model`` from a model function and (data, covariates).

    The horizon is derived from the time coords: covariates span the full
    horizon, data covers the observed prefix. Registered coords: ``"time"``
    (observed steps), ``"time_future"`` (forecast steps, only when
    forecasting), every non-time dim of data/covariates, plus any user
    ``coords``.

    Parameters
    ----------
    model_fn
        The model body ``(covariates, data) -> None`` or a
        :class:`ForecastingModel` instance.
    data
        Observed data (DataArray / Series / DataFrame / ndarray), or ``None``
        for a prior-only build over the whole covariate span.
    covariates
        Covariates spanning the full horizon (use
        :func:`~pymc_forecast.data.null_covariates` if the model has none).
    coords
        Extra coords to register on the model.
    """
    cov_da = as_dataarray(covariates, role="covariates")
    data_da = None if data is None else as_dataarray(data, role="data")
    h = Horizon.from_data(cov_da, data_da)

    model_coords: dict[str, object] = {TIME_DIM: h.time}
    if h.future > 0:
        model_coords[FUTURE_DIM] = h.time_future
    model_coords.update(_coords_from([data_da, cov_da]))
    if coords:
        model_coords.update(coords)

    with pm.Model(coords=model_coords) as model:
        model_fn(cov_da, data_da)
    if OBS_VAR not in model.named_vars:
        msg = (
            f"the model registered no '{OBS_VAR}' variable; call predict() "
            "exactly once in the model body"
        )
        raise HorizonError(msg)
    return model

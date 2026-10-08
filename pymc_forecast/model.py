"""Model-building core: the train/forecast Horizon and the primitives built on it.

The primitives here register time-series latents and observation variables
against the :class:`Horizon`.

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
import functools
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.random.basic import NormalRV, StudentTRV

from pymc_forecast._dist import (
    check_unnamed_dist,
    expand_dist,
    is_dist,
    is_zero_constant,
    resize_dist,
)
from pymc_forecast.data import (
    FUTURE_DIM,
    TIME_DIM,
    as_dataarray,
    validate_alignment,
)
from pymc_forecast.exceptions import HorizonError
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
"""Name of the observed (in-sample) variable registered by :func:`predict`.

Also registered by :func:`~pymc_forecast.gaussian.predict_mvn`.
"""

FORECAST_VAR = "forecast"
"""Name of the forecast-horizon variable registered by :func:`predict`.

Also registered by :func:`~pymc_forecast.gaussian.predict_mvn`.
"""

EXPECTED_OBSERVATION_VAR = "expected_observation"
"""Reserved name of the in-sample expected observation registered by :func:`predict`.

It is the conditional expected observation in observed outcome units,
excluding observation noise.
"""

EXPECTED_OBSERVATION_FORECAST_VAR = "expected_observation_future"
"""Reserved name of the forecast-horizon expected observation registered by :func:`predict`.

See :data:`EXPECTED_OBSERVATION_VAR`.
"""

MU_VAR = "mu"
"""Reserved name of the in-sample noise-free latent predictor registered by :func:`predict`.

It is the latent passed to :func:`predict`, before observation noise (for
GLM-style models this is the linear predictor, not the distribution mean).
"""

MU_FORECAST_VAR = "mu_future"
"""Reserved name of the forecast-horizon noise-free latent predictor from :func:`predict`.

See :data:`MU_VAR`.
"""

RVFactory = Callable[[str, tuple[str, ...]], pt.TensorVariable]
"""``(name, dims) -> RV``: creates the named model variable ``name`` with these dims.

Produced by :func:`~pymc_forecast.priors.prior_rv_factory`;
:func:`innovations` and ``ssoe`` do not accept such callables directly.
"""

ObsFactory = Callable[..., pt.TensorVariable]
"""``(name, latent, dims, observed) -> RV``: creates the observation variable.

``latent`` is the time slice of the full-horizon predictor for this variable
(time on axis 0), ``dims`` the variable's dims, and ``observed`` the observed
values (``None`` for the forecast suffix and during prior-only builds).
"""


@dataclass(frozen=True)
class Horizon:
    """The train/forecast split of a single model build, derived from coords.

    Each parameter is available as an attribute of the same name.

    Parameters
    ----------
    data : xarray.DataArray or None
        Observed data, stored as given (time-first when built by
        :func:`build_model`), or ``None`` for prior-only builds.
    time : numpy.ndarray
        Coordinate values of the in-sample window (length ``t_obs``); in
        prior-only builds, the whole covariate span.
    time_future : numpy.ndarray, optional
        Coordinate values of the forecast horizon. Empty whenever there is no
        horizon. Defaults to an empty array.
    """

    data: xr.DataArray | None
    time: np.ndarray
    time_future: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def t_obs(self) -> int:
        """Number of in-sample time steps (the full covariate span in prior-only builds)."""
        return len(self.time)

    @property
    def future(self) -> int:
        """Number of forecast time steps (``0`` when there is no forecast horizon)."""
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
        covariate span counts as observed time. No normalization is applied
        here (:func:`build_model` normalizes with
        :func:`~pymc_forecast.data.as_dataarray` first).

        Parameters
        ----------
        covariates : xarray.DataArray
            Normalized covariates with a ``"time"`` coord spanning the full
            horizon.
        data : xarray.DataArray or None
            Normalized observed data with a ``"time"`` dim, or ``None`` for a
            prior-only build.

        Returns
        -------
        Horizon
            The derived horizon; ``data`` is stored as given.

        Raises
        ------
        pymc_forecast.exceptions.AlignmentError
            If ``data`` is given and is not aligned with ``covariates``
            (see :func:`~pymc_forecast.data.validate_alignment`).
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
    including one shared draw of nested hyper-priors.
    A ``.dist()`` is resized to ``("time", *dims)`` and registered as ``name``,
    plus ``("time_future", *dims)`` as ``{name}_future`` when forecasting.
    Its parameters broadcast against the trailing ``dims`` (e.g. a
    per-series scale), and a multivariate dist's support fills the last
    ``dims``. The time axis belongs to this helper, so a ``.dist()`` with
    more axes than ``dims`` is rejected. So are a model variable (pass
    ``pm.Normal.dist(...)``, not ``pm.Normal(name, ...)``) and a ``.dist()``
    whose parameters are unnamed random variables: create a random scale as
    a model variable (``sigma = pm.HalfNormal("sigma", 1)``) and pass
    ``pm.Normal.dist(0, sigma)``.

    Parameters
    ----------
    h : Horizon
        The horizon of the current model build.
    name : str
        Base variable name of the in-sample latent; the forecast segment is
        registered as ``f"{name}_future"``.
    dist : pymc_extras.prior.Prior or pytensor.tensor.TensorVariable
        A pymc-extras ``Prior`` or an unnamed ``.dist()`` tensor.
    dims : tuple of str, default ``()``
        Extra (non-time) dims of the per-step latent; each must be a model
        coord.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The latent over the full horizon, time on axis 0. When
        ``h.future == 0`` this is the registered model variable ``name``
        itself; otherwise it is an unnamed concatenation of ``name`` and
        ``{name}_future`` with shape ``(h.duration, *dim lengths)``.

    Raises
    ------
    pymc_forecast.exceptions.HorizonError
        If ``dist`` is neither a ``Prior`` nor an unnamed ``.dist()``, is a
        model variable, depends on unnamed random variables, or has more axes
        than ``dims``; if a ``.dist()`` latent needs a dim that is not a model
        coord; or if a ``Prior`` has a time-dimmed hyper-prior.
    """
    if is_prior_like(dist):
        rv_fn = prior_rv_factory(dist, name)
        prefix = rv_fn(name, (TIME_DIM, *dims))
        if h.future == 0:
            return prefix
        suffix = rv_fn(f"{name}_future", (FUTURE_DIM, *dims))
        return pt.concatenate([prefix, suffix], axis=0)

    if not is_dist(dist):
        msg = "innovations expects a Prior or an unnamed .dist() tensor"
        raise HorizonError(msg)
    check_unnamed_dist(dist, owner="innovations")
    model = pm.modelcontext(None)
    prefix = model.register_rv(
        expand_dist(dist, _segment_shape((TIME_DIM, *dims)), owner="innovations"),
        name,
        dims=(TIME_DIM, *dims),
    )
    if h.future == 0:
        return prefix
    suffix = model.register_rv(
        expand_dist(dist, _segment_shape((FUTURE_DIM, *dims)), owner="innovations"),
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


def _dist_classmethod(fn) -> type | None:
    """The PyMC distribution class behind a bare ``.dist`` classmethod.

    Also unwraps ``functools.partial``. Covers ``pm.Distribution`` subclasses
    and the plain helper classes in ``pymc.distributions`` (zero-inflated,
    hurdle, ``NormalMixture``).
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    owner = getattr(fn, "__self__", None)
    if (
        inspect.ismethod(fn)
        and fn.__name__ == "dist"
        and isinstance(owner, type)
        and (
            issubclass(owner, pm.Distribution) or owner.__module__.startswith("pymc.distributions")
        )
    ):
        return owner
    return None


def _first_positional_parameter(fn) -> str | None:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return next((p.name for p in parameters if p.kind in kinds), None)


def _is_observation_factory(fn) -> bool:
    """Whether ``fn`` is the 4-argument ``(name, latent, dims, observed)`` factory.

    A bare PyMC ``.dist`` classmethod never is. Any other callable is the
    factory iff its signature has at least four positional parameters, or
    binds four positional arguments and cannot bind one (e.g. a
    ``functools.partial``). Decided by arity, not parameter names. Callables
    with fewer than four positional parameters that also accept a single
    argument (e.g. through ``*args``), and callables whose signature cannot be
    inspected, take the 1-argument path.
    """
    if _dist_classmethod(fn) is not None:
        return False
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if sum(p.kind in kinds for p in signature.parameters.values()) >= 4:
        return True
    try:
        signature.bind(None, None, None, None)
    except TypeError:
        return False
    try:
        signature.bind(None)
    except TypeError:
        return True
    return False


def _check_dist_classmethod(fn) -> None:
    """Reject a bare ``.dist`` classmethod whose first parameter is not ``mu``."""
    cls = _dist_classmethod(fn)
    if cls is None:
        return
    first = _first_positional_parameter(fn)
    if first == "mu":
        return
    name = cls.__name__
    msg = (
        f"pm.{name}.dist takes {first!r} as its first parameter; a bare .dist "
        "classmethod is accepted only when its first parameter is 'mu'. Pass a "
        "1-argument callable that binds the latent by keyword, e.g. "
        f"lambda latent: pm.{name}.dist(..., <parameter>=latent)"
    )
    raise HorizonError(msg)


def _rebuild_zero_centered(dist, latent) -> pt.TensorVariable:
    if not is_dist(dist):
        raise HorizonError(_DIST_LOC_ERROR)
    op = dist.owner.op
    params = op.dist_params(dist.owner)
    if isinstance(op, NormalRV) and is_zero_constant(params[0]):
        check_unnamed_dist(dist, owner="predict")
        return pm.Normal.dist(latent, params[1])
    if isinstance(op, StudentTRV) and is_zero_constant(params[1]):
        check_unnamed_dist(dist, owner="predict")
        nu, _, sigma = params
        return pm.StudentT.dist(nu, latent, sigma=sigma)
    raise HorizonError(_DIST_LOC_ERROR)


def _register_unnamed(name: str, dist, dims: tuple[str, ...], observed) -> pt.TensorVariable:
    if not is_dist(dist):
        msg = "observation callable must return an unnamed .dist()"
        raise HorizonError(msg)
    check_unnamed_dist(dist, owner="predict")
    model = pm.modelcontext(None)
    shape = tuple(model.dim_lengths[dim] for dim in dims)
    return model.register_rv(resize_dist(dist, shape), name, dims=dims, observed=observed)


def _call_one_argument(obs, segment) -> pt.TensorVariable:
    """Call ``obs(segment)``, rejecting model variables it creates."""
    _check_dist_classmethod(obs)
    model = pm.modelcontext(None)
    before = set(model.named_vars)
    dist = obs(segment)
    created = sorted(set(model.named_vars) - before)
    if created:
        msg = (
            f"the observation callable created model variables {_join_names(created)}; "
            "it is called once per segment and must not create model variables — "
            "create them in the model body and close over them"
        )
        raise HorizonError(msg)
    return dist


def _emit_observation(obs, name: str, segment, dims: tuple[str, ...], observed):
    """Dispatch one observation segment. The 4-argument factory path is unchanged."""
    if callable(obs) and _is_observation_factory(obs):
        return obs(name, segment, dims, observed)
    if callable(obs):
        return _register_unnamed(name, _call_one_argument(obs, segment), dims, observed)
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
    h : Horizon
        The horizon of the current model build.
    obs : pymc_extras.prior.Prior, callable or pytensor.tensor.TensorVariable
        Observation specification, dispatched in order:

        - a pymc-extras ``Prior`` (``mu`` must be left unset; nested
          hyper-priors are shared across segments);
        - a 4-argument factory ``(name, latent, dims, observed) -> RV``: a
          callable with at least four positional parameters, or one that
          binds four positional arguments but not one (e.g. a
          ``functools.partial``). Parameter names do not matter;
        - any other callable, called once per segment with the segment
          latent as its first positional argument and returning an unnamed
          ``.dist()``. A bare ``.dist`` classmethod is accepted only when that
          parameter is ``mu`` (e.g. ``pm.Poisson.dist``); ``pm.StudentT.dist``
          takes ``nu`` first and is rejected. The callable must not create
          model variables: create them in the model body and close over them;
        - a zero-centered ``pm.Normal.dist`` / ``pm.StudentT.dist`` whose
          location is replaced by the segment latent.
    latent : pytensor.tensor.TensorVariable
        Full-horizon predictor with time on axis 0; its length along axis 0
        must be ``h.duration``.
    expected_observation : pytensor.tensor.TensorVariable, optional
        Full-horizon conditional expected observation in observed outcome
        units, with time on axis 0. For a Poisson log-link model whose
        ``latent`` is ``eta``, pass ``pt.exp(eta)``. ``None`` registers no
        expected-observation variables; their names are reserved either way.
    dims : tuple of str, optional
        Extra (non-time) dims of the observation. ``None`` infers them from
        the data's non-time dims (``()`` for prior-only builds).

    Raises
    ------
    pymc_forecast.exceptions.HorizonError
        If the model already defines ``"mu"``, ``"mu_future"``,
        ``"expected_observation"`` or ``"expected_observation_future"``
        (including a second ``predict`` call in one model); or if ``obs`` is
        invalid: any other dist, a non-zero location, a bare ``.dist``
        classmethod whose first parameter is not ``mu``, a callable that
        creates model variables or does not return a ``.dist()``, a model
        variable instead of a ``.dist()``, a ``.dist()`` depending on unnamed
        random variables, or a ``Prior`` with a time-dimmed hyper-prior.
    ValueError
        If ``obs`` is a ``Prior`` with ``mu`` set.
    KeyError
        If an entry of ``dims`` is not a model coord.
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
    :meth:`innovations`, :meth:`predict` and :meth:`markov_series`, which thread
    the current :class:`Horizon` (available as :attr:`horizon`) automatically.
    An instance is a valid model function for :func:`build_model` and the
    forecaster classes.

    Priors are user-injectable (see :class:`~pymc_forecast.priors.PriorConfig`):
    a subclass declares its overridable defaults in
    :attr:`~pymc_forecast.priors.PriorConfig.default_priors` and reads
    ``self.prior_config[...]`` in the model body; callers override any subset
    at construction time::

        import pytensor.tensor as pt
        from pymc_extras.prior import Prior

        from pymc_forecast.model import ForecastingModel

        class LocalLevel(ForecastingModel):
            default_priors = {
                "drift": Prior("Normal", mu=0, sigma=0.1),
                "noise": Prior("Normal", sigma=Prior("HalfNormal", sigma=1)),
            }

            def model(self, covariates, data=None):
                drift = self.innovations("drift", self.prior_config["drift"])
                self.predict(self.prior_config["noise"], pt.cumsum(drift))

        LocalLevel(priors={"drift": Prior("StudentT", nu=4, mu=0, sigma=0.2)})

    Parameters
    ----------
    priors : mapping, optional
        Named prior overrides merged over ``default_priors``. ``None`` keeps
        the defaults.
    """

    _horizon: Horizon | None = None

    @abc.abstractmethod
    def model(self, covariates, data=None) -> None:
        """Define the generative model.

        The body must register the ``"obs"`` variable exactly once, normally by
        calling :meth:`predict` once (a second call raises
        :class:`~pymc_forecast.exceptions.HorizonError`).

        Parameters
        ----------
        covariates : xarray.DataArray
            Normalized (time-first) covariates spanning the full horizon.
        data : xarray.DataArray, optional
            Normalized (time-first) observed data; ``None`` for prior-only
            builds.
        """

    @property
    def horizon(self) -> Horizon:
        """The :class:`Horizon` of the model build currently in progress.

        Raises :class:`~pymc_forecast.exceptions.HorizonError` when accessed
        outside a model build.
        """
        return self._require_horizon()

    def _require_horizon(self) -> Horizon:
        if self._horizon is None:
            msg = "horizon is only available during a model build"
            raise HorizonError(msg)
        return self._horizon

    def innovations(self, name, dist, *, dims=()) -> pt.TensorVariable:
        """Bound :func:`~pymc_forecast.model.innovations` using the current build's horizon.

        Parameters
        ----------
        name : str
            Base variable name; the forecast segment is ``f"{name}_future"``.
        dist : pymc_extras.prior.Prior or pytensor.tensor.TensorVariable
            A pymc-extras ``Prior`` or an unnamed ``.dist()`` tensor.
        dims : tuple of str, default ``()``
            Extra (non-time) dims of the per-step latent; each must be a model
            coord.

        Returns
        -------
        pytensor.tensor.TensorVariable
            The latent over the full horizon, time on axis 0 (see
            :func:`~pymc_forecast.model.innovations`).

        Raises
        ------
        pymc_forecast.exceptions.HorizonError
            If called outside a model build, or for the conditions listed in
            :func:`~pymc_forecast.model.innovations`.
        """
        return innovations(self._require_horizon(), name, dist, dims=dims)

    def predict(
        self,
        obs,
        latent: pt.TensorVariable,
        *,
        expected_observation: pt.TensorVariable | None = None,
        dims: tuple[str, ...] | None = None,
    ) -> None:
        """Bound :func:`~pymc_forecast.model.predict` using the current build's horizon.

        Parameters
        ----------
        obs : pymc_extras.prior.Prior, callable or pytensor.tensor.TensorVariable
            Observation specification (see :func:`~pymc_forecast.model.predict`).
        latent : pytensor.tensor.TensorVariable
            Full-horizon predictor with time on axis 0 and length
            ``horizon.duration``.
        expected_observation : pytensor.tensor.TensorVariable, optional
            Full-horizon conditional expected observation in observed outcome
            units. ``None`` registers no expected-observation variables.
        dims : tuple of str, optional
            Extra (non-time) dims of the observation. ``None`` infers them from
            the data's non-time dims (``()`` for prior-only builds).

        Raises
        ------
        pymc_forecast.exceptions.HorizonError
            If called outside a model build, or for the conditions listed in
            :func:`~pymc_forecast.model.predict`.
        ValueError
            If ``obs`` is a ``Prior`` with ``mu`` set.
        KeyError
            If an entry of ``dims`` is not a model coord.
        """
        predict(
            self._require_horizon(),
            obs,
            latent,
            expected_observation=expected_observation,
            dims=dims,
        )

    def markov_series(self, name, init, transition, *, params=(), xs=None, dims=()):
        """Bound :func:`~pymc_forecast.markov.markov_series` using this build's horizon.

        Parameters
        ----------
        name : str
            Base variable name; the forecast segment is ``f"{name}_future"``.
        init : float, array_like or pytensor.tensor.TensorVariable
            Initial state fed to the first transition (cast to float64).
        transition : callable
            ``(z_prev, *params) -> dist`` (with ``xs``:
            ``(z_prev, x_t, *params) -> dist``) returning a ``.dist()``.
        params : sequence, default ``()``
            Random variables the transition uses, passed as explicit inputs.
        xs : array_like, optional
            Exogenous inputs, positional with time on axis 0 and at least
            ``horizon.duration`` rows. ``None`` means no inputs.
        dims : tuple of str, default ``()``
            Extra (non-time) dims of the per-step state.

        Returns
        -------
        pytensor.tensor.TensorVariable
            The latent over the full horizon, time on axis 0 (see
            :func:`~pymc_forecast.markov.markov_series`).

        Raises
        ------
        pymc_forecast.exceptions.HorizonError
            If called outside a model build, or for the conditions listed in
            :func:`~pymc_forecast.markov.markov_series`.
        """
        from pymc_forecast.markov import markov_series

        return markov_series(
            self.horizon,
            name,
            init,
            transition,
            params=params,
            xs=xs,
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
    forecasting), every non-time dim of data/covariates (data dims take
    precedence over covariate dims of the same name; unlabeled dims get
    integer coords), plus any user ``coords``, which are applied last and
    override the derived coords.

    Parameters
    ----------
    model_fn : callable or ForecastingModel
        The model body ``(covariates, data) -> None`` or a
        :class:`ForecastingModel` instance. It is called with the normalized
        (time-first) ``xarray.DataArray`` covariates and data (or ``None``).
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim). ``None`` gives a prior-only build over the
        whole covariate span (``obs`` is then unobserved).
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like
        Covariates spanning the full horizon, normalized with ``as_dataarray``
        (2-D input gets a ``"covariate"`` dim). Rows past the end of ``data``
        define the forecast horizon (use
        :func:`~pymc_forecast.data.null_covariates` if the model has none).
    coords : mapping, optional
        Extra coords to register on the model; they override derived coords of
        the same name. ``None`` adds none.

    Returns
    -------
    pymc.Model
        The built model (its context is no longer active on return).

    Raises
    ------
    pymc_forecast.exceptions.AlignmentError
        If ``data`` or ``covariates`` cannot be normalized (a DataArray without
        a ``"time"`` dim, or an array that is not 1-D or 2-D), or if they are
        misaligned (see :meth:`Horizon.from_data`).
    pymc_forecast.exceptions.HorizonError
        If the model body registered no ``"obs"`` variable. Errors raised by
        the model body itself (e.g. by :func:`predict` or :func:`innovations`)
        propagate unchanged.
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

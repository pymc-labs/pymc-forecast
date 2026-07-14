"""Model-building core: the train/forecast :class:`Horizon` and the primitives
that register time-series latents and observation variables against it.

The package's central invariant (inherited from pyro / numpyro_forecast): **one
model definition both trains and forecasts**. In-sample time latents live on
variables dimmed ``"time"``; the forecast horizon lives on separate
``{name}_future`` variables dimmed ``"time_future"``. Those future variables
are absent from the fitted posterior, so ``pm.sample_posterior_predictive``
replays the posterior in-sample and draws the future from the prior —
conditioned on the replayed parents (see ``tests/test_replay_mechanism.py``).

A model is a callable ``(Horizon, covariates) -> None`` executed inside a
managed ``pm.Model`` whose coords carry real time coordinates. The horizon is
derived from the *coords*: ``future = len(covariates.time) - len(data.time)``.
"""

import abc
from collections.abc import Callable, Iterable, Mapping
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
from pymc_forecast.priors import (
    PriorConfig,
    is_prior_like,
    prior_obs_factory,
    prior_rv_factory,
)

__all__ = [
    "FORECAST_VAR",
    "OBS_VAR",
    "ForecastingModel",
    "Horizon",
    "build_model",
    "predict",
    "time_series",
]

OBS_VAR = "obs"
"""Reserved name of the observed (in-sample) variable registered by :func:`predict`."""

FORECAST_VAR = "forecast"
"""Reserved name of the forecast-horizon variable registered by :func:`predict`."""

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
    def from_arrays(cls, covariates: xr.DataArray, data: xr.DataArray | None) -> "Horizon":
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


def time_series(
    h: Horizon,
    name: str,
    rv_fn: RVFactory,
    *,
    dims: tuple[str, ...] = (),
) -> pt.TensorVariable:
    """Sample a per-step latent over the full horizon.

    Calls ``rv_fn(name, ("time", *dims))`` for the observed window and — when
    forecasting — ``rv_fn(f"{name}_future", ("time_future", *dims))`` for the
    horizon, concatenating along time (axis 0). The future variable is what
    keeps the posterior blind to the horizon (see module docstring).

    Parameters
    ----------
    h
        The horizon of the current model build.
    name
        Base variable name for the in-sample latent.
    rv_fn
        Factory creating the variable, e.g.
        ``lambda name, dims: pm.Normal(name, 0, drift_scale, dims=dims)``.
        It must create the variable with exactly the dims it is given.
        A pymc-extras ``Prior`` is also accepted (e.g. ``Prior("Normal",
        mu=0, sigma=0.1)``); nested hyper-priors are created once under
        ``name`` and shared by the in-sample and future segments (see
        :mod:`pymc_forecast.priors`).
    dims
        Extra (non-time) dims of the latent, e.g. ``("series",)``.

    Returns
    -------
    TensorVariable
        The latent over the full horizon, time on axis 0.
    """
    if is_prior_like(rv_fn):
        rv_fn = prior_rv_factory(rv_fn, name)
    prefix = rv_fn(name, (TIME_DIM, *dims))
    if h.future == 0:
        return prefix
    suffix = rv_fn(f"{name}_future", (FUTURE_DIM, *dims))
    return pt.concatenate([prefix, suffix], axis=0)


def predict(
    h: Horizon,
    obs_fn: ObsFactory,
    latent: pt.TensorVariable,
    *,
    dims: tuple[str, ...] | None = None,
) -> None:
    """Register the observation and forecast variables of the model.

    ``latent`` is the deterministic full-horizon predictor (time on axis 0).
    The observed prefix becomes the likelihood (``"obs"``, dims
    ``("time", *dims)``); when forecasting, the suffix becomes the unobserved
    ``"forecast"`` variable (dims ``("time_future", *dims)``) that
    ``pm.sample_posterior_predictive`` draws.

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
    obs_fn
        Observation factory ``(name, latent, dims, observed) -> RV``. Must
        create the variable with exactly the dims it is given, and pass
        ``observed`` through. A pymc-extras ``Prior`` is also accepted (e.g.
        ``Prior("Normal", sigma=Prior("HalfNormal", sigma=1))``): its
        distribution becomes the likelihood with the latent as location, and
        nested hyper-priors are created once under ``"obs"`` and shared by
        the observed and forecast segments (see :mod:`pymc_forecast.priors`).
    latent
        Full-horizon predictor with time on axis 0.
    dims
        Extra (non-time) dims of the observation. Default: inferred from the
        data's non-time dims (``()`` for prior-only builds).
    """
    if is_prior_like(obs_fn):
        obs_fn = prior_obs_factory(obs_fn, OBS_VAR)
    if dims is None:
        dims = () if h.data is None else tuple(d for d in h.data.dims if d != TIME_DIM)
    observed = None if h.data is None else h.data.transpose(TIME_DIM, ...).values
    obs_fn(OBS_VAR, latent[: h.t_obs], (TIME_DIM, *dims), observed)
    if h.future > 0:
        obs_fn(FORECAST_VAR, latent[h.t_obs :], (FUTURE_DIM, *dims), None)


ModelFunction = Callable[[Horizon, xr.DataArray], None]
"""A model body: ``(Horizon, covariates) -> None``, called inside a ``pm.Model``."""


class ForecastingModel(PriorConfig, abc.ABC):
    """Object-oriented facade over the functional primitives.

    Subclasses implement :meth:`model` and use the bound helpers
    :meth:`time_series` / :meth:`predict`, which thread the current
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

            def model(self, h, covariates):
                drift = self.time_series("drift", self.prior_config["drift"])
                self.predict(self.prior_config["noise"], pt.cumsum(drift))

        LocalLevel(priors={"drift": Prior("StudentT", nu=4, mu=0, sigma=0.2)})
    """

    _horizon: Horizon | None = None

    @abc.abstractmethod
    def model(self, h: Horizon, covariates: xr.DataArray) -> None:
        """Define the generative model; call :meth:`predict` exactly once."""

    def _require_horizon(self) -> Horizon:
        if self._horizon is None:
            msg = "horizon is only available during a model build"
            raise HorizonError(msg)
        return self._horizon

    def time_series(
        self, name: str, rv_fn: RVFactory, *, dims: tuple[str, ...] = ()
    ) -> pt.TensorVariable:
        """Bound :func:`time_series` using the current build's horizon."""
        return time_series(self._require_horizon(), name, rv_fn, dims=dims)

    def predict(
        self,
        obs_fn: ObsFactory,
        latent: pt.TensorVariable,
        *,
        dims: tuple[str, ...] | None = None,
    ) -> None:
        """Bound :func:`predict` using the current build's horizon."""
        predict(self._require_horizon(), obs_fn, latent, dims=dims)

    def __call__(self, h: Horizon, covariates: xr.DataArray) -> None:
        """Run the model body with the horizon bound (used by :func:`build_model`)."""
        self._horizon = h
        try:
            self.model(h, covariates)
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
        The model body ``(Horizon, covariates) -> None`` or a
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
    h = Horizon.from_arrays(cov_da, data_da)

    model_coords: dict[str, object] = {TIME_DIM: h.time}
    if h.future > 0:
        model_coords[FUTURE_DIM] = h.time_future
    model_coords.update(_coords_from([data_da, cov_da]))
    if coords:
        model_coords.update(coords)

    with pm.Model(coords=model_coords) as model:
        model_fn(h, cov_da)
    if OBS_VAR not in model.named_vars:
        msg = (
            f"the model registered no '{OBS_VAR}' variable; call predict() "
            "exactly once in the model body"
        )
        raise HorizonError(msg)
    return model

"""Interop with ``pymc_extras.statespace`` models.

Statespace structural models (level/trend, seasonality, cycles, AR, regression;
SARIMAX; VARMAX) cover the linear-Gaussian slice of what
:func:`~pymc_forecast.markov.markov_time_series` is used for — with
Kalman-filter marginalization instead of sampling per-step latents, which
usually gives better posteriors *and* faster sampling.

Their lifecycle differs from this package's model functions: components are
combined and ``.build()`` into a ``PyMCStateSpace``, parameter priors are
declared inside a ``pm.Model`` carrying the statespace coords, and
``build_statespace_graph`` inserts the Kalman-filter likelihood. The
:class:`StatespaceForecaster` adapter maps that lifecycle onto the package's
``fit + forecast(horizon)`` protocol, so a statespace model drops into
:func:`~pymc_forecast.evaluate.backtest` and the metrics layer wherever a
hand-written forecasting model is accepted.

Coords stay this package's responsibility: the statespace internals only ever
see positional (integer-indexed) data, and real time coordinates are stamped
onto the outputs — ``"time_future"`` on forecasts, ``"time"`` on in-sample
predictions — exactly like the core forecasters.
"""

import abc
import warnings
from collections.abc import Mapping

import numpy as np
import pymc as pm
import xarray as xr

from pymc_forecast.data import (
    FUTURE_DIM,
    TIME_DIM,
    as_dataarray,
    extend_time_index,
    null_covariates,
    validate_alignment,
)
from pymc_forecast.exceptions import HorizonError, OptionalDependencyError
from pymc_forecast.prediction import thin_draws

__all__ = ["StatespaceForecaster", "StatespaceModel"]

OBSERVED_STATE_DIM = "observed_state"
"""Dim name pymc-extras puts on the observed-series axis of its outputs."""

_NO_TIME_INDEX_MESSAGE = "No time index found on the supplied data"
"""pymc-extras warns when data has a plain range index — expected here, since
the adapter deliberately feeds positional data and stamps coords itself."""


class StatespaceModel(abc.ABC):
    """A pymc-extras statespace model definition for :class:`StatespaceForecaster`.

    The statespace lifecycle is two-phase — the component graph must exist
    before its coords can be registered on the ``pm.Model`` that holds the
    parameter priors — so the definition is split accordingly. Example::

        import pymc as pm
        import pytensor.tensor as pt
        from pymc_extras.statespace import structural as st

        class LocalLinearTrend(StatespaceModel):
            def statespace(self, covariates):
                trend = st.LevelTrend(order=2, innovations_order=[0, 1])
                return (trend + st.MeasurementError()).build(verbose=False)

            def priors(self, ss_mod, covariates):
                P0_diag = pm.Gamma("P0_diag", alpha=2, beta=5, dims="state")
                pm.Deterministic("P0", pt.diag(P0_diag), dims=("state", "state_aux"))
                pm.Normal("initial_level_trend", 0, 1, dims="state_level_trend")
                pm.Gamma("sigma_level_trend", alpha=2, beta=10, dims="shock_level_trend")
                pm.Gamma("sigma_MeasurementError", alpha=2, beta=10)
    """

    @abc.abstractmethod
    def statespace(self, covariates: xr.DataArray):
        """Build and return the ``PyMCStateSpace`` (component sum + ``.build()``)."""

    @abc.abstractmethod
    def priors(self, ss_mod, covariates: xr.DataArray) -> None:
        """Declare the parameter priors listed by ``ss_mod.param_info``.

        Called inside a ``pm.Model`` whose coords are ``ss_mod.coords``; the
        adapter calls ``build_statespace_graph`` afterwards.
        """


def _observed_frame(data: xr.DataArray):
    """Convert normalized data to the positional DataFrame statespace expects.

    The real time coords are dropped on purpose (statespace requires a
    zero-based index and would regenerate one anyway); the adapter re-attaches
    them to every output.
    """
    import pandas as pd

    values = data.transpose(TIME_DIM, ...).values
    if values.ndim == 1:
        values = values[:, None]
    return pd.DataFrame(values)


def _posterior_tree(posterior: xr.Dataset) -> xr.DataTree:
    """Wrap a posterior ``Dataset`` in the idata shape statespace methods take."""
    return xr.DataTree.from_dict({"posterior": posterior})


def _predictive_dataset(result) -> xr.Dataset:
    """Extract the predictive ``Dataset`` from a statespace sampling result.

    pymc-extras returns an ArviZ tree whose predictive variables live either at
    the root or under a ``posterior_predictive`` child, depending on version.
    """
    children = getattr(result, "children", None) or {}
    if "posterior_predictive" in children:
        result = result["posterior_predictive"]
    elif hasattr(result, "posterior_predictive") and not hasattr(result, "data_vars"):
        result = result.posterior_predictive
    return result.to_dataset() if hasattr(result, "to_dataset") else result


class StatespaceForecaster:
    """Fit and forecast a pymc-extras statespace model behind the forecaster protocol.

    Satisfies the same interface as the :mod:`~pymc_forecast.forecaster`
    classes — fit on construction, :meth:`draw_posterior`, :meth:`forecast`
    returning a labeled ``predictions`` group, :meth:`predict_in_sample` — so
    it drops into :func:`~pymc_forecast.evaluate.backtest` via
    ``forecaster_cls=StatespaceForecaster``.

    pymc-extras is imported lazily, so constructing this class is the opt-in
    that requires it.

    Parameters
    ----------
    model_fn
        The model definition, a :class:`StatespaceModel` (or any object with
        its ``statespace`` / ``priors`` methods).
    data
        Observed training data (univariate series or 2-d with a named series
        dim).
    covariates
        Covariates covering (at least) the training window, passed through to
        the model definition; surplus future steps are ignored during fitting.
        ``None`` for models without covariates.
    draws, tune, chains
        MCMC schedule (defaults ``1000`` / ``1000`` / ``2``).
    nuts_sampler
        NUTS backend: ``"pymc"`` (default), ``"nutpie"``, ``"numpyro"``, or
        ``"blackjax"``.
    random_seed
        Seed for the fit.
    sample_kwargs
        Extra keyword arguments for ``pm.sample``.

    Attributes
    ----------
    ss_mod
        The built ``PyMCStateSpace``.
    model
        The ``pm.Model`` holding priors and the Kalman-filter likelihood.
    idata
        The full MCMC result.
    """

    def __init__(
        self,
        model_fn,
        data,
        covariates=None,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        nuts_sampler: str = "pymc",
        random_seed=None,
        sample_kwargs: Mapping | None = None,
    ) -> None:
        try:
            import pymc_extras.statespace  # noqa: F401
        except ImportError as err:
            raise OptionalDependencyError("pymc-extras", "extras", "StatespaceForecaster") from err

        self.model_fn = model_fn
        self._data = as_dataarray(data, role="data")
        if covariates is None:
            cov = null_covariates(self._data[TIME_DIM].values)
        else:
            cov = as_dataarray(covariates, role="covariates")
            cov = cov.isel({TIME_DIM: slice(None, self._data.sizes[TIME_DIM])})
        self._covariates = cov

        self.ss_mod = model_fn.statespace(self._covariates)
        with pm.Model(coords=self.ss_mod.coords) as self.model:
            model_fn.priors(self.ss_mod, self._covariates)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_NO_TIME_INDEX_MESSAGE)
                self.ss_mod.build_statespace_graph(_observed_frame(self._data))

        sample_kwargs = dict(sample_kwargs or {})
        self.idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler=nuts_sampler,
            model=self.model,
            random_seed=random_seed,
            progressbar=sample_kwargs.pop("progressbar", False),
            **sample_kwargs,
        )

    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the MCMC posterior."""
        return thin_draws(self.idata, num_samples, random_seed)

    def _relabel(self, da: xr.DataArray, time_dim: str, time_coords) -> xr.DataArray:
        """Stamp real coords on a statespace output: rename its positional
        ``"time"`` dim and map ``"observed_state"`` back to the data's series
        dim (squeezed away for univariate data)."""
        if "time" != time_dim:
            da = da.rename({"time": time_dim})
        da = da.assign_coords({time_dim: np.asarray(time_coords)})
        extra_dims = [d for d in self._data.dims if d != TIME_DIM]
        if OBSERVED_STATE_DIM in da.dims:
            if not extra_dims:
                da = da.squeeze(OBSERVED_STATE_DIM, drop=True)
            else:
                series_dim = extra_dims[0]
                da = da.rename({OBSERVED_STATE_DIM: series_dim})
                da = da.assign_coords({series_dim: self._data[series_dim].values})
        sample_dims = [d for d in ("chain", "draw") if d in da.dims]
        return da.transpose(*sample_dims, time_dim, ...)

    def forecast(
        self,
        covariates=None,
        num_samples: int = 100,
        *,
        horizon: int | None = None,
        random_seed=None,
        progressbar: bool = False,
    ) -> xr.DataTree:
        """Sample forecasts beyond the training window.

        Provide the horizon in one of two ways: pass ``covariates`` spanning
        the training window plus the forecast steps, or pass ``horizon=N`` to
        forecast ``N`` steps past the training data (its time coord is
        extended at the inferred spacing).

        The forecast draws the terminal state from its smoothed posterior and
        iterates the statespace forward — the Kalman analogue of the core
        mechanism of seeding ``*_future`` latents from the in-sample state.

        Parameters
        ----------
        covariates
            Covariates spanning training window + forecast horizon (time
            coords must extend the training data's). Mutually exclusive with
            ``horizon``.
        num_samples
            Number of posterior draws (and forecast samples).
        horizon
            Number of steps to forecast past the training data. Mutually
            exclusive with ``covariates``.
        random_seed, progressbar
            Passed through to ``PyMCStateSpace.forecast``.

        Returns
        -------
        DataTree
            With a ``predictions`` group holding ``"forecast"`` (dims
            ``(chain, draw, time_future, ...)``) and the latent state
            trajectories as ``"forecast_latent"``.
        """
        if (covariates is None) == (horizon is None):
            msg = "pass exactly one of covariates or horizon"
            raise ValueError(msg)
        t_obs = self._data.sizes[TIME_DIM]
        if horizon is not None:
            full_index = extend_time_index(self._data[TIME_DIM].values, horizon)
            future_coords = np.asarray(full_index)[t_obs:]
        else:
            cov = as_dataarray(covariates, role="covariates")
            validate_alignment(self._data, cov)
            future_coords = cov[TIME_DIM].values[t_obs:]
        if len(future_coords) == 0:
            msg = (
                "no forecast horizon: pass covariates longer than the data "
                f"along '{TIME_DIM}', or horizon >= 1"
            )
            raise HorizonError(msg)

        posterior = self.draw_posterior(num_samples, random_seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_NO_TIME_INDEX_MESSAGE)
            result = self.ss_mod.forecast(
                _posterior_tree(posterior),
                start=t_obs - 1,
                periods=len(future_coords),
                random_seed=random_seed,
                progressbar=progressbar,
                verbose=False,
            )
        ds = _predictive_dataset(result)
        predictions = xr.Dataset(
            {
                "forecast": self._relabel(ds["forecast_observed"], FUTURE_DIM, future_coords),
                "forecast_latent": self._relabel(ds["forecast_latent"], FUTURE_DIM, future_coords),
            }
        )
        return xr.DataTree.from_dict({"predictions": predictions})

    def predict_in_sample(
        self,
        num_samples: int = 100,
        *,
        random_seed=None,
        progressbar: bool = False,
    ) -> xr.DataTree:
        """Sample the in-sample predictive of the observed series.

        Draws observations from the smoothed state posterior (conditioned on
        the full training window) — the statespace analogue of replaying
        in-sample latents and resampling the observation noise.

        Returns
        -------
        DataTree
            With a ``posterior_predictive`` group holding ``"obs"`` (dims
            ``(chain, draw, time, ...)``).
        """
        posterior = self.draw_posterior(num_samples, random_seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_NO_TIME_INDEX_MESSAGE)
            result = self.ss_mod.sample_conditional_posterior(
                _posterior_tree(posterior),
                random_seed=random_seed,
                progressbar=progressbar,
            )
        ds = _predictive_dataset(result)
        obs = self._relabel(
            ds["smoothed_posterior_observed"], TIME_DIM, self._data[TIME_DIM].values
        )
        return xr.DataTree.from_dict({"posterior_predictive": xr.Dataset({"obs": obs})})

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
from collections.abc import Mapping, Sequence

import numpy as np
import pymc as pm
import xarray as xr

from pymc_forecast.data import (
    FUTURE_DIM,
    TIME_DIM,
    as_dataarray,
    concat_covariates,
    concat_time_index,
    extend_time_index,
    null_covariates,
    validate_alignment,
)
from pymc_forecast.exceptions import AlignmentError, HorizonError, OptionalDependencyError
from pymc_forecast.forecaster import HMCForecaster

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
            def statespace(self, data, covariates):
                trend = st.LevelTrend(order=2, innovations_order=[0, 1])
                return (trend + st.MeasurementError()).build(verbose=False)

            def priors(self, ss_mod, data, covariates):
                P0_diag = pm.Gamma("P0_diag", alpha=2, beta=5, dims="state")
                pm.Deterministic("P0", pt.diag(P0_diag), dims=("state", "state_aux"))
                pm.Normal("initial_level_trend", float(data[0]), 1, dims="state_level_trend")
                pm.Gamma("sigma_level_trend", alpha=2, beta=10, dims="shock_level_trend")
                pm.Gamma("sigma_MeasurementError", alpha=2, beta=10)

    Both phases receive the normalized training ``data`` and ``covariates`` as
    labeled DataArrays — ``statespace`` so the component graph can be sized
    from them (series count, regression features), ``priors`` so priors can be
    informed by them (as with ``initial_level_trend`` above). A model with a
    ``st.Regression`` component registers its feature matrix with
    ``pm.Data(name, ...)`` inside ``priors``, where ``name`` is the entry in
    ``ss_mod.data_names``; at forecast time the adapter feeds the future
    covariate values through as the scenario.
    """

    @abc.abstractmethod
    def statespace(self, data: xr.DataArray, covariates: xr.DataArray):
        """Build and return the ``PyMCStateSpace`` (component sum + ``.build()``)."""

    @abc.abstractmethod
    def priors(self, ss_mod, data: xr.DataArray, covariates: xr.DataArray) -> None:
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


def _posterior_like(fit_result, posterior: xr.Dataset):
    """Wrap a thinned posterior ``Dataset`` in the same container type as the
    fit result (``DataTree`` or ``arviz.InferenceData``, depending on the
    pymc/arviz generation), which is by construction the idata flavor the
    installed statespace methods accept."""
    if isinstance(fit_result, xr.DataTree):
        return xr.DataTree.from_dict({"posterior": posterior})
    return type(fit_result)(posterior=posterior)


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


class StatespaceForecaster(HMCForecaster):
    """Fit and forecast a pymc-extras statespace model behind the forecaster protocol.

    An :class:`~pymc_forecast.forecaster.HMCForecaster` whose training model
    is built through the statespace lifecycle instead of
    :func:`~pymc_forecast.model.build_model`: fit on construction with NUTS,
    :meth:`draw_posterior`, :meth:`forecast` returning a labeled
    ``predictions`` group, :meth:`predict_in_sample` — so it drops into
    :func:`~pymc_forecast.evaluate.backtest` via
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
    build_kwargs
        Extra keyword arguments for ``build_statespace_graph``, such as
        ``mvn_method`` for the likelihood decomposition.
    forecast_kwargs
        Extra keyword arguments for ``PyMCStateSpace.forecast``, such as
        ``filter_output`` or ``mvn_method``. Horizon, scenario, seed,
        verbosity, and progress are managed by the adapter and cannot be
        overridden here.

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
        build_kwargs: Mapping | None = None,
        forecast_kwargs: Mapping | None = None,
    ) -> None:
        try:
            import pymc_extras.statespace  # noqa: F401
        except ImportError as err:
            raise OptionalDependencyError("pymc-extras", "extras", "StatespaceForecaster") from err

        self._build_kwargs = dict(build_kwargs or {})
        self._forecast_kwargs = dict(forecast_kwargs or {})
        reserved = {"start", "periods", "end", "scenario", "random_seed", "verbose", "progressbar"}
        overlap = reserved.intersection(self._forecast_kwargs)
        if overlap:
            msg = f"forecast_kwargs cannot override adapter-managed arguments: {sorted(overlap)}"
            raise ValueError(msg)
        super().__init__(
            model_fn,
            data,
            covariates,
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed,
            sample_kwargs=sample_kwargs,
        )

    def _build_model(self) -> pm.Model:
        """Drive the statespace lifecycle: components, priors, Kalman graph.

        Validation happens here — after input normalization, before the
        (expensive) fit.
        """
        if self._data.ndim > 2:
            msg = (
                "pymc-extras statespace models take one- or two-dimensional "
                f"data, got dims {self._data.dims}"
            )
            raise AlignmentError(msg)
        validate_alignment(self._data, self._covariates)
        self.ss_mod = self.model_fn.statespace(self._data, self._covariates)
        data_width = 1 if self._data.ndim == 1 else self._data.sizes[self._series_dim]
        if self.ss_mod.k_endog != data_width:
            msg = (
                "statespace observed width does not match the training data: "
                f"k_endog={self.ss_mod.k_endog}, data width={data_width}"
            )
            raise AlignmentError(msg)
        with pm.Model(coords=self.ss_mod.coords) as model:
            self.model_fn.priors(self.ss_mod, self._data, self._covariates)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_NO_TIME_INDEX_MESSAGE)
                self.ss_mod.build_statespace_graph(
                    _observed_frame(self._data), **self._build_kwargs
                )
        return model

    @property
    def _series_dim(self) -> str | None:
        """The data's non-time dim, or ``None`` for a univariate series."""
        extra_dims = [d for d in self._data.dims if d != TIME_DIM]
        return extra_dims[0] if extra_dims else None

    def _relabel(self, da: xr.DataArray, time_dim: str, time_coords) -> xr.DataArray:
        """Stamp real coords on a statespace output: rename its positional
        ``"time"`` dim and map ``"observed_state"`` back to the data's series
        dim (squeezed away for univariate data)."""
        if "time" != time_dim:
            da = da.rename({"time": time_dim})
        da = da.assign_coords({time_dim: np.asarray(time_coords)})
        series_dim = self._series_dim
        if OBSERVED_STATE_DIM in da.dims:
            if series_dim is None:
                da = da.squeeze(OBSERVED_STATE_DIM, drop=True)
            else:
                da = da.rename({OBSERVED_STATE_DIM: series_dim})
                da = da.assign_coords({series_dim: self._data[series_dim].values})
        sample_dims = [d for d in ("chain", "draw") if d in da.dims]
        return da.transpose(*sample_dims, time_dim, ...)

    def _scenario(self, future_covariates: xr.DataArray):
        """Future exogenous values for ``PyMCStateSpace.forecast(scenario=...)``.

        Returns ``None`` for models without exogenous data. A single
        registered data variable (one ``st.Regression`` component) is fed the
        future slice of the covariate matrix.
        """
        data_names = tuple(self.ss_mod.data_names)
        if not data_names:
            return None
        if len(data_names) > 1:
            msg = (
                "one covariate DataArray cannot populate a statespace model "
                f"with multiple exogenous inputs {data_names}; combine them "
                "into one Regression component"
            )
            raise NotImplementedError(msg)
        feature_dims = [d for d in future_covariates.dims if d != TIME_DIM]
        if len(feature_dims) != 1 or future_covariates.sizes[feature_dims[0]] == 0:
            msg = (
                f"the statespace model needs future values for {data_names[0]!r}: "
                "pass full-horizon covariates instead of horizon=/future_index="
            )
            raise AlignmentError(msg)
        return np.asarray(future_covariates.transpose(TIME_DIM, ...).values)

    def forecast(
        self,
        covariates=None,
        num_samples: int = 100,
        *,
        horizon: int | None = None,
        future_index=None,
        future_covariates=None,
        var_names: Sequence[str] | None = None,
        random_seed=None,
        progressbar: bool = False,
    ) -> xr.DataTree:
        """Sample forecasts beyond the training window.

        The horizon is supplied at forecast time, in one of four mutually
        exclusive ways: pass ``covariates`` spanning the training window plus
        the forecast steps, ``future_covariates`` covering only the forecast
        steps, or — for a model without exogenous inputs — pass ``horizon=N``
        to forecast ``N`` steps past the training data (its time coord is
        extended at the inferred spacing) or ``future_index=`` to forecast
        over an arbitrary later time index (strictly increasing values lying
        after the training window; the horizon length is derived from it).
        Forecast steps are always iterated consecutively from the end of
        training and labeled with the supplied coordinates.

        The forecast draws the terminal state from its smoothed posterior and
        iterates the statespace forward — the Kalman analogue of the core
        mechanism of seeding ``*_future`` latents from the in-sample state.
        For a model with a ``st.Regression`` component, the future slice of
        the covariates is fed through as the scenario.

        Parameters
        ----------
        covariates
            Covariates spanning training window + forecast horizon (time
            coords must extend the training data's).
        num_samples
            Number of posterior draws (and forecast samples).
        horizon
            Number of steps to forecast past the training data.
        future_index
            Time coordinate values of the forecast horizon, supplied at
            forecast time (models without exogenous inputs only) — the
            covariate-free half of the predict-time horizon capability;
            ``future_covariates`` is the with-exogenous half.
        future_covariates
            Covariates covering only the forecast horizon, with a time index
            lying after the training window; fed through as the forecast
            scenario. Structure (dims, covariate names and order) must match
            the training covariates.
        var_names
            Subset of prediction variables to keep (``"forecast"``,
            ``"forecast_latent"``). Default: both.
        random_seed, progressbar
            Passed through to ``PyMCStateSpace.forecast``.

        Returns
        -------
        DataTree
            With a ``predictions`` group holding ``"forecast"`` (dims
            ``(chain, draw, time_future, ...)``) and the latent state
            trajectories as ``"forecast_latent"``.
        """
        provided = sum(
            arg is not None for arg in (covariates, horizon, future_index, future_covariates)
        )
        if provided != 1:
            msg = "pass exactly one of covariates, horizon, future_index, or future_covariates"
            raise ValueError(msg)
        t_obs = self._data.sizes[TIME_DIM]
        if horizon is not None:
            full_index = extend_time_index(self._data[TIME_DIM].values, horizon)
            cov = null_covariates(np.asarray(full_index))
        elif future_index is not None:
            full_index = concat_time_index(self._data[TIME_DIM].values, future_index)
            cov = null_covariates(np.asarray(full_index))
        elif future_covariates is not None:
            cov = concat_covariates(self._covariates, future_covariates)
        else:
            cov = as_dataarray(covariates, role="covariates")
            validate_alignment(self._data, cov)
        future_covariates = cov.isel({TIME_DIM: slice(t_obs, None)})
        future_coords = future_covariates[TIME_DIM].values
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
                _posterior_like(self.idata, posterior),
                start=t_obs - 1,
                periods=len(future_coords),
                scenario=self._scenario(future_covariates),
                random_seed=random_seed,
                progressbar=progressbar,
                verbose=False,
                **self._forecast_kwargs,
            )
        ds = _predictive_dataset(result)
        predictions = xr.Dataset(
            {
                "forecast": self._relabel(ds["forecast_observed"], FUTURE_DIM, future_coords),
                "forecast_latent": self._relabel(ds["forecast_latent"], FUTURE_DIM, future_coords),
            }
        )
        if var_names is not None:
            names = list(var_names)
            unknown = set(names).difference(predictions.data_vars)
            if unknown:
                msg = f"unknown statespace prediction variables: {sorted(unknown)}"
                raise KeyError(msg)
            predictions = predictions[names]
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
                _posterior_like(self.idata, posterior),
                random_seed=random_seed,
                progressbar=progressbar,
            )
        ds = _predictive_dataset(result)
        obs = self._relabel(
            ds["smoothed_posterior_observed"], TIME_DIM, self._data[TIME_DIM].values
        )
        return xr.DataTree.from_dict({"posterior_predictive": xr.Dataset({"obs": obs})})

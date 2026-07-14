"""Forecaster classes: fit a forecasting model, then draw probabilistic forecasts.

Three inference backends behind one interface: :class:`Forecaster`
(variational, ADVI by default), :class:`HMCForecaster` (MCMC via
``pm.sample``), and :class:`PathfinderForecaster` (pymc-extras Pathfinder).
Passing data at construction fits the model immediately. Alternatively,
construct with only the model function and call :meth:`BaseForecaster.fit`
later. :meth:`~BaseForecaster.forecast` then rebuilds the model over extended
covariates and samples the horizon.
"""

import abc
from collections.abc import Mapping

import pymc as pm
import xarray as xr

from pymc_forecast.data import (
    TIME_DIM,
    as_dataarray,
    concat_covariates,
    concat_time_index,
    extend_time_index,
    null_covariates,
)
from pymc_forecast.exceptions import AlignmentError, MethodResolutionError, OptionalDependencyError
from pymc_forecast.model import build_model
from pymc_forecast.prediction import (
    forecast as _forecast,
)
from pymc_forecast.prediction import (
    posterior_dataset,
    predict_in_sample,
    thin_draws,
)

__all__ = ["Forecaster", "HMCForecaster", "PathfinderForecaster"]

DEFAULT_LEARNING_RATE = 0.01
"""Default Adam learning rate for variational fits (matches upstream)."""


class BaseForecaster(abc.ABC):
    """Shared fit/forecast plumbing.

    Parameters
    ----------
    model_fn
        The model body (``(Horizon, covariates) -> None`` or a
        :class:`~pymc_forecast.model.ForecastingModel`).
    data
        Observed training data. Omit it to configure now and call :meth:`fit`
        later.
    covariates
        Covariates covering (at least) the training window; surplus future
        steps are ignored during fitting. ``None`` for models without
        covariates.
    random_seed
        Default seed for the fit. A seed passed to :meth:`fit` overrides and
        replaces it for subsequent refits.
    """

    def __init__(self, model_fn, data=None, covariates=None, *, random_seed=None) -> None:
        self.model_fn = model_fn
        self._random_seed = random_seed
        self._is_fitted = False
        self.model = None
        if data is None:
            if covariates is not None:
                msg = "covariates cannot be supplied without training data; pass both to fit()"
                raise ValueError(msg)
            return
        self.fit(data, covariates)

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed successfully."""
        return self._is_fitted

    def fit(self, data, covariates=None, *, random_seed=None):
        """Fit or refit this configured forecaster on training data.

        This is the deferred counterpart to passing ``data`` and
        ``covariates`` to the constructor. Backend options configured at
        construction are reused. Returns ``self`` for fluent adapter
        lifecycles.
        """
        if random_seed is not None:
            self._random_seed = random_seed
        self._is_fitted = False
        self._data = as_dataarray(data, role="data")
        if covariates is None:
            cov = null_covariates(self._data[TIME_DIM].values)
        else:
            cov = as_dataarray(covariates, role="covariates")
            cov = cov.isel({TIME_DIM: slice(None, self._data.sizes[TIME_DIM])})
        self._covariates = cov
        self.model = self._build_model()
        self._fit(self._random_seed)
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            msg = "forecaster is not fit; call fit(data, covariates) before prediction"
            raise RuntimeError(msg)

    def _build_model(self) -> pm.Model:
        """Build the training model from the normalized data (called once
        from :meth:`fit`); adapters for other model lifecycles override this."""
        return build_model(self.model_fn, self._data, self._covariates)

    @abc.abstractmethod
    def _fit(self, random_seed) -> None:
        """Fit the training model (called by :meth:`fit`)."""

    @abc.abstractmethod
    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Return ``num_samples`` posterior draws as a posterior ``Dataset``."""

    def forecast(
        self,
        covariates=None,
        num_samples: int = 100,
        *,
        horizon: int | None = None,
        future_index=None,
        future_covariates=None,
        var_names=None,
        random_seed=None,
        progressbar: bool = False,
    ):
        """Sample forecasts beyond the training window.

        The horizon is supplied at forecast time, in one of four mutually
        exclusive ways: pass ``covariates`` spanning the training window plus
        the forecast steps, ``future_covariates`` covering only the forecast
        steps, or — for a covariate-free model — ``horizon=N`` to forecast
        ``N`` steps past the training data (its time coord is extended at the
        inferred spacing) or ``future_index=`` to forecast over an arbitrary
        later time index.

        Parameters
        ----------
        covariates
            Covariates spanning training window + forecast horizon (time coords
            must extend the training data's).
        num_samples
            Number of posterior draws (and forecast samples).
        horizon
            Number of steps to forecast past the training data (covariate-free
            models only).
        future_index
            Time coordinate values of the forecast horizon (covariate-free
            models only): strictly increasing values lying after the training
            window, e.g. a ``DatetimeIndex`` of the period to predict. The
            horizon length is derived from it, so it need not be known at fit
            time. Forecast steps are drawn consecutively and labeled with
            these coordinates. The covariate-free half of the predict-time
            horizon capability; ``future_covariates`` is the with-covariates
            half.
        future_covariates
            Covariates covering only the forecast horizon, with a time index
            lying after the training window; the forecast is conditioned on
            them — the with-covariates half of the predict-time horizon
            capability (``future_index`` is the covariate-free half).
            Structure (dims, covariate names and order) must match the
            training covariates. The horizon length is derived from it, so it
            need not be known at fit time.
        var_names, random_seed, progressbar
            Passed through to :func:`pymc_forecast.prediction.forecast`.

        Returns
        -------
        DataTree
            With a ``predictions`` group carrying ``time_future`` coords.
        """
        self._require_fitted()
        provided = sum(
            arg is not None for arg in (covariates, horizon, future_index, future_covariates)
        )
        if provided != 1:
            msg = "pass exactly one of covariates, horizon, future_index, or future_covariates"
            raise ValueError(msg)
        if horizon is not None or future_index is not None:
            if self._covariates.size > 0:
                msg = (
                    "this model was fit with covariates, so the forecast needs their "
                    "future values: pass future_covariates= (or full-horizon "
                    "covariates=) instead of horizon=/future_index="
                )
                raise AlignmentError(msg)
            if horizon is not None:
                full_index = extend_time_index(self._data[TIME_DIM].values, horizon)
            else:
                full_index = concat_time_index(self._data[TIME_DIM].values, future_index)
            covariates = null_covariates(full_index)
        elif future_covariates is not None:
            covariates = concat_covariates(self._covariates, future_covariates)
        posterior = self.draw_posterior(num_samples, random_seed)
        return _forecast(
            self.model_fn,
            posterior,
            self._data,
            covariates,
            var_names=var_names,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    def predict_in_sample(
        self,
        num_samples: int = 100,
        *,
        random_seed=None,
        progressbar: bool = False,
    ):
        """Sample the in-sample posterior predictive of ``"obs"``."""
        self._require_fitted()
        posterior = self.draw_posterior(num_samples, random_seed)
        return predict_in_sample(
            self.model_fn,
            posterior,
            self._data,
            self._covariates,
            random_seed=random_seed,
            progressbar=progressbar,
        )


def _resolve_optimizer(optimizer):
    """Normalize an optimizer spec: ``None`` → Adam(0.01), scalar → Adam(lr)."""
    if optimizer is None:
        return pm.adam(learning_rate=DEFAULT_LEARNING_RATE)
    if isinstance(optimizer, int | float):
        learning_rate = float(optimizer)
        if learning_rate <= 0:
            msg = f"learning rate must be positive, got {learning_rate}"
            raise MethodResolutionError(msg)
        return pm.adam(learning_rate=learning_rate)
    if callable(optimizer):
        return optimizer
    msg = (
        "optimizer must be None, a positive learning rate, or a PyMC optimizer "
        f"(e.g. pm.adam(learning_rate=...)); got {type(optimizer).__name__}"
    )
    raise MethodResolutionError(msg)


def _resolve_progressbar(progressbar, kwargs: dict, kwargs_name: str) -> bool:
    """Hoist a legacy backend-kwargs ``progressbar`` to the common option."""
    if "progressbar" in kwargs:
        if progressbar is not None:
            msg = f"pass progressbar directly or through {kwargs_name}, not both"
            raise ValueError(msg)
        progressbar = kwargs.pop("progressbar")
    return False if progressbar is None else bool(progressbar)


class Forecaster(BaseForecaster):
    """Fit a forecasting model with variational inference (ADVI by default).

    Parameters
    ----------
    model_fn, data, covariates, random_seed
        See :class:`BaseForecaster`.
    method
        VI method: ``"advi"`` (mean-field, default) or ``"fullrank_advi"``, or
        any ``pm.fit``-compatible inference object.
    optimizer
        ``None`` (Adam with lr ``0.01``), a positive learning rate, or a PyMC
        optimizer such as ``pm.adam(learning_rate=...)``.
    num_steps
        Number of optimization steps.
    fit_kwargs
        Extra keyword arguments for ``pm.fit``. ``progressbar`` is accepted
        here for compatibility, but the direct argument is preferred.
    progressbar
        Show the fitting progress bar.

    Attributes
    ----------
    approx
        The fitted ``pm.Approximation``.
    losses
        The ELBO loss history (one value per step).
    """

    def __init__(
        self,
        model_fn,
        data=None,
        covariates=None,
        *,
        method="advi",
        optimizer=None,
        num_steps: int = 10_000,
        random_seed=None,
        fit_kwargs: Mapping | None = None,
        progressbar: bool | None = None,
    ) -> None:
        self._method = method
        self._optimizer = _resolve_optimizer(optimizer)
        self._num_steps = num_steps
        self._fit_kwargs = dict(fit_kwargs or {})
        self._progressbar = _resolve_progressbar(progressbar, self._fit_kwargs, "fit_kwargs")
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        try:
            self.approx = pm.fit(
                n=self._num_steps,
                method=self._method,
                model=self.model,
                random_seed=random_seed,
                obj_optimizer=self._optimizer,
                progressbar=self._progressbar,
                **self._fit_kwargs,
            )
        except KeyError as err:
            msg = (
                f"unknown VI method {self._method!r}; use 'advi', 'fullrank_advi', "
                "or a pm.fit-compatible inference object"
            )
            raise MethodResolutionError(msg) from err
        self.losses = self.approx.hist

    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Draw ``num_samples`` posterior samples from the approximation."""
        self._require_fitted()
        idata = self.approx.sample(draws=num_samples, random_seed=random_seed)
        return posterior_dataset(idata)


class HMCForecaster(BaseForecaster):
    """Fit a forecasting model with MCMC (NUTS by default).

    Parameters
    ----------
    model_fn, data, covariates, random_seed
        See :class:`BaseForecaster`.
    draws, tune, chains
        MCMC schedule (defaults ``1000`` / ``1000`` / ``2``).
    nuts_sampler
        NUTS backend: ``"pymc"`` (default), ``"nutpie"``, ``"numpyro"``, or
        ``"blackjax"``.
    sample_kwargs
        Extra keyword arguments for ``pm.sample``. ``progressbar`` is accepted
        here for compatibility, but the direct argument is preferred.
    progressbar
        Show the sampling progress bar.

    Attributes
    ----------
    idata
        The full MCMC result (posterior, sample stats, ...).
    """

    def __init__(
        self,
        model_fn,
        data=None,
        covariates=None,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        nuts_sampler: str = "pymc",
        random_seed=None,
        sample_kwargs: Mapping | None = None,
        progressbar: bool | None = None,
    ) -> None:
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._nuts_sampler = nuts_sampler
        self._sample_kwargs = dict(sample_kwargs or {})
        self._progressbar = _resolve_progressbar(progressbar, self._sample_kwargs, "sample_kwargs")
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        self.idata = pm.sample(
            draws=self._draws,
            tune=self._tune,
            chains=self._chains,
            nuts_sampler=self._nuts_sampler,
            model=self.model,
            random_seed=random_seed,
            progressbar=self._progressbar,
            **self._sample_kwargs,
        )

    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the MCMC posterior."""
        self._require_fitted()
        return thin_draws(self.idata, num_samples, random_seed)


class PathfinderForecaster(BaseForecaster):
    """Fit a forecasting model with Pathfinder variational inference.

    A thin wrapper over ``pymc_extras.fit_pathfinder``. pymc-extras is imported
    lazily, so constructing this class is the opt-in that requires it.

    Parameters
    ----------
    model_fn, data, covariates, random_seed
        See :class:`BaseForecaster`.
    pathfinder_kwargs
        Extra keyword arguments for ``pymc_extras.fit_pathfinder``
        (e.g. ``num_paths``, ``num_draws``). ``progressbar`` is accepted here
        for compatibility, but the direct argument is preferred.
    progressbar
        Show the fitting progress bar.

    Attributes
    ----------
    idata
        The Pathfinder result with its ``posterior`` group.
    """

    def __init__(
        self,
        model_fn,
        data=None,
        covariates=None,
        *,
        random_seed=None,
        pathfinder_kwargs: Mapping | None = None,
        progressbar: bool | None = None,
    ) -> None:
        self._pathfinder_kwargs = dict(pathfinder_kwargs or {})
        self._progressbar = _resolve_progressbar(
            progressbar, self._pathfinder_kwargs, "pathfinder_kwargs"
        )
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        try:
            from pymc_extras import fit_pathfinder
        except ImportError as err:
            raise OptionalDependencyError("pymc-extras", "extras", "PathfinderForecaster") from err
        self.idata = fit_pathfinder(
            model=self.model,
            random_seed=random_seed,
            progressbar=self._progressbar,
            **self._pathfinder_kwargs,
        )

    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the Pathfinder posterior."""
        self._require_fitted()
        return thin_draws(self.idata, num_samples, random_seed)

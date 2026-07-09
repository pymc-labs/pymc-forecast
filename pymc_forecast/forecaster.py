"""Forecaster classes: fit a forecasting model, then draw probabilistic forecasts.

Three inference backends behind one interface: :class:`Forecaster`
(variational, ADVI by default), :class:`HMCForecaster` (MCMC via
``pm.sample``), and :class:`PathfinderForecaster` (pymc-extras Pathfinder).
Construction fits the model on ``(data, covariates)``; :meth:`~BaseForecaster.forecast`
then rebuilds the model over extended covariates and samples the horizon.
"""

import abc
from collections.abc import Mapping

import pymc as pm
import xarray as xr

from pymc_forecast.data import TIME_DIM, as_dataarray, null_covariates
from pymc_forecast.exceptions import MethodResolutionError, OptionalDependencyError
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
        Observed training data.
    covariates
        Covariates covering (at least) the training window; surplus future
        steps are ignored during fitting. ``None`` for models without
        covariates.
    random_seed
        Seed for the fit.
    """

    def __init__(self, model_fn, data, covariates=None, *, random_seed=None) -> None:
        self.model_fn = model_fn
        self._data = as_dataarray(data, role="data")
        if covariates is None:
            cov = null_covariates(self._data[TIME_DIM].values)
        else:
            cov = as_dataarray(covariates, role="covariates")
            cov = cov.isel({TIME_DIM: slice(None, self._data.sizes[TIME_DIM])})
        self._covariates = cov
        self.model = build_model(model_fn, self._data, self._covariates)
        self._fit(random_seed)

    @abc.abstractmethod
    def _fit(self, random_seed) -> None:
        """Fit the training model (called once from ``__init__``)."""

    @abc.abstractmethod
    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Return ``num_samples`` posterior draws as a posterior ``Dataset``."""

    def forecast(
        self,
        covariates,
        num_samples: int = 100,
        *,
        var_names=None,
        random_seed=None,
        progressbar: bool = False,
    ):
        """Sample forecasts for the covariate steps beyond the training window.

        Parameters
        ----------
        covariates
            Covariates spanning the training window plus the forecast horizon
            (time coords must extend the training data's).
        num_samples
            Number of posterior draws (and forecast samples).
        var_names, random_seed, progressbar
            Passed through to :func:`pymc_forecast.prediction.forecast`.

        Returns
        -------
        DataTree
            With a ``predictions`` group carrying ``time_future`` coords.
        """
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
        Extra keyword arguments for ``pm.fit``.

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
        data,
        covariates=None,
        *,
        method="advi",
        optimizer=None,
        num_steps: int = 10_000,
        random_seed=None,
        fit_kwargs: Mapping | None = None,
    ) -> None:
        self._method = method
        self._optimizer = _resolve_optimizer(optimizer)
        self._num_steps = num_steps
        self._fit_kwargs = dict(fit_kwargs or {})
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        try:
            self.approx = pm.fit(
                n=self._num_steps,
                method=self._method,
                model=self.model,
                random_seed=random_seed,
                obj_optimizer=self._optimizer,
                progressbar=self._fit_kwargs.pop("progressbar", False),
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
        Extra keyword arguments for ``pm.sample``.

    Attributes
    ----------
    idata
        The full MCMC result (posterior, sample stats, ...).
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
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._nuts_sampler = nuts_sampler
        self._sample_kwargs = dict(sample_kwargs or {})
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        self.idata = pm.sample(
            draws=self._draws,
            tune=self._tune,
            chains=self._chains,
            nuts_sampler=self._nuts_sampler,
            model=self.model,
            random_seed=random_seed,
            progressbar=self._sample_kwargs.pop("progressbar", False),
            **self._sample_kwargs,
        )

    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the MCMC posterior."""
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
        (e.g. ``num_paths``, ``num_draws``).

    Attributes
    ----------
    idata
        The Pathfinder result with its ``posterior`` group.
    """

    def __init__(
        self,
        model_fn,
        data,
        covariates=None,
        *,
        random_seed=None,
        pathfinder_kwargs: Mapping | None = None,
    ) -> None:
        self._pathfinder_kwargs = dict(pathfinder_kwargs or {})
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        try:
            from pymc_extras import fit_pathfinder
        except ImportError as err:
            raise OptionalDependencyError("pymc-extras", "extras", "PathfinderForecaster") from err
        self.idata = fit_pathfinder(
            model=self.model,
            random_seed=random_seed,
            progressbar=self._pathfinder_kwargs.pop("progressbar", False),
            **self._pathfinder_kwargs,
        )

    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the Pathfinder posterior."""
        return thin_draws(self.idata, num_samples, random_seed)

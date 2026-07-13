"""Interop with linear-Gaussian state-space models from ``pymc-extras``.

``pymc-extras`` state-space models own their model-building and forecasting
lifecycle. :class:`StatespaceForecaster` adapts that lifecycle to the same
fit/forecast protocol as the native forecasters and normalizes its output to a
``predictions/forecast`` group with the package's named dimensions.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

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
from pymc_forecast.exceptions import AlignmentError, HorizonError, OptionalDependencyError
from pymc_forecast.forecaster import BaseForecaster
from pymc_forecast.prediction import thin_draws

__all__ = ["StatespaceBuilder", "StatespaceForecaster"]

_OBSERVED_STATE_DIM = "observed_state"
_STATE_DIM = "state"


class StatespaceBuilder(Protocol):
    """Build a fresh ``(statespace_model, pymc_model)`` pair for one fit.

    Builders receive normalized training data and covariates as labeled
    :class:`xarray.DataArray` objects. They should instantiate a fresh
    ``pymc_extras.statespace.core.PyMCStateSpace``, define its parameter priors
    inside a :class:`pymc.Model`, call ``build_statespace_graph``, and return
    both objects.
    """

    def __call__(self, data: xr.DataArray, covariates: xr.DataArray) -> tuple[Any, pm.Model]: ...


def _dataset(output) -> xr.Dataset:
    """Extract the dataset from a state-space post-estimation result."""
    if isinstance(output, xr.Dataset):
        return output
    if isinstance(output, xr.DataTree):
        return output.to_dataset()
    if hasattr(output, "to_dataset"):
        return output.to_dataset()
    msg = f"expected an xarray Dataset/DataTree, got {type(output).__name__}"
    raise TypeError(msg)


def _group_dataset(output, group: str) -> xr.Dataset:
    """Extract a named group from a PyMC predictive result."""
    try:
        result = output[group]
    except (KeyError, TypeError, IndexError):
        result = getattr(output, group, None)
    if result is None:
        msg = f"predictive result has no {group!r} group"
        raise TypeError(msg)
    return _dataset(result)


def _restore_observed_dims(
    values: xr.DataArray,
    data: xr.DataArray,
    *,
    output_time_dim: str,
    output_time_coord,
) -> xr.DataArray:
    """Map pymc-extras' observed-state dims back to the input data dims."""
    if TIME_DIM != output_time_dim:
        values = values.rename({TIME_DIM: output_time_dim})
    values = values.assign_coords({output_time_dim: output_time_coord})

    data_dims = tuple(dim for dim in data.dims if dim != TIME_DIM)
    if not data_dims:
        if _OBSERVED_STATE_DIM in values.dims:
            if values.sizes[_OBSERVED_STATE_DIM] != 1:
                msg = (
                    "a one-dimensional training series produced "
                    f"{values.sizes[_OBSERVED_STATE_DIM]} observed states"
                )
                raise AlignmentError(msg)
            values = values.squeeze(_OBSERVED_STATE_DIM, drop=True)
    else:
        [data_dim] = data_dims
        if _OBSERVED_STATE_DIM not in values.dims:
            msg = "multivariate state-space output has no 'observed_state' dimension"
            raise AlignmentError(msg)
        if values.sizes[_OBSERVED_STATE_DIM] != data.sizes[data_dim]:
            msg = (
                "state-space output width does not match training data: "
                f"{values.sizes[_OBSERVED_STATE_DIM]} != {data.sizes[data_dim]}"
            )
            raise AlignmentError(msg)
        values = values.rename({_OBSERVED_STATE_DIM: data_dim})
        values = values.assign_coords({data_dim: data[data_dim].values})

    sample_dims = [dim for dim in ("chain", "draw") if dim in values.dims]
    trailing_dims = [dim for dim in values.dims if dim not in (*sample_dims, output_time_dim)]
    return values.transpose(*sample_dims, output_time_dim, *trailing_dims)


class StatespaceForecaster(BaseForecaster):
    """Adapt a ``pymc-extras`` state-space model to the forecaster protocol.

    Parameters
    ----------
    model_fn
        A :class:`StatespaceBuilder`. It is called once per fit with normalized
        ``(data, covariates)`` and must return a fresh
        ``(statespace_model, pymc_model)`` pair. Returning a fresh state-space
        object is important because ``build_statespace_graph`` stores fit data
        used by its post-estimation methods.
    data, covariates, random_seed
        As in :class:`~pymc_forecast.forecaster.BaseForecaster`.
    draws, tune, chains, nuts_sampler
        MCMC schedule and backend passed to :func:`pymc.sample`.
    sample_kwargs
        Additional keyword arguments for :func:`pymc.sample`.
    forecast_kwargs
        Additional keyword arguments for the state-space model's ``forecast``
        method, such as ``filter_output`` or ``mvn_method``. Horizon, scenario,
        seed, verbosity, and progress are managed by the adapter.

    Notes
    -----
    ``pymc-extras`` is imported lazily. Install the ``extras`` optional extra
    before constructing this class.
    """

    def __init__(
        self,
        model_fn: StatespaceBuilder,
        data,
        covariates=None,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        nuts_sampler: str = "pymc",
        random_seed=None,
        sample_kwargs: Mapping | None = None,
        forecast_kwargs: Mapping | None = None,
    ) -> None:
        try:
            from pymc_extras.statespace.core import PyMCStateSpace
        except ImportError as err:
            raise OptionalDependencyError("pymc-extras", "extras", "StatespaceForecaster") from err

        self.model_fn = model_fn
        self._data = as_dataarray(data, role="data")
        if self._data.ndim > 2:
            msg = (
                "pymc-extras state-space models require one- or two-dimensional "
                f"data, got dims {self._data.dims}"
            )
            raise AlignmentError(msg)

        if covariates is None:
            cov = null_covariates(self._data[TIME_DIM].values)
        else:
            cov = as_dataarray(covariates, role="covariates")
            validate_alignment(self._data, cov)
            cov = cov.isel({TIME_DIM: slice(None, self._data.sizes[TIME_DIM])})
        self._covariates = cov

        built = model_fn(self._data, self._covariates)
        if not isinstance(built, tuple) or len(built) != 2:
            msg = "a state-space builder must return (statespace_model, pymc_model)"
            raise TypeError(msg)
        statespace_model, model = built
        if not isinstance(statespace_model, PyMCStateSpace):
            msg = "the first builder result must be a pymc_extras.statespace.core.PyMCStateSpace"
            raise TypeError(msg)
        if not isinstance(model, pm.Model):
            msg = "the second builder result must be a pymc.Model"
            raise TypeError(msg)

        data_dims = tuple(dim for dim in self._data.dims if dim != TIME_DIM)
        data_width = 1 if not data_dims else self._data.sizes[data_dims[0]]
        if statespace_model.k_endog != data_width:
            msg = (
                "state-space observed width does not match training data: "
                f"k_endog={statespace_model.k_endog}, data width={data_width}"
            )
            raise AlignmentError(msg)

        self.statespace_model = statespace_model
        self.model = model
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._nuts_sampler = nuts_sampler
        self._sample_kwargs = dict(sample_kwargs or {})
        self._forecast_kwargs = dict(forecast_kwargs or {})
        self._fit(random_seed)

    def _fit(self, random_seed) -> None:
        kwargs = dict(self._sample_kwargs)
        self.idata = pm.sample(
            draws=self._draws,
            tune=self._tune,
            chains=self._chains,
            nuts_sampler=self._nuts_sampler,
            model=self.model,
            random_seed=random_seed,
            progressbar=kwargs.pop("progressbar", False),
            **kwargs,
        )

    def draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the parameter posterior."""
        return thin_draws(self.idata, num_samples, random_seed)

    def _forecast_scenario(self, future_covariates: xr.DataArray):
        data_names = tuple(self.statespace_model.data_names)
        if not data_names:
            return None
        if len(data_names) > 1:
            msg = (
                "one covariate DataArray cannot populate a state-space model with "
                f"multiple exogenous inputs {data_names}; combine them into one "
                "Regression component"
            )
            raise NotImplementedError(msg)
        covariate_dims = tuple(dim for dim in future_covariates.dims if dim != TIME_DIM)
        if len(covariate_dims) != 1 or future_covariates.sizes[covariate_dims[0]] == 0:
            msg = "the state-space model requires future exogenous covariates"
            raise AlignmentError(msg)
        return np.asarray(future_covariates.values)

    def forecast(
        self,
        covariates=None,
        num_samples: int = 100,
        *,
        horizon: int | None = None,
        var_names: Sequence[str] | None = None,
        random_seed=None,
        progressbar: bool = False,
    ):
        """Draw a forecast and return the standard ``predictions`` group.

        The horizon contract matches :meth:`BaseForecaster.forecast`: provide
        either full-span covariates or ``horizon=N`` for a covariate-free model.
        ``pymc-extras``' ``forecast_observed`` and ``forecast_latent`` outputs
        are exposed as ``forecast`` and ``latent_state`` respectively.
        """
        if (covariates is None) == (horizon is None):
            msg = "pass exactly one of covariates or horizon"
            raise ValueError(msg)
        if horizon is not None:
            full_index = extend_time_index(self._data[TIME_DIM].values, horizon)
            cov_da = null_covariates(full_index)
        else:
            cov_da = as_dataarray(covariates, role="covariates")
            validate_alignment(self._data, cov_da)

        t_obs = self._data.sizes[TIME_DIM]
        future_covariates = cov_da.isel({TIME_DIM: slice(t_obs, None)})
        periods = future_covariates.sizes[TIME_DIM]
        if periods < 1:
            msg = "forecast horizon must contain at least one future step"
            raise HorizonError(msg)

        kwargs = dict(self._forecast_kwargs)
        reserved = {"start", "periods", "end", "scenario", "random_seed", "verbose"}
        overlap = reserved.intersection(kwargs)
        if overlap:
            msg = f"forecast_kwargs cannot override adapter-managed arguments: {sorted(overlap)}"
            raise ValueError(msg)
        kwargs["progressbar"] = progressbar
        posterior = self.draw_posterior(num_samples, random_seed)
        raw = self.statespace_model.forecast(
            posterior,
            start=-1,
            periods=periods,
            scenario=self._forecast_scenario(future_covariates),
            random_seed=random_seed,
            verbose=False,
            **kwargs,
        )
        raw_ds = _dataset(raw)

        future_coord = future_covariates[TIME_DIM].values
        observed = _restore_observed_dims(
            raw_ds["forecast_observed"],
            self._data,
            output_time_dim=FUTURE_DIM,
            output_time_coord=future_coord,
        )
        latent = raw_ds["forecast_latent"].rename({TIME_DIM: FUTURE_DIM})
        latent = latent.assign_coords({FUTURE_DIM: future_coord})
        latent = latent.transpose("chain", "draw", FUTURE_DIM, _STATE_DIM)

        predictions = xr.Dataset({"forecast": observed, "latent_state": latent})
        if var_names is not None:
            names = list(var_names)
            unknown = set(names).difference(predictions.data_vars)
            if unknown:
                msg = f"unknown state-space prediction variables: {sorted(unknown)}"
                raise KeyError(msg)
            predictions = predictions[names]
        return xr.DataTree.from_dict({"predictions": predictions})

    def predict_in_sample(
        self,
        num_samples: int = 100,
        *,
        random_seed=None,
        progressbar: bool = False,
    ):
        """Draw the in-sample posterior predictive as ``posterior_predictive/obs``."""
        posterior = self.draw_posterior(num_samples, random_seed)
        raw = pm.sample_posterior_predictive(
            posterior,
            model=self.model,
            var_names=["obs"],
            random_seed=random_seed,
            progressbar=progressbar,
        )
        raw_ds = _group_dataset(raw, "posterior_predictive")
        observed = _restore_observed_dims(
            raw_ds["obs"],
            self._data,
            output_time_dim=TIME_DIM,
            output_time_coord=self._data[TIME_DIM].values,
        )
        return xr.DataTree.from_dict({"posterior_predictive": xr.Dataset({"obs": observed})})

"""Predictive drivers: forecasting and in-sample posterior prediction.

Both drivers rebuild the model via :func:`~pymc_forecast.model.build_model`
(forecasting with extended covariates, in-sample with the observed window) and
run ``pm.sample_posterior_predictive`` over a posterior. Posteriors are
accepted in any of the shapes the fitting paths produce — an ArviZ
``DataTree``/``InferenceData`` with a ``posterior`` group, or a bare posterior
``Dataset``.

Prediction outputs are draw-level by contract: every variable in the
``predictions`` (out-of-sample) and ``posterior_predictive`` (in-sample)
groups carries ``chain``/``draw`` dims with the full posterior-predictive
samples — nothing is reduced to means or quantiles on the default path.
:func:`prediction_samples` extracts that samples ``Dataset`` from any result
shape the drivers produce.
"""

from collections.abc import Sequence

import numpy as np
import pymc as pm
import xarray as xr

from pymc_forecast.data import TIME_DIM, as_dataarray, null_covariates
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import FORECAST_VAR, OBS_VAR, build_model

__all__ = [
    "forecast",
    "posterior_dataset",
    "predict_in_sample",
    "prediction_samples",
    "thin_draws",
]

PREDICTIVE_GROUPS = ("predictions", "posterior_predictive")
"""Result groups holding draw-level predictive samples, in lookup order."""


def prediction_samples(result) -> xr.Dataset:
    """Extract the draw-level predictive samples from a prediction result.

    Accepts any result shape the predictive drivers produce — an ArviZ
    ``DataTree`` / ``InferenceData`` with a ``predictions`` group (from
    :func:`forecast`) or a ``posterior_predictive`` group (from
    :func:`predict_in_sample`) — or a bare ``Dataset`` (returned unchanged),
    and returns the samples as an ``xarray.Dataset`` whose variables retain
    the full ``chain`` / ``draw`` dims. Point summaries are the caller's
    choice, e.g. ``prediction_samples(result)["forecast"].mean(("chain",
    "draw"))``.

    Raises
    ------
    TypeError
        If ``result`` carries none of the predictive groups.
    """
    if isinstance(result, xr.Dataset):
        return result
    for group in PREDICTIVE_GROUPS:
        try:
            ds = result[group]
        except (KeyError, TypeError, IndexError):
            ds = getattr(result, group, None)
        if ds is not None:
            return ds.to_dataset() if hasattr(ds, "to_dataset") else ds
    msg = (
        f"cannot extract prediction samples from {type(result).__name__}: "
        f"no {' or '.join(repr(g) for g in PREDICTIVE_GROUPS)} group"
    )
    raise TypeError(msg)


def posterior_dataset(posterior) -> xr.Dataset:
    """Extract the posterior group as a plain ``xarray.Dataset``.

    Accepts an ArviZ ``DataTree`` / ``InferenceData`` (uses its ``posterior``
    group) or a bare ``Dataset`` (returned unchanged).
    """
    if isinstance(posterior, xr.Dataset):
        return posterior
    try:
        group = posterior["posterior"]
    except (KeyError, TypeError, IndexError):
        group = getattr(posterior, "posterior", None)
    if group is None:
        msg = f"cannot extract a posterior group from {type(posterior).__name__}"
        raise TypeError(msg)
    return group.to_dataset() if hasattr(group, "to_dataset") else group


def thin_draws(posterior, num_samples: int, random_seed=None) -> xr.Dataset:
    """Subsample a posterior to ``num_samples`` draws (chain-flattened).

    Draws are selected uniformly without replacement from the flattened
    ``(chain, draw)`` axes (with replacement only if more draws are requested
    than exist). The result is a posterior ``Dataset`` with ``chain=1``,
    directly consumable by ``pm.sample_posterior_predictive``.
    """
    if num_samples <= 0:
        msg = f"num_samples must be positive, got {num_samples}"
        raise ValueError(msg)
    ds = posterior_dataset(posterior).stack(__sample__=("chain", "draw"))
    total = ds.sizes["__sample__"]
    rng = np.random.default_rng(random_seed)
    index = rng.choice(total, size=num_samples, replace=num_samples > total)
    return (
        ds.isel(__sample__=index)
        .reset_index("__sample__")
        .drop_vars(["chain", "draw"], errors="ignore")
        .rename({"__sample__": "draw"})
        .assign_coords(draw=np.arange(num_samples))
        .expand_dims(chain=[0])
        .transpose("chain", "draw", ...)
    )


def _default_var_names(model: pm.Model) -> list[str]:
    """The forecast variable plus every ``*_future`` latent, for the output."""
    names = [FORECAST_VAR]
    names += [
        rv.name for rv in model.free_RVs if rv.name.endswith("_future") and rv.name != FORECAST_VAR
    ]
    return names


def forecast(
    model_fn,
    posterior,
    data,
    covariates,
    *,
    num_samples: int | None = None,
    var_names: Sequence[str] | None = None,
    random_seed=None,
    progressbar: bool = False,
):
    """Sample probabilistic forecasts over the covariate horizon.

    Rebuilds the model with ``covariates`` extending ``data`` along
    ``"time"``; the surplus covariate steps are the forecast horizon. The
    posterior is replayed for in-sample latents while ``*_future`` variables
    (absent from it) are drawn fresh.

    Parameters
    ----------
    model_fn
        The model body (``(Horizon, covariates) -> None`` or a
        :class:`~pymc_forecast.model.ForecastingModel`).
    posterior
        A fitted posterior (``DataTree``/``InferenceData`` or ``Dataset``).
    data
        Observed data over the training window.
    covariates
        Covariates spanning training window plus forecast horizon.
    num_samples
        If given, subsample the posterior to this many draws first.
    var_names
        Variables to record. Default: ``"forecast"`` plus all ``*_future``
        latents.
    random_seed
        Seed for thinning and predictive sampling.
    progressbar
        Show the sampling progress bar.

    Returns
    -------
    DataTree
        With a ``predictions`` group carrying ``time_future`` coords.
    """
    model = build_model(model_fn, data, covariates)
    if FORECAST_VAR not in model.named_vars:
        msg = (
            "the rebuilt model has no forecast horizon: covariates must be "
            f"longer than data along '{TIME_DIM}'"
        )
        raise HorizonError(msg)
    if num_samples is not None:
        posterior = thin_draws(posterior, num_samples, random_seed)
    return pm.sample_posterior_predictive(
        posterior_dataset(posterior),
        model=model,
        var_names=list(var_names) if var_names is not None else _default_var_names(model),
        predictions=True,
        random_seed=random_seed,
        progressbar=progressbar,
    )


def predict_in_sample(
    model_fn,
    posterior,
    data,
    covariates=None,
    *,
    num_samples: int | None = None,
    random_seed=None,
    progressbar: bool = False,
):
    """Sample the in-sample posterior predictive of the ``"obs"`` variable.

    The in-sample counterpart of :func:`forecast`: the model is rebuilt over
    the observed window only (no forecast horizon) and the observed variable
    is resampled given replayed latents.

    Parameters
    ----------
    model_fn, posterior, data
        As in :func:`forecast`.
    covariates
        Covariates covering (at least) the observed window; surplus future
        steps are dropped. ``None`` for models without covariates.
    num_samples, random_seed, progressbar
        As in :func:`forecast`.

    Returns
    -------
    DataTree
        With a ``posterior_predictive`` group holding ``"obs"``.
    """
    data_da = as_dataarray(data, role="data")
    if covariates is None:
        cov_da = null_covariates(data_da[TIME_DIM].values)
    else:
        cov_da = as_dataarray(covariates, role="covariates")
        cov_da = cov_da.isel({TIME_DIM: slice(None, data_da.sizes[TIME_DIM])})
    model = build_model(model_fn, data_da, cov_da)
    if num_samples is not None:
        posterior = thin_draws(posterior, num_samples, random_seed)
    return pm.sample_posterior_predictive(
        posterior_dataset(posterior),
        model=model,
        var_names=[OBS_VAR],
        random_seed=random_seed,
        progressbar=progressbar,
    )

"""Functional fitters: one sampling path shared with the forecaster classes.

Each fitter accepts an already-built ``model=``. When it is omitted, the
fitter calls :func:`~pymc_forecast.model.build_model` once and samples that
model. Class ``_fit`` methods pass ``model=self.model`` and do not build again.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from pymc_forecast.exceptions import MethodResolutionError, OptionalDependencyError
from pymc_forecast.model import build_model
from pymc_forecast.prediction import posterior_dataset, thin_draws

__all__ = ["FitResult", "draw_posterior", "fit_mcmc", "fit_pathfinder", "fit_vi"]


@dataclass(frozen=True)
class FitResult:
    """A completed fit.

    Variational results carry ``approx`` and ``losses`` and leave ``idata``
    as ``None`` until something draws. MCMC and Pathfinder set ``idata``.
    """

    idata: az.InferenceData | None
    approx: pm.Approximation | None
    losses: np.ndarray | None
    method: str


def _training_model(model, model_fn, data, covariates):
    """Return ``model`` unchanged, or build it once from the training inputs."""
    if model is not None:
        return model
    if covariates is None:
        from pymc_forecast.data import TIME_DIM, as_dataarray, null_covariates

        data = as_dataarray(data, role="data")
        covariates = null_covariates(data[TIME_DIM].values)
    return build_model(model_fn, data, covariates)


def _vi_helpers():
    from pymc_forecast.forecaster import (
        DEFAULT_LEARNING_RATE,
        _check_vi_convergence,
        _resolve_optimizer,
        _resolve_progressbar,
    )

    return DEFAULT_LEARNING_RATE, _check_vi_convergence, _resolve_optimizer, _resolve_progressbar


def fit_vi(
    model_fn,
    data=None,
    covariates=None,
    *,
    method="advi",
    optimizer=None,
    backend=None,
    num_steps=10_000,
    random_seed=None,
    progressbar=None,
    fit_kwargs=None,
    model=None,
) -> FitResult:
    """Fit ``model_fn`` (or an already-built ``model``) with variational inference."""
    default_lr, check_convergence, resolve_optimizer, resolve_progressbar = _vi_helpers()
    kwargs = dict(fit_kwargs or {})
    progressbar = resolve_progressbar(progressbar, kwargs, "fit_kwargs")
    if backend not in (None, "pytensor", "jax"):
        msg = f"unknown VI backend {backend!r}; use None, 'pytensor', or 'jax'"
        raise MethodResolutionError(msg)
    if backend == "jax":
        if method != "advi":
            msg = "the JAX backend currently supports method='advi' only"
            raise MethodResolutionError(msg)
        if optimizer is None:
            learning_rate = default_lr
        elif isinstance(optimizer, int | float) and optimizer > 0:
            learning_rate = float(optimizer)
        else:
            msg = "the JAX backend requires optimizer=None or a positive learning rate"
            raise MethodResolutionError(msg)
        if kwargs:
            msg = "fit_kwargs are not supported by the JAX backend"
            raise MethodResolutionError(msg)
        from pymc_forecast.jax_backend import fit_advi_jax

        fitted = _training_model(model, model_fn, data, covariates)
        approx = fit_advi_jax(
            fitted,
            num_steps=num_steps,
            learning_rate=learning_rate,
            random_seed=random_seed,
        )
    else:
        obj_optimizer = resolve_optimizer(optimizer)
        fitted = _training_model(model, model_fn, data, covariates)
        try:
            approx = pm.fit(
                n=num_steps,
                method=method,
                model=fitted,
                random_seed=random_seed,
                obj_optimizer=obj_optimizer,
                progressbar=progressbar,
                **kwargs,
            )
        except KeyError as err:
            msg = (
                f"unknown VI method {method!r}; use 'advi', 'fullrank_advi', "
                "or a pm.fit-compatible inference object"
            )
            raise MethodResolutionError(msg) from err
    losses = np.asarray(approx.hist)
    check_convergence(losses, num_steps)
    return FitResult(idata=None, approx=approx, losses=losses, method=method)


def fit_mcmc(
    model_fn,
    data=None,
    covariates=None,
    *,
    draws=1000,
    tune=1000,
    chains=2,
    nuts_sampler="pymc",
    random_seed=None,
    progressbar=None,
    sample_kwargs=None,
    model=None,
) -> FitResult:
    """Fit ``model_fn`` (or an already-built ``model``) with MCMC."""
    from pymc_forecast.forecaster import _resolve_progressbar

    kwargs = dict(sample_kwargs or {})
    progressbar = _resolve_progressbar(progressbar, kwargs, "sample_kwargs")
    fitted = _training_model(model, model_fn, data, covariates)
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        nuts_sampler=nuts_sampler,
        model=fitted,
        random_seed=random_seed,
        progressbar=progressbar,
        **kwargs,
    )
    return FitResult(idata=idata, approx=None, losses=None, method="mcmc")


def fit_pathfinder(
    model_fn,
    data=None,
    covariates=None,
    *,
    random_seed=None,
    progressbar=None,
    pathfinder_kwargs=None,
    model=None,
) -> FitResult:
    """Fit ``model_fn`` (or an already-built ``model``) with Pathfinder."""
    from pymc_forecast.forecaster import _resolve_progressbar

    try:
        from pymc_extras import fit_pathfinder as _fit_pathfinder
    except ImportError as err:
        raise OptionalDependencyError("pymc-extras", "extras", "PathfinderForecaster") from err
    kwargs = dict(pathfinder_kwargs or {})
    progressbar = _resolve_progressbar(progressbar, kwargs, "pathfinder_kwargs")
    fitted = _training_model(model, model_fn, data, covariates)
    idata = _fit_pathfinder(
        model=fitted,
        random_seed=random_seed,
        progressbar=progressbar,
        **kwargs,
    )
    return FitResult(idata=idata, approx=None, losses=None, method="pathfinder")


def _draw_once(result: FitResult, num_samples: int, random_seed=None) -> xr.Dataset:
    if result.idata is None:
        idata = result.approx.sample(draws=num_samples, random_seed=random_seed)
        return posterior_dataset(idata)
    return thin_draws(result.idata, num_samples, random_seed)


def draw_posterior(
    result: FitResult,
    num_samples: int,
    random_seed=None,
    *,
    batch_size: int | None = None,
) -> xr.Dataset:
    """Draw or thin ``num_samples`` posterior samples from a :class:`FitResult`.

    Variational results are sampled from the approximation, optionally in
    host-side chunks of ``batch_size``. MCMC and Pathfinder results are thinned
    once from ``idata.posterior``.
    """
    if batch_size is not None and batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    generated = result.idata is None
    if batch_size is None or not generated or batch_size >= num_samples:
        return _draw_once(result, num_samples, random_seed)

    rng = (
        random_seed
        if isinstance(random_seed, np.random.RandomState)
        else np.random.default_rng(random_seed)
    )
    chunks: list[xr.Dataset] = []
    offset = 0
    while offset < num_samples:
        size = min(batch_size, num_samples - offset)
        chunk = _draw_once(result, size, rng)
        if chunk.sizes.get("chain") != 1:
            msg = (
                "generated posterior batches must have one chain; got "
                f"sizes {dict(chunk.sizes)}"
            )
            raise ValueError(msg)
        chunk = chunk.assign_coords(draw=np.arange(offset, offset + size))
        chunks.append(chunk)
        offset += size
    return xr.concat(
        chunks,
        dim="draw",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )

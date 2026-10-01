"""Functional fitters: one sampling path shared with the forecaster classes.

Each fitter accepts an already-built ``model=``. When it is omitted, the
fitter calls :func:`~pymc_forecast.model.build_model` once and samples that
model. Class ``_fit`` methods pass ``model=self.model`` and do not build again.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from pymc_forecast.data import TIME_DIM, as_dataarray, null_covariates
from pymc_forecast.exceptions import MethodResolutionError, OptionalDependencyError
from pymc_forecast.model import build_model
from pymc_forecast.prediction import posterior_dataset, thin_draws

__all__ = ["FitResult", "draw_posterior", "fit_mcmc", "fit_pathfinder", "fit_vi"]

DEFAULT_LEARNING_RATE = 0.01
"""Default Adam learning rate for variational fits (matches upstream)."""

CONVERGENCE_WINDOW_FRACTION = 0.1
"""Fraction of the ELBO loss history in each of the two windows (one at the
midpoint of the run, one at the end) compared by the post-fit ADVI
convergence check (see :func:`_check_vi_convergence`)."""

CONVERGENCE_MIN_WINDOW = 10
"""Minimum steps per convergence-check window; shorter loss histories are
too noisy to assess and are skipped."""


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
    """Hoist a backend-kwargs ``progressbar`` to the uniform direct option."""
    if "progressbar" in kwargs:
        if progressbar is not None:
            msg = f"pass progressbar directly or through {kwargs_name}, not both"
            raise ValueError(msg)
        progressbar = kwargs.pop("progressbar")
    return False if progressbar is None else bool(progressbar)


def _check_vi_convergence(losses, num_steps: int) -> None:
    """Warn when the ELBO loss is still clearly descending at the end of a fit.

    Heuristic: compare the median loss over the last
    ``CONVERGENCE_WINDOW_FRACTION`` of the steps against the median over the
    same-sized window starting at the midpoint of the history. The fit is
    flagged when the improvement between the two windows exceeds both the
    fluctuation within the final window (its median absolute deviation) and
    twice the standard error of the median difference — i.e. the optimizer
    was still making clear progress, beyond the stochastic-ELBO noise floor,
    over the second half of the run. Medians are used because the raw ELBO
    history is spiky early in a fit. A slow descent can hide inside the
    noise, so the absence of a warning is not proof of convergence.
    """
    hist = np.asarray(losses, dtype=float)
    if not np.isfinite(hist).all():
        msg = "ADVI convergence could not be assessed: the loss history contains non-finite values."
        warnings.warn(msg, UserWarning, stacklevel=2)
        return
    n = max(int(len(hist) * CONVERGENCE_WINDOW_FRACTION), CONVERGENCE_MIN_WINDOW)
    if len(hist) // 2 + n > len(hist) - n:
        return
    last = hist[-n:]
    mid = hist[len(hist) // 2 :][:n]
    improvement = float(np.median(mid) - np.median(last))
    noise = float(np.median(np.abs(last - np.median(last))))
    # standard error of the difference of two window medians, MAD-scaled
    sem = 1.858 * noise * float(np.sqrt(2.0 / n))
    if improvement > max(noise, 2.0 * sem):
        msg = (
            f"ADVI has not converged after {num_steps} steps: the ELBO loss "
            f"is still descending (median over the last {n} steps improved "
            f"by {improvement:.3g} since mid-run, more than the within-window "
            f"fluctuation {noise:.3g}). The forecast may be confidently wrong "
            "— increase num_steps, raise the learning rate, or use "
            "HMCForecaster; inspect the loss history via the `losses` "
            "attribute."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)


@dataclass(frozen=True)
class FitResult:
    """A completed fit.

    Variational results carry ``approx`` and ``losses`` and leave ``idata``
    as ``None`` until something draws. MCMC and Pathfinder set ``idata``.
    ``method`` is ``"mcmc"``, ``"pathfinder"``, or the VI method passed to
    :func:`fit_vi` (a name or an inference object).
    """

    idata: az.InferenceData | None
    approx: pm.Approximation | None
    losses: np.ndarray | None
    method: str | pm.variational.Inference


def _training_inputs(data, covariates) -> tuple[xr.DataArray, xr.DataArray]:
    """Normalize training inputs and cut covariates to the observed window.

    Covariate rows past the data would give the training model a forecast
    horizon, registering ``{name}_future`` latents as free variables that the
    fit then puts into the posterior.
    """
    data_da = as_dataarray(data, role="data")
    if covariates is None:
        return data_da, null_covariates(data_da[TIME_DIM].values)
    cov_da = as_dataarray(covariates, role="covariates")
    return data_da, cov_da.isel({TIME_DIM: slice(None, data_da.sizes[TIME_DIM])})


def _training_model(model, model_fn, data, covariates):
    """Return ``model`` unchanged, or build it once from the training inputs."""
    if model is not None:
        return model
    return build_model(model_fn, *_training_inputs(data, covariates))


class _VIOptions(NamedTuple):
    optimizer: object
    """Resolved PyMC optimizer, or a float learning rate for ``backend="jax"``."""
    fit_kwargs: dict
    progressbar: bool


def _resolve_vi_options(method, optimizer, backend, fit_kwargs, progressbar) -> _VIOptions:
    """Validate VI options once for :func:`fit_vi` and ``Forecaster``.

    Idempotent: resolving an already-resolved optimizer, learning rate, or
    hoisted progressbar returns them unchanged.
    """
    if backend not in (None, "pytensor", "jax"):
        msg = f"unknown VI backend {backend!r}; use None, 'pytensor', or 'jax'"
        raise MethodResolutionError(msg)
    kwargs = dict(fit_kwargs or {})
    if backend == "jax":
        if method != "advi":
            msg = "the JAX backend currently supports method='advi' only"
            raise MethodResolutionError(msg)
        if optimizer is None:
            resolved = DEFAULT_LEARNING_RATE
        elif isinstance(optimizer, int | float) and optimizer > 0:
            resolved = float(optimizer)
        else:
            msg = "the JAX backend requires optimizer=None or a positive learning rate"
            raise MethodResolutionError(msg)
        if kwargs:
            msg = "fit_kwargs are not supported by the JAX backend"
            raise MethodResolutionError(msg)
    else:
        resolved = _resolve_optimizer(optimizer)
    return _VIOptions(resolved, kwargs, _resolve_progressbar(progressbar, kwargs, "fit_kwargs"))


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
    options = _resolve_vi_options(method, optimizer, backend, fit_kwargs, progressbar)
    fitted = _training_model(model, model_fn, data, covariates)
    if backend == "jax":
        from pymc_forecast.jax_backend import fit_advi_jax

        approx = fit_advi_jax(
            fitted,
            num_steps=num_steps,
            learning_rate=options.optimizer,
            random_seed=random_seed,
        )
    else:
        try:
            approx = pm.fit(
                n=num_steps,
                method=method,
                model=fitted,
                random_seed=random_seed,
                obj_optimizer=options.optimizer,
                progressbar=options.progressbar,
                **options.fit_kwargs,
            )
        except KeyError as err:
            msg = (
                f"unknown VI method {method!r}; use 'advi', 'fullrank_advi', "
                "or a pm.fit-compatible inference object"
            )
            raise MethodResolutionError(msg) from err
    losses = np.asarray(approx.hist)
    _check_vi_convergence(losses, num_steps)
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
    try:
        from pymc_extras import fit_pathfinder as _fit_pathfinder
    except ImportError as err:
        raise OptionalDependencyError("pymc-extras", "extras", "Pathfinder inference") from err
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


def _draw_batched(draw, num_samples, random_seed, *, batch_size, generated) -> xr.Dataset:
    """Call ``draw(size, seed)`` once, or in ``batch_size`` chunks when ``generated``.

    One Generator threads through all chunks, so each chunk gets a fresh,
    deterministic child seed without mutating global NumPy state. PyMC's legacy
    ``RandomState`` passes through as-is because ``default_rng`` cannot wrap it.
    """
    if batch_size is not None and batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if batch_size is None or not generated or batch_size >= num_samples:
        return draw(num_samples, random_seed)
    rng = (
        random_seed
        if isinstance(random_seed, np.random.RandomState)
        else np.random.default_rng(random_seed)
    )
    chunks: list[xr.Dataset] = []
    offset = 0
    while offset < num_samples:
        size = min(batch_size, num_samples - offset)
        chunk = draw(size, rng)
        if chunk.sizes.get("chain") != 1:
            msg = f"generated posterior batches must have one chain; got sizes {dict(chunk.sizes)}"
            raise ValueError(msg)
        chunks.append(chunk.assign_coords(draw=np.arange(offset, offset + size)))
        offset += size
    return xr.concat(
        chunks,
        dim="draw",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )


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
    return _draw_batched(
        lambda size, seed: _draw_once(result, size, seed),
        num_samples,
        random_seed,
        batch_size=batch_size,
        generated=result.idata is None,
    )

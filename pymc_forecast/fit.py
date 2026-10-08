"""Functional fitters: one sampling path shared with the forecaster classes.

Each fitter accepts an already-built ``model=``, which must be a
training-window model: a model with forecast-horizon variables (``forecast``
or ``{name}_future`` free variables) is rejected with
:class:`~pymc_forecast.exceptions.HorizonError`. When ``model`` is given,
``model_fn``, ``data`` and ``covariates`` are ignored (``model_fn`` is still a
required positional argument). When ``model`` is omitted, the fitter drops
covariate rows past the training window, calls
:func:`~pymc_forecast.model.build_model` once, and samples that model; only
the ``model_fn, data, covariates`` path performs that trim. The forecaster
classes pass their own training model and do not build again.
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
from pymc_forecast.exceptions import HorizonError, MethodResolutionError, OptionalDependencyError
from pymc_forecast.model import FORECAST_VAR, build_model
from pymc_forecast.prediction import posterior_dataset, thin_draws

__all__ = ["FitResult", "draw_posterior", "fit_mcmc", "fit_pathfinder", "fit_vi"]

DEFAULT_LEARNING_RATE = 0.01
"""Default Adam learning rate for variational fits (matches upstream)."""

CONVERGENCE_WINDOW_FRACTION = 0.1
"""Fraction of the ELBO loss history in each of the two windows (one at the
midpoint of the run, one at the end) compared by the post-fit ADVI
convergence check."""

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

    Variational results carry ``approx`` and ``losses`` and keep
    ``idata=None``; the result is frozen and never filled in, so pass
    ``draw_posterior(result, n)`` (a posterior ``Dataset``) to ``forecast``.
    MCMC and Pathfinder set ``idata``, which can also be passed to
    ``forecast`` directly.
    ``method`` is ``"mcmc"``, ``"pathfinder"``, or the VI method passed to
    :func:`fit_vi` (a name or an inference object).

    Each parameter is available as an attribute of the same name.

    Parameters
    ----------
    idata : xarray.DataTree or arviz.InferenceData or None
        Output of ``pm.sample`` (MCMC) or ``pymc_extras.fit_pathfinder``
        (Pathfinder) with a ``posterior`` group: a DataTree with current
        PyMC/ArviZ, InferenceData with older releases. ``None`` for VI.
    approx : pymc.variational.Approximation or None
        The fitted variational approximation for VI; ``None`` otherwise.
    losses : numpy.ndarray or None
        Loss history (``approx.hist``) for VI: one value per step for ADVI and
        full-rank ADVI, empty for ``"svgd"``/``"asvgd"`` (no loss is
        recorded); ``None`` otherwise.
    method : str or pymc.variational.Inference
        ``"mcmc"``, ``"pathfinder"``, or the ``method`` argument of
        :func:`fit_vi` as passed.
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
    """Return a validated ``model``, or build it once from the training inputs.

    A supplied model is not trimmed, so one built on full-horizon covariates
    is rejected: its future latents would be fit and then replayed.
    """
    if model is not None:
        horizon = [
            rv.name
            for rv in model.free_RVs
            if rv.name == FORECAST_VAR or rv.name.endswith("_future")
        ]
        if horizon:
            msg = (
                f"the supplied model has forecast-horizon free variables {horizon}; "
                "fitting would put them in the posterior. Pass a model built on the "
                "training window only (covariates cut to the data's time steps), or "
                "pass model_fn, data, covariates and let the fitter build it."
            )
            raise HorizonError(msg)
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
    """Fit ``model_fn`` (or an already-built ``model``) with variational inference.

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        Model body called as ``model_fn(covariates, data)``; it must register
        ``obs`` (normally via :func:`~pymc_forecast.model.predict`). Ignored
        when ``model`` is given.
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim). Required unless ``model`` is given.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data`` (normalized
        with ``as_dataarray``; 2-D input gets a ``"covariate"`` dim). Rows past
        the training window are dropped. ``None`` means no covariates.
    method : str or pymc.variational.Inference, default "advi"
        Forwarded to ``pm.fit`` (e.g. ``"advi"`` or ``"fullrank_advi"``, or an
        inference object, which fits the model it was built on and ignores
        ``random_seed``). The JAX backend accepts only ``"advi"``.
    optimizer : callable or float, optional
        ``None`` uses Adam with learning rate 0.01; a positive number is an
        Adam learning rate; a callable is passed to ``pm.fit`` as
        ``obj_optimizer`` (PyTensor backend only).
    backend : {"pytensor", "jax"}, optional
        ``None`` or ``"pytensor"`` runs ``pm.fit``; ``"jax"`` runs
        ``pymc_forecast.jax_backend.fit_advi_jax``.
    num_steps : int, default 10_000
        Number of optimization steps.
    random_seed : int, optional
        Seed for the fit.
    progressbar : bool, optional
        Show the fit progress bar; ``None`` means off. May instead be given
        as a ``fit_kwargs`` key, but not both. Ignored by the JAX backend.
    fit_kwargs : mapping, optional
        Extra keyword arguments forwarded to ``pm.fit``. Not supported with
        ``backend="jax"``.
    model : pymc.Model, optional
        An already-built training-window model to fit instead of building
        one from ``model_fn``, ``data`` and ``covariates``.

    Returns
    -------
    FitResult
        Result with ``approx`` and ``losses`` set and ``idata=None``.

    Raises
    ------
    pymc_forecast.exceptions.MethodResolutionError
        If ``backend``, ``optimizer``, ``fit_kwargs`` or ``method`` cannot be
        resolved (unknown backend or VI method name, non-positive learning
        rate, unsupported optimizer, or a JAX backend with a method other
        than ``"advi"``, a non-numeric optimizer, or ``fit_kwargs``).
    ValueError
        If ``progressbar`` is given both directly and in ``fit_kwargs``, or
        ``num_steps`` is not positive with ``backend="jax"``.
    pymc_forecast.exceptions.HorizonError
        If ``model`` has forecast-horizon variables, or the built model does
        not register ``obs``.
    pymc_forecast.exceptions.AlignmentError
        If ``data``/``covariates`` cannot be normalized or do not align.
    pymc_forecast.exceptions.OptionalDependencyError
        If ``backend="jax"`` and JAX is not installed.

    Warns
    -----
    UserWarning
        If the loss is still clearly decreasing at the end of the fit (a
        heuristic convergence check), or if the loss history contains
        non-finite values so convergence cannot be assessed.
    """
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
    """Fit ``model_fn`` (or an already-built ``model``) with MCMC.

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        Model body called as ``model_fn(covariates, data)``; it must register
        ``obs`` (normally via :func:`~pymc_forecast.model.predict`). Ignored
        when ``model`` is given.
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim). Required unless ``model`` is given.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data`` (normalized
        with ``as_dataarray``; 2-D input gets a ``"covariate"`` dim). Rows past
        the training window are dropped. ``None`` means no covariates.
    draws : int, default 1000
        Posterior draws per chain, forwarded to ``pm.sample``.
    tune : int, default 1000
        Tuning steps per chain, forwarded to ``pm.sample``.
    chains : int, default 2
        Number of chains, forwarded to ``pm.sample``.
    nuts_sampler : {"pymc", "nutpie", "numpyro", "blackjax"}, default "pymc"
        NUTS implementation, forwarded to ``pm.sample``; non-PyMC samplers
        need their optional package.
    random_seed : int or numpy.random.Generator, optional
        Seed forwarded to ``pm.sample``.
    progressbar : bool, optional
        Show the sampling progress bar; ``None`` means off. May instead be
        given as a ``sample_kwargs`` key, but not both.
    sample_kwargs : mapping, optional
        Extra keyword arguments forwarded to ``pm.sample``.
    model : pymc.Model, optional
        An already-built training-window model to sample instead of building
        one from ``model_fn``, ``data`` and ``covariates``.

    Returns
    -------
    FitResult
        Result with ``idata`` set and ``method="mcmc"``.

    Raises
    ------
    ValueError
        If ``progressbar`` is given both directly and in ``sample_kwargs``.
    pymc_forecast.exceptions.HorizonError
        If ``model`` has forecast-horizon variables, or the built model does
        not register ``obs``.
    pymc_forecast.exceptions.AlignmentError
        If ``data``/``covariates`` cannot be normalized or do not align.
    """
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
    """Fit ``model_fn`` (or an already-built ``model``) with Pathfinder.

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        Model body called as ``model_fn(covariates, data)``; it must register
        ``obs`` (normally via :func:`~pymc_forecast.model.predict`). Ignored
        when ``model`` is given.
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim). Required unless ``model`` is given.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data`` (normalized
        with ``as_dataarray``; 2-D input gets a ``"covariate"`` dim). Rows past
        the training window are dropped. ``None`` means no covariates.
    random_seed : int, optional
        Seed forwarded to ``pymc_extras.fit_pathfinder``.
    progressbar : bool, optional
        Show the progress bar; ``None`` means off. May instead be given as a
        ``pathfinder_kwargs`` key, but not both.
    pathfinder_kwargs : mapping, optional
        Extra keyword arguments forwarded to ``pymc_extras.fit_pathfinder``.
    model : pymc.Model, optional
        An already-built training-window model to fit instead of building one
        from ``model_fn``, ``data`` and ``covariates``.

    Returns
    -------
    FitResult
        Result with ``idata`` set and ``method="pathfinder"``.

    Raises
    ------
    pymc_forecast.exceptions.OptionalDependencyError
        If pymc-extras is not installed.
    ValueError
        If ``progressbar`` is given both directly and in
        ``pathfinder_kwargs``.
    pymc_forecast.exceptions.HorizonError
        If ``model`` has forecast-horizon variables, or the built model does
        not register ``obs``.
    pymc_forecast.exceptions.AlignmentError
        If ``data``/``covariates`` cannot be normalized or do not align.
    """
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

    Parameters
    ----------
    result : FitResult
        A result of :func:`fit_vi`, :func:`fit_mcmc` or :func:`fit_pathfinder`.
    num_samples : int
        Number of posterior draws to return.
    random_seed : int or numpy.random.Generator, optional
        Seed for sampling the approximation or thinning ``idata``.
    batch_size : int, optional
        For variational results, draw at most this many samples at once and
        concatenate the chunks; ``None`` draws all at once. Ignored (but still
        validated) for MCMC and Pathfinder results.

    Returns
    -------
    xarray.Dataset
        Posterior samples with a single ``chain`` and ``num_samples`` draws.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive, or ``num_samples`` is not positive
        for an MCMC or Pathfinder result.
    """
    return _draw_batched(
        lambda size, seed: _draw_once(result, size, seed),
        num_samples,
        random_seed,
        batch_size=batch_size,
        generated=result.idata is None,
    )

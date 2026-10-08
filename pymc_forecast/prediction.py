"""Predictive drivers: forecasting and in-sample posterior prediction.

Both drivers rebuild the model via :func:`~pymc_forecast.model.build_model`
(forecasting with extended covariates, in-sample with the observed window) and
run ``pm.sample_posterior_predictive`` over a posterior. Posteriors are
accepted in any of the shapes the fitting paths produce — an
``xarray.DataTree`` or ``arviz.InferenceData`` with a ``posterior`` group, or
a bare posterior ``Dataset``.

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

from pymc_forecast.data import DRAW_DIM, TIME_DIM, as_dataarray, null_covariates
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import (
    EXPECTED_OBSERVATION_FORECAST_VAR,
    EXPECTED_OBSERVATION_VAR,
    FORECAST_VAR,
    MU_FORECAST_VAR,
    MU_VAR,
    OBS_VAR,
    build_model,
)

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

    Accepts any result shape the predictive drivers produce — an
    ``xarray.DataTree`` / ``arviz.InferenceData`` with a ``predictions`` group
    (from :func:`forecast`) or a ``posterior_predictive`` group (from
    :func:`predict_in_sample`) — or a bare ``Dataset`` (returned unchanged),
    and returns the samples as an ``xarray.Dataset`` whose variables retain
    the full ``chain`` / ``draw`` dims. Point summaries are the caller's
    choice, e.g. ``prediction_samples(result)["forecast"].mean(("chain",
    "draw"))``.

    Parameters
    ----------
    result : xarray.Dataset, xarray.DataTree or arviz.InferenceData
        A prediction result; ``predictions`` is looked up before
        ``posterior_predictive``.

    Returns
    -------
    xarray.Dataset
        The draw-level predictive samples.

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

    Accepts an ``xarray.DataTree`` / ``arviz.InferenceData`` (or any object
    with a ``posterior`` item or attribute; uses its ``posterior`` group) or a
    bare ``Dataset`` (returned unchanged).

    Parameters
    ----------
    posterior : xarray.Dataset, xarray.DataTree or arviz.InferenceData
        The posterior container.

    Returns
    -------
    xarray.Dataset
        The posterior samples.

    Raises
    ------
    TypeError
        If ``posterior`` is not a Dataset and has no ``posterior`` group.
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

    Parameters
    ----------
    posterior : xarray.Dataset, xarray.DataTree or arviz.InferenceData
        A posterior Dataset or any object with a ``posterior`` group (see
        :func:`posterior_dataset`), with ``chain`` and ``draw`` dims.
    num_samples : int
        Number of draws to keep; must be positive.
    random_seed : int or numpy.random.Generator, optional
        Seed for the draw selection (passed to ``numpy.random.default_rng``).

    Returns
    -------
    xarray.Dataset
        Posterior with dims ``(chain, draw, ...)``, ``chain`` coord ``[0]``
        and ``draw`` coord ``0 .. num_samples - 1``.

    Raises
    ------
    ValueError
        If ``num_samples`` is not positive.
    TypeError
        If ``posterior`` has no ``posterior`` group.
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


def _chunk_seeds(random_seed, num_chunks: int) -> list:
    """Derive one independent per-chunk seed from the caller's seed.

    ``None`` stays ``None`` per chunk (non-deterministic, matching the
    unbatched semantics); a ``numpy`` ``Generator`` or legacy ``RandomState``
    draws the chunk seeds off its own stream; an integer seed spawns them
    deterministically via ``SeedSequence``.
    """
    if random_seed is None:
        return [None] * num_chunks
    if isinstance(random_seed, np.random.Generator):
        return [int(s) for s in random_seed.integers(0, 2**63, size=num_chunks)]
    if isinstance(random_seed, np.random.RandomState):
        # ``SeedSequence`` cannot wrap a legacy ``RandomState``; draw the chunk
        # seeds directly (``randint`` high is bounded by ``2**31 - 1``). This
        # mirrors ``draw_posterior``'s ``RandomState`` support on the posterior
        # side so both batching knobs accept the same seed types.
        return [int(s) for s in random_seed.randint(0, 2**31 - 1, size=num_chunks)]
    children = np.random.SeedSequence(random_seed).spawn(num_chunks)
    return [int(child.generate_state(1)[0]) for child in children]


def _group_datasets(result) -> dict[str, xr.Dataset]:
    """Group-name → ``Dataset`` mapping of a predictive result (``DataTree``
    or legacy ``InferenceData``)."""
    if hasattr(result, "children"):  # xarray DataTree
        return {name: node.to_dataset() for name, node in result.children.items()}
    return {group: result[group] for group in result.groups()}


def _concat_draw_chunks(chunks: list):
    """Reassemble chunked predictive results in posterior draw order.

    Groups carrying a ``draw`` dim retain the posterior's draw coordinates;
    draw-free groups (constant data, observed data) are taken from the first
    chunk — they are identical across chunks by construction. The result has
    the same type and group structure as a single-pass call.
    """
    template = chunks[0]
    chunk_groups = [_group_datasets(chunk) for chunk in chunks]
    groups: dict[str, xr.Dataset] = {}
    for name, first in chunk_groups[0].items():
        if DRAW_DIM not in first.dims:
            groups[name] = first
            continue
        groups[name] = xr.concat([chunk[name] for chunk in chunk_groups], dim=DRAW_DIM)
    if hasattr(template, "children"):
        tree = xr.DataTree.from_dict(groups)
        tree.attrs.update(template.attrs)
        return tree
    return type(template)(**groups)


def _sample_predictive(
    posterior_ds: xr.Dataset,
    model: pm.Model,
    var_names: list[str],
    *,
    predictions: bool,
    batch_size: int | None,
    random_seed,
    progressbar: bool,
):
    """Run ``pm.sample_posterior_predictive``, optionally in draw batches.

    With ``batch_size`` set, the posterior is split into consecutive blocks of
    at most ``batch_size`` draws (per chain) and the predictive runs once per
    block against the same model; block results are concatenated along
    ``draw``. This bounds the working memory of each predictive pass — the
    port of upstream ``numpyro_forecast``'s chunk-and-offload prediction
    (juanitorduz/numpyro_forecast#65) — which matters on very wide panels
    (many series) where a single pass over all draws can exhaust memory.
    """
    if batch_size is not None and batch_size < 1:
        msg = f"batch_size must be a positive integer, got {batch_size}"
        raise ValueError(msg)
    kwargs = dict(model=model, var_names=var_names, progressbar=progressbar)
    if predictions:
        kwargs["predictions"] = True
    num_draws = posterior_ds.sizes[DRAW_DIM]
    if batch_size is None or num_draws <= batch_size:
        return pm.sample_posterior_predictive(posterior_ds, random_seed=random_seed, **kwargs)
    # Materialize implicit draw indices before slicing, otherwise PyMC starts
    # a new zero-based index in each batch of an unlabeled posterior Dataset.
    if DRAW_DIM not in posterior_ds.coords:
        posterior_ds = posterior_ds.assign_coords({DRAW_DIM: np.arange(num_draws)})
    starts = range(0, num_draws, batch_size)
    seeds = _chunk_seeds(random_seed, len(starts))
    chunks = [
        pm.sample_posterior_predictive(
            posterior_ds.isel({DRAW_DIM: slice(start, start + batch_size)}),
            random_seed=seed,
            **kwargs,
        )
        for start, seed in zip(starts, seeds, strict=True)
    ]
    return _concat_draw_chunks(chunks)


def _default_var_names(model: pm.Model) -> list[str]:
    """The forecast variable, every ``*_future`` latent, and the noise-free
    predictors registered as Deterministics, for the output."""
    names = [FORECAST_VAR]
    names += [
        rv.name for rv in model.free_RVs if rv.name.endswith("_future") and rv.name != FORECAST_VAR
    ]
    for name in (MU_FORECAST_VAR, EXPECTED_OBSERVATION_FORECAST_VAR):
        if name in model.named_vars and name not in names:
            names.append(name)
    return names


def forecast(
    model_fn,
    posterior,
    data,
    covariates,
    *,
    num_samples: int | None = None,
    var_names: Sequence[str] | None = None,
    batch_size: int | None = None,
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
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body (``(covariates, data=None) -> None`` or a
        :class:`~pymc_forecast.model.ForecastingModel`).
    posterior : xarray.Dataset, xarray.DataTree or arviz.InferenceData
        A posterior Dataset or any object with a ``posterior`` group
        (:func:`posterior_dataset`); thinned only if ``num_samples`` is given.
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like
        Observed series over the training window, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim).
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like
        Covariates on the same ``"time"`` coordinate as ``data``, spanning the
        training window plus the forecast horizon (normalized with
        ``as_dataarray``; 2-D input gets a ``"covariate"`` dim); the steps past
        ``data`` define the horizon. Required: for covariate-free models pass
        :func:`~pymc_forecast.data.null_covariates` over the full time index.
    num_samples : int, optional
        Thin ``posterior`` to this many draws first (see :func:`thin_draws`);
        ``None`` uses every draw.
    var_names : sequence of str, optional
        Variables to record (a bare string is not accepted as a single name).
        Default: ``"forecast"``, all ``*_future`` latents, and — for models
        registered through :func:`~pymc_forecast.model.predict` — the
        noise-free ``"mu_future"`` predictor plus
        ``"expected_observation_future"`` when supplied by the model. On very
        wide panels, restricting this to ``["forecast"]`` also shrinks the
        result's memory footprint.
    batch_size : int, optional
        Maximum posterior draws (per chain) per predictive pass. When set,
        the posterior is processed in consecutive blocks of at most this many
        draws and the blocks are concatenated along ``draw`` — bounding the
        working memory of each pass on very wide panels (the port of
        upstream's chunked prediction, juanitorduz/numpyro_forecast#65).
        Per-block seeds are derived from ``random_seed``, so a batched run is
        deterministic given the seed but draws different (equally valid)
        noise than an unbatched run. ``None`` runs a single pass.
    random_seed : int or numpy.random.Generator, optional
        Seed for thinning and predictive sampling.
    progressbar : bool, default False
        Show the sampling progress bar.

    Returns
    -------
    xarray.DataTree or arviz.InferenceData
        Result of ``pm.sample_posterior_predictive`` (a DataTree with current
        PyMC/ArviZ, InferenceData with older releases), with a
        ``predictions`` group carrying ``time_future`` coords.

    Raises
    ------
    pymc_forecast.exceptions.HorizonError
        If ``covariates`` has the same length as ``data`` along ``"time"`` (no
        forecast horizon; shorter covariates raise ``AlignmentError``), or the
        model body does not register ``"obs"``.
    pymc_forecast.exceptions.AlignmentError
        If ``data`` or ``covariates`` cannot be normalized or do not align.
    ValueError
        If ``num_samples`` or ``batch_size`` is not positive.
    TypeError
        If ``posterior`` has no ``posterior`` group.
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
    return _sample_predictive(
        posterior_dataset(posterior),
        model,
        list(var_names) if var_names is not None else _default_var_names(model),
        predictions=True,
        batch_size=batch_size,
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
    batch_size: int | None = None,
    random_seed=None,
    progressbar: bool = False,
):
    """Sample the in-sample posterior predictive of ``"obs"`` and registered predictors.

    The in-sample counterpart of :func:`forecast`: the model is rebuilt over
    the observed window only (no forecast horizon) and the observed variable
    is resampled given replayed latents. For models registered through
    :func:`~pymc_forecast.model.predict`, the noise-free ``"mu"`` predictor
    is recorded alongside ``"obs"``; ``"expected_observation"`` is also
    recorded when supplied by the model.

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body (``(covariates, data=None) -> None`` or a
        :class:`~pymc_forecast.model.ForecastingModel`).
    posterior : xarray.Dataset, xarray.DataTree or arviz.InferenceData
        A posterior Dataset or any object with a ``posterior`` group
        (:func:`posterior_dataset`); thinned only if ``num_samples`` is given.
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like
        Observed series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim).
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data``, covering (at
        least) the observed window (normalized with ``as_dataarray``; 2-D
        input gets a ``"covariate"`` dim); rows past ``data`` are dropped.
        ``None`` for models without covariates.
    num_samples : int, optional
        Thin ``posterior`` to this many draws first (see :func:`thin_draws`);
        ``None`` uses every draw.
    batch_size : int, optional
        Maximum posterior draws (per chain) per predictive pass; ``None`` runs
        a single pass (see :func:`forecast`).
    random_seed : int or numpy.random.Generator, optional
        Seed for thinning and predictive sampling.
    progressbar : bool, default False
        Show the sampling progress bar.

    Returns
    -------
    xarray.DataTree or arviz.InferenceData
        Result of ``pm.sample_posterior_predictive`` (a DataTree with current
        PyMC/ArviZ, InferenceData with older releases), with a
        ``posterior_predictive`` group holding ``"obs"``, plus ``"mu"`` and
        ``"expected_observation"`` when the model registers them.

    Raises
    ------
    pymc_forecast.exceptions.AlignmentError
        If ``data`` or ``covariates`` cannot be normalized or do not align.
    pymc_forecast.exceptions.HorizonError
        If the model body does not register ``"obs"``.
    ValueError
        If ``num_samples`` or ``batch_size`` is not positive.
    TypeError
        If ``posterior`` has no ``posterior`` group.
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
    var_names = [OBS_VAR]
    if MU_VAR in model.named_vars:
        var_names.append(MU_VAR)
    if EXPECTED_OBSERVATION_VAR in model.named_vars:
        var_names.append(EXPECTED_OBSERVATION_VAR)
    return _sample_predictive(
        posterior_dataset(posterior),
        model,
        var_names,
        predictions=False,
        batch_size=batch_size,
        random_seed=random_seed,
        progressbar=progressbar,
    )

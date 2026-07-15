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

from pymc_forecast.data import DRAW_DIM, TIME_DIM, as_dataarray, null_covariates
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import FORECAST_VAR, MU_FORECAST_VAR, MU_VAR, OBS_VAR, build_model

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


def _chunk_seeds(random_seed, num_chunks: int) -> list:
    """Derive one independent per-chunk seed from the caller's seed.

    ``None`` stays ``None`` per chunk (non-deterministic, matching the
    unbatched semantics); a ``numpy`` ``Generator`` draws the chunk seeds; an
    integer seed spawns them deterministically via ``SeedSequence``.
    """
    if random_seed is None:
        return [None] * num_chunks
    if isinstance(random_seed, np.random.Generator):
        return [int(s) for s in random_seed.integers(0, 2**63, size=num_chunks)]
    children = np.random.SeedSequence(random_seed).spawn(num_chunks)
    return [int(child.generate_state(1)[0]) for child in children]


def _group_datasets(result) -> dict[str, xr.Dataset]:
    """Group-name → ``Dataset`` mapping of a predictive result (``DataTree``
    or legacy ``InferenceData``)."""
    if hasattr(result, "children"):  # xarray DataTree
        return {name: node.to_dataset() for name, node in result.children.items()}
    return {group: result[group] for group in result.groups()}


def _concat_draw_chunks(chunks: list):
    """Reassemble chunked predictive results into one draw-contiguous result.

    Groups carrying a ``draw`` dim are concatenated with renumbered,
    contiguous draw coords; draw-free groups (constant data, observed data)
    are taken from the first chunk — they are identical across chunks by
    construction. The result has the same type and group structure as a
    single-pass call.
    """
    template = chunks[0]
    chunk_groups = [_group_datasets(chunk) for chunk in chunks]
    groups: dict[str, xr.Dataset] = {}
    for name, first in chunk_groups[0].items():
        if DRAW_DIM not in first.dims:
            groups[name] = first
            continue
        parts = []
        offset = 0
        for chunk in chunk_groups:
            ds = chunk[name]
            draws = np.arange(offset, offset + ds.sizes[DRAW_DIM])
            parts.append(ds.assign_coords({DRAW_DIM: draws}))
            offset += ds.sizes[DRAW_DIM]
        groups[name] = xr.concat(parts, dim=DRAW_DIM)
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
    ``mu_future`` predictor (a Deterministic, so collected explicitly), for
    the output."""
    names = [FORECAST_VAR]
    names += [
        rv.name for rv in model.free_RVs if rv.name.endswith("_future") and rv.name != FORECAST_VAR
    ]
    if MU_FORECAST_VAR in model.named_vars:
        names.append(MU_FORECAST_VAR)
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
        Variables to record. Default: ``"forecast"``, all ``*_future``
        latents, and — for models registered through
        :func:`~pymc_forecast.model.predict` — the noise-free ``"mu_future"``
        predictor. On very wide panels, restricting this to
        ``["forecast"]`` also shrinks the result's memory footprint.
    batch_size
        Maximum posterior draws (per chain) per predictive pass. When set,
        the posterior is processed in consecutive blocks of at most this many
        draws and the blocks are concatenated along ``draw`` — bounding the
        working memory of each pass on very wide panels (the port of
        upstream's chunked prediction, juanitorduz/numpyro_forecast#65).
        Per-block seeds are derived from ``random_seed``, so a batched run is
        deterministic given the seed but draws different (equally valid)
        noise than an unbatched run.
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
    """Sample the in-sample posterior predictive of the ``"obs"`` variable.

    The in-sample counterpart of :func:`forecast`: the model is rebuilt over
    the observed window only (no forecast horizon) and the observed variable
    is resampled given replayed latents. For models registered through
    :func:`~pymc_forecast.model.predict`, the noise-free ``"mu"`` predictor
    is recorded alongside ``"obs"``.

    Parameters
    ----------
    model_fn, posterior, data
        As in :func:`forecast`.
    covariates
        Covariates covering (at least) the observed window; surplus future
        steps are dropped. ``None`` for models without covariates.
    num_samples, batch_size, random_seed, progressbar
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
    var_names = [OBS_VAR]
    if MU_VAR in model.named_vars:
        var_names.append(MU_VAR)
    return _sample_predictive(
        posterior_dataset(posterior),
        model,
        var_names,
        predictions=False,
        batch_size=batch_size,
        random_seed=random_seed,
        progressbar=progressbar,
    )

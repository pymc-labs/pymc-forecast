"""Interop with the pymc-extras ``Prior`` API: declarative, user-injectable priors.

A pymc-extras :class:`~pymc_extras.prior.Prior` is accepted anywhere the model
primitives take a factory callable, so priors live as inspectable data on the
model object instead of inside lambdas::

    from pymc_extras.prior import Prior

    drift = time_series(h, "drift", Prior("Normal", mu=0, sigma=0.1))
    predict(h, Prior("Normal", sigma=Prior("HalfNormal", sigma=1)), pt.cumsum(drift))

The adapters preserve the package's replay mechanism: nested hyper-priors
(``Prior``-valued parameters) are materialized **once** per base name — e.g.
``drift_mu``, ``obs_sigma`` — and shared between the in-sample and
``*_future`` segments, so ``pm.sample_posterior_predictive`` replays them from
the posterior while the future latents are drawn conditional on them. A naive
per-segment ``create_variable`` would instead give the forecast segment fresh
hyper-priors silently drawn from the prior.

pymc-extras is never imported here — ``Prior`` objects are recognized
structurally — so the core package keeps that dependency optional.
"""

from pymc_forecast.data import FUTURE_DIM, TIME_DIM
from pymc_forecast.exceptions import HorizonError

__all__ = ["is_prior_like", "prior_obs_factory", "prior_rv_factory"]

_PRIOR_ATTRS = ("create_variable", "create_likelihood_variable", "deepcopy", "parameters", "dims")


def is_prior_like(obj) -> bool:
    """Whether ``obj`` structurally matches the pymc-extras ``Prior`` API."""
    return all(hasattr(obj, attr) for attr in _PRIOR_ATTRS)


def _materialize_hyperpriors(prior, base_name: str):
    """Create nested ``Prior`` parameters once, as ``{base_name}_{param}`` variables.

    Returns a copy of ``prior`` whose ``Prior``-valued parameters are replaced
    by model variables, so every segment built from the copy shares them. Must
    run inside a model context.
    """
    materialized = prior.deepcopy()
    for key, value in list(materialized.parameters.items()):
        if not is_prior_like(value):
            continue
        dims = tuple(value.dims or ())
        if TIME_DIM in dims or FUTURE_DIM in dims:
            msg = (
                f"hyper-prior {base_name}_{key} has per-step dims {dims}: a "
                "time-dimmed hyper-prior cannot be shared across the "
                "train/forecast split; use a callable factory instead of a Prior"
            )
            raise HorizonError(msg)
        materialized.parameters[key] = value.create_variable(f"{base_name}_{key}")
    return materialized


def _segment(materialized, dims: tuple[str, ...]):
    """The materialized prior stamped with one segment's dims.

    Re-stamped in place (segments are created sequentially): a ``deepcopy``
    here would clone the materialized hyper-prior tensors out of the model
    graph, leaving unregistered random variables in the likelihood.
    """
    materialized.dims = tuple(dims)
    return materialized


def prior_rv_factory(prior, base_name: str):
    """Adapt a ``Prior`` to the :data:`~pymc_forecast.model.RVFactory` protocol.

    The returned factory creates each segment variable (``base_name``,
    ``{base_name}_future``) from ``prior`` with the dims it is handed,
    materializing nested hyper-priors once under ``base_name`` on first use.
    """
    state = {}

    def rv_fn(name: str, dims: tuple[str, ...]):
        if "materialized" not in state:
            state["materialized"] = _materialize_hyperpriors(prior, base_name)
        return _segment(state["materialized"], dims).create_variable(name)

    return rv_fn


def prior_obs_factory(prior, base_name: str):
    """Adapt a ``Prior`` to the :data:`~pymc_forecast.model.ObsFactory` protocol.

    The prior's distribution becomes the likelihood: its location (``mu``) is
    the latent predictor handed in by :func:`~pymc_forecast.model.predict`,
    and nested hyper-priors — e.g. the noise scale — are materialized once
    under ``base_name`` and shared by the observed and forecast segments.

    The likelihood is built directly from the materialized parameters rather
    than through ``Prior.create_likelihood_variable``, which deep-copies the
    prior and would clone the shared hyper-prior variables out of the model
    graph.
    """
    if "mu" in prior.parameters:
        msg = (
            "an observation Prior must leave 'mu' unset — the latent predictor "
            f"registered by predict() becomes the location; got {prior}"
        )
        raise ValueError(msg)
    state = {}

    def obs_fn(name: str, latent, dims: tuple[str, ...], observed):
        if "materialized" not in state:
            state["materialized"] = _materialize_hyperpriors(prior, base_name)
        segment = state["materialized"]
        return segment.pymc_distribution(
            name, mu=latent, observed=observed, dims=tuple(dims), **segment.parameters
        )

    return obs_fn

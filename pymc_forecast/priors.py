"""Interop with the pymc-extras ``Prior`` API: declarative, user-injectable priors.

``innovations`` and ``predict`` accept a pymc-extras
``pymc_extras.prior.Prior``, so priors live as inspectable data on the
model object instead of inside lambdas::

    from pymc_extras.prior import Prior

    drift = innovations(h, "drift", Prior("Normal", mu=0, sigma=0.1))
    predict(h, Prior("Normal", sigma=Prior("HalfNormal", sigma=1)), pt.cumsum(drift))

The adapters preserve the package's replay mechanism: nested hyper-priors
(``Prior``-valued parameters) are materialized **once** per base name as
``{base_name}_{param}`` — e.g. ``obs_sigma`` above, or ``drift_mu`` if the
drift prior had a ``Prior``-valued ``mu`` — and shared between the in-sample and
``*_future`` segments, so ``pm.sample_posterior_predictive`` replays them from
the posterior while the future latents are drawn conditional on them. A naive
per-segment ``create_variable`` would instead give the forecast segment fresh
hyper-priors silently drawn from the prior.

On model objects, :class:`PriorConfig` turns those specifications into a
named, overridable configuration: subclasses declare ``default_priors`` and
callers replace any subset with ``priors={...}`` at construction time.

pymc-extras is never imported here — ``Prior`` objects are recognized
structurally — so the core package keeps that dependency optional.
"""

from collections.abc import Mapping

from pymc_forecast.data import FUTURE_DIM, TIME_DIM
from pymc_forecast.exceptions import HorizonError

__all__ = ["PriorConfig", "is_prior_like", "prior_obs_factory", "prior_rv_factory"]

_PRIOR_ATTRS = ("create_variable", "create_likelihood_variable", "deepcopy", "parameters", "dims")


def is_prior_like(obj) -> bool:
    """Whether ``obj`` structurally matches the pymc-extras ``Prior`` API.

    Parameters
    ----------
    obj : object
        Any object.

    Returns
    -------
    bool
        ``True`` if ``obj`` has all of ``create_variable``,
        ``create_likelihood_variable``, ``deepcopy``, ``parameters`` and
        ``dims``.
    """
    return all(hasattr(obj, attr) for attr in _PRIOR_ATTRS)


class PriorConfig:
    """Mixin: named, user-overridable priors on a model object.

    Subclasses declare their defaults in :attr:`default_priors`; callers
    override any subset with the ``priors=`` constructor argument, and the
    model body reads the effective mapping from :attr:`prior_config` — e.g.
    ``self.innovations("drift", self.prior_config["drift"])`` — or creates a
    standalone variable with :meth:`create_prior`. Mixed into
    :class:`~pymc_forecast.model.ForecastingModel` and
    :class:`~pymc_forecast.statespace.StatespaceModel`.

    Parameters
    ----------
    priors : mapping, optional
        Named overrides merged over :attr:`default_priors` (``None`` keeps the
        defaults); values are pymc-extras ``Prior`` objects, the factory
        callables the model primitives accept, or — for :meth:`create_prior` —
        any ``name -> RV`` callable. Values are not validated.
    """

    default_priors: Mapping[str, object] = {}
    """Class-level default priors, overridable per instance via ``priors=``."""

    def __init__(self, priors: Mapping[str, object] | None = None) -> None:
        self._prior_config = {**self.default_priors, **(priors or {})}

    @property
    def prior_config(self) -> dict[str, object]:
        """The effective priors, :attr:`default_priors` merged with overrides.

        A ``dict`` mapping names to specifications. This is the instance's
        live mapping, not a copy, so mutating it changes the configuration.
        Falls back to a fresh copy of the defaults when a subclass
        ``__init__`` does not call ``super().__init__()``.
        """
        config = getattr(self, "_prior_config", None)
        return dict(self.default_priors) if config is None else config

    def create_prior(self, name: str):
        """Create the model variable ``name`` from its configured prior.

        Must run inside a ``pm.Model`` context. The configured specification
        either exposes ``create_variable(name)`` (a pymc-extras ``Prior``,
        created with its own dims) or is a callable ``name -> RV``, e.g.
        ``lambda name: pm.Normal(name, 0, 1)``.

        Parameters
        ----------
        name : str
            Key of :attr:`prior_config` and name of the created variable.

        Returns
        -------
        pytensor.tensor.TensorVariable
            Whatever the specification creates, normally the registered model
            random variable.

        Raises
        ------
        KeyError
            If no prior is configured for ``name``.
        TypeError
            If the specification neither exposes ``create_variable`` nor is
            callable.
        """
        try:
            spec = self.prior_config[name]
        except KeyError as err:
            available = sorted(self.prior_config)
            msg = f"no prior configured for {name!r}; available priors: {available}"
            raise KeyError(msg) from err
        create_variable = getattr(spec, "create_variable", None)
        if create_variable is not None:
            return create_variable(name)
        if callable(spec):
            return spec(name)
        msg = (
            f"prior {name!r} must expose create_variable(name) or be a callable "
            f"taking the variable name; got {type(spec).__name__}"
        )
        raise TypeError(msg)


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
    """Adapt a ``Prior`` to the ``pymc_forecast.model.RVFactory`` protocol.

    The returned factory creates whatever variable ``name`` it is called with
    (e.g. ``base_name`` and ``{base_name}_future``) from ``prior`` with the dims
    it is handed, materializing nested hyper-priors once under ``base_name`` on
    first use. It is stateful, so build a new factory for each model.

    Parameters
    ----------
    prior : pymc_extras.prior.Prior
        The prior to adapt; it is deep-copied, not mutated.
    base_name : str
        Prefix for the hyper-prior variables, named ``{base_name}_{param}``.

    Returns
    -------
    callable
        A factory ``rv_fn(name, dims)`` that registers and returns the variable
        in the active ``pm.Model``. Its first call raises
        :class:`~pymc_forecast.exceptions.HorizonError` if a nested hyper-prior
        has a ``"time"`` or ``"time_future"`` dim.
    """
    state = {}

    def rv_fn(name: str, dims: tuple[str, ...]):
        if "materialized" not in state:
            state["materialized"] = _materialize_hyperpriors(prior, base_name)
        return _segment(state["materialized"], dims).create_variable(name)

    return rv_fn


def prior_obs_factory(prior, base_name: str):
    """Adapt a ``Prior`` to the ``pymc_forecast.model.ObsFactory`` protocol.

    The prior's distribution becomes the likelihood: its location (``mu``) is
    the latent predictor handed in by :func:`~pymc_forecast.model.predict`,
    and nested hyper-priors — e.g. the noise scale — are materialized once
    under ``base_name`` and shared by the observed and forecast segments.

    The likelihood is built directly from the materialized parameters rather
    than through ``Prior.create_likelihood_variable``, which deep-copies the
    prior and would clone the shared hyper-prior variables out of the model
    graph.

    Parameters
    ----------
    prior : pymc_extras.prior.Prior
        The observation prior; ``mu`` must be unset and its distribution must
        accept a ``mu`` keyword.
    base_name : str
        Prefix for the hyper-prior variables, named ``{base_name}_{param}``.

    Returns
    -------
    callable
        A factory ``obs_fn(name, latent, dims, observed)`` that registers and
        returns the likelihood variable in the active ``pm.Model``. Its first
        call raises :class:`~pymc_forecast.exceptions.HorizonError` if a nested
        hyper-prior has a ``"time"`` or ``"time_future"`` dim.

    Raises
    ------
    ValueError
        If ``prior`` sets ``mu`` (checked immediately).
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

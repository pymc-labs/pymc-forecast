"""Private distribution helpers. Not part of the public API."""

import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.shape_utils import change_dist_size
from pytensor.tensor.basic import get_underlying_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError

from pymc_forecast.exceptions import HorizonError


def is_dist(value) -> bool:
    """Whether ``value`` is a ``.dist()``-style random variable (its op has ``ndim_supp``)."""
    return (
        isinstance(value, pt.TensorVariable)
        and value.owner is not None
        and hasattr(value.owner.op, "ndim_supp")
    )


def is_zero_constant(var) -> bool:
    """Whether ``var`` is a compile-time constant equal to zero everywhere."""
    try:
        return get_underlying_scalar_constant_value(var) == 0
    except NotScalarConstantError:
        return False


def resize_dist(dist: pt.TensorVariable, shape: tuple) -> pt.TensorVariable:
    """Resize ``dist`` to the full registered ``shape``.

    Registration attaches dim names but does not infer the size, so the batch
    part of ``shape`` replaces the dist's size. Multivariate support axes stay
    out of the batch size, as PyMC constructors do, and the parameters
    broadcast against it (NumPy rules, right-aligned).
    """
    shape = tuple(shape)
    batch = shape[: len(shape) - dist.owner.op.ndim_supp]
    return change_dist_size(dist, batch, expand=False)


def expand_dist(dist: pt.TensorVariable, shape: tuple, *, owner: str) -> pt.TensorVariable:
    """Resize ``dist`` to ``shape`` = ``(time_axis, *dims)``.

    See :func:`resize_dist`. A per-series scale or a multivariate support
    axis maps onto the trailing ``dims`` instead of gaining a duplicate axis.
    The leading time axis belongs to the caller, so a dist with more axes
    than ``dims`` is rejected.
    """
    n_dims = len(shape) - 1
    if dist.ndim > n_dims:
        msg = (
            f"{owner} owns the time axis: the .dist() has {dist.ndim} axes but "
            f"only {n_dims} non-time dims were declared"
        )
        raise HorizonError(msg)
    return resize_dist(dist, shape)


def _unregistered_random_ancestors(dist: pt.TensorVariable, model) -> list:
    """Random variables behind ``dist`` that ``model`` does not register.

    Registered model variables are first replaced by their value variables, so
    symbolic distributions (``Censored``, ``CustomDist(dist=...)``) that clone
    their inputs while deriving a logp are not mistaken for unregistered ones.
    Distributions without a logp (e.g. a random-only ``CustomDist``) are
    checked through their inputs instead.
    """
    registered = {id(rv) for rv in model.basic_RVs}
    found, seen = [], set()
    (replaced,) = model.replace_rvs_by_values([dist])
    try:
        stack = [pm.logp(replaced, replaced.type(), warn_rvs=False)]
    except NotImplementedError:
        stack = list(replaced.owner.inputs)
    while stack:
        var = stack.pop()
        if id(var) in seen:
            continue
        seen.add(id(var))
        if var.owner is None:
            continue
        if isinstance(var, pt.TensorVariable) and hasattr(var.owner.op, "ndim_supp"):
            if id(var) not in registered:
                found.append(var)
            continue
        stack.extend(var.owner.inputs)
    return found


def check_unnamed_dist(dist: pt.TensorVariable, *, owner: str) -> None:
    """Reject a ``.dist()`` the active model cannot register as an unnamed dist.

    Raises :class:`~pymc_forecast.exceptions.HorizonError` when ``dist`` is
    itself a model variable (``.dist()`` was dropped) or depends on random
    variables the model does not register, which would leave the model's
    logp with unvalued random variables.
    """
    model = pm.modelcontext(None)
    if any(dist is rv for rv in model.basic_RVs):
        msg = (
            f"{owner} received the model variable {dist.name!r}, not an unnamed "
            "distribution; pass e.g. pm.Normal.dist(...) instead of pm.Normal(name, ...)"
        )
        raise HorizonError(msg)
    unregistered = _unregistered_random_ancestors(dist, model)
    if unregistered:
        ops = ", ".join(sorted({type(rv.owner.op).__name__ for rv in unregistered}))
        msg = (
            f"the {owner} .dist() depends on random variables that are not model "
            f"variables ({ops}); create each as a model variable, e.g. "
            "sigma = pm.HalfNormal('sigma', 1), and pass that in"
        )
        raise HorizonError(msg)

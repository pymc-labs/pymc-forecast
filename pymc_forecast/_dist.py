"""Private distribution helpers. Not part of the public API."""

import pytensor.tensor as pt
from pymc.distributions.shape_utils import change_dist_size

from pymc_forecast.exceptions import HorizonError


def is_dist(value) -> bool:
    """Whether ``value`` is an unnamed ``.dist()`` random variable."""
    return (
        isinstance(value, pt.TensorVariable)
        and value.owner is not None
        and hasattr(value.owner.op, "ndim_supp")
    )


def expand_dist(dist: pt.TensorVariable, shape: tuple, *, owner: str) -> pt.TensorVariable:
    """Resize ``dist`` to ``shape`` = ``(time_axis, *dims)``.

    The batch part of ``shape`` replaces the dist's size, and its parameters
    broadcast against it (NumPy rules, right-aligned). A per-series scale or a
    multivariate support axis maps onto the trailing ``dims`` instead of
    gaining a duplicate axis. The leading time axis belongs to the caller,
    so a dist with more axes than ``dims`` is rejected.
    """
    n_dims = len(shape) - 1
    if dist.ndim > n_dims:
        msg = (
            f"{owner} owns the time axis: the .dist() has {dist.ndim} axes but "
            f"only {n_dims} non-time dims were declared"
        )
        raise HorizonError(msg)
    batch = tuple(shape)[: len(shape) - dist.owner.op.ndim_supp]
    return change_dist_size(dist, batch, expand=False)

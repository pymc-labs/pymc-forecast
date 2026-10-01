"""Private distribution helpers. Not part of the public API."""

import pytensor.tensor as pt


def expand_dist(dist: pt.TensorVariable, shape) -> pt.TensorVariable:
    """Resize an unnamed ``.dist()`` tensor. Not public."""
    from pymc.distributions.distribution import _change_dist_size

    return _change_dist_size(dist.owner.op, dist, shape, expand=True)

"""Markov (state-space) time-series latents via ``pytensor.scan``.

The scan analogue of :func:`~pymc_forecast.model.time_series`: in-sample steps
run in one scan-backed ``pm.CustomDist`` under the base name; when forecasting,
horizon steps run in a second scan-backed ``CustomDist`` named
``{name}_future`` whose initial state is the **final in-sample value** — so
under posterior replay the forecast is conditioned through the state, exactly
upstream's "seeded by the final carry" invariant.

This follows PyMC's documented pattern for time series derived from a
generative graph (``CustomDist`` + ``scan`` + ``collect_default_updates``),
which is also what makes the in-sample latent's logp derivable for NUTS/ADVI.
Two constraints follow from it:

- every random *parameter* used by the transition must be passed via
  ``params=`` (PyMC needs them as explicit inputs of the generative graph);
- the transition returns a **distribution** for the next step
  (``pm.Normal.dist(...)``), not a sample — the wrapper owns the randomness,
  so the Markov structure cannot be broken by user resampling.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.pytensorf import collect_default_updates

from pymc_forecast.data import FUTURE_DIM, TIME_DIM
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import Horizon

__all__ = ["markov_time_series"]

Transition = Callable[..., pt.TensorVariable]
"""``(z_prev, *params) -> dist`` (or ``(z_prev, x_t, *params) -> dist`` with
exogenous inputs): returns the next step's distribution as a ``.dist()``
variable, e.g. ``lambda z, mu, sigma: pm.Normal.dist(z + mu, sigma)``."""


def _make_dist(transition: Transition, n_steps: int, xs_slice: np.ndarray | None):
    """Build the ``CustomDist`` generative function for one scan segment."""

    def dist_fn(init_value, *params_and_size):
        # CustomDist appends `size` as the final positional argument; the scan
        # produces a fixed-length sequence, so we split it off and ignore it.
        *param_values, _size = params_and_size

        def step(*args):
            if xs_slice is not None:
                x_t, z_prev, *params = args
                next_dist = transition(z_prev, x_t, *params)
            else:
                z_prev, *params = args
                next_dist = transition(z_prev, *params)
            return next_dist, collect_default_updates(next_dist)

        # The carry dtype must match the step output. The transition builds its
        # distribution from float64 parameters/data, so pin the initial state
        # (which may arrive as a floatX=float32 constant) to float64 to avoid a
        # scan upcast/downcast mismatch.
        init = pt.as_tensor_variable(init_value).astype("float64")
        sequences = (
            None if xs_slice is None else [pt.as_tensor_variable(xs_slice).astype("float64")]
        )
        seq, _ = pytensor.scan(
            step,
            sequences=sequences,
            outputs_info=[init],
            non_sequences=list(param_values),
            n_steps=n_steps if xs_slice is None else None,
            strict=False,
        )
        return seq

    return dist_fn


def markov_time_series(
    h: Horizon,
    name: str,
    init,
    transition: Transition,
    *,
    params: Sequence = (),
    xs=None,
    dims: tuple[str, ...] = (),
) -> pt.TensorVariable:
    """Sample a Markov latent over the full horizon.

    Parameters
    ----------
    h
        The horizon of the current model build.
    name
        Base variable name of the in-sample latent; the forecast segment is
        named ``f"{name}_future"``.
    init
        Initial state fed to the first transition — a constant or a model
        variable (e.g. ``pm.Normal("z0", 0, 1)``).
    transition
        ``(z_prev, *params) -> dist`` (with ``xs``:
        ``(z_prev, x_t, *params) -> dist``). Must return a ``.dist()``
        distribution for the next step.
    params
        Every random variable the transition uses (beyond the state), passed
        as explicit generative-graph inputs. Deterministic constants may be
        closed over freely.
    xs
        Optional exogenous inputs over the full horizon, time on axis 0
        (numpy array or DataArray).
    dims
        Extra (non-time) dims of the per-step state, e.g. ``("series",)``.

    Returns
    -------
    TensorVariable
        The latent over the full horizon, time on axis 0.

    Raises
    ------
    HorizonError
        When forecasting without observed data, or when ``xs`` does not span
        the full horizon.
    """
    if h.future > 0 and h.data is None:
        msg = "markov_time_series requires observed data when forecasting"
        raise HorizonError(msg)

    xs_pre = xs_fut = None
    if xs is not None:
        xs_values = np.asarray(getattr(xs, "values", xs))
        # xs spans the full forecast horizon; each build consumes its own
        # prefix. The training build has future=0 (duration == t_obs) and reads
        # only the observed slice; the forecast build reads through to the
        # horizon end. Anything beyond the current duration is left for a later
        # build, so require >= duration rather than an exact match.
        if xs_values.shape[0] < h.duration:
            msg = (
                f"xs must cover at least the horizon ({h.duration} steps along "
                f"axis 0), got {xs_values.shape[0]}"
            )
            raise HorizonError(msg)
        xs_pre = xs_values[: h.t_obs]
        xs_fut = xs_values[h.t_obs : h.duration]

    ndim_supp = 1 + len(dims)
    prefix = pm.CustomDist(
        name,
        init,
        *params,
        dist=_make_dist(transition, h.t_obs, xs_pre),
        ndim_supp=ndim_supp,
        dims=(TIME_DIM, *dims),
    )
    if h.future == 0:
        return prefix
    suffix = pm.CustomDist(
        f"{name}_future",
        prefix[-1],  # conditioning through the final in-sample state
        *params,
        dist=_make_dist(transition, h.future, xs_fut),
        ndim_supp=ndim_supp,
        dims=(FUTURE_DIM, *dims),
    )
    return pt.concatenate([prefix, suffix], axis=0)

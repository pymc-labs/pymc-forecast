"""Seasonal feature builders: Fourier design matrices and periodic tiling."""

import numpy as np
import xarray as xr

from pymc_forecast.data import TIME_DIM

__all__ = ["fourier_features", "periodic_repeat"]


def fourier_features(time, *, period: float, num_terms: int) -> xr.DataArray:
    """Build a labeled Fourier seasonality design matrix.

    Parameters
    ----------
    time : int or array_like
        Either a Python ``int`` duration (phases ``0..duration-1``) or an array
        of numeric time positions (e.g. ``np.arange(len(index))`` or fractional
        day-of-week positions). Only a built-in ``int`` takes the duration
        branch; anything else (including a NumPy integer scalar) is converted
        with ``np.asarray(time, dtype=float)``. The positions set the phase;
        the returned ``"time"`` coord carries them.
    period : float
        Seasonal period, in the same units as ``time``.
    num_terms : int
        Number of harmonics; the output has ``2 * num_terms`` columns.

    Returns
    -------
    xarray.DataArray
        Dims ``("time", "fourier")`` with shape ``(n, 2 * num_terms)``. The
        ``"time"`` coord holds the positions (not the data's time coords) and
        the ``"fourier"`` coord labels each harmonic
        (``sin_1 .. sin_k, cos_1 .. cos_k``).

    Raises
    ------
    ValueError
        If ``num_terms < 1``.
    """
    if num_terms < 1:
        msg = f"num_terms must be >= 1, got {num_terms}"
        raise ValueError(msg)
    positions = np.arange(time) if isinstance(time, int) else np.asarray(time, dtype=float)
    angles = 2.0 * np.pi * np.arange(1, num_terms + 1)[None, :] * positions[:, None] / period
    values = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    labels = [f"sin_{k}" for k in range(1, num_terms + 1)] + [
        f"cos_{k}" for k in range(1, num_terms + 1)
    ]
    return xr.DataArray(
        values,
        dims=(TIME_DIM, "fourier"),
        coords={TIME_DIM: positions, "fourier": labels},
    )


def periodic_repeat(pattern, duration: int, *, axis: int = 0, period: int | None = None):
    """Tile a seasonal pattern to cover ``duration`` time steps.

    Works on numpy arrays and on PyTensor variables (e.g. a sampled seasonal
    latent inside a model). For PyTensor variables the length along ``axis``
    is always symbolic, so ``period`` must be passed explicitly.

    Parameters
    ----------
    pattern : numpy.ndarray or pytensor.tensor.TensorVariable
        The seasonal pattern; its length along ``axis`` is the period unless
        ``period`` is given.
    duration : int
        Target length along ``axis``.
    axis : int, default 0
        Axis to repeat along.
    period : int, optional
        Explicit period; only the first ``period`` entries of ``pattern``
        along ``axis`` are repeated. ``None`` reads it from a NumPy
        ``pattern``'s shape; it is required for every PyTensor ``pattern``.

    Returns
    -------
    numpy.ndarray or pytensor.tensor.TensorVariable
        The tiled pattern with length ``duration`` along ``axis``: a NumPy
        array for a NumPy ``pattern``, otherwise a symbolic tensor.

    Raises
    ------
    ValueError
        If ``period`` is ``None`` and the length of ``pattern`` along ``axis``
        is symbolic (any PyTensor ``pattern``).
    """
    if period is None:
        size = pattern.shape[axis]
        try:
            period = int(size)
        except TypeError as err:  # symbolic dimension
            msg = (
                "pattern length along the repeat axis is not statically known; "
                "pass period= explicitly"
            )
            raise ValueError(msg) from err
    indices = np.arange(duration) % period
    if isinstance(pattern, np.ndarray):
        return np.take(pattern, indices, axis=axis)
    import pytensor.tensor as pt

    return pt.take(pattern, indices, axis=axis)

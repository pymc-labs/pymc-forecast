"""Seasonal feature builders: Fourier design matrices and periodic tiling."""

import numpy as np
import xarray as xr

from pymc_forecast.data import TIME_DIM

__all__ = ["fourier_features", "periodic_repeat"]


def fourier_features(time, *, period: float, num_terms: int) -> xr.DataArray:
    """Build a labeled Fourier seasonality design matrix.

    Parameters
    ----------
    time
        Either an integer duration (phases ``0..duration-1``) or an array of
        numeric time positions (e.g. ``np.arange(len(index))`` or fractional
        day-of-week positions). The positions set the phase; the returned
        ``"time"`` coord carries them.
    period
        Seasonal period, in the same units as ``time``.
    num_terms
        Number of harmonics; the output has ``2 * num_terms`` columns.

    Returns
    -------
    xarray.DataArray
        Dims ``("time", "fourier")`` with coords labeling each harmonic
        (``sin_1 .. sin_k, cos_1 .. cos_k``).
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
    latent inside a model). For symbolic tensors whose length along ``axis`` is
    not statically known, pass ``period`` explicitly.

    Parameters
    ----------
    pattern
        The seasonal pattern; its length along ``axis`` is the period.
    duration
        Target length along ``axis``.
    axis
        Axis to repeat along (defaults to ``0``, the package's time axis).
    period
        Explicit period, required only when it cannot be read from
        ``pattern``'s static shape.
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

"""Correlated-over-time observation noise: the Gaussian prefix conditional.

For elementwise noise the forecast distribution is the horizon marginal (what
:func:`~pymc_forecast.model.predict` does implicitly). When the observation
noise is a ``MultivariateNormal`` over the *time* axis — GP-style correlated
residuals — the forecast must instead be the Gaussian conditional given the
observed prefix. This module ports numpyro_forecast's ``_mvn_prefix_condition``
(Cholesky of the prefix block, triangular solves for the conditional mean and
covariance, jitter + symmetrization) to PyTensor and wires it into a
``predict``-style primitive.

Univariate series only (the MVN event dimension is time), matching upstream.
"""

import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.linalg import cholesky, solve_triangular

from pymc_forecast.data import FUTURE_DIM, TIME_DIM
from pymc_forecast.exceptions import HorizonError
from pymc_forecast.model import FORECAST_VAR, OBS_VAR, Horizon

__all__ = ["conditional_mvn", "predict_mvn"]

DEFAULT_JITTER = 1e-6
"""Default additive diagonal jitter of :func:`conditional_mvn` and :func:`predict_mvn`."""


def _symmetrize(cov):
    return 0.5 * (cov + cov.T)


def _jittered(cov, jitter: float):
    return _symmetrize(cov) + jitter * pt.eye(cov.shape[0])


def _chol_solve(chol, b):
    """Solve ``A x = b`` given the lower Cholesky factor of ``A``."""
    y = solve_triangular(chol, b, lower=True)
    return solve_triangular(chol.T, y, lower=False)


def conditional_mvn(loc, cov, observed_prefix, *, jitter: float = DEFAULT_JITTER):
    """Condition a length-``t+f`` MVN on its first ``t`` observed values.

    Parameters
    ----------
    loc : numpy.ndarray or pytensor.tensor.TensorVariable
        Full-horizon mean, shape ``(t + f,)``; must be a tensor when
        ``observed_prefix`` is a tensor.
    cov : numpy.ndarray or pytensor.tensor.TensorVariable
        Full-horizon covariance, shape ``(t + f, t + f)``; it is symmetrized
        before use.
    observed_prefix : numpy.ndarray or pytensor.tensor.TensorVariable
        Observed values, shape ``(t,)``.
    jitter : float, default 1e-6
        Additive diagonal term, added to the symmetrized joint covariance and
        again to the conditional covariance.

    Returns
    -------
    cond_mean : pytensor.tensor.TensorVariable
        Mean of the forecast conditional, shape ``(f,)``.
    cond_cov : pytensor.tensor.TensorVariable
        Covariance of the forecast conditional, shape ``(f, f)``.
    """
    t = observed_prefix.shape[0]
    cov = _jittered(cov, jitter)
    cov_pp = cov[:t, :t]
    cov_pf = cov[:t, t:]
    cov_fp = cov[t:, :t]
    cov_ff = cov[t:, t:]
    chol = cholesky(cov_pp, lower=True)
    resid = observed_prefix - loc[:t]
    cond_mean = loc[t:] + cov_fp @ _chol_solve(chol, resid)
    cond_cov = _jittered(cov_ff - cov_fp @ _chol_solve(chol, cov_pf), jitter)
    return cond_mean, cond_cov


def predict_mvn(h: Horizon, loc, cov, *, jitter: float = DEFAULT_JITTER) -> None:
    """Register MVN-over-time observation/forecast variables.

    The correlated-noise counterpart of :func:`~pymc_forecast.model.predict`:
    the likelihood is ``MvNormal`` over the observed window (the prefix block
    of ``cov``) and — when forecasting — the ``"forecast"`` variable is the
    exact Gaussian conditional given the observed prefix, not the horizon
    marginal.

    Only ``"obs"`` and (when forecasting) ``"forecast"`` are registered; unlike
    :func:`~pymc_forecast.model.predict`, no ``"mu"`` / ``"mu_future"``
    Deterministics are recorded.

    Parameters
    ----------
    h : pymc_forecast.model.Horizon
        The horizon of the current model build; its data, if any, must be 1-D.
    loc : array_like or pytensor.tensor.TensorVariable
        Full-horizon mean with shape ``(duration,)`` (time on axis 0).
    cov : array_like or pytensor.tensor.TensorVariable
        Full-horizon covariance with shape ``(duration, duration)`` — e.g. a
        GP kernel gram matrix over the time positions.
    jitter : float, default 1e-6
        Additive diagonal term for numerical stability, added to the
        symmetrized covariance and again to the conditional forecast
        covariance (see :func:`conditional_mvn`).

    Raises
    ------
    pymc_forecast.exceptions.HorizonError
        If the data are multivariate (the MVN event dim must be time), or if
        forecasting without observed data.
    """
    loc = pt.as_tensor_variable(loc)
    cov = pt.as_tensor_variable(cov)
    if h.data is not None and h.data.ndim != 1:
        msg = (
            "predict_mvn supports univariate series only (data dims "
            f"('time',)); got dims {h.data.dims}"
        )
        raise HorizonError(msg)
    if h.future == 0:
        observed = None if h.data is None else h.data.values
        pm.MvNormal(
            OBS_VAR, mu=loc, cov=_jittered(cov, jitter), observed=observed, dims=(TIME_DIM,)
        )
        return
    if h.data is None:
        msg = "forecasting requires observed data"
        raise HorizonError(msg)
    observed = h.data.values
    t = h.t_obs
    full_cov = _jittered(cov, jitter)
    pm.MvNormal(
        OBS_VAR,
        mu=loc[:t],
        cov=full_cov[:t, :t],
        observed=observed,
        dims=(TIME_DIM,),
    )
    cond_mean, cond_cov = conditional_mvn(loc, cov, pt.as_tensor_variable(observed), jitter=jitter)
    pm.MvNormal(FORECAST_VAR, mu=cond_mean, cov=cond_cov, dims=(FUTURE_DIM,))

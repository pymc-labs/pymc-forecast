"""Numerical parity of the MVN prefix conditional against a numpy reference."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc_forecast.exceptions import HorizonError
from pymc_forecast.gaussian import conditional_mvn, predict_mvn
from pymc_forecast.model import build_model

T_OBS = 12
HORIZON = 4
DURATION = T_OBS + HORIZON


def rbf_kernel(n: int, ell: float = 3.0, amp: float = 1.0) -> np.ndarray:
    x = np.arange(n, dtype=float)
    return amp * np.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2)


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(11)
    loc = np.linspace(0.0, 2.0, DURATION)
    cov = rbf_kernel(DURATION) + 1e-6 * np.eye(DURATION)
    full_draw = rng.multivariate_normal(loc, cov)
    observed = full_draw[:T_OBS]
    return loc, cov, observed


def numpy_conditional(loc, cov, observed, jitter=1e-6):
    """Reference Gaussian conditional using the same jitter as the port.

    The RBF kernel is near-singular, so the diagonal floor materially affects
    the solve; matching it here makes this an apples-to-apples parity check of
    the PyTensor algorithm rather than of two different regularizations.
    """
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T) + jitter * np.eye(n)
    t = len(observed)
    cov_pp, cov_pf = cov[:t, :t], cov[:t, t:]
    cov_fp, cov_ff = cov[t:, :t], cov[t:, t:]
    solve = np.linalg.solve
    mean = loc[t:] + cov_fp @ solve(cov_pp, observed - loc[:t])
    sigma = cov_ff - cov_fp @ solve(cov_pp, cov_pf)
    sigma = 0.5 * (sigma + sigma.T) + jitter * np.eye(sigma.shape[0])
    return mean, sigma


class TestConditionalMvn:
    def test_matches_numpy_reference(self, problem):
        loc, cov, observed = problem
        mean_ref, cov_ref = numpy_conditional(loc, cov, observed)
        mean_pt, cov_pt = conditional_mvn(loc, cov, observed)
        np.testing.assert_allclose(mean_pt.eval(), mean_ref, atol=1e-8)
        np.testing.assert_allclose(cov_pt.eval(), cov_ref, atol=1e-8)


def mvn_model_factory(loc, cov):
    def model_fn(h, covariates):
        predict_mvn(h, loc, cov)

    return model_fn


class TestPredictMvn:
    def test_forecast_samples_match_conditional(self, problem):
        loc, cov, observed = problem
        data = xr.DataArray(observed, dims=("time",), coords={"time": np.arange(T_OBS)})
        covariates = xr.DataArray(
            np.zeros((DURATION, 0)),
            dims=("time", "covariate"),
            coords={"time": np.arange(DURATION)},
        )
        model = build_model(mvn_model_factory(loc, cov), data, covariates)
        # all parameters are constants, so the forecast RV *is* the conditional
        with model:
            draws = pm.sample_prior_predictive(draws=4000, random_seed=5)
        samples = draws["prior"]["forecast"].stack(s=("chain", "draw")).transpose("s", ...).values
        mean_ref, cov_ref = numpy_conditional(loc, cov, observed)
        np.testing.assert_allclose(samples.mean(axis=0), mean_ref, atol=0.05)
        np.testing.assert_allclose(np.cov(samples.T), cov_ref, atol=0.05)

    def test_conditional_differs_from_marginal(self, problem):
        # regression guard: correlated noise must NOT reduce to the marginal
        loc, cov, observed = problem
        mean_ref, cov_ref = numpy_conditional(loc, cov, observed)
        assert np.abs(mean_ref - loc[T_OBS:]).max() > 0.1
        assert np.abs(np.diag(cov_ref) - np.diag(cov[T_OBS:, T_OBS:])).max() > 0.1

    def test_multivariate_data_rejected(self, problem):
        loc, cov, _ = problem
        data = np.zeros((T_OBS, 2))
        covariates = np.zeros((DURATION, 0))
        with pytest.raises(HorizonError, match="univariate"):
            build_model(mvn_model_factory(loc, cov), data, covariates)

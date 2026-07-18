"""Forecaster classes: fit a forecasting model, then draw probabilistic forecasts.

Three inference backends behind one interface: :class:`Forecaster`
(variational, ADVI by default), :class:`HMCForecaster` (MCMC via
``pm.sample``), and :class:`PathfinderForecaster` (pymc-extras Pathfinder).
Construction fits the model on ``(data, covariates)``; :meth:`~BaseForecaster.forecast`
then rebuilds the model over extended covariates and samples the horizon.
Construct without data to defer the fit (:meth:`~BaseForecaster.fit`).
"""

import abc
import warnings
from collections.abc import Mapping

import numpy as np
import pymc as pm
import xarray as xr

from pymc_forecast.data import (
    TIME_DIM,
    as_dataarray,
    concat_covariates,
    concat_time_index,
    extend_time_index,
    null_covariates,
)
from pymc_forecast.exceptions import (
    AlignmentError,
    MethodResolutionError,
    NotFittedError,
    OptionalDependencyError,
)
from pymc_forecast.model import build_model
from pymc_forecast.prediction import (
    forecast as _forecast,
)
from pymc_forecast.prediction import (
    posterior_dataset,
    predict_in_sample,
    thin_draws,
)

__all__ = ["Forecaster", "HMCForecaster", "PathfinderForecaster"]

DEFAULT_LEARNING_RATE = 0.01
"""Default Adam learning rate for variational fits (matches upstream)."""

DEFAULT_NUM_SAMPLES = 100
"""Default number of posterior draws for the predictive methods."""

CONVERGENCE_WINDOW_FRACTION = 0.1
"""Fraction of the ELBO loss history in each of the two windows (one at the
midpoint of the run, one at the end) compared by the post-fit ADVI
convergence check (see :func:`_check_vi_convergence`)."""

CONVERGENCE_MIN_WINDOW = 10
"""Minimum steps per convergence-check window; shorter loss histories are
too noisy to assess and are skipped."""


class BaseForecaster(abc.ABC):
    """Shared fit/forecast plumbing.

    Fitting happens on construction when ``data`` is given. Constructing
    without data defers it — configure now, :meth:`fit` later (the
    sklearn-style lifecycle adapter authors need)::

        fc = HMCForecaster(model_fn, draws=500)   # not fitted yet
        fc.fit(data, covariates)                  # returns self

    Parameters
    ----------
    model_fn
        The model body (``(Horizon, covariates) -> None`` or a
        :class:`~pymc_forecast.model.ForecastingModel`).
    data
        Observed training data, or ``None`` to construct unfitted and call
        :meth:`fit` later.
    covariates
        Covariates covering (at least) the training window; surplus future
        steps are ignored during fitting. ``None`` for models without
        covariates.
    random_seed
        Seed for the fit.
    """

    def __init__(self, model_fn, data=None, covariates=None, *, random_seed=None) -> None:
        self.model_fn = model_fn
        self._random_seed = random_seed
        self._is_fitted = False
        self.model = None
        if data is None:
            if covariates is not None:
                msg = (
                    "covariates were given without data; pass both to fit on "
                    "construction, or neither and call fit(data, covariates) later"
                )
                raise ValueError(msg)
            return
        self.fit(data, covariates, random_seed=random_seed)

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed successfully."""
        return self._is_fitted

    def fit(self, data, covariates=None, *, random_seed=None) -> "BaseForecaster":
        """Fit the model on ``(data, covariates)`` and return ``self``.

        Called automatically when the forecaster is constructed with data;
        call it explicitly on a forecaster constructed without data (or to
        refit an existing one on new data — the backend configuration is
        reused).

        Parameters
        ----------
        data, covariates
            As in the constructor (``data`` is required here).
        random_seed
            Seed for the fit; defaults to the constructor's ``random_seed``.
        """
        self._is_fitted = False
        self._data = as_dataarray(data, role="data")
        if covariates is None:
            cov = null_covariates(self._data[TIME_DIM].values)
        else:
            cov = as_dataarray(covariates, role="covariates")
            cov = cov.isel({TIME_DIM: slice(None, self._data.sizes[TIME_DIM])})
        self._covariates = cov
        self.model = self._build_model()
        self._fit(self._random_seed if random_seed is None else random_seed)
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            msg = (
                f"this {type(self).__name__} is not fitted yet; construct it "
                "with data or call fit(data, covariates) first"
            )
            raise NotFittedError(msg)

    def _build_model(self) -> pm.Model:
        """Build the training model from the normalized data (called once per
        :meth:`fit`); adapters for other model lifecycles override this."""
        return build_model(self.model_fn, self._data, self._covariates)

    @abc.abstractmethod
    def _fit(self, random_seed) -> None:
        """Fit the training model (called once per :meth:`fit`)."""

    _batch_generated_posterior = False
    """Whether posterior draws are generated on demand and benefit from batching."""

    def draw_posterior(
        self,
        num_samples: int,
        random_seed=None,
        *,
        batch_size: int | None = None,
    ) -> xr.Dataset:
        """Return ``num_samples`` posterior draws as a posterior ``Dataset``.

        Feed the result to the ``posterior=`` argument of :meth:`forecast`
        and :meth:`predict_in_sample` to condition several predictive calls
        on the same draws (see those methods).

        Parameters
        ----------
        num_samples
            Number of posterior draws.
        random_seed
            Seed for posterior sampling.
        batch_size
            For backends that generate posterior draws on demand (currently
            :class:`Forecaster`), draw at most this many at once and
            concatenate the host-backed xarray chunks. On wide panels this
            bounds the peak allocation made by ``pm.Approximation.sample`` —
            the posterior side of upstream's chunked sampling
            (juanitorduz/numpyro_forecast#65); the predictive side is the
            ``batch_size`` argument of :meth:`forecast` /
            :meth:`predict_in_sample`. ``None`` keeps the single-shot path.
            Backends whose posterior is already materialized (HMC and
            Pathfinder) thin it once and do not need this memory knob.
        """
        self._require_fitted()
        if batch_size is not None and batch_size <= 0:
            msg = f"batch_size must be positive, got {batch_size}"
            raise ValueError(msg)
        if batch_size is None or not self._batch_generated_posterior:
            return self._draw_posterior(num_samples, random_seed)
        if batch_size >= num_samples:
            return self._draw_posterior(num_samples, random_seed)

        # Passing one Generator through all calls gives every chunk a fresh,
        # deterministic child seed without mutating global NumPy state. PyMC's
        # legacy RandomState is kept as-is because ``default_rng`` cannot wrap it.
        rng = (
            random_seed
            if isinstance(random_seed, np.random.RandomState)
            else np.random.default_rng(random_seed)
        )
        chunks: list[xr.Dataset] = []
        offset = 0
        while offset < num_samples:
            size = min(batch_size, num_samples - offset)
            chunk = self._draw_posterior(size, rng)
            if chunk.sizes.get("chain") != 1:
                msg = (
                    "generated posterior batches must have one chain; got "
                    f"sizes {dict(chunk.sizes)}"
                )
                raise ValueError(msg)
            chunk = chunk.assign_coords(draw=np.arange(offset, offset + size))
            chunks.append(chunk)
            offset += size
        return xr.concat(
            chunks,
            dim="draw",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )

    @abc.abstractmethod
    def _draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Backend-specific posterior sampling (fit is guaranteed)."""

    def _resolve_posterior(self, posterior, num_samples, random_seed) -> xr.Dataset:
        """One posterior for a predictive call: the caller's, or fresh draws."""
        if posterior is not None:
            if num_samples is not None:
                msg = "pass either posterior= or num_samples=, not both"
                raise ValueError(msg)
            return posterior_dataset(posterior)
        if num_samples is None:
            num_samples = DEFAULT_NUM_SAMPLES
        return self.draw_posterior(num_samples, random_seed)

    def forecast(
        self,
        covariates=None,
        num_samples: int | None = None,
        *,
        horizon: int | None = None,
        future_index=None,
        future_covariates=None,
        posterior=None,
        var_names=None,
        batch_size: int | None = None,
        random_seed=None,
        progressbar: bool = False,
    ):
        """Sample forecasts beyond the training window.

        The horizon is supplied at forecast time, in one of four mutually
        exclusive ways: pass ``covariates`` spanning the training window plus
        the forecast steps, ``future_covariates`` covering only the forecast
        steps, or — for a covariate-free model — ``horizon=N`` to forecast
        ``N`` steps past the training data (its time coord is extended at the
        inferred spacing) or ``future_index=`` to forecast over an arbitrary
        later time index.

        Parameters
        ----------
        covariates
            Covariates spanning training window + forecast horizon (time coords
            must extend the training data's).
        num_samples
            Number of posterior draws (and forecast samples); default 100.
            Mutually exclusive with ``posterior``.
        horizon
            Number of steps to forecast past the training data (covariate-free
            models only).
        future_index
            Time coordinate values of the forecast horizon (covariate-free
            models only): strictly increasing values lying after the training
            window, e.g. a ``DatetimeIndex`` of the period to predict. The
            horizon length is derived from it, so it need not be known at fit
            time. Forecast steps are drawn consecutively and labeled with
            these coordinates. The covariate-free half of the predict-time
            horizon capability; ``future_covariates`` is the with-covariates
            half.
        future_covariates
            Covariates covering only the forecast horizon, with a time index
            lying after the training window; the forecast is conditioned on
            them — the with-covariates half of the predict-time horizon
            capability (``future_index`` is the covariate-free half).
            Structure (dims, covariate names and order) must match the
            training covariates. The horizon length is derived from it, so it
            need not be known at fit time.
        posterior
            A fixed posterior to condition on, in any shape
            :func:`~pymc_forecast.prediction.posterior_dataset` accepts
            (typically from :meth:`draw_posterior`). Passing the same
            posterior to :meth:`predict_in_sample` and :meth:`forecast` makes
            the calls draw-coherent: draw *i* in both results comes from the
            same parameter draw. Without it, each call draws
            ``num_samples`` fresh subsamples.
        var_names, batch_size, random_seed, progressbar
            Passed through to :func:`pymc_forecast.prediction.forecast`
            (``batch_size`` bounds the working memory of predictive sampling
            on very wide panels by processing the posterior in draw blocks).

        Returns
        -------
        DataTree
            With a ``predictions`` group carrying ``time_future`` coords.
        """
        self._require_fitted()
        provided = sum(
            arg is not None for arg in (covariates, horizon, future_index, future_covariates)
        )
        if provided != 1:
            msg = "pass exactly one of covariates, horizon, future_index, or future_covariates"
            raise ValueError(msg)
        if horizon is not None or future_index is not None:
            if self._covariates.size > 0:
                msg = (
                    "this model was fit with covariates, so the forecast needs their "
                    "future values: pass future_covariates= (or full-horizon "
                    "covariates=) instead of horizon=/future_index="
                )
                raise AlignmentError(msg)
            if horizon is not None:
                full_index = extend_time_index(self._data[TIME_DIM].values, horizon)
            else:
                full_index = concat_time_index(self._data[TIME_DIM].values, future_index)
            covariates = null_covariates(full_index)
        elif future_covariates is not None:
            covariates = concat_covariates(self._covariates, future_covariates)
        posterior = self._resolve_posterior(posterior, num_samples, random_seed)
        return _forecast(
            self.model_fn,
            posterior,
            self._data,
            covariates,
            var_names=var_names,
            batch_size=batch_size,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    def predict_in_sample(
        self,
        num_samples: int | None = None,
        *,
        posterior=None,
        batch_size: int | None = None,
        random_seed=None,
        progressbar: bool = False,
    ):
        """Sample the in-sample posterior predictive and registered predictors.

        The result contains ``"obs"`` and ``"mu"`` plus
        ``"expected_observation"`` when the model supplies it to
        :func:`~pymc_forecast.model.predict`.

        Parameters
        ----------
        num_samples
            Number of posterior draws; default 100. Mutually exclusive with
            ``posterior``.
        posterior
            A fixed posterior to condition on (see :meth:`forecast` for the
            draw-coherence semantics).
        batch_size, random_seed, progressbar
            Passed through to :func:`pymc_forecast.prediction.predict_in_sample`.
        """
        self._require_fitted()
        posterior = self._resolve_posterior(posterior, num_samples, random_seed)
        return predict_in_sample(
            self.model_fn,
            posterior,
            self._data,
            self._covariates,
            batch_size=batch_size,
            random_seed=random_seed,
            progressbar=progressbar,
        )


def _resolve_optimizer(optimizer):
    """Normalize an optimizer spec: ``None`` → Adam(0.01), scalar → Adam(lr)."""
    if optimizer is None:
        return pm.adam(learning_rate=DEFAULT_LEARNING_RATE)
    if isinstance(optimizer, int | float):
        learning_rate = float(optimizer)
        if learning_rate <= 0:
            msg = f"learning rate must be positive, got {learning_rate}"
            raise MethodResolutionError(msg)
        return pm.adam(learning_rate=learning_rate)
    if callable(optimizer):
        return optimizer
    msg = (
        "optimizer must be None, a positive learning rate, or a PyMC optimizer "
        f"(e.g. pm.adam(learning_rate=...)); got {type(optimizer).__name__}"
    )
    raise MethodResolutionError(msg)


def _resolve_progressbar(progressbar, kwargs: dict, kwargs_name: str) -> bool:
    """Hoist a backend-kwargs ``progressbar`` to the uniform direct option."""
    if "progressbar" in kwargs:
        if progressbar is not None:
            msg = f"pass progressbar directly or through {kwargs_name}, not both"
            raise ValueError(msg)
        progressbar = kwargs.pop("progressbar")
    return False if progressbar is None else bool(progressbar)


def _check_vi_convergence(losses, num_steps: int) -> None:
    """Warn when the ELBO loss is still clearly descending at the end of a fit.

    Heuristic: compare the median loss over the last
    ``CONVERGENCE_WINDOW_FRACTION`` of the steps against the median over the
    same-sized window starting at the midpoint of the history. The fit is
    flagged when the improvement between the two windows exceeds both the
    fluctuation within the final window (its median absolute deviation) and
    twice the standard error of the median difference — i.e. the optimizer
    was still making clear progress, beyond the stochastic-ELBO noise floor,
    over the second half of the run. Medians are used because the raw ELBO
    history is spiky early in a fit. A slow descent can hide inside the
    noise, so the absence of a warning is not proof of convergence.
    """
    hist = np.asarray(losses, dtype=float)
    if not np.isfinite(hist).all():
        msg = "ADVI convergence could not be assessed: the loss history contains non-finite values."
        warnings.warn(msg, UserWarning, stacklevel=2)
        return
    n = max(int(len(hist) * CONVERGENCE_WINDOW_FRACTION), CONVERGENCE_MIN_WINDOW)
    if len(hist) // 2 + n > len(hist) - n:
        return
    last = hist[-n:]
    mid = hist[len(hist) // 2 :][:n]
    improvement = float(np.median(mid) - np.median(last))
    noise = float(np.median(np.abs(last - np.median(last))))
    # standard error of the difference of two window medians, MAD-scaled
    sem = 1.858 * noise * float(np.sqrt(2.0 / n))
    if improvement > max(noise, 2.0 * sem):
        msg = (
            f"ADVI has not converged after {num_steps} steps: the ELBO loss "
            f"is still descending (median over the last {n} steps improved "
            f"by {improvement:.3g} since mid-run, more than the within-window "
            f"fluctuation {noise:.3g}). The forecast may be confidently wrong "
            "— increase num_steps, raise the learning rate, or use "
            "HMCForecaster; inspect the loss history via the `losses` "
            "attribute."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)


class Forecaster(BaseForecaster):
    """Fit a forecasting model with variational inference (ADVI by default).

    Mean-field ADVI can underconverge silently — the posterior looks fine but
    is biased and overconfident. A post-fit heuristic warns when the ELBO is
    still descending (see :func:`_check_vi_convergence`); absence of the
    warning is *not* proof of convergence, so check :attr:`losses` has
    plateaued before trusting results, and prefer :class:`HMCForecaster` when
    accuracy matters more than speed.

    Parameters
    ----------
    model_fn, data, covariates, random_seed
        See :class:`BaseForecaster`.
    method
        VI method: ``"advi"`` (mean-field, default) or ``"fullrank_advi"``, or
        any ``pm.fit``-compatible inference object.
    optimizer
        ``None`` (Adam with lr ``0.01``), a positive learning rate, or a PyMC
        optimizer such as ``pm.adam(learning_rate=...)``. The JAX backend
        accepts ``None`` or a positive learning rate.
    backend
        ``None`` or ``"pytensor"`` uses ``pm.fit``. ``"jax"`` runs mean-field
        ADVI and Adam as one JAX ``lax.scan`` on the selected accelerator
        (GPU when a CUDA JAX is installed), while retaining PyMC's ordinary
        approximation object for posterior sampling. The optional ``jax``
        extra is required.
    num_steps
        Number of optimization steps.
    progressbar
        Show the fit progress bar.
    fit_kwargs
        Extra keyword arguments for ``pm.fit``. ``progressbar`` is accepted
        here for compatibility, but the direct argument is preferred (passing
        both raises).

    Attributes
    ----------
    approx
        The fitted ``pm.Approximation``.
    losses
        The ELBO loss history (one value per step).
    """

    _batch_generated_posterior = True

    def __init__(
        self,
        model_fn,
        data=None,
        covariates=None,
        *,
        method="advi",
        optimizer=None,
        backend: str | None = None,
        num_steps: int = 10_000,
        random_seed=None,
        progressbar: bool | None = None,
        fit_kwargs: Mapping | None = None,
    ) -> None:
        self._method = method
        if backend not in (None, "pytensor", "jax"):
            msg = f"unknown VI backend {backend!r}; use None, 'pytensor', or 'jax'"
            raise MethodResolutionError(msg)
        self._backend = backend
        if backend == "jax":
            if method != "advi":
                msg = "the JAX backend currently supports method='advi' only"
                raise MethodResolutionError(msg)
            if optimizer is None:
                self._learning_rate = DEFAULT_LEARNING_RATE
            elif isinstance(optimizer, int | float) and optimizer > 0:
                self._learning_rate = float(optimizer)
            else:
                msg = "the JAX backend requires optimizer=None or a positive learning rate"
                raise MethodResolutionError(msg)
            self._optimizer = None
        else:
            self._optimizer = _resolve_optimizer(optimizer)
        self._num_steps = num_steps
        self._fit_kwargs = dict(fit_kwargs or {})
        if backend == "jax" and self._fit_kwargs:
            msg = "fit_kwargs are not supported by the JAX backend"
            raise MethodResolutionError(msg)
        self._progressbar = _resolve_progressbar(progressbar, self._fit_kwargs, "fit_kwargs")
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        if self._backend == "jax":
            from pymc_forecast.jax_backend import fit_advi_jax

            self.approx = fit_advi_jax(
                self.model,
                num_steps=self._num_steps,
                learning_rate=self._learning_rate,
                random_seed=random_seed,
            )
            self.losses = self.approx.hist
            _check_vi_convergence(self.losses, self._num_steps)
            return
        try:
            self.approx = pm.fit(
                n=self._num_steps,
                method=self._method,
                model=self.model,
                random_seed=random_seed,
                obj_optimizer=self._optimizer,
                progressbar=self._progressbar,
                **self._fit_kwargs,
            )
        except KeyError as err:
            msg = (
                f"unknown VI method {self._method!r}; use 'advi', 'fullrank_advi', "
                "or a pm.fit-compatible inference object"
            )
            raise MethodResolutionError(msg) from err
        self.losses = self.approx.hist
        _check_vi_convergence(self.losses, self._num_steps)

    def _draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Draw ``num_samples`` posterior samples from the approximation."""
        idata = self.approx.sample(draws=num_samples, random_seed=random_seed)
        return posterior_dataset(idata)


class HMCForecaster(BaseForecaster):
    """Fit a forecasting model with MCMC (NUTS by default).

    Parameters
    ----------
    model_fn, data, covariates, random_seed
        See :class:`BaseForecaster`.
    draws, tune, chains
        MCMC schedule (defaults ``1000`` / ``1000`` / ``2``).
    nuts_sampler
        NUTS backend: ``"pymc"`` (default), ``"nutpie"``, ``"numpyro"``, or
        ``"blackjax"``.
    progressbar
        Show the sampling progress bar.
    sample_kwargs
        Extra keyword arguments for ``pm.sample``. ``progressbar`` is
        accepted here for compatibility, but the direct argument is preferred
        (passing both raises).

    Attributes
    ----------
    idata
        The full MCMC result (posterior, sample stats, ...).
    """

    def __init__(
        self,
        model_fn,
        data=None,
        covariates=None,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        nuts_sampler: str = "pymc",
        random_seed=None,
        progressbar: bool | None = None,
        sample_kwargs: Mapping | None = None,
    ) -> None:
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._nuts_sampler = nuts_sampler
        self._sample_kwargs = dict(sample_kwargs or {})
        self._progressbar = _resolve_progressbar(progressbar, self._sample_kwargs, "sample_kwargs")
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        self.idata = pm.sample(
            draws=self._draws,
            tune=self._tune,
            chains=self._chains,
            nuts_sampler=self._nuts_sampler,
            model=self.model,
            random_seed=random_seed,
            progressbar=self._progressbar,
            **self._sample_kwargs,
        )

    def _draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the MCMC posterior."""
        return thin_draws(self.idata, num_samples, random_seed)


class PathfinderForecaster(BaseForecaster):
    """Fit a forecasting model with Pathfinder variational inference.

    A thin wrapper over ``pymc_extras.fit_pathfinder``. pymc-extras is imported
    lazily, so constructing this class is the opt-in that requires it.

    Parameters
    ----------
    model_fn, data, covariates, random_seed
        See :class:`BaseForecaster`.
    progressbar
        Show the fit progress bar.
    pathfinder_kwargs
        Extra keyword arguments for ``pymc_extras.fit_pathfinder``
        (e.g. ``num_paths``, ``num_draws``). ``progressbar`` is accepted here
        for compatibility, but the direct argument is preferred (passing both
        raises).

    Attributes
    ----------
    idata
        The Pathfinder result with its ``posterior`` group.
    """

    def __init__(
        self,
        model_fn,
        data=None,
        covariates=None,
        *,
        random_seed=None,
        progressbar: bool | None = None,
        pathfinder_kwargs: Mapping | None = None,
    ) -> None:
        self._pathfinder_kwargs = dict(pathfinder_kwargs or {})
        self._progressbar = _resolve_progressbar(
            progressbar, self._pathfinder_kwargs, "pathfinder_kwargs"
        )
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        try:
            from pymc_extras import fit_pathfinder
        except ImportError as err:
            raise OptionalDependencyError("pymc-extras", "extras", "PathfinderForecaster") from err
        self.idata = fit_pathfinder(
            model=self.model,
            random_seed=random_seed,
            progressbar=self._progressbar,
            **self._pathfinder_kwargs,
        )

    def _draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the Pathfinder posterior."""
        return thin_draws(self.idata, num_samples, random_seed)

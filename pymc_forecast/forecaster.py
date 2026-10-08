"""Forecaster classes: fit a forecasting model, then draw probabilistic forecasts.

Three inference backends behind one interface: :class:`Forecaster`
(variational, ADVI by default), :class:`HMCForecaster` (MCMC via
``pm.sample``), and :class:`PathfinderForecaster` (pymc-extras Pathfinder).
Construction fits the model on ``(data, covariates)``; ``forecast``
then rebuilds the model over extended covariates and samples the horizon.
Construct without data to defer the fit (call ``fit`` later).
"""

import abc
from collections.abc import Mapping

import pymc as pm
import xarray as xr

from pymc_forecast.data import (
    TIME_DIM,
    _validate_covariate_structure,
    as_dataarray,
    concat_covariates,
    concat_time_index,
    extend_time_index,
    null_covariates,
    validate_alignment,
)
from pymc_forecast.exceptions import (
    AlignmentError,
    NotFittedError,
)
from pymc_forecast.fit import (
    _draw_batched,
    _resolve_progressbar,
    _resolve_vi_options,
    _training_inputs,
    fit_mcmc,
    fit_pathfinder,
    fit_vi,
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

DEFAULT_NUM_SAMPLES = 100
"""Default number of posterior draws for the predictive methods."""


class BaseForecaster(abc.ABC):
    """Shared fit/forecast plumbing.

    Fitting happens on construction when ``data`` is given. Constructing
    without data defers it — configure now, :meth:`fit` later (the
    sklearn-style lifecycle adapter authors need)::

        fc = HMCForecaster(model_fn, draws=500)   # not fitted yet
        fc.fit(data, covariates)                  # returns self

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body (``(covariates, data=None) -> None`` or a
        :class:`~pymc_forecast.model.ForecastingModel`).
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D input
        gets a ``"series"`` dim). ``None`` constructs the forecaster unfitted;
        call :meth:`fit` later.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data``, covering (at
        least) the training window (normalized with ``as_dataarray``; 2-D input
        gets a ``"covariate"`` dim). Rows past the training window are dropped
        during fitting. ``None`` for models without covariates.
    random_seed : int, optional
        Seed for the fit on construction; also the default seed for later
        :meth:`fit` calls.

    Attributes
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body passed to the constructor.
    model : pymc.Model or None
        The training model built by the last :meth:`fit`; ``None`` before the
        first fit.

    Raises
    ------
    ValueError
        If ``covariates`` is given without ``data``.
    pymc_forecast.exceptions.AlignmentError
        If (when fitting) ``data`` or ``covariates`` cannot be normalized or do
        not align along ``"time"``.
    pymc_forecast.exceptions.HorizonError
        If (when fitting) the model body does not register ``"obs"`` or
        misuses the model primitives.
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
        data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like
            Observed training series, normalized with
            :func:`~pymc_forecast.data.as_dataarray` (``"time"`` first; 2-D
            input gets a ``"series"`` dim).
        covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
            Covariates on the same ``"time"`` coordinate as ``data``, covering
            (at least) the training window (normalized with ``as_dataarray``;
            2-D input gets a ``"covariate"`` dim). Rows past the training
            window are dropped. ``None`` for models without covariates.
        random_seed : int, optional
            Seed for the fit; ``None`` uses the constructor's ``random_seed``.

        Returns
        -------
        BaseForecaster
            The fitted instance (``self``).

        Raises
        ------
        pymc_forecast.exceptions.AlignmentError
            If ``data`` or ``covariates`` cannot be normalized, or the
            covariates do not align with ``data`` along ``"time"``.
        pymc_forecast.exceptions.HorizonError
            If the model body does not register ``"obs"`` or misuses the model
            primitives.
        pymc_forecast.exceptions.MethodResolutionError
            If the backend cannot resolve the configured inference method
            (e.g. an unknown VI method name for ``Forecaster``).
        pymc_forecast.exceptions.OptionalDependencyError
            If the backend's optional dependency (JAX for
            ``Forecaster(backend="jax")``, pymc-extras for
            ``PathfinderForecaster``) is not installed.

        Warns
        -----
        UserWarning
            For ``Forecaster``, when the post-fit heuristic finds the ELBO
            still descending (possible underconvergence) or the loss history
            contains non-finite values.
        """
        self._is_fitted = False
        self._data, self._covariates = _training_inputs(data, covariates)
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
        num_samples : int
            Number of posterior draws.
        random_seed : int or numpy.random.Generator, optional
            Seed for posterior sampling.
        batch_size : int, optional
            For backends that generate posterior draws on demand (currently
            :class:`Forecaster`), draw at most this many at once and
            concatenate the host-backed xarray chunks. On wide panels this
            bounds the peak allocation made by ``pm.Approximation.sample`` —
            the posterior side of upstream's chunked sampling
            (juanitorduz/numpyro_forecast#65); the predictive side is the
            ``batch_size`` argument of :meth:`forecast` /
            :meth:`predict_in_sample`. ``None`` keeps the single-shot path.
            Backends whose posterior is already materialized (HMC and
            Pathfinder) thin it once and do not need this memory knob; the
            value is still validated on every backend.

        Returns
        -------
        xarray.Dataset
            Posterior with a single chain (``chain`` size 1) and
            ``num_samples`` draws.

        Raises
        ------
        pymc_forecast.exceptions.NotFittedError
            If the forecaster has not been fitted.
        ValueError
            If ``batch_size`` is not positive, or (HMC/Pathfinder) if
            ``num_samples`` is not positive.
        """
        self._require_fitted()
        return _draw_batched(
            self._draw_posterior,
            num_samples,
            random_seed,
            batch_size=batch_size,
            generated=self._batch_generated_posterior,
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
        covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
            Covariates spanning training window + forecast horizon (time coords
            must extend the training data's), normalized with
            :func:`~pymc_forecast.data.as_dataarray`; the steps past the
            training window define the horizon. Non-time dimensions and
            coordinate names/order must match the training covariates, as with
            ``future_covariates``; reordered or renamed columns are rejected
            before posterior sampling. ``None`` when the horizon comes from
            another argument.
        num_samples : int, optional
            Number of posterior draws (and forecast samples) to sample when
            ``posterior`` is not given; ``None`` means 100. Mutually exclusive
            with ``posterior``.
        horizon : int, optional
            Number of steps to forecast past the training data (models fit
            without covariates only). Must be at least 1, and the training time
            index must have an inferable spacing.
        future_index : array_like, optional
            Time coordinate values of the forecast horizon (models fit without
            covariates only): strictly increasing values lying after the
            training window, e.g. a ``DatetimeIndex`` of the period to predict.
            The horizon length is derived from it, so it need not be known at
            fit time. Forecast steps are drawn consecutively and labeled with
            these coordinates. The covariate-free half of the predict-time
            horizon capability; ``future_covariates`` is the with-covariates
            half.
        future_covariates : xarray.DataArray, pandas.DataFrame or array_like, optional
            Covariates (any input :func:`~pymc_forecast.data.as_dataarray`
            accepts, including a ``pandas.Series``) covering only the forecast
            horizon, with a time index
            lying after the training window; the forecast is conditioned on
            them — the with-covariates half of the predict-time horizon
            capability (``future_index`` is the covariate-free half).
            Structure (dims, covariate names and order) must match the
            training covariates. The horizon length is derived from it, so it
            need not be known at fit time.
        posterior : xarray.Dataset, xarray.DataTree or arviz.InferenceData, optional
            A fixed posterior to condition on: a posterior Dataset or any
            object with a ``posterior`` group
            (:func:`~pymc_forecast.prediction.posterior_dataset`), typically
            from ``draw_posterior``; used as given, not thinned. Passing the
            same posterior to ``predict_in_sample`` and ``forecast`` makes the
            calls draw-coherent: draw *i* in both results comes from the same
            parameter draw. Without it, each call draws ``num_samples`` fresh
            subsamples. Mutually exclusive with ``num_samples``.
        var_names : sequence of str, optional
            Variables to record (a bare string is not accepted as a single
            name); ``None`` uses the defaults of
            :func:`pymc_forecast.prediction.forecast`.
        batch_size : int, optional
            Maximum posterior draws (per chain) per predictive pass, bounding
            the working memory on very wide panels; ``None`` runs a single
            pass. See :func:`pymc_forecast.prediction.forecast`.
        random_seed : int or numpy.random.Generator, optional
            Seeds both the posterior draws (when ``posterior`` is not given)
            and the predictive sampling.
        progressbar : bool, default False
            Show the predictive sampling progress bar.

        Returns
        -------
        xarray.DataTree or arviz.InferenceData
            Result of ``pm.sample_posterior_predictive`` (a DataTree with
            current PyMC/ArviZ, InferenceData with older releases), with a
            ``predictions`` group carrying ``time_future`` coords.

        Raises
        ------
        pymc_forecast.exceptions.NotFittedError
            If the forecaster has not been fitted.
        ValueError
            If not exactly one of ``covariates``, ``horizon``, ``future_index``
            and ``future_covariates`` is given; if both ``posterior`` and
            ``num_samples`` are given; if ``batch_size`` is not positive; or
            (HMC/Pathfinder) if ``num_samples`` is not positive.
        pymc_forecast.exceptions.AlignmentError
            If ``horizon`` or ``future_index`` is given for a model fit with
            covariates; if ``horizon`` is negative, not an integer, or the
            time spacing cannot be inferred; if ``future_index`` is empty, not
            strictly increasing, or does not start after the training window;
            if the covariate structure does not match the training covariates;
            or if the covariates cannot be normalized or do not align with the
            training data.
        pymc_forecast.exceptions.HorizonError
            If the resulting horizon is empty (e.g. ``horizon=0``).
        TypeError
            If ``posterior`` has no ``posterior`` group.
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
                full_index = extend_time_index(self._data.get_index(TIME_DIM), horizon)
            else:
                full_index = concat_time_index(self._data[TIME_DIM].values, future_index)
            covariates = null_covariates(full_index)
        elif future_covariates is not None:
            covariates = concat_covariates(self._covariates, future_covariates)
        else:
            covariates = as_dataarray(covariates, role="covariates")
            _validate_covariate_structure(self._covariates, covariates)
        validate_alignment(self._data, covariates)
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

        The result contains ``"obs"``, plus ``"mu"`` and
        ``"expected_observation"`` when the model registers them (models
        built with :func:`~pymc_forecast.model.predict` register ``"mu"``;
        ``"expected_observation"`` only when supplied to it).

        Parameters
        ----------
        num_samples : int, optional
            Number of posterior draws to sample when ``posterior`` is not
            given; ``None`` means 100. Mutually exclusive with ``posterior``.
        posterior : xarray.Dataset, xarray.DataTree or arviz.InferenceData, optional
            A fixed posterior to condition on: a posterior Dataset or any
            object with a ``posterior`` group
            (:func:`~pymc_forecast.prediction.posterior_dataset`); used as
            given, not thinned (see ``forecast`` for the draw-coherence
            semantics). Mutually exclusive with ``num_samples``.
        batch_size : int, optional
            Maximum posterior draws (per chain) per predictive pass; ``None``
            runs a single pass. See
            :func:`pymc_forecast.prediction.predict_in_sample`.
        random_seed : int or numpy.random.Generator, optional
            Seeds both the posterior draws (when ``posterior`` is not given)
            and the predictive sampling.
        progressbar : bool, default False
            Show the predictive sampling progress bar.

        Returns
        -------
        xarray.DataTree or arviz.InferenceData
            Result of ``pm.sample_posterior_predictive`` (a DataTree with
            current PyMC/ArviZ, InferenceData with older releases), with a
            ``posterior_predictive`` group holding the variables above over
            ``"time"``.

        Raises
        ------
        pymc_forecast.exceptions.NotFittedError
            If the forecaster has not been fitted.
        ValueError
            If both ``posterior`` and ``num_samples`` are given, if
            ``batch_size`` is not positive, or (HMC/Pathfinder) if
            ``num_samples`` is not positive.
        TypeError
            If ``posterior`` has no ``posterior`` group.
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


class Forecaster(BaseForecaster):
    """Fit a forecasting model with variational inference (ADVI by default).

    Mean-field ADVI can underconverge silently — the posterior looks fine but
    is biased and overconfident. A post-fit heuristic warns when the median
    ELBO loss over the final 10% of steps (at least 10) is still clearly
    below the median over a same-sized window starting mid-run; absence of
    the warning is *not* proof of convergence, so check :attr:`losses` has
    plateaued before trusting results, and prefer :class:`HMCForecaster` when
    accuracy matters more than speed.

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body (see ``BaseForecaster``).
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray`. ``None`` constructs the
        forecaster unfitted; call ``fit`` later.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data``; rows past the
        training window are dropped during fitting. ``None`` for models
        without covariates.
    method : str or pymc.variational.Inference, default "advi"
        VI method: ``"advi"`` (mean-field), ``"fullrank_advi"``, ``"svgd"`` or
        ``"asvgd"`` (case-insensitive, forwarded to ``pm.fit``), or a
        ``pymc.variational.Inference`` instance — which fits the model it was
        built on and ignores this forecaster's model and ``random_seed``. The
        JAX backend accepts only ``"advi"``.
    optimizer : float or callable, optional
        ``None`` (Adam with lr ``0.01``), a positive learning rate, or a PyMC
        optimizer such as ``pm.adam(learning_rate=...)``. The JAX backend
        accepts ``None`` or a positive learning rate.
    backend : {"pytensor", "jax"}, optional
        ``None`` or ``"pytensor"`` uses ``pm.fit``. ``"jax"`` runs mean-field
        ADVI and Adam as one JAX ``lax.scan`` on the selected accelerator
        (GPU when a CUDA JAX is installed), while retaining PyMC's ordinary
        approximation object for posterior sampling. The optional ``jax``
        extra is required.
    num_steps : int, default 10_000
        Number of optimization steps.
    random_seed : int, optional
        Seed for the fit; also the default seed for later ``fit`` calls.
    progressbar : bool, optional
        Show the fit progress bar; ``None`` means off. Ignored by the JAX
        backend. May instead be given in ``fit_kwargs``, but not both.
    fit_kwargs : mapping, optional
        Extra keyword arguments for ``pm.fit``. ``progressbar`` is accepted
        here for compatibility, but the direct argument is preferred (passing
        both raises). Must be empty with ``backend="jax"``.

    Attributes
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body passed to the constructor.
    model : pymc.Model or None
        The training model; ``None`` before the first fit.
    approx : pymc.variational.Approximation
        The fitted approximation (set by fitting).
    losses : numpy.ndarray
        The per-step loss history (``approx.hist``); empty for ``"svgd"`` and
        ``"asvgd"``, which record no loss (set by fitting).
    idata : None
        Always ``None`` for variational fits (set by fitting).

    Raises
    ------
    pymc_forecast.exceptions.MethodResolutionError
        If, on construction, ``backend`` is unknown, ``optimizer`` is not a
        positive learning rate or (pytensor backend) a callable, or, with
        ``backend="jax"``, ``method`` is not ``"advi"`` or ``fit_kwargs`` is
        non-empty; or if, when fitting, ``method`` is an unknown name.
    ValueError
        If ``progressbar`` is given both directly and in ``fit_kwargs``, if
        ``covariates`` is given without ``data``, or (JAX backend) if
        ``num_steps`` is not positive.
    pymc_forecast.exceptions.AlignmentError
        If (when fitting) ``data`` or ``covariates`` cannot be normalized or do
        not align along ``"time"``.
    pymc_forecast.exceptions.HorizonError
        If (when fitting) the model body does not register ``"obs"`` or
        misuses the model primitives.
    pymc_forecast.exceptions.OptionalDependencyError
        If ``backend="jax"`` and JAX is not installed (when fitting).

    Warns
    -----
    UserWarning
        When the post-fit heuristic finds the ELBO still descending, or the
        loss history contains non-finite values so convergence cannot be
        assessed.
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
        options = _resolve_vi_options(method, optimizer, backend, fit_kwargs, progressbar)
        self._method = method
        self._backend = backend
        self._optimizer = options.optimizer
        self._num_steps = num_steps
        self._fit_kwargs = options.fit_kwargs
        self._progressbar = options.progressbar
        super().__init__(model_fn, data, covariates, random_seed=random_seed)

    def _fit(self, random_seed) -> None:
        result = fit_vi(
            self.model_fn,
            method=self._method,
            optimizer=self._optimizer,
            backend=self._backend,
            num_steps=self._num_steps,
            random_seed=random_seed,
            progressbar=self._progressbar,
            fit_kwargs=self._fit_kwargs,
            model=self.model,
        )
        self.approx = result.approx
        self.losses = result.losses
        self.idata = result.idata

    def _draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Draw ``num_samples`` posterior samples from the approximation."""
        idata = self.approx.sample(draws=num_samples, random_seed=random_seed)
        return posterior_dataset(idata)


class HMCForecaster(BaseForecaster):
    """Fit a forecasting model with MCMC (NUTS by default).

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body (see ``BaseForecaster``).
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray`. ``None`` constructs the
        forecaster unfitted; call ``fit`` later.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data``; rows past the
        training window are dropped during fitting. ``None`` for models
        without covariates.
    draws : int, default 1000
        Number of posterior draws per chain.
    tune : int, default 1000
        Number of tuning steps per chain.
    chains : int, default 2
        Number of chains.
    nuts_sampler : {"pymc", "nutpie", "numpyro", "blackjax"}, default "pymc"
        NUTS backend, forwarded to ``pm.sample``; non-PyMC samplers need
        their optional package installed.
    random_seed : int or numpy.random.Generator, optional
        Seed for the fit, forwarded to ``pm.sample``; also the default seed
        for later ``fit`` calls.
    progressbar : bool, optional
        Show the sampling progress bar; ``None`` means off. May instead be
        given in ``sample_kwargs``, but not both.
    sample_kwargs : mapping, optional
        Extra keyword arguments for ``pm.sample``. ``progressbar`` is
        accepted here for compatibility, but the direct argument is preferred
        (passing both raises).

    Attributes
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body passed to the constructor.
    model : pymc.Model or None
        The training model; ``None`` before the first fit.
    idata : xarray.DataTree or arviz.InferenceData
        The full MCMC result (posterior, sample stats, ...), as returned by
        ``pm.sample`` (set by fitting).
    approx : None
        Always ``None`` for MCMC fits (set by fitting).
    losses : None
        Always ``None`` for MCMC fits (set by fitting).

    Raises
    ------
    ValueError
        If ``progressbar`` is given both directly and in ``sample_kwargs``, or
        ``covariates`` is given without ``data``.
    pymc_forecast.exceptions.AlignmentError
        If (when fitting) ``data`` or ``covariates`` cannot be normalized or do
        not align along ``"time"``.
    pymc_forecast.exceptions.HorizonError
        If (when fitting) the model body does not register ``"obs"`` or
        misuses the model primitives.
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
        result = fit_mcmc(
            self.model_fn,
            draws=self._draws,
            tune=self._tune,
            chains=self._chains,
            nuts_sampler=self._nuts_sampler,
            random_seed=random_seed,
            progressbar=self._progressbar,
            sample_kwargs=self._sample_kwargs,
            model=self.model,
        )
        self.approx = result.approx
        self.losses = result.losses
        self.idata = result.idata

    def _draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the MCMC posterior."""
        return thin_draws(self.idata, num_samples, random_seed)


class PathfinderForecaster(BaseForecaster):
    """Fit a forecasting model with Pathfinder variational inference.

    A thin wrapper over ``pymc_extras.fit_pathfinder``. pymc-extras is imported
    only when fitting, so constructing without data does not require it.

    Parameters
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body (see ``BaseForecaster``).
    data : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Observed training series, normalized with
        :func:`~pymc_forecast.data.as_dataarray`. ``None`` constructs the
        forecaster unfitted; call ``fit`` later.
    covariates : xarray.DataArray, pandas.Series, pandas.DataFrame or array_like, optional
        Covariates on the same ``"time"`` coordinate as ``data``; rows past the
        training window are dropped during fitting. ``None`` for models
        without covariates.
    random_seed : int, optional
        Seed for the fit, forwarded to ``pymc_extras.fit_pathfinder``; also the
        default seed for later ``fit`` calls.
    progressbar : bool, optional
        Show the fit progress bar; ``None`` means off. May instead be given in
        ``pathfinder_kwargs``, but not both.
    pathfinder_kwargs : mapping, optional
        Extra keyword arguments for ``pymc_extras.fit_pathfinder``
        (e.g. ``num_paths``, ``num_draws``). ``progressbar`` is accepted here
        for compatibility, but the direct argument is preferred (passing both
        raises).

    Attributes
    ----------
    model_fn : callable or pymc_forecast.model.ForecastingModel
        The model body passed to the constructor.
    model : pymc.Model or None
        The training model; ``None`` before the first fit.
    idata : xarray.DataTree or arviz.InferenceData
        The Pathfinder result with its ``posterior`` group (set by fitting).
    approx : None
        Always ``None`` for Pathfinder fits (set by fitting).
    losses : None
        Always ``None`` for Pathfinder fits (set by fitting).

    Raises
    ------
    ValueError
        If ``progressbar`` is given both directly and in
        ``pathfinder_kwargs``, or ``covariates`` is given without ``data``.
    pymc_forecast.exceptions.AlignmentError
        If (when fitting) ``data`` or ``covariates`` cannot be normalized or do
        not align along ``"time"``.
    pymc_forecast.exceptions.HorizonError
        If (when fitting) the model body does not register ``"obs"`` or
        misuses the model primitives.
    pymc_forecast.exceptions.OptionalDependencyError
        If pymc-extras is not installed (when fitting).
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
        result = fit_pathfinder(
            self.model_fn,
            random_seed=random_seed,
            progressbar=self._progressbar,
            pathfinder_kwargs=self._pathfinder_kwargs,
            model=self.model,
        )
        self.approx = result.approx
        self.losses = result.losses
        self.idata = result.idata

    def _draw_posterior(self, num_samples: int, random_seed=None) -> xr.Dataset:
        """Subsample ``num_samples`` draws from the Pathfinder posterior."""
        return thin_draws(self.idata, num_samples, random_seed)

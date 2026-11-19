import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from example_models import linear_model, make_trend_data

from pymc_forecast.evaluate import (
    BacktestResult,
    _windows,
    backtest,
    results_to_dataframe,
)
from pymc_forecast.exceptions import AlignmentError, BacktestWindowError
from pymc_forecast.fit import draw_posterior, fit_vi
from pymc_forecast.forecaster import Forecaster
from pymc_forecast.metrics import eval_mae, make_mase
from pymc_forecast.prediction import forecast, predict_in_sample

SEED = 321
FAST = {"num_steps": 1_500}


def window_list(duration, **kwargs):
    defaults = dict(
        window_type="expanding",
        train_window=None,
        min_train_window=1,
        test_window=None,
        min_test_window=1,
        stride=1,
    )
    defaults.update(kwargs)
    return list(_windows(duration, **defaults))


class TestWindows:
    @pytest.mark.parametrize("test_window", [0, -1, 1])
    def test_test_window_must_meet_minimum(self, test_window):
        with pytest.raises(BacktestWindowError, match=r"test_window .* must be >="):
            window_list(10, min_train_window=5, min_test_window=3, test_window=test_window)

    @pytest.mark.parametrize(
        "name", ["train_window", "min_train_window", "test_window", "min_test_window", "stride"]
    )
    @pytest.mark.parametrize("value", [1.5, True])
    def test_window_sizes_must_be_integers(self, name, value):
        with pytest.raises(BacktestWindowError, match=f"{name} must be an integer"):
            window_list(10, **{name: value})

    @pytest.mark.parametrize("name", ["min_train_window", "min_test_window", "stride"])
    def test_required_window_sizes_cannot_be_none(self, name):
        with pytest.raises(BacktestWindowError, match=f"{name} must be an integer"):
            window_list(10, **{name: None})

    def test_expanding(self):
        splits = window_list(10, min_train_window=6, test_window=2, stride=2)
        assert splits == [(0, 6, 8), (0, 8, 10)]

    def test_rolling(self):
        splits = window_list(10, window_type="rolling", train_window=5, test_window=2, stride=2)
        assert splits == [(0, 5, 7), (2, 7, 9), (4, 9, 10)]

    def test_test_window_clipped_at_end(self):
        splits = window_list(8, min_train_window=6, test_window=5)
        assert splits == [(0, 6, 8), (0, 7, 8)]

    def test_no_windows_raises(self):
        with pytest.raises(BacktestWindowError, match="no valid windows"):
            window_list(5, min_train_window=5, min_test_window=1)

    def test_parameter_validation(self):
        with pytest.raises(BacktestWindowError, match="stride"):
            window_list(10, stride=0)
        with pytest.raises(BacktestWindowError, match="min_train_window"):
            window_list(10, min_train_window=0)
        with pytest.raises(BacktestWindowError, match="min_test_window"):
            window_list(10, min_test_window=0)
        with pytest.raises(BacktestWindowError, match=r"train_window .* must be >="):
            window_list(10, window_type="rolling", train_window=2, min_train_window=5)


class TestWindowTypeResolution:
    def test_expanding_with_train_window_rejected(self):
        data, cov = make_trend_data()
        with pytest.raises(BacktestWindowError, match="mutually exclusive"):
            backtest(data, cov, linear_model, window_type="expanding", train_window=5)

    def test_rolling_requires_train_window(self):
        data, cov = make_trend_data()
        with pytest.raises(BacktestWindowError, match="requires train_window"):
            backtest(data, cov, linear_model, window_type="rolling")


class TestBacktest:
    @pytest.fixture(scope="class")
    def results(self):
        data, cov = make_trend_data(t_obs=30, horizon=0)
        return backtest(
            data,
            cov,
            linear_model,
            min_train_window=20,
            test_window=5,
            stride=5,
            num_samples=80,
            forecaster_options=FAST,
            eval_train=True,
            keep_predictions=True,
            random_seed=SEED,
        )

    def test_windows_and_metrics(self, results):
        assert [(r.t0, r.t1, r.t2) for r in results] == [(0, 20, 25), (0, 25, 30)]
        for r in results:
            assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}
            assert r.metrics["mae"] < 0.5  # trend model on clean trend data
            assert r.train_metrics["rmse"] < 0.5
            assert r.train_walltime > 0 and r.test_walltime > 0

    def test_predictions_kept_with_coords(self, results):
        pred = results[0].prediction
        assert pred is not None
        np.testing.assert_array_equal(pred["time_future"].values, np.arange(20, 25))

    def test_per_window_metrics(self):
        data, cov = make_trend_data(t_obs=30, horizon=0)
        results = backtest(
            data,
            cov,
            linear_model,
            min_train_window=25,
            test_window=5,
            num_samples=50,
            forecaster_options=FAST,
            per_window_metrics=lambda t0, t1, t2: {
                "mase": make_mase(data.isel(time=slice(t0, t1)))
            },
            metrics={},
            random_seed=SEED,
        )
        assert set(results[0].metrics) == {"mase"}
        # MASE is finite and order-1: on a clean smooth trend the naive
        # random-walk baseline is itself strong, so ~1 is expected.
        assert 0.0 < results[0].metrics["mase"] < 2.0

    def test_transform_applied(self):
        data, cov = make_trend_data(t_obs=30, horizon=0)
        seen = []

        def spy(pred, truth):
            seen.append(True)
            return pred, truth

        backtest(
            data,
            cov,
            linear_model,
            min_train_window=25,
            num_samples=20,
            forecaster_options=FAST,
            transform=spy,
            random_seed=SEED,
        )
        assert seen

    def test_deterministic_given_seed(self):
        data, cov = make_trend_data(t_obs=28, horizon=0)
        kwargs = dict(
            min_train_window=24,
            test_window=4,
            num_samples=40,
            forecaster_options=FAST,
            random_seed=SEED,
        )
        first = backtest(data, cov, linear_model, **kwargs)
        second = backtest(data, cov, linear_model, **kwargs)
        assert [r.metrics for r in first] == [r.metrics for r in second]

    def test_results_to_dataframe(self, results):
        df = results_to_dataframe(results)
        assert list(df["t1"]) == [20, 25]
        assert {"mae", "rmse", "crps", "coverage", "train_mae"} <= set(df.columns)


class TestBacktestResultDataclass:
    def test_defaults(self):
        r = BacktestResult(
            t0=0,
            t1=5,
            t2=8,
            num_samples=10,
            train_walltime=0.1,
            test_walltime=0.1,
            metrics={"mae": 1.0},
        )
        assert r.train_metrics == {} and r.prediction is None


class LastValueForecaster:
    """The forecaster protocol of ``backtest`` and nothing else.

    Repeats the last training value with a little noise. ``forecast`` takes no
    ``batch_size``, like the adapters written against v0.3.0.
    """

    def __init__(self, model_fn, data, covariates, *, random_seed=None, log=None):
        self.data = data
        self.log = [] if log is None else log
        self.log.append(("fit", random_seed))

    def forecast(self, covariates, num_samples=None, random_seed=None):
        self.log.append(("forecast", random_seed))
        num_train = self.data.sizes["time"]
        horizon = covariates.sizes["time"] - num_train
        rng = np.random.default_rng(random_seed)
        draws = float(self.data.isel(time=-1)) + rng.normal(0.0, 0.01, (1, num_samples, horizon))
        future = covariates["time"].values[num_train:]
        pred = xr.DataArray(
            draws, dims=("chain", "draw", "time_future"), coords={"time_future": future}
        )
        return {"predictions": {"forecast": pred}}


class BatchedLastValueForecaster(LastValueForecaster):
    def forecast(self, covariates, num_samples=None, random_seed=None, batch_size=None):
        self.log.append(("batch_size", batch_size))
        return super().forecast(covariates, num_samples=num_samples, random_seed=random_seed)


def last_value_forecast_fn(calls=None, extra_draws=0):
    """A ``forecast_fn`` that repeats the last value and records its arguments."""

    def forecast_fn(
        rng,
        model_fn,
        train_data,
        train_covariates,
        full_covariates,
        num_samples,
        /,
        *,
        batch_size=None,
    ):
        if calls is not None:
            calls.append(
                {
                    "rng": rng,
                    "model_fn": model_fn,
                    "train_data": train_data,
                    "train_covariates": train_covariates,
                    "full_covariates": full_covariates,
                    "num_samples": num_samples,
                    "batch_size": batch_size,
                }
            )
        num_train = train_data.sizes["time"]
        horizon = full_covariates.sizes["time"] - num_train
        draws = float(train_data.isel(time=-1)) + rng.normal(
            0.0, 0.01, (2, num_samples + extra_draws, horizon)
        )
        future = full_covariates["time"].values[num_train:]
        return xr.DataArray(
            draws, dims=("chain", "draw", "time_future"), coords={"time_future": future}
        )

    return forecast_fn


def last_value_in_sample_fn(
    rng, model_fn, train_data, train_covariates, num_samples, /, *, batch_size=None
):
    draws = train_data.values + rng.normal(0.0, 0.01, (1, num_samples, train_data.sizes["time"]))
    return xr.DataArray(
        draws, dims=("chain", "draw", "time"), coords={"time": train_data["time"].values}
    )


def never_called(*args, **kwargs):
    raise AssertionError("backtest called the closure before rejecting its arguments")


class TestForecasterProtocol:
    """Item A of #99: the variable that ``backtest`` scores."""

    def test_var_name_scores_another_predictions_variable(self):
        data, cov = make_trend_data(t_obs=30, horizon=0)
        results = backtest(
            data,
            cov,
            linear_model,
            var_name="mu_future",
            min_train_window=25,
            stride=5,
            num_samples=30,
            forecaster_options=FAST,
            keep_predictions=True,
            random_seed=SEED,
        )
        (result,) = results
        assert result.prediction.name == "mu_future"
        truth = data.isel(time=slice(25, 30)).rename(time="time_future")
        assert result.metrics["mae"] == eval_mae(result.prediction, truth)

    def test_batch_size_forwarded_to_forecaster(self):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        log = []
        backtest(
            data,
            cov,
            None,
            forecaster_cls=BatchedLastValueForecaster,
            forecaster_options={"log": log},
            min_train_window=10,
            num_samples=8,
            batch_size=4,
            random_seed=SEED,
        )
        assert [entry for entry in log if entry[0] == "batch_size"] == [("batch_size", 4)] * 2

    def test_minimal_forecaster_without_batch_size(self):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        results = backtest(
            data,
            cov,
            None,
            forecaster_cls=LastValueForecaster,
            min_train_window=10,
            num_samples=8,
            random_seed=SEED,
        )
        assert [r.num_samples for r in results] == [8, 8]

    def test_var_name_with_eval_train_rejected(self):
        # In-sample scoring reads ``obs``, so train and test scores would be of
        # two different variables.
        data, cov = make_trend_data(t_obs=12, horizon=0)
        with pytest.raises(ValueError, match="var_name"):
            backtest(
                data,
                cov,
                None,
                forecaster_cls=never_called,
                var_name="mu_future",
                eval_train=True,
                min_train_window=10,
            )


class TestForecastFn:
    """Item B of #99: a closure fits and forecasts each window."""

    def test_closures_get_the_window(self):
        data, cov = make_trend_data(t_obs=30, horizon=0)
        calls, in_sample_calls = [], []

        def in_sample_fn(rng, model_fn, train_data, train_covariates, num_samples, /, **kwargs):
            in_sample_calls.append((model_fn, train_data, train_covariates, kwargs))
            return last_value_in_sample_fn(
                rng, model_fn, train_data, train_covariates, num_samples, **kwargs
            )

        results = backtest(
            data,
            cov,
            "model body",
            forecast_fn=last_value_forecast_fn(calls),
            in_sample_fn=in_sample_fn,
            window_type="rolling",
            train_window=20,
            test_window=5,
            stride=5,
            num_samples=10,
            batch_size=4,
            eval_train=True,
            random_seed=SEED,
        )
        assert [(r.t0, r.t1, r.t2) for r in results] == [(0, 20, 25), (5, 25, 30)]
        for call, in_sample_call, r in zip(calls, in_sample_calls, results, strict=True):
            assert isinstance(call["rng"], np.random.Generator)
            assert call["model_fn"] == "model body"
            np.testing.assert_array_equal(call["train_data"]["time"], np.arange(r.t0, r.t1))
            np.testing.assert_array_equal(call["train_covariates"]["time"], np.arange(r.t0, r.t1))
            np.testing.assert_array_equal(call["full_covariates"]["time"], np.arange(r.t0, r.t2))
            assert call["num_samples"] == 10
            assert call["batch_size"] == 4
            model_fn, train_data, train_covariates, kwargs = in_sample_call
            assert model_fn == "model body"
            np.testing.assert_array_equal(train_data["time"], np.arange(r.t0, r.t1))
            np.testing.assert_array_equal(train_covariates["time"], np.arange(r.t0, r.t1))
            assert kwargs == {"batch_size": 4}

    def test_result_records_the_returned_draws_and_one_walltime(self):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        (result,) = backtest(
            data,
            cov,
            None,
            forecast_fn=last_value_forecast_fn(extra_draws=3),
            min_train_window=10,
            test_window=2,
            stride=5,
            num_samples=10,
            random_seed=SEED,
        )
        assert result.num_samples == 2 * 13
        assert np.isnan(result.train_walltime)
        assert result.test_walltime > 0
        assert set(result.metrics) == {"mae", "rmse", "crps", "coverage"}

    def test_library_functions_in_the_closures(self):
        data, cov = make_trend_data(t_obs=30, horizon=0)

        def fit(rng, model_fn, train_data, train_covariates, num_samples):
            seed = int(rng.integers(2**31))
            result = fit_vi(model_fn, train_data, train_covariates, random_seed=seed, **FAST)
            return draw_posterior(result, num_samples, rng)

        def forecast_fn(
            rng, model_fn, train_data, train_cov, full_cov, num_samples, /, *, batch_size=None
        ):
            posterior = fit(rng, model_fn, train_data, train_cov, num_samples)
            tree = forecast(
                model_fn,
                posterior,
                train_data,
                full_cov,
                var_names=["forecast"],
                batch_size=batch_size,
                random_seed=rng,
            )
            return tree["predictions"]["forecast"]

        def in_sample_fn(rng, model_fn, train_data, train_cov, num_samples, /, *, batch_size=None):
            posterior = fit(rng, model_fn, train_data, train_cov, num_samples)
            tree = predict_in_sample(
                model_fn, posterior, train_data, train_cov, batch_size=batch_size, random_seed=rng
            )
            return tree["posterior_predictive"]["obs"]

        (result,) = backtest(
            data,
            cov,
            linear_model,
            forecast_fn=forecast_fn,
            in_sample_fn=in_sample_fn,
            min_train_window=25,
            stride=5,
            num_samples=40,
            eval_train=True,
            keep_predictions=True,
            random_seed=SEED,
        )
        assert result.metrics["mae"] < 0.5
        assert result.train_metrics["rmse"] < 0.5
        np.testing.assert_array_equal(result.prediction["time_future"], np.arange(25, 30))

    def test_each_closure_call_gets_its_own_stream(self):
        data, cov = make_trend_data(t_obs=14, horizon=0)
        states = []

        def recording(closure):
            def wrapped(rng, *args, **kwargs):
                states.append(copy.deepcopy(rng).random())
                return closure(rng, *args, **kwargs)

            return wrapped

        backtest(
            data,
            cov,
            None,
            forecast_fn=recording(last_value_forecast_fn()),
            in_sample_fn=recording(last_value_in_sample_fn),
            min_train_window=10,
            test_window=2,
            stride=2,
            num_samples=5,
            eval_train=True,
            random_seed=SEED,
        )
        assert len(states) == 4
        assert len(set(states)) == 4

    @pytest.mark.parametrize(
        "misalign",
        [
            lambda cov: cov.isel(time=slice(0, 11)),
            lambda cov: cov.assign_coords(time=cov["time"].values + 1),
        ],
        ids=["short", "shifted"],
    )
    def test_misaligned_covariates_rejected_at_entry(self, misalign):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        with pytest.raises(AlignmentError):
            backtest(data, misalign(cov), None, forecast_fn=never_called, min_train_window=10)

    def test_eval_train_does_not_change_the_forecast(self):
        data, cov = make_trend_data(t_obs=14, horizon=0)
        kwargs = dict(
            forecast_fn=last_value_forecast_fn(),
            in_sample_fn=last_value_in_sample_fn,
            min_train_window=10,
            test_window=2,
            stride=2,
            num_samples=5,
            random_seed=SEED,
        )
        without = backtest(data, cov, None, **kwargs)
        with_train = backtest(data, cov, None, eval_train=True, **kwargs)
        assert [r.metrics for r in without] == [r.metrics for r in with_train]
        assert all(set(r.train_metrics) == {"mae", "rmse", "crps", "coverage"} for r in with_train)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"forecaster_cls": Forecaster}, "forecaster_cls"),
            ({"forecaster_options": {}}, "forecaster_options"),
            ({"var_name": "mu_future"}, "var_name"),
            ({"eval_train": True}, "in_sample_fn"),
        ],
    )
    def test_conflicting_arguments_rejected_at_entry(self, kwargs, match):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        with pytest.raises(ValueError, match=match):
            backtest(data, cov, None, forecast_fn=never_called, min_train_window=10, **kwargs)

    def test_in_sample_fn_requires_forecast_fn(self):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        with pytest.raises(ValueError, match="in_sample_fn"):
            backtest(
                data,
                cov,
                None,
                forecaster_cls=LastValueForecaster,
                in_sample_fn=never_called,
                eval_train=True,
                min_train_window=10,
            )

    @pytest.mark.parametrize(
        ("returned", "error"),
        [
            (np.zeros((5, 2)), TypeError),
            (xr.DataArray(np.zeros((5, 2)), dims=("draw", "time_future")), ValueError),
        ],
    )
    def test_closure_must_return_chain_draw_dataarray(self, returned, error):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        with pytest.raises(error, match="forecast_fn must return"):
            backtest(data, cov, None, forecast_fn=lambda *a, **k: returned, min_train_window=10)


def make_panel(tz=None):
    """Two observed tables by month and cohort age, as in #99."""
    rng = np.random.default_rng(0)
    months = pd.date_range("2023-01-01", periods=24, freq="MS", tz=tz)
    return xr.Dataset(
        {
            "retained": (("time", "age"), rng.poisson(100, size=(24, 3))),
            "revenue": (("time", "age"), rng.gamma(50.0, 2.0, size=(24, 3))),
        },
        coords={"time": months, "age": [1, 2, 3]},
    )


def panel_forecast_fn(seen=None):
    def forecast_fn(
        rng, model_fn, train_data, train_cov, full_cov, num_samples, /, *, batch_size=None
    ):
        if seen is not None:
            seen.append((train_data, full_cov))
        mean = train_data["revenue"].mean("time").values
        draws = rng.normal(mean, 10.0, size=(1, num_samples, mean.size))
        return xr.DataArray(draws, dims=("chain", "draw", "age"), coords={"age": train_data["age"]})

    return forecast_fn


def mean_revenue(pred, truth):
    time_dim = "time_future" if "time_future" in truth.dims else "time"
    return pred, truth["revenue"].mean(time_dim)


class TestDatasetData:
    """Item C of #99: several observed tables in one Dataset."""

    @pytest.mark.parametrize("tz", [None, "America/Los_Angeles"])
    def test_tables_cut_at_the_origin_with_their_dtypes(self, tz):
        panel = make_panel(tz)
        months = panel.get_index("time")
        seen, truths = [], []

        def transform(pred, truth):
            truths.append(truth)
            return mean_revenue(pred, truth)

        results = backtest(
            panel,
            None,
            None,
            forecast_fn=panel_forecast_fn(seen),
            transform=transform,
            min_train_window=12,
            test_window=6,
            min_test_window=6,
            stride=6,
            keep_predictions=True,
            random_seed=SEED,
        )
        assert [(r.t0, r.t1, r.t2) for r in results] == [(0, 12, 18), (0, 18, 24)]
        for (train, full_cov), truth, r in zip(seen, truths, results, strict=True):
            assert isinstance(train, xr.Dataset)
            assert train["retained"].dtype == panel["retained"].dtype
            assert train.get_index("time").equals(months[r.t0 : r.t1])
            assert full_cov.get_index("time").equals(months[r.t0 : r.t2])
            assert isinstance(truth, xr.Dataset)
            assert truth.get_index("time_future").equals(months[r.t1 : r.t2].rename("time_future"))
            assert r.prediction.dims == ("chain", "draw", "age")
            assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}

    def test_eval_train_passes_the_training_tables_to_transform(self):
        panel = make_panel()
        truths = []

        def transform(pred, truth):
            truths.append(truth)
            return mean_revenue(pred, truth)

        def in_sample_fn(rng, model_fn, train_data, train_cov, num_samples, /, *, batch_size=None):
            return panel_forecast_fn()(rng, model_fn, train_data, train_cov, train_cov, num_samples)

        (result,) = backtest(
            panel,
            None,
            None,
            forecast_fn=panel_forecast_fn(),
            in_sample_fn=in_sample_fn,
            transform=transform,
            min_train_window=12,
            test_window=6,
            stride=12,
            eval_train=True,
            random_seed=SEED,
        )
        (train_truth,) = [truth for truth in truths if "time" in truth.dims]
        assert isinstance(train_truth, xr.Dataset)
        assert train_truth.get_index("time").equals(panel.get_index("time")[:12])
        assert set(result.train_metrics) == {"mae", "rmse", "crps", "coverage"}

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"forecaster_cls": LastValueForecaster}, TypeError, "requires forecast_fn"),
            ({"forecast_fn": never_called}, ValueError, "transform"),
        ],
    )
    def test_unsupported_combinations_rejected_at_entry(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            backtest(make_panel(), None, None, min_train_window=12, **kwargs)

    def test_variable_without_time_rejected(self):
        panel = make_panel().assign(cohort_size=("age", [10, 20, 30]))
        with pytest.raises(AlignmentError, match="cohort_size"):
            backtest(
                panel,
                None,
                None,
                forecast_fn=never_called,
                transform=mean_revenue,
                min_train_window=12,
            )


class TestRandomSeed:
    """Item D of #99: a ``numpy.random.Generator`` as the base seed."""

    def run(self, random_seed):
        data, cov = make_trend_data(t_obs=14, horizon=0)
        return backtest(
            data,
            cov,
            None,
            forecast_fn=last_value_forecast_fn(),
            min_train_window=10,
            test_window=2,
            stride=2,
            num_samples=5,
            keep_predictions=True,
            random_seed=random_seed,
        )

    def test_fresh_generators_with_one_seed_agree(self):
        first = self.run(np.random.default_rng(7))
        second = self.run(np.random.default_rng(7))
        for a, b in zip(first, second, strict=True):
            xr.testing.assert_identical(a.prediction, b.prediction)

    def test_reused_generator_spawns_new_streams(self):
        rng = np.random.default_rng(7)
        first = self.run(rng)
        second = self.run(rng)
        assert not np.array_equal(first[0].prediction.values, second[0].prediction.values)

    def test_integer_seed_is_reproducible_on_the_closure_path(self):
        first = self.run(SEED)
        second = self.run(SEED)
        for a, b in zip(first, second, strict=True):
            xr.testing.assert_identical(a.prediction, b.prediction)

    def test_forecaster_gets_integer_seeds_from_a_generator(self):
        data, cov = make_trend_data(t_obs=12, horizon=0)
        logs = []
        for _ in range(2):
            log = []
            backtest(
                data,
                cov,
                None,
                forecaster_cls=LastValueForecaster,
                forecaster_options={"log": log},
                min_train_window=10,
                num_samples=5,
                random_seed=np.random.default_rng(7),
            )
            logs.append(log)
        assert all(isinstance(seed, int) for _, seed in logs[0])
        assert logs[0] == logs[1]

    def test_integer_seed_gives_the_forecaster_the_seeds_of_v0_3(self):
        # Seeded forecaster backtests must not change results across versions.
        data, cov = make_trend_data(t_obs=12, horizon=0)
        log = []
        backtest(
            data,
            cov,
            None,
            forecaster_cls=LastValueForecaster,
            forecaster_options={"log": log},
            min_train_window=10,
            num_samples=5,
            random_seed=SEED,
        )
        expected = []
        for child in np.random.SeedSequence(SEED).spawn(2):
            fit_seed, forecast_seed = (int(s) for s in child.generate_state(2))
            expected += [("fit", fit_seed), ("forecast", forecast_seed)]
        assert log == expected

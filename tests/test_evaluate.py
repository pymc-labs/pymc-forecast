import numpy as np
import pytest
from example_models import linear_model, make_trend_data

from pymc_forecast.evaluate import (
    BacktestResult,
    _windows,
    backtest,
    results_to_dataframe,
)
from pymc_forecast.exceptions import BacktestWindowError
from pymc_forecast.metrics import make_mase

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

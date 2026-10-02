import numpy as np
import pandas as pd
import pytensor
import pytest
import scipy.stats
import xarray as xr

from pymc_forecast.data import TIME_DIM
from pymc_forecast.datasets import load_m5
from pymc_forecast.m5 import (
    M5_QUANTILES,
    T0,
    BottomUpModel,
    MiddleOutModel,
    TopDownModel,
    aggregate,
    bottom_covariates,
    bounded_exp,
    build_aggregation,
    calendar_features,
    disaggregate,
    evaluation_origins,
    forecast_draws,
    lagged_log_moving_average,
    last_28_day_shares,
    m5_scales,
    m5_weights,
    mean_level_score,
    reconcile_middle_out,
    reconcile_top_down,
    require_history,
    score_bottom,
    select_series,
    top_covariates,
    weight_gap,
)
from pymc_forecast.model import build_model

_KEYS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
_CALENDAR_HEADER = (
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
)


def _write_csv(path, rows):
    path.write_text("".join(",".join(map(str, row)) + "\n" for row in rows))


def _calendar_row(date, week, snap):
    return [date, week, "Saturday", 1, 1, 2011, "NA", "NA", "NA", "NA", *snap]


def _plant_m5(directory, *, train_days, test_days, extra_test=False, drop_test=False):
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(
        directory / "calendar.csv",
        [
            _CALENDAR_HEADER,
            _calendar_row("2011-01-29", 11101, (0, 1, 0)),
            _calendar_row("2011-01-30", 11101, (0, 0, 1)),
            _calendar_row("2011-01-31", 11103, (1, 0, 0)),
            _calendar_row("2011-02-01", 11102, (0, 0, 0)),
        ],
    )
    train = [
        ["item_id", "dept_id", "cat_id", "store_id", "state_id", *train_days],
        ["HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA", 1, 2, 3],
        ["FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX", 4, 5, 6],
    ]
    test_rows = [
        ["FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX", 8],
        ["HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA", 7],
    ]
    if drop_test:
        test_rows = test_rows[:1]
    if extra_test:
        test_rows.append(["HOUSEHOLD_1_001", "HOUSEHOLD_1", "HOUSEHOLD", "WI_1", "WI", 9])
    _write_csv(directory / "sales_train_evaluation.csv", train)
    _write_csv(
        directory / "sales_test_evaluation.csv",
        [["item_id", "dept_id", "cat_id", "store_id", "state_id", *test_days], *test_rows],
    )
    _write_csv(
        directory / "sell_prices.csv",
        [
            ["store_id", "item_id", "wm_yr_wk", "sell_price"],
            ["CA_1", "HOBBIES_1_001", 11101, 2.5],
            ["CA_1", "HOBBIES_1_001", 11102, 4.5],
            ["TX_1", "FOODS_1_001", 11101, 3.5],
        ],
    )
    _write_csv(
        directory / "weights_evaluation.csv",
        [
            ["Level_id", "Agg_Level_1", "Agg_Level_2", "Dollar_Sales", "weight"],
            ["Level1", "Total", "X", 1, 1],
        ],
    )


def _keys(*rows):
    frame = pd.DataFrame(rows, columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"])
    frame.insert(0, "id", frame["item_id"] + "_" + frame["store_id"])
    return frame


def _calendar(n, start="2011-01-29"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": np.arange(n),
            "snap_CA": np.arange(n) % 2,
            "snap_TX": np.arange(n) % 3,
            "snap_WI": np.arange(n) % 5,
        }
    )


def _series_data(values, time, names):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return xr.DataArray(
        values,
        dims=(TIME_DIM, "series"),
        coords={TIME_DIM: np.asarray(time), "series": list(names)},
    )


def _eval(model, name, point):
    target = model[name]
    ancestors = set(pytensor.graph.ancestors([target]))
    replacements = {
        model[key]: np.asarray(value)
        for key, value in point.items()
        if key in model.named_vars and model[key] in ancestors
    }
    return np.asarray(pytensor.graph.clone_replace(target, replacements).eval())


def _logp(model, point, var="obs"):
    transformed = {}
    for rv, value in model.rvs_to_values.items():
        transform = model.rvs_to_transforms[rv]
        if rv.name in point:
            raw = np.asarray(point[rv.name])
            if transform is None:
                transformed[value.name] = raw
            else:
                forward = transform.forward(raw, *rv.owner.inputs)
                transformed[value.name] = np.asarray(forward.eval())
        elif value.name in point:
            transformed[value.name] = point[value.name]
    return float(model.compile_logp(vars=[model[var]])(transformed))


def test_load_m5_keeps_train_order_and_maps_weeks_explicitly(tmp_path, monkeypatch):
    _plant_m5(tmp_path, train_days=["d_1", "d_2", "d_3"], test_days=["d_4"])
    monkeypatch.setattr(
        "pymc_forecast.datasets.pooch.create",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fetch")),
    )

    loaded = load_m5(tmp_path)

    assert list(loaded.keys["id"]) == ["HOBBIES_1_001_CA_1", "FOODS_1_001_TX_1"]
    np.testing.assert_array_equal(
        loaded.sales,
        np.array([[1, 4], [2, 5], [3, 6], [7, 8]], dtype=np.float32),
    )
    expected_price = np.array(
        [[2.5, 3.5], [2.5, 3.5], [np.nan, np.nan], [4.5, np.nan]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(loaded.price, expected_price, equal_nan=True)
    assert loaded.calendar["date"].iloc[0] == pd.Timestamp("2011-01-29")
    assert list(loaded.weights["Level_id"]) == ["Level1"]


def test_load_m5_ignores_test_only_rows_and_leaves_missing_test_days_nan(tmp_path):
    extra = tmp_path / "extra"
    _plant_m5(extra, train_days=["d_1", "d_2", "d_3"], test_days=["d_4"], extra_test=True)
    loaded = load_m5(extra)
    assert loaded.sales.shape == (4, 2)

    missing = tmp_path / "missing"
    _plant_m5(missing, train_days=["d_1", "d_2", "d_3"], test_days=["d_4"], drop_test=True)
    loaded = load_m5(missing)
    assert np.isnan(loaded.sales[3, 0])
    assert loaded.sales[3, 1] == 8


def test_load_m5_rejects_bad_sales_files(tmp_path):
    overlap = tmp_path / "overlap"
    _plant_m5(overlap, train_days=["d_1", "d_2", "d_3"], test_days=["d_3"])
    with pytest.raises(ValueError, match="overlap"):
        load_m5(overlap)

    gap = tmp_path / "gap"
    _plant_m5(gap, train_days=["d_1", "d_3"], test_days=["d_4"])
    with pytest.raises(ValueError, match="contiguous"):
        load_m5(gap)

    duplicate = tmp_path / "duplicate"
    _plant_m5(duplicate, train_days=["d_1", "d_2", "d_3"], test_days=["d_4"])
    train = (duplicate / "sales_train_evaluation.csv").read_text().splitlines()
    train.append(train[1])
    (duplicate / "sales_train_evaluation.csv").write_text("\n".join(train) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_m5(duplicate)


def test_scales_exclude_the_jump_into_the_active_period():
    values = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
        ]
    )
    # Column 0 is active for two days: the entry jump is excluded, leaving |4-2|.
    np.testing.assert_allclose(m5_scales(values), [2.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="at least 2"):
        m5_scales(values[:1])


def test_aggregation_uses_sorted_labels_and_sums_each_level():
    keys = _keys(
        ("B", "D2", "C", "S2", "TX"),
        ("A", "D1", "C", "S1", "CA"),
    )
    aggregation = build_aggregation(keys)
    assert aggregation.labels_for("Level2") == ["CA/X", "TX/X"]
    np.testing.assert_array_equal(aggregation.group_index["Level2"], [1, 0])
    assert aggregation.matrix.shape == (2, len(aggregation.labels))
    assert aggregation.slices["Level1"].stop == 1
    assert aggregation.slices["Level4"].stop - aggregation.slices["Level4"].start == 1
    assert aggregation.slices["Level2"].stop - aggregation.slices["Level2"].start == 2
    sales = np.array([[1.0, 3.0], [2.0, 4.0]])
    totals = aggregate(sales, aggregation.matrix)
    level1 = aggregation.slices["Level1"]
    np.testing.assert_allclose(totals[:, level1], [[4.0], [6.0]])
    with pytest.raises(ValueError, match="last axis"):
        aggregate(sales[:, :1], aggregation.matrix)


def test_weights_sum_to_one_and_a_zero_dollar_level_is_uniform():
    keys = _keys(("A", "D", "C", "S", "CA"), ("B", "D", "C", "S", "CA"))
    aggregation = build_aggregation(keys)
    sales = np.ones((30, 2))
    price = np.ones((30, 2))
    price[:, 1] = 0.0
    weights = m5_weights(sales, price, aggregation, 28)
    for sl in aggregation.slices.values():
        assert weights[sl].sum() == pytest.approx(1.0)

    zeros = m5_weights(np.zeros((30, 2)), np.zeros((30, 2)), aggregation, 28)
    for sl in aggregation.slices.values():
        width = sl.stop - sl.start
        np.testing.assert_allclose(zeros[sl], np.full(width, 1.0 / width))
    with pytest.raises(ValueError, match="28 days"):
        m5_weights(sales, price, aggregation, 10)


def test_weight_gap_matches_on_the_official_label():
    official = pd.DataFrame(
        {
            "Level_id": ["Level2", "Level1", "Level2"],
            "Agg_Level_1": ["CA", "Total", "TX"],
            "Agg_Level_2": ["X", "X", "X"],
            "Dollar_Sales": [1.0, 2.0, 1.0],
            "weight": [0.4, 1.0, 0.5],
        }
    )
    labels = ["Level1/Total/X", "Level2/CA/X", "Level2/TX/X"]
    assert weight_gap(official, labels, np.array([1.0, 0.41, 0.5])) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="did not match"):
        weight_gap(official.iloc[:2], labels, np.ones(3))


def test_shares_are_uniform_when_a_group_sold_nothing():
    train = np.zeros((28, 4))
    train[-1, 0] = 3.0
    train[-1, 1] = 1.0
    shares = last_28_day_shares(train, np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(shares, [0.75, 0.25, 0.5, 0.5])
    with pytest.raises(ValueError, match="negative"):
        last_28_day_shares(train, np.array([-1, 0, 1, 1]))


def test_disaggregate_follows_one_generator_stream_at_any_chunk_size():
    draws = np.array([[1.0, 4.0], [2.0, 0.0], [np.nan, np.inf]])
    shares = np.array([0.25, 0.75, 1.0])
    groups = np.array([0, 1, 0])
    fine = disaggregate(7, draws, shares, groups, chunk=1)
    coarse = disaggregate(7, draws, shares, groups, chunk=50)
    rate = draws[..., groups] * shares
    rate = np.clip(np.nan_to_num(rate, nan=0.0, posinf=1e9, neginf=0.0), 0.0, 1e9)
    expected = np.random.default_rng(7).poisson(np.ascontiguousarray(rate).ravel())
    expected = expected.astype(np.float32).reshape(rate.shape)
    np.testing.assert_array_equal(fine, coarse)
    np.testing.assert_array_equal(fine, expected)
    with pytest.raises(ValueError, match="outside"):
        disaggregate(0, np.ones((2, 2)), np.ones(1), np.array([5]))


def test_lagged_moving_average_at_day_121_uses_earlier_history():
    sales = np.zeros((200, 1))
    sales[:121, 0] = 1.0
    sales[121:, 0] = 100.0
    full = lagged_log_moving_average(sales, 28, 28)
    sliced = lagged_log_moving_average(sales[121:], 28, 28)
    assert full[121, 0] == pytest.approx(np.log(1.0))
    assert sliced[0, 0] == pytest.approx(np.log(1e-3))
    assert full[10, 0] == pytest.approx(np.log(1e-3))


def test_bounded_exp_matches_the_capped_sigmoid():
    values = np.array([-100.0, 0.0, np.log(1000.0), 100.0])
    expected = (1.0 / (1.0 + np.exp(-(values - np.log(1000.0))))) * 1000.0
    np.testing.assert_allclose(bounded_exp(values), expected)
    capped = bounded_exp(np.array([100.0]))[0]
    assert 999.0 < capped <= 1000.0


def test_calendar_features_and_bottom_channels():
    calendar = _calendar(40, start="2011-12-20")
    features = calendar_features(calendar)
    assert features.loc[0, "dow"] == 1  # Tuesday, 2011-12-20
    christmas = features.index[features["christmas"] == 1]
    assert calendar.loc[christmas, "date"].item() == pd.Timestamp("2011-12-25")
    assert features.loc[calendar["date"].dt.day == 1, "dom_1"].iloc[0] == 1
    np.testing.assert_allclose(features["years"], np.arange(40) / 365.0)

    sales = np.arange(40 * 3, dtype=float).reshape(40, 3)
    price = np.ones_like(sales)
    price[0, 1] = np.nan
    keys = _keys(
        ("A", "D", "C", "S1", "CA"),
        ("B", "D", "C", "S2", "TX"),
        ("C", "D", "C", "S3", "WI"),
    )
    covariates = bottom_covariates(sales, price, calendar, keys)
    assert list(covariates.coords["channel"].values) == [
        "dow",
        "snap",
        "saled",
        "ma28",
        "ma56",
        "ma84",
    ]
    np.testing.assert_array_equal(
        covariates.sel(channel="snap").values[0],
        features.loc[0, ["snap_CA", "snap_TX", "snap_WI"]],
    )
    assert covariates.sel(channel="saled").values[0, 1] == 0.0
    with pytest.raises(ValueError, match="unknown state_id"):
        bottom_covariates(sales[:, :1], price[:, :1], calendar, _keys(("A", "D", "C", "S", "ZZ")))


def test_select_series_keeps_file_order():
    keys = _keys(
        ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_003", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
        ("FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("FOODS_1_003", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "TX_2", "TX"),
    )
    np.testing.assert_array_equal(
        select_series(keys, stores=["CA_1", "TX_1"], items_per_dept=2),
        [0, 1, 3, 4, 5],
    )
    with pytest.raises(ValueError, match="unknown store"):
        select_series(keys, stores=["WI_1"], items_per_dept=1)
    with pytest.raises(ValueError, match="items_per_dept"):
        select_series(keys, stores=["CA_1"], items_per_dept=0)


def test_evaluation_origins_match_the_upstream_backtest():
    assert evaluation_origins(1941) == [1843, 1878, 1913]
    with pytest.raises(ValueError, match="cannot host"):
        evaluation_origins(30, n_windows=2)
    with pytest.raises(ValueError, match="at least"):
        require_history(T0)


def test_scores_match_a_two_sample_hand_calculation():
    np.testing.assert_allclose(
        M5_QUANTILES,
        [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995],
    )
    keys = _keys(("A", "D", "C", "S", "CA"))
    aggregation = build_aggregation(keys)
    pred = np.array([[[1.0]], [[3.0]]])
    truth = np.array([[2.0]])
    weights = np.ones(aggregation.matrix.shape[1])
    scales = np.full(weights.shape, 2.0)
    scores, wspl = score_bottom(pred, truth, aggregation, weights, scales)
    assert list(scores) == list(aggregation.slices)
    assert all(value == pytest.approx(0.25) for value in scores.values())
    assert mean_level_score(scores) == pytest.approx(0.25)

    quantiles = 1.0 + 2.0 * M5_QUANTILES
    error = quantiles - 2.0
    pinball = np.where(error <= 0.0, -M5_QUANTILES * error, (1.0 - M5_QUANTILES) * error)
    assert wspl == pytest.approx(float(pinball.mean()) / 2.0)


def test_reconcile_top_down_exponentiates_before_the_poisson_split():
    forecast = xr.Dataset(
        {
            "forecast": xr.DataArray(
                np.log([[[8.0], [4.0]]]),
                dims=("sample", "time_future", "series"),
            )
        }
    )
    train = np.zeros((28, 2))
    train[-1] = [3.0, 1.0]
    draws = reconcile_top_down(forecast, train, seed=3)
    units = np.array([[[8.0, 8.0], [4.0, 4.0]]])
    shares = last_28_day_shares(train, np.zeros(2, dtype=np.int64))
    expected = disaggregate(3, units, shares, np.zeros(2, dtype=np.int64))
    np.testing.assert_array_equal(draws, expected)
    assert draws.shape == (1, 2, 2)


def test_reconcile_middle_out_clips_and_restores_units():
    forecast = xr.Dataset(
        {
            "forecast": xr.DataArray(
                [[[-2.0, 3.0], [1.0, 0.5]]],
                dims=("sample", "time_future", "series"),
            )
        }
    )
    train = np.ones((28, 3))
    groups = np.array([0, 0, 1])
    scale = np.array([2.0, 4.0])
    draws = reconcile_middle_out(forecast, train, groups, scale, seed=5)
    units = np.clip(forecast["forecast"].values, 0.0, None) * scale
    expected = disaggregate(5, units, last_28_day_shares(train, groups), groups)
    np.testing.assert_array_equal(draws, expected)
    assert draws.shape == (1, 2, 3)


def _top_point(model):
    point = model.initial_point()
    point["bias"] = np.array(0.4)
    point["trend"] = np.array(1.2)
    point["weight"] = np.linspace(-0.2, 0.2, point["weight"].size)
    point["seasonal"] = np.arange(7, dtype=float)
    point["dof"] = np.array(4.0)
    point["noise_scale"] = np.array(0.5)
    return point


def test_top_down_mu_matches_the_numpy_regression_on_a_forecast_build():
    calendar = _calendar(8)
    covariates = top_covariates(calendar)
    data = _series_data(np.linspace(0.1, 0.8, 5), covariates[TIME_DIM].values[:5], ["total"])
    model = build_model(TopDownModel(), data, covariates)
    point = _top_point(model)
    years = covariates.sel(feature="years").values
    dow = covariates.sel(feature="dow").values.astype(int)
    dom_names = [
        name for name in covariates.coords["feature"].values if str(name).startswith("dom_")
    ]
    dom = covariates.sel(feature=dom_names).values
    expected = (
        point["bias"] + point["trend"] * years + point["seasonal"][dow] + dom @ point["weight"]
    )
    np.testing.assert_allclose(_eval(model, "mu", point)[:, 0], expected[:5])
    np.testing.assert_allclose(_eval(model, "mu_future", point)[:, 0], expected[5:])
    sigma = float(point["noise_scale"])
    dof = float(point["dof"])
    logp = _logp(model, point)
    expected_logp = scipy.stats.t.logpdf(
        data.values[:, 0], dof, loc=expected[:5], scale=sigma
    ).sum()
    assert logp == pytest.approx(expected_logp)


def _bottom_channels(n_days, n_series):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(n_days, n_series, 6))
    values[..., 0] = np.resize(np.arange(7), n_days)[:, None]
    values[..., 1] = rng.integers(0, 2, size=(n_days, n_series))
    values[..., 2] = rng.integers(0, 2, size=(n_days, n_series))
    return xr.DataArray(
        values,
        dims=(TIME_DIM, "series", "channel"),
        coords={
            TIME_DIM: np.arange(n_days),
            "series": [f"s{i}" for i in range(n_series)],
            "channel": ["dow", "snap", "saled", "ma28", "ma56", "ma84"],
        },
    )


def _bottom_numpy(point, covariates, store, dept):
    dow = covariates.sel(channel="dow").values[:, 0].astype(int)
    snap = covariates.sel(channel="snap").values
    saled = covariates.sel(channel="saled").values
    log_ma = np.stack(
        [covariates.sel(channel=name).values for name in ("ma28", "ma56", "ma84")],
        axis=-1,
    )
    moving = np.einsum("nhk,tnk->tnh", point["ma_weight"][store, :, :, dept], log_ma)
    snap_effect = point["snap_weight"][store, :, dept][None] * snap[:, :, None]
    seasonal = point["seasonal"][store, :, :, dept][:, dow, :].transpose(1, 0, 2)
    combined = moving + snap_effect + seasonal
    mean = bounded_exp(combined[..., 0]) * saled + 1e-3
    scale = bounded_exp(combined[..., 1]) * saled + 1e-3
    return mean, scale


def test_bottom_up_gamma_uses_the_training_scale_on_a_forecast_build():
    covariates = _bottom_channels(6, 2)
    data = _series_data(
        np.array([[1.0, 2.0], [1.5, 0.5], [2.0, 3.0], [0.4, 1.2]]),
        np.arange(4),
        ["s0", "s1"],
    )
    spec = BottomUpModel(np.array([0, 0]), np.array([1, 0]), ["CA_1"], ["D0", "D1"])
    model = build_model(spec, data, covariates)
    point = model.initial_point()
    point["ma_weight"] = np.linspace(-0.3, 0.3, point["ma_weight"].size).reshape(
        point["ma_weight"].shape
    )
    point["snap_weight"] = np.linspace(-0.2, 0.4, point["snap_weight"].size).reshape(
        point["snap_weight"].shape
    )
    point["seasonal"] = np.linspace(-0.5, 0.5, point["seasonal"].size).reshape(
        point["seasonal"].shape
    )
    mean, scale = _bottom_numpy(point, covariates, spec.store_index, spec.dept_index)
    np.testing.assert_allclose(_eval(model, "mu", point), mean[:4])
    np.testing.assert_allclose(_eval(model, "mu_future", point), mean[4:])
    logp = _logp(model, point)
    expected = scipy.stats.gamma.logpdf(data.values, a=mean[:4] / scale[:4], scale=scale[:4]).sum()
    future_scale = np.resize(scale[4:], mean[:4].shape)
    wrong = scipy.stats.gamma.logpdf(
        data.values, a=mean[:4] / future_scale, scale=future_scale
    ).sum()
    assert logp == pytest.approx(expected)
    assert logp != pytest.approx(wrong)


def test_middle_out_mu_matches_the_per_series_regression():
    from pymc_forecast.m5 import middle_covariates

    calendar = _calendar(7)
    full = middle_covariates(calendar)
    # Keep two harmonics so the test model stays small and still exercises the contraction.
    names = ["years", "dow", "sin_1", "cos_1"]
    covariates = full.sel(feature=names)
    data = _series_data(np.ones((4, 2)), covariates[TIME_DIM].values[:4], ["g0", "g1"])
    model = build_model(MiddleOutModel(), data, covariates)
    point = model.initial_point()
    point["bias"] = np.array([0.2, -0.4])
    point["trend"] = np.array([1.1, 0.7])
    point["weight"] = np.arange(4, dtype=float).reshape(2, 2)
    point["seasonal"] = np.arange(14, dtype=float).reshape(7, 2)
    point["noise_scale"] = np.array([0.3, 0.6])
    point["dof"] = np.array(5.0)
    years = covariates.sel(feature="years").values
    dow = covariates.sel(feature="dow").values.astype(int)
    feature = covariates.sel(feature=["sin_1", "cos_1"]).values
    expected = (
        point["bias"]
        + point["trend"] * years[:, None]
        + point["seasonal"][dow]
        + feature @ point["weight"].T
    )
    np.testing.assert_allclose(_eval(model, "mu", point), expected[:4])
    np.testing.assert_allclose(_eval(model, "mu_future", point), expected[4:])


def test_forecasters_emit_finite_draws_and_reconciliation_shapes():
    from pymc_forecast import Forecaster

    calendar = _calendar(6)
    from pymc_forecast.m5 import top_covariates

    top_cov = top_covariates(calendar)
    top_data = _series_data(np.linspace(1.0, 1.4, 4), top_cov[TIME_DIM].values[:4], ["total"])
    top = Forecaster(
        TopDownModel(),
        top_data,
        top_cov.isel({TIME_DIM: slice(None, 4)}),
        num_steps=2,
        progressbar=False,
        random_seed=0,
        optimizer=0.1,
    )
    top_fc = top.forecast(
        future_covariates=top_cov.isel({TIME_DIM: slice(4, None)}),
        num_samples=2,
        random_seed=1,
    )
    top_draws = forecast_draws(top_fc)
    assert top_draws.sizes["sample"] == 2
    assert np.isfinite(top_draws.values).all()
    reconciled = reconcile_top_down(top_fc, np.ones((28, 3)), seed=1)
    assert reconciled.shape == (2, 2, 3)
    assert np.isfinite(reconciled).all()

    covariates = _bottom_channels(6, 2)
    bottom_data = _series_data(np.ones((4, 2)), np.arange(4), ["s0", "s1"])
    spec = BottomUpModel(np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.int32), ["S"], ["D"])
    bottom = Forecaster(
        spec,
        bottom_data,
        covariates.isel({TIME_DIM: slice(None, 4)}),
        num_steps=2,
        progressbar=False,
        random_seed=0,
        optimizer=0.1,
    )
    bottom_fc = bottom.forecast(
        future_covariates=covariates.isel({TIME_DIM: slice(4, None)}),
        num_samples=2,
        random_seed=1,
    )
    bottom_draws = forecast_draws(bottom_fc)
    assert bottom_draws.sizes["time_future"] == 2
    assert bottom_draws.sizes["series"] == 2
    assert np.isfinite(bottom_draws.values).all()

    from pymc_forecast.m5 import middle_covariates

    mid_cov = middle_covariates(calendar).sel(feature=["years", "dow", "sin_1", "cos_1"])
    mid_data = _series_data(np.ones((4, 2)), mid_cov[TIME_DIM].values[:4], ["g0", "g1"])
    middle = Forecaster(
        MiddleOutModel(),
        mid_data,
        mid_cov.isel({TIME_DIM: slice(None, 4)}),
        num_steps=2,
        progressbar=False,
        random_seed=0,
        optimizer=0.1,
    )
    mid_fc = middle.forecast(
        future_covariates=mid_cov.isel({TIME_DIM: slice(4, None)}),
        num_samples=2,
        random_seed=1,
    )
    mid_draws = forecast_draws(mid_fc)
    assert np.isfinite(mid_draws.values).all()
    split = reconcile_middle_out(mid_fc, np.ones((28, 3)), np.array([0, 1, 1]), np.ones(2), seed=2)
    assert split.shape == (2, mid_draws.sizes["time_future"], 3)

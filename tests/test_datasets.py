import numpy as np
import pandas as pd
import pytest

from pymc_forecast.datasets import (
    load_bart_od,
    load_bart_weekly,
    load_bart_weekly_by_origin,
    load_m5,
    load_victoria_electricity,
)


def test_bart_loaders(monkeypatch, tmp_path):
    stations = np.array(["A", "B"])
    start_date = np.array([np.datetime64("2011-01-01T00:00")])
    paths = []
    for index, value in enumerate((1, 2)):
        path = tmp_path / f"bart_{index}.npz"
        np.savez(
            path,
            stations=stations,
            start_date=start_date,
            counts=np.full((7 * 24, 2, 2), value, dtype=np.int16),
        )
        paths.append(path)
    monkeypatch.setattr("pymc_forecast.datasets._bart_file_paths", lambda: paths)

    od = load_bart_od()
    assert od.dims == ("time", "origin", "destination")
    assert od.shape == (2 * 7 * 24, 2, 2)
    np.testing.assert_array_equal(od["origin"], stations)
    assert od["time"].values[0] == np.datetime64("2011-01-01T00:00")

    rides = load_bart_weekly()
    assert rides.dims == ("time",)
    assert rides.sizes["time"] == 2
    np.testing.assert_array_equal(rides["time"], np.arange(2))
    np.testing.assert_allclose(rides, np.log([7 * 24 * 4, 7 * 24 * 8]))
    assert rides.name == "log_rides"

    panel = load_bart_weekly_by_origin(num_series=1)
    assert panel.dims == ("time", "series")
    assert panel.shape == (2, 1)
    np.testing.assert_array_equal(panel["series"], ["B"])
    np.testing.assert_allclose(panel[:, 0], np.log1p([7 * 24 * 2, 7 * 24 * 4]))

    all_stations = load_bart_weekly_by_origin(num_series=None)
    np.testing.assert_array_equal(all_stations["series"], stations)

    with pytest.raises(ValueError, match="positive or None"):
        load_bart_weekly_by_origin(num_series=0)


def test_victoria_electricity():
    demand, temperature = load_victoria_electricity()
    assert demand.dims == ("time",) and temperature.dims == ("time",)
    assert demand.sizes["time"] == temperature.sizes["time"] == 8 * 7 * 24
    index = pd.DatetimeIndex(demand["time"].values)
    assert index[0] == pd.Timestamp("2014-01-01")
    assert (index[1] - index[0]) == pd.Timedelta(hours=1)
    assert 0 < float(demand.mean()) < 10  # GW scale
    assert np.isfinite(temperature.values).all()


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


def test_load_m5_keeps_train_order_and_maps_weeks_explicitly(tmp_path, monkeypatch):
    _plant_m5(tmp_path, train_days=["d_1", "d_2", "d_3"], test_days=["d_4"])
    monkeypatch.setattr(
        "pymc_forecast.datasets.pooch.create",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fetch")),
    )

    loaded = load_m5(tmp_path)

    assert list(loaded.keys["id"]) == ["HOBBIES_1_001_CA_1", "FOODS_1_001_TX_1"]
    assert loaded.sales.dims == ("time", "series")
    assert loaded.price.dims == ("time", "series")
    assert list(loaded.sales["series"].values) == list(loaded.keys["id"])
    np.testing.assert_array_equal(
        loaded.sales["time"].values,
        pd.to_datetime(["2011-01-29", "2011-01-30", "2011-01-31", "2011-02-01"]).to_numpy(),
    )
    np.testing.assert_array_equal(
        loaded.sales.values,
        np.array([[1, 4], [2, 5], [3, 6], [7, 8]], dtype=np.float32),
    )
    expected_price = np.array(
        [[2.5, 3.5], [2.5, 3.5], [np.nan, np.nan], [4.5, np.nan]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(loaded.price.values, expected_price, equal_nan=True)
    assert loaded.sales.dtype == np.float32
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
    assert np.isnan(loaded.sales.values[3, 0])
    assert loaded.sales.values[3, 1] == 8


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

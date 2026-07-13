import numpy as np
import pandas as pd
import pytest

from pymc_forecast.datasets import (
    load_bart_od,
    load_bart_weekly,
    load_bart_weekly_by_origin,
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

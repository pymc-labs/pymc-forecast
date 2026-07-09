import numpy as np
import pandas as pd

from pymc_forecast.datasets import load_victoria_electricity


def test_victoria_electricity():
    demand, temperature = load_victoria_electricity()
    assert demand.dims == ("time",) and temperature.dims == ("time",)
    assert demand.sizes["time"] == temperature.sizes["time"] == 8 * 7 * 24
    index = pd.DatetimeIndex(demand["time"].values)
    assert index[0] == pd.Timestamp("2014-01-01")
    assert (index[1] - index[0]) == pd.Timedelta(hours=1)
    assert 0 < float(demand.mean()) < 10  # GW scale
    assert np.isfinite(temperature.values).all()

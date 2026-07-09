import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_forecast.data import as_dataarray, null_covariates, validate_alignment
from pymc_forecast.exceptions import AlignmentError


class TestAsDataarray:
    def test_numpy_1d(self):
        da = as_dataarray(np.arange(5.0))
        assert da.dims == ("time",)
        np.testing.assert_array_equal(da["time"].values, np.arange(5))

    def test_numpy_2d_roles(self):
        assert as_dataarray(np.zeros((4, 2)), role="data").dims == ("time", "series")
        assert as_dataarray(np.zeros((4, 2)), role="covariates").dims == (
            "time",
            "covariate",
        )

    def test_numpy_3d_rejected(self):
        with pytest.raises(AlignmentError, match="1-d or 2-d"):
            as_dataarray(np.zeros((2, 2, 2)))

    def test_series_keeps_index(self):
        idx = pd.date_range("2026-01-01", periods=4, freq="W")
        da = as_dataarray(pd.Series(np.arange(4.0), index=idx))
        assert da.dims == ("time",)
        np.testing.assert_array_equal(da["time"].values, idx.values)

    def test_dataframe_columns_become_coord(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        da = as_dataarray(df, role="covariates")
        assert da.dims == ("time", "covariate")
        assert list(da["covariate"].values) == ["a", "b"]

    def test_dataarray_transposed_time_first(self):
        da_in = xr.DataArray(np.zeros((2, 3)), dims=("series", "time"), coords={"time": [0, 1, 2]})
        da = as_dataarray(da_in)
        assert da.dims == ("time", "series")

    def test_dataarray_without_time_dim_rejected(self):
        with pytest.raises(AlignmentError, match="'time' dim"):
            as_dataarray(xr.DataArray(np.zeros(3), dims=("t",)))

    def test_missing_time_coord_gets_range(self):
        da = as_dataarray(xr.DataArray(np.zeros(3), dims=("time",)))
        np.testing.assert_array_equal(da["time"].values, [0, 1, 2])


class TestNullCovariates:
    def test_zero_width_with_coords(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="D")
        da = null_covariates(idx)
        assert da.shape == (6, 0)
        assert da.dims == ("time", "covariate")
        np.testing.assert_array_equal(da["time"].values, idx.values)


class TestValidateAlignment:
    def test_ok_when_covariates_extend(self):
        data = as_dataarray(np.arange(3.0))
        cov = as_dataarray(np.zeros((5, 1)), role="covariates")
        validate_alignment(data, cov)

    def test_short_covariates_rejected(self):
        data = as_dataarray(np.arange(5.0))
        cov = as_dataarray(np.zeros((3, 1)), role="covariates")
        with pytest.raises(AlignmentError, match="must extend"):
            validate_alignment(data, cov)

    def test_mismatched_coords_rejected(self):
        data = as_dataarray(
            xr.DataArray(np.zeros(3), dims=("time",), coords={"time": [10, 11, 12]})
        )
        cov = as_dataarray(np.zeros((5, 1)), role="covariates")
        with pytest.raises(AlignmentError, match="misaligned"):
            validate_alignment(data, cov)

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_forecast.data import (
    as_dataarray,
    concat_covariates,
    concat_time_index,
    extend_time_index,
    null_covariates,
    validate_alignment,
)
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


class TestExtendTimeIndex:
    def test_datetime_inferred_freq(self):
        idx = pd.date_range("2024-01-07", periods=5, freq="W")
        out = extend_time_index(idx, 3)
        assert len(out) == 8
        assert out[5] == idx[-1] + pd.Timedelta(weeks=1)
        assert out[-1] == idx[-1] + pd.Timedelta(weeks=3)

    def test_numeric_constant_step(self):
        out = extend_time_index(np.array([0, 2, 4, 6]), 2)
        np.testing.assert_array_equal(np.asarray(out), [0, 2, 4, 6, 8, 10])

    def test_horizon_zero_is_identity(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        assert len(extend_time_index(idx, 0)) == 3

    def test_negative_horizon_rejected(self):
        with pytest.raises(AlignmentError, match="non-negative"):
            extend_time_index(np.arange(3), -1)

    def test_uneven_numeric_rejected(self):
        with pytest.raises(AlignmentError, match="evenly spaced"):
            extend_time_index(np.array([0, 1, 4]), 2)


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


class TestConcatTimeIndex:
    def test_numeric_gap_allowed(self):
        full = concat_time_index(np.arange(5), [10, 20, 30])
        np.testing.assert_array_equal(full, [0, 1, 2, 3, 4, 10, 20, 30])

    def test_datetime(self):
        train = pd.date_range("2024-01-01", periods=4, freq="D")
        future = pd.date_range("2024-02-01", periods=2, freq="D")
        full = concat_time_index(train, future)
        assert len(full) == 6
        assert full[-1] == pd.Timestamp("2024-02-02")

    def test_empty_future_rejected(self):
        with pytest.raises(AlignmentError, match="empty"):
            concat_time_index(np.arange(5), [])

    def test_not_increasing_rejected(self):
        with pytest.raises(AlignmentError, match="strictly increasing"):
            concat_time_index(np.arange(5), [7, 6])
        with pytest.raises(AlignmentError, match="strictly increasing"):
            concat_time_index(np.arange(5), [7, 7])

    def test_overlap_with_training_rejected(self):
        with pytest.raises(AlignmentError, match="strictly after"):
            concat_time_index(np.arange(5), [4, 5, 6])

    def test_incomparable_types_rejected(self):
        train = pd.date_range("2024-01-01", periods=3, freq="D")
        with pytest.raises(AlignmentError, match="not comparable"):
            concat_time_index(train, [1, 2, 3])


class TestConcatCovariates:
    def _train_cov(self):
        return as_dataarray(
            xr.DataArray(
                np.zeros((5, 2)),
                dims=("time", "covariate"),
                coords={"time": np.arange(5), "covariate": ["a", "b"]},
            ),
            role="covariates",
        )

    def test_appends_future_rows(self):
        fut = xr.DataArray(
            np.ones((3, 2)),
            dims=("time", "covariate"),
            coords={"time": [5, 6, 7], "covariate": ["a", "b"]},
        )
        full = concat_covariates(self._train_cov(), fut)
        assert full.sizes == {"time": 8, "covariate": 2}
        np.testing.assert_array_equal(full["time"].values, np.arange(8))
        np.testing.assert_array_equal(full.values[5:], np.ones((3, 2)))

    def test_dataframe_input_normalized(self):
        pdf = pd.DataFrame(np.ones((2, 2)), index=[5, 6], columns=["a", "b"])
        full = concat_covariates(self._train_cov(), pdf)
        assert full.sizes["time"] == 7

    def test_mismatched_names_rejected(self):
        fut = xr.DataArray(
            np.ones((2, 2)),
            dims=("time", "covariate"),
            coords={"time": [5, 6], "covariate": ["b", "a"]},
        )
        with pytest.raises(AlignmentError, match="coords must match"):
            concat_covariates(self._train_cov(), fut)

    def test_mismatched_width_rejected(self):
        fut = xr.DataArray(
            np.ones((2, 1)),
            dims=("time", "covariate"),
            coords={"time": [5, 6], "covariate": ["a"]},
        )
        with pytest.raises(AlignmentError, match="size mismatch"):
            concat_covariates(self._train_cov(), fut)

    def test_mismatched_dims_rejected(self):
        fut = xr.DataArray(np.ones(2), dims=("time",), coords={"time": [5, 6]})
        with pytest.raises(AlignmentError, match="same dims"):
            concat_covariates(self._train_cov(), fut)

    def test_overlapping_time_rejected(self):
        fut = xr.DataArray(
            np.ones((2, 2)),
            dims=("time", "covariate"),
            coords={"time": [4, 5], "covariate": ["a", "b"]},
        )
        with pytest.raises(AlignmentError, match="strictly after"):
            concat_covariates(self._train_cov(), fut)

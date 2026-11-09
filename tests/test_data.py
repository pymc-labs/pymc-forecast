import datetime as dt
import zoneinfo

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_forecast.data import (
    as_dataarray,
    as_time_index,
    concat_covariates,
    concat_time_index,
    extend_time_index,
    null_covariates,
    validate_alignment,
)
from pymc_forecast.exceptions import AlignmentError


class TestAsDataarray:
    @pytest.mark.parametrize("container", [pd.Series, pd.DataFrame])
    def test_short_datetime_frequency_survives_normalization(self, container):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        data = as_dataarray(container([1.0, 2.0], index=index))
        assert data.get_index("time").freq == index.freq

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
    @pytest.mark.parametrize("freq", ["M", "Q", "2M"])
    def test_period_index(self, freq):
        index = pd.period_range("2024-01", periods=3, freq=freq)
        result = extend_time_index(index.values, 2)
        assert result.equals(pd.period_range("2024-01", periods=5, freq=freq))

    @pytest.mark.parametrize("index", [[], [0, 0], [2, 1]])
    def test_empty_or_non_increasing_index_rejected(self, index):
        with pytest.raises(AlignmentError, match="non-empty and strictly increasing"):
            extend_time_index(index, 2)

    @pytest.mark.parametrize("horizon", [1.5, True])
    def test_non_integer_horizon_rejected(self, horizon):
        with pytest.raises(AlignmentError, match="integer"):
            extend_time_index([0, 1], horizon)

    def test_two_dates_with_frequency(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        assert extend_time_index(index, 1)[-1] == pd.Timestamp("2024-01-03")

    def test_short_datetime_without_frequency_has_actionable_error(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D").values
        with pytest.raises(AlignmentError, match="future_index="):
            extend_time_index(index, 1)

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
    @pytest.mark.parametrize("future", [[3, 2], [3, 3], [1, 2]])
    def test_invalid_future_coordinates_rejected(self, future):
        data = as_dataarray(np.arange(3.0))
        cov = null_covariates([0, 1, 2, *future])
        with pytest.raises(AlignmentError, match="strictly"):
            validate_alignment(data, cov)

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

    def test_raw_training_input_normalized(self):
        training = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2], columns=["a", "b"])
        future = pd.DataFrame(np.ones((2, 2)), index=[3, 4], columns=["a", "b"])
        full = concat_covariates(training, future)
        assert full.dims == ("time", "covariate")
        assert full.sizes["time"] == 5

    def test_asymmetric_coord_labeling_rejected(self):
        unlabeled = xr.DataArray(
            np.ones((2, 2)),
            dims=("time", "covariate"),
            coords={"time": [5, 6]},
        )
        with pytest.raises(AlignmentError, match="one side is unlabeled"):
            concat_covariates(self._train_cov(), unlabeled)


def _zone(index):
    """Timezone of an index, including an object index of aware timestamps."""
    if getattr(index, "tz", None) is not None:
        return index.tz
    stamps = [pd.Timestamp(value) for value in index]
    zones = {stamp.tz for stamp in stamps}
    assert len(zones) == 1
    return zones.pop()


class TestTimezonePreserved:
    @pytest.mark.parametrize("tz", ["America/Los_Angeles", "UTC", "Europe/Berlin"])
    def test_helpers_keep_zone_and_clock(self, tz):
        train = pd.date_range("2020-01-10 22:00", periods=3, freq="h", tz=tz)
        future = pd.date_range(train[-1] + pd.Timedelta(hours=1), periods=2, freq="h", tz=tz)
        da = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": train})

        assert as_time_index(da).equals(train)
        assert as_time_index(train).equals(train)
        assert null_covariates(train).get_index("time").equals(train)
        assert extend_time_index(train, 2).equals(train.append(future))
        assert concat_time_index(train, future).equals(train.append(future))
        extended = null_covariates(train.append(future))
        validate_alignment(da, extended)
        assert _zone(extended.get_index("time")) == train.tz

    def test_naive_and_aware_stay_incomparable(self):
        naive = pd.date_range("2024-01-01", periods=3, freq="D")
        aware = pd.date_range("2024-01-04", periods=2, freq="D", tz="UTC")
        with pytest.raises(AlignmentError, match="not comparable"):
            concat_time_index(naive, aware)
        data = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": naive})
        cov = null_covariates(naive.append(pd.date_range("2024-01-04", periods=2, freq="D")))
        # Same wall-clock values in UTC are not the naive index. Do not call
        # that a misalignment, and do not treat the zone as UTC.
        aware_prefix = null_covariates(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
        with pytest.raises(AlignmentError, match="localize both to one zone"):
            validate_alignment(data, aware_prefix)
        validate_alignment(data, cov)

    def test_aware_data_with_naive_covariates_names_the_mismatch(self):
        utc = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
        data = xr.DataArray(np.arange(4.0), dims="time", coords={"time": utc})
        cov = xr.DataArray(
            np.ones((4, 1)),
            dims=("time", "covariate"),
            coords={"time": utc.tz_localize(None), "covariate": ["x"]},
        )
        with pytest.raises(AlignmentError, match="timezone-aware"):
            validate_alignment(data, cov)

    def test_non_datetime_beside_aware_is_not_called_naive(self):
        utc = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        data = xr.DataArray(np.arange(3.0), dims="time", coords={"time": utc})
        integer = xr.DataArray(
            np.ones((4, 1)),
            dims=("time", "covariate"),
            coords={"time": np.arange(4), "covariate": ["x"]},
        )
        periods = xr.DataArray(
            np.ones((4, 1)),
            dims=("time", "covariate"),
            coords={"time": pd.period_range("2024-01-01", periods=4, freq="D"), "covariate": ["x"]},
        )
        with pytest.raises(AlignmentError, match="misaligned"):
            validate_alignment(data, integer)
        with pytest.raises(AlignmentError, match="misaligned"):
            validate_alignment(data, periods)

    def test_mixed_naive_and_aware_object_index_is_rejected(self):
        mixed = pd.Index([pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05", tz="UTC")])
        train = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        with pytest.raises(AlignmentError, match="mixes timezones"):
            concat_time_index(train, mixed)

    def test_future_in_another_zone_is_converted_to_the_training_zone(self):
        train = pd.date_range("2020-01-10 22:00", periods=3, freq="h", tz="Asia/Kolkata")
        future = pd.date_range("2020-01-11 02:00", periods=2, freq="h", tz="America/New_York")
        full = concat_time_index(train, future)
        assert str(full.tz) == "Asia/Kolkata"
        assert full[3:].equals(future.tz_convert("Asia/Kolkata"))

    @pytest.mark.parametrize(
        ("train_tz", "future_tz"),
        [
            ("UTC", dt.UTC),
            ("UTC", zoneinfo.ZoneInfo("UTC")),
            (dt.UTC, zoneinfo.ZoneInfo("UTC")),
            ("America/New_York", zoneinfo.ZoneInfo("America/New_York")),
            ("America/New_York", "dateutil/America/New_York"),
            (zoneinfo.ZoneInfo("America/New_York"), "dateutil/America/New_York"),
        ],
    )
    def test_same_zone_from_different_tz_libraries_is_accepted(self, train_tz, future_tz):
        def build(start, periods, tz):
            try:
                return pd.date_range(start, periods=periods, freq="D", tz=tz)
            except AttributeError as err:
                # pandas 3.0.6 + dateutil 4, and some 3.1.0 builds, raise
                # inside date_range before this package is called.
                if "dateutil" not in str(tz):
                    raise
                pytest.skip(f"pandas cannot localize {tz!r}: {err}")

        train = build("2024-01-01", 3, train_tz)
        future = build("2024-01-04", 2, future_tz)
        full = concat_time_index(train, future)
        assert full.tz == train.tz
        assert list(full) == list(train) + list(future)

    def test_data_and_covariates_in_different_zones_align_on_instants(self):
        utc = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        data = as_dataarray(pd.Series([1.0, 2.0, 3.0], index=utc))
        berlin = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC").tz_convert(
            "Europe/Berlin"
        )
        cov = as_dataarray(
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}, index=berlin), role="covariates"
        )
        validate_alignment(data, cov)
        shifted = as_dataarray(
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}, index=berlin + pd.Timedelta(hours=1)),
            role="covariates",
        )
        with pytest.raises(AlignmentError, match="misaligned"):
            validate_alignment(data, shifted)

    def test_horizon_stamps_covariates_in_the_data_zone(self):
        from pymc_forecast.model import Horizon

        utc = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        data = as_dataarray(pd.Series([1.0, 2.0, 3.0], index=utc))
        berlin = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC").tz_convert(
            "Europe/Berlin"
        )
        cov = as_dataarray(pd.DataFrame({"x": np.arange(5.0)}, index=berlin), role="covariates")
        h = Horizon.from_data(cov, data)
        assert h.time.equals(utc)
        assert h.time_future.equals(pd.date_range("2024-01-04", periods=2, freq="D", tz="UTC"))

    def test_mixed_zones_rejected(self):
        train = pd.date_range("2020-01-10 22:00", periods=3, freq="h", tz="Asia/Kolkata")
        mixed = pd.Index(
            [
                pd.Timestamp("2020-01-11 02:00", tz="Pacific/Chatham"),
                pd.Timestamp("2020-01-11 03:00", tz="Pacific/Auckland"),
            ]
        )
        with pytest.raises(AlignmentError, match="mixes timezones"):
            concat_time_index(train, mixed)

    def test_future_covariates_in_another_zone_use_the_training_zone(self):
        utc = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
        berlin = pd.date_range("2024-01-05", periods=2, freq="D", tz="UTC").tz_convert(
            "Europe/Berlin"
        )
        train = xr.DataArray(
            np.arange(4.0)[:, None],
            dims=("time", "covariate"),
            coords={"time": utc, "covariate": ["x"]},
        )
        fut = xr.DataArray(
            np.array([[10.0], [11.0]]),
            dims=("time", "covariate"),
            coords={"time": berlin, "covariate": ["x"]},
        )
        full = concat_covariates(train, fut)
        expected = utc.append(berlin.tz_convert("UTC"))
        assert isinstance(full.get_index("time"), pd.DatetimeIndex)
        assert full.get_index("time").equals(expected)
        np.testing.assert_array_equal(full.values[-2:, 0], [10.0, 11.0])


def test_concat_covariates_stamps_future_rows_in_the_training_zone():
    train = pd.date_range("2020-01-10 22:00", periods=3, freq="h", tz="Asia/Kolkata")
    future = pd.date_range("2020-01-11 02:00", periods=2, freq="h", tz="America/New_York")
    cov = pd.DataFrame({"x": [0.0, 1.0, 2.0]}, index=train)
    fut = pd.DataFrame({"x": [3.0, 4.0]}, index=future)
    out = concat_covariates(cov, fut)
    index = out.indexes["time"]
    assert str(index.tz) == "Asia/Kolkata"
    assert index[3:].equals(future.tz_convert("Asia/Kolkata"))
    np.testing.assert_array_equal(out.values.ravel(), [0.0, 1.0, 2.0, 3.0, 4.0])

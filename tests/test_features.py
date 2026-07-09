import numpy as np
import pytensor.tensor as pt
import pytest

from pymc_forecast.features import fourier_features, periodic_repeat


class TestFourierFeatures:
    def test_shape_and_labels(self):
        feats = fourier_features(10, period=7, num_terms=2)
        assert feats.dims == ("time", "fourier")
        assert feats.shape == (10, 4)
        assert list(feats["fourier"].values) == ["sin_1", "sin_2", "cos_1", "cos_2"]

    def test_periodicity(self):
        feats = fourier_features(14, period=7, num_terms=3).values
        np.testing.assert_allclose(feats[:7], feats[7:], atol=1e-12)

    def test_explicit_positions(self):
        feats = fourier_features(np.array([0.0, 3.5, 7.0]), period=7, num_terms=1)
        np.testing.assert_allclose(feats.values[0], feats.values[2], atol=1e-12)
        np.testing.assert_allclose(feats.values[1, 0], 0.0, atol=1e-12)  # sin(pi)

    def test_num_terms_validated(self):
        with pytest.raises(ValueError, match="num_terms"):
            fourier_features(10, period=7, num_terms=0)


class TestPeriodicRepeat:
    def test_numpy_tiling(self):
        pattern = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            periodic_repeat(pattern, 7), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
        )

    def test_axis(self):
        pattern = np.arange(6.0).reshape(3, 2)
        out = periodic_repeat(pattern, 5, axis=0)
        assert out.shape == (5, 2)
        np.testing.assert_array_equal(out[3], pattern[0])

    def test_tensor_with_explicit_period(self):
        pattern = pt.vector("p")
        out = periodic_repeat(pattern, 5, period=2)
        np.testing.assert_array_equal(
            out.eval({pattern: np.array([1.0, 2.0])}), [1.0, 2.0, 1.0, 2.0, 1.0]
        )

    def test_symbolic_length_needs_period(self):
        pattern = pt.vector("p")
        with pytest.raises(ValueError, match="pass period="):
            periodic_repeat(pattern, 5)

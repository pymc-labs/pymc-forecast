import pymc_forecast


def test_version() -> None:
    assert pymc_forecast.__version__


def test_make_mase_is_a_public_export() -> None:
    assert "make_mase" in pymc_forecast.__all__

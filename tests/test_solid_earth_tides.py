from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from dolphin import io

from disp_nisar.solid_earth_tides import (
    calculate_solid_earth_tides_correction,
    resample_to_target,
)

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize("orbit_direction", ["ascending", "descending"])
def test_calculate_solid_earth_tides_correction(orbit_direction):
    ifgram_filename = TEST_DATA_DIR / "20160716_20160809.unw.tif"
    los_east_file = TEST_DATA_DIR / "los_east.tif"
    los_north_file = TEST_DATA_DIR / "los_north.tif"
    reference_start_time = datetime(2016, 7, 16, 13, 27, 39, 698599)
    reference_stop_time = datetime(2016, 7, 16, 13, 27, 42, 145748)
    secondary_start_time = datetime(2016, 8, 9, 10, 45, 20, 562106)
    secondary_stop_time = datetime(2016, 8, 9, 10, 45, 23, 9255)

    solid_earth_t = calculate_solid_earth_tides_correction(
        ifgram_filename,
        reference_start_time,
        reference_stop_time,
        secondary_start_time,
        secondary_stop_time,
        los_east_file,
        los_north_file,
        orbit_direction=orbit_direction,
    )

    assert solid_earth_t.shape == io.get_raster_xysize(ifgram_filename)[::-1]
    assert np.nanmax(solid_earth_t) < 0.1


def test_resample_to_target_same_shape():
    """If array already matches target shape, return it unchanged."""
    arr = np.ones((10, 10), dtype=np.float64)
    result = resample_to_target(arr, (10, 10))
    assert result is arr


def test_resample_to_target_ndarray():
    """Test upsampling a plain ndarray."""
    arr = np.arange(100, dtype=np.float64).reshape(10, 10)
    result = resample_to_target(arr, (20, 20))
    assert result.shape == (20, 20)
    assert not isinstance(result, np.ma.MaskedArray)
    # Corner values should be close to originals
    np.testing.assert_allclose(result[0, 0], arr[0, 0], atol=1)
    np.testing.assert_allclose(result[-1, -1], arr[-1, -1], atol=1)


def test_resample_to_target_masked_array():
    """Test upsampling a MaskedArray preserves and resamples mask."""
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:, 5:] = True
    arr = np.ma.MaskedArray(data=data, mask=mask)

    result = resample_to_target(arr, (20, 20))
    assert result.shape == (20, 20)
    assert isinstance(result, np.ma.MaskedArray)
    # The bottom-right quadrant should have masked pixels
    assert result.mask[-1, -1]
    # The top-left corner should not be masked
    assert not result.mask[0, 0]

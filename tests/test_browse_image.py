import numpy as np
import pytest

from disp_nisar.browse_image import _resize_to_max_pixel_dim


def test_resize_to_max_pixel_dim():
    """Test that resize scales array and preserves NaNs."""
    arr = np.random.rand(100, 100).astype(np.float32)
    arr[10:20, 10:20] = np.nan

    result = _resize_to_max_pixel_dim(arr, max_dim_allowed=50)
    assert result.shape == (50, 50)
    # NaN values should be preserved in the resized output
    assert np.any(np.isnan(result))


def test_resize_invalid_max_dim():
    """Invalid max_dim should raise ValueError."""
    arr = np.ones((10, 10))
    with pytest.raises(ValueError):
        _resize_to_max_pixel_dim(arr, max_dim_allowed=0)

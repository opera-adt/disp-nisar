"""Tests for pure helpers in disp_nisar._baselines."""

import numpy as np

from disp_nisar._baselines import _get_grids


class TestGetGrids:
    def test_output_shape_matches_meshgrid(self):
        x = np.array([500000.0, 501000.0, 502000.0])  # UTM easting (3)
        y = np.array([4000000.0, 4001000.0])  # UTM northing (2)
        lon, lat = _get_grids(x, y, 32611)  # UTM zone 11N
        # meshgrid(x, y) -> shape (len(y), len(x))
        assert lon.shape == (2, 3)
        assert lat.shape == (2, 3)

    def test_utm_zone_11n_maps_to_expected_lonlat(self):
        # Use a small multi-point grid (pyproj warns on single-element arrays).
        x = np.array([500000.0, 501000.0])  # ~central meridian of zone 11N
        y = np.array([4000000.0, 4001000.0])
        lon, lat = _get_grids(x, y, 32611)
        # 500000 easting is the false-easting central meridian -> ~ -117 lon
        assert abs(lon[0, 0] - (-117.0)) < 0.5
        assert 30 < lat[0, 0] < 40

    def test_grid_varies_with_easting(self):
        x = np.array([500000.0, 510000.0])
        y = np.array([4000000.0])
        lon, _ = _get_grids(x, y, 32611)
        # Larger easting -> larger (less negative) longitude
        assert lon[0, 1] > lon[0, 0]

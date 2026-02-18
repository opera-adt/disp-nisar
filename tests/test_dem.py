import numpy as np
from dolphin import Bbox
from shapely.geometry import Polygon

from disp_nisar._dem import (
    check_dateline,
    margin_km_to_deg,
    margin_km_to_longitude_deg,
    polygon_from_bounding_box,
)


def test_margin_km_to_deg():
    # 111 km ~ 1 degree of latitude
    assert abs(margin_km_to_deg(111.0) - 1.0) < 0.01
    assert margin_km_to_deg(0) == 0.0


def test_margin_km_to_longitude_deg():
    # At equator (lat=0), same as latitude conversion
    assert abs(margin_km_to_longitude_deg(111.0, 0.0) - 1.0) < 0.01
    # At 60 degrees, longitude degrees should be ~2x latitude degrees
    result_60 = margin_km_to_longitude_deg(111.0, 60.0)
    assert abs(result_60 - 2.0) < 0.05


def test_polygon_from_bounding_box():
    bbox = Bbox(-120.0, 34.0, -118.0, 36.0)
    poly = polygon_from_bounding_box(bbox, margin_km=0)
    assert isinstance(poly, Polygon)
    bounds = poly.bounds
    np.testing.assert_allclose(bounds, (-120.0, 34.0, -118.0, 36.0), atol=1e-6)

    # With margin, the polygon should be larger
    poly_margin = polygon_from_bounding_box(bbox, margin_km=10)
    assert poly_margin.area > poly.area


def test_check_dateline_no_crossing():
    """Polygon not crossing dateline returns a single polygon."""
    poly = Polygon([(-120, 34), (-118, 34), (-118, 36), (-120, 36)])
    result = check_dateline(poly)
    assert len(result) == 1
    assert result[0].equals(poly)


def test_check_dateline_crossing():
    """Polygon crossing dateline returns two polygons."""
    poly = Polygon([(170, -10), (190, -10), (190, 10), (170, 10)])
    result = check_dateline(poly)
    assert len(result) == 2

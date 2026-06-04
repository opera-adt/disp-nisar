"""Tests for the pure geospatial helpers in disp_nisar._water."""

import pytest
from shapely.geometry import Polygon, box

from disp_nisar._water import (
    EARTH_APPROX_CIRCUMFERENCE,
    check_dateline,
    margin_km_to_deg,
    margin_km_to_longitude_deg,
    polygon_from_bounding_box,
)

# Kilometers spanning one degree of latitude at the equator.
KM_PER_DEG = EARTH_APPROX_CIRCUMFERENCE / 360.0 / 1000.0


class TestMarginKmToDeg:
    def test_zero(self):
        assert margin_km_to_deg(0) == 0.0

    def test_one_degree_worth(self):
        assert margin_km_to_deg(KM_PER_DEG) == pytest.approx(1.0)

    def test_linear(self):
        assert margin_km_to_deg(20) == pytest.approx(2 * margin_km_to_deg(10))


class TestMarginKmToLongitudeDeg:
    def test_equator_matches_latitude_degree(self):
        # At the equator a degree of longitude ~ a degree of latitude.
        assert margin_km_to_longitude_deg(100, lat=0) == pytest.approx(
            margin_km_to_deg(100), rel=1e-3
        )

    def test_default_latitude_is_equator(self):
        assert margin_km_to_longitude_deg(100) == margin_km_to_longitude_deg(100, lat=0)

    def test_higher_latitude_gives_larger_longitude_span(self):
        # Longitude degrees grow as 1/cos(lat); at 60 deg it doubles.
        ratio = margin_km_to_longitude_deg(100, lat=60) / margin_km_to_longitude_deg(
            100, lat=0
        )
        assert ratio == pytest.approx(2.0, rel=1e-3)


class TestCheckDateline:
    def test_non_crossing_returns_single_polygon(self):
        poly = box(10, 10, 20, 20)
        result = check_dateline(poly)
        assert len(result) == 1
        assert result[0].equals(poly)

    def test_crossing_returns_two_polygons(self):
        crossing = Polygon([(170, 10), (190, 10), (190, 20), (170, 20)])
        result = check_dateline(crossing)
        assert len(result) == 2

    def test_crossing_pieces_within_valid_longitude(self):
        crossing = Polygon([(170, 10), (190, 10), (190, 20), (170, 20)])
        for piece in check_dateline(crossing):
            xs, _ = piece.exterior.coords.xy
            assert all(-180.0 <= x <= 180.0 for x in xs)


class TestPolygonFromBoundingBox:
    def test_zero_margin_matches_bbox(self):
        poly = polygon_from_bounding_box((10, 10, 20, 20), margin_in_km=0)
        assert poly.bounds == pytest.approx((10.0, 10.0, 20.0, 20.0))

    def test_positive_margin_expands_bounds(self):
        bbox = (10, 10, 20, 20)
        poly = polygon_from_bounding_box(bbox, margin_in_km=10)
        west, south, east, north = poly.bounds
        assert west < 10 and south < 10
        assert east > 20 and north > 20

    def test_latitude_clamped_to_pole(self):
        poly = polygon_from_bounding_box((10, -89.9, 20, 89.9), margin_in_km=100)
        _, south, _, north = poly.bounds
        assert south == -90.0
        assert north == 90.0

    def test_returns_polygon(self):
        poly = polygon_from_bounding_box((0, 0, 1, 1), margin_in_km=5)
        assert isinstance(poly, Polygon)
        assert poly.area > 0

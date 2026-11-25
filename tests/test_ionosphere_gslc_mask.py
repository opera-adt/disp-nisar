import h5py
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from disp_nisar.ionosphere._main_diff_io import Grid
from disp_nisar.ionosphere.gslc_mask import (
    prepare_gslc_mask_cache,
    reduce_gslc_mask,
)

CRS_ = CRS.from_epsg(32611)
NATIVE = Grid(80, 120, CRS_, Affine(10, 0, 500000, 0, -10, 3800000))
TARGET = Grid(20, 30, CRS_, Affine(40, 0, 500000, 0, -40, 3800000))


@pytest.fixture
def codes():
    rng = np.random.default_rng(0)
    values = rng.choice([0, 1, 2, 255], size=(80, 120), p=[0.002, 0.6, 0.397, 0.001])
    values = values.astype(np.uint8)
    values[:12] = 255
    return values


def _blocks(codes):
    return codes.reshape(20, 4, 30, 4).transpose(0, 2, 1, 3)


def _reduce(codes, mode, tile_size=7):
    good, inside = np.zeros((2, 20, 30), bool)
    for win, g, i in reduce_gslc_mask(codes, NATIVE, TARGET, mode, tile_size):
        rows, cols = win.toslices()
        good[rows, cols], inside[rows, cols] = g, i
    return good, inside


def test_all_valid_matches_brute_force(codes):
    good, inside = _reduce(codes, "all_valid")
    blocks = _blocks(codes)
    np.testing.assert_array_equal(good, ((blocks > 0) & (blocks < 255)).all((2, 3)))
    np.testing.assert_array_equal(inside, (blocks != 255).any((2, 3)))


def test_nearest_reads_the_center_sample(codes):
    good, _ = _reduce(codes, "nearest")
    center = codes[2::4, 2::4]
    np.testing.assert_array_equal(good, (center > 0) & (center < 255))


def test_result_does_not_depend_on_tile_size(codes):
    for mode in ("all_valid", "nearest"):
        np.testing.assert_array_equal(
            _reduce(codes, mode, 5)[0], _reduce(codes, mode, 64)[0]
        )


def test_unknown_mode_is_rejected(codes):
    with pytest.raises(ValueError, match="Unknown reduction"):
        list(reduce_gslc_mask(codes, NATIVE, TARGET, "mean", 8))


def test_mask_codes_must_be_uint8_values():
    bad = np.full((80, 120), 300, dtype=np.int32)
    dataset = _Named(bad)
    with pytest.raises(ValueError, match="Expected uint8 GSLC subswath codes"):
        list(reduce_gslc_mask(dataset, NATIVE, TARGET, "all_valid", 8))


class _Named(np.ndarray):
    """An array carrying the `.name` an HDF5 dataset would have."""

    name = "mask"

    def __new__(cls, array):
        return np.asarray(array).view(cls)


def _write_gslc(path, codes):
    with h5py.File(path, "w") as f:
        group = f.create_group("science/LSAR/GSLC/grids/frequencyA")
        group["mask"] = codes
        group["xCoordinates"] = 500000 + 5.0 + 10.0 * np.arange(120)
        group["yCoordinates"] = 3800000 - 5.0 - 10.0 * np.arange(80)
        group["xCoordinateSpacing"] = 10.0
        group["yCoordinateSpacing"] = -10.0
        group["projection"] = 32611


def test_prepare_cache_writes_reduced_masks_and_reuses_them(tmp_path, codes):
    source = tmp_path / "NISAR_L2_GSLC_20260101.h5"
    _write_gslc(source, codes)
    cache = tmp_path / "masks"

    outputs = prepare_gslc_mask_cache(source, "A", TARGET, cache)

    assert outputs["valid"].name == "20260101_A_valid.tif"
    blocks = _blocks(codes)
    with rasterio.open(outputs["valid"]) as ds:
        assert ds.transform == TARGET.transform
        np.testing.assert_array_equal(
            ds.read(1).astype(bool), ((blocks > 0) & (blocks < 255)).all((2, 3))
        )

    before = outputs["valid"].stat().st_mtime_ns
    prepare_gslc_mask_cache(source, "A", TARGET, cache)
    assert outputs["valid"].stat().st_mtime_ns == before


def test_prepare_cache_rejects_a_bad_band(tmp_path):
    with pytest.raises(ValueError, match="band must be A or B"):
        prepare_gslc_mask_cache(tmp_path / "x.h5", "C", TARGET, tmp_path)

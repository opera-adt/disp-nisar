import zipfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from dolphin import io

from disp_nisar._masking import create_mask_from_distance


@pytest.fixture
def mask_file_zipped():
    return Path(__file__).parent / "data/water_distance.tif.zip"


def test_bounds(tmp_path, mask_file_zipped):
    # Unzip to tmp_path
    with zipfile.ZipFile(mask_file_zipped, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    # Get the path of the extracted TIF file
    input_tif = tmp_path / "water_distance.tif"
    outpath = tmp_path / "mask.tif"
    create_mask_from_distance(
        input_tif, output_file=outpath, land_buffer=0, ocean_buffer=0
    )
    expected0 = io.load_gdal(Path(__file__).parent / "data/expected_mask0.tif")
    npt.assert_array_equal(expected0, io.load_gdal(outpath))

    outpath = tmp_path / "mask2.tif"
    create_mask_from_distance(
        input_tif, output_file=outpath, land_buffer=2, ocean_buffer=2
    )
    expected2 = io.load_gdal(Path(__file__).parent / "data/expected_mask2.tif")
    npt.assert_array_equal(expected2, io.load_gdal(outpath))


def _write_gslc_with_mask(path, mask, epsg=32611, x0=500000.0, y0=4000000.0, dx=10.0):
    """Write a minimal GSLC-like file with a `mask` dataset."""
    import h5py

    ny, nx = mask.shape
    grp_path = "science/LSAR/GSLC/grids/frequencyA"
    with h5py.File(path, "w") as hf:
        g = hf.create_group(grp_path)
        g["mask"] = mask.astype("uint8")
        g["xCoordinates"] = x0 + dx * np.arange(nx)
        g["yCoordinates"] = y0 - dx * np.arange(ny)
        g["projection"] = np.int32(epsg)
    return path


def test_combine_gslc_masks_is_intersection(tmp_path):
    """A pixel is valid only if it is valid in every input (0 and 255 are invalid)."""
    from disp_nisar._masking import combine_gslc_masks

    a = np.ones((6, 8), dtype="uint8")
    a[0, :] = 255  # fill outside the radar extent
    b = np.ones((6, 8), dtype="uint8") * 2  # subswath 2 is valid
    b[:, 0] = 0  # invalid focus
    c = np.ones((6, 8), dtype="uint8")
    c[3, 3] = 0
    files = [
        _write_gslc_with_mask(tmp_path / f"{n}.h5", m)
        for n, m in zip("abc", (a, b, c))
    ]

    combined = combine_gslc_masks(files)

    expected = np.ones((6, 8), dtype="uint8")
    expected[0, :] = 0
    expected[:, 0] = 0
    expected[3, 3] = 0
    np.testing.assert_array_equal(combined.values, expected)
    assert combined.rio.crs.to_epsg() == 32611


def test_combine_gslc_masks_strides_and_skips_unreadable(tmp_path):
    from disp_nisar._masking import combine_gslc_masks

    m = np.ones((8, 8), dtype="uint8")
    good = _write_gslc_with_mask(tmp_path / "good.h5", m)
    no_mask = tmp_path / "nomask.h5"
    import h5py

    with h5py.File(no_mask, "w") as hf:
        hf.create_group("science/LSAR/GSLC/grids/frequencyA")

    combined = combine_gslc_masks([good, no_mask], strides=(2, 4))
    assert combined.shape == (4, 2)

    with pytest.raises(ValueError, match="No readable"):
        combine_gslc_masks([no_mask])


def test_combine_gslc_masks_different_grid(tmp_path):
    """An input on a shifted grid is resampled; uncovered pixels do not mask."""
    from disp_nisar._masking import combine_gslc_masks

    full = np.ones((6, 8), dtype="uint8")
    # Smaller footprint, shifted by 2 pixels in x, with an invalid pixel
    small = np.ones((6, 4), dtype="uint8")
    small[2, 1] = 0
    a = _write_gslc_with_mask(tmp_path / "a.h5", full)
    b = _write_gslc_with_mask(tmp_path / "b.h5", small, x0=500020.0)

    combined = combine_gslc_masks([a, b]).values

    assert combined.shape == (6, 8)
    assert combined[2, 3] == 0  # the invalid pixel of b (x index 1 + 2)
    assert combined.sum() == 6 * 8 - 1

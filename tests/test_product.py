from pathlib import Path

import h5py
import pytest

from disp_nisar import product
from disp_nisar.pge_runconfig import RunConfig

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_GSLC_FILE = (
    TEST_DATA_DIR
    / "NISAR_L2_GSLC_NI_F150_20060630T061920Z_20240504T181714Z_NI_HH_v0.1.h5"  # noqa: E501
)

TEST_OUTPUT_CGSLC_FILE = TEST_DATA_DIR / "compressed_20060630_20230101_20230113.tif"


# Shapely runtime warning
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_output_product(
    tmp_path,
    test_data_dir,
    scratch_dir,
):
    # Get the test run scratch directory
    # $ ls scratch/forward/
    # interferograms  log_sas.log  t042_088905_iw1  t042_088906_iw1  unwrapped
    # $ ls scratch/forward/unwrapped/
    # 20221119_20221213.unw.conncomp  20221119_20221213.unw.conncomp.hdr  ...
    unw_dir = scratch_dir / "forward/unwrapped"
    unw_filename = unw_dir / "20221119_20221213.unw.tif"
    conncomp_filename = unw_dir / "20221119_20221213.unw.conncomp"

    ifg_dir = scratch_dir / "forward/interferograms/stitched"
    tcorr_filename = ifg_dir / "tcorr.tif"
    ps_mask_filename = ifg_dir / "ps_mask_looked.tif"
    ifg_corr_filename = ifg_dir / "20221119_20221213.cor"

    gslc_files = sorted(
        (test_data_dir / "delivery_data_small/input_slcs").glob("t*.h5")
    )
    assert len(gslc_files) > 0

    output_name = tmp_path / "20221119_20221213.nc"

    pge_runconfig = RunConfig.from_yaml(
        test_data_dir / "delivery_data_small/config_files/runconfig_forward.yaml"
    )

    product.create_output_product(
        unw_filename=unw_filename,
        conncomp_filename=conncomp_filename,
        tcorr_filename=tcorr_filename,
        ifg_corr_filename=ifg_corr_filename,
        output_name=output_name,
        corrections={},
        ps_mask_filename=ps_mask_filename,
        pge_runconfig=pge_runconfig,
        cslc_files=gslc_files,
    )


def test_create_compressed_slc(tmp_path):
    # NISAR_L2_GSLC_NI_F150_20221228T161651Z_20240504T181714Z_NI_HH_v0.1.h5
    # compressed_20221228_20230101_20230113.tif
    date_str = "20060630_20230101_20230113"
    processed_cgslc_file = TEST_OUTPUT_CGSLC_FILE
    comp_slc_dict = {"": [processed_cgslc_file]}

    product.create_compressed_products(
        comp_slc_dict, cslc_file_list=[TEST_GSLC_FILE], output_dir=tmp_path
    )

    expected_name = tmp_path / product.COMPRESSED_SLC_TEMPLATE.format(date_str=date_str)
    assert expected_name.exists()
    # Check product structure
    with h5py.File(expected_name) as hf:
        assert hf["/science/LSAR/GSLC/grids/frequencyA/HH"].size > 0
        assert "/science/LSAR/GSLC/metadata/orbit" in hf
        assert "/science/LSAR/identification/zeroDopplerStartTime" in hf
        assert (
            "/science/LSAR/GSLC/metadata/sourceData/swaths/frequencyA/centerFrequency"
            in hf
        )
    # assert "/metadata/processing_information/input_burst_metadata/wavelength" in hf


def test_compressed_slc_has_gslc_layout(tmp_path):
    """Compressed SLCs are readable by the same code that reads real GSLCs."""
    import numpy as np
    from dolphin import io

    from disp_nisar._common import NISAR_DATASET_NAME
    from disp_nisar._ps import run_combine
    from disp_nisar._utils import _frequency_to_wavelength, get_nisar_frame_bbox

    [out] = product.create_compressed_products(
        {"": [TEST_OUTPUT_CGSLC_FILE]},
        cslc_file_list=[TEST_GSLC_FILE],
        output_dir=tmp_path,
    )
    grid = "/science/LSAR/GSLC/grids/frequencyA"
    with h5py.File(out) as hf:
        for name in (
            "HH",
            "amplitude_dispersion",
            "xCoordinates",
            "yCoordinates",
            "xCoordinateSpacing",
            "yCoordinateSpacing",
            "projection",
            "centerFrequency",
        ):
            assert f"{grid}/{name}" in hf, name
        assert (
            int(hf[f"{grid}/projection"][()])
            == hf[f"{grid}/projection"].attrs["epsg_code"]
        )
        # The old COMPASS-style names are gone
        assert "x_coordinates" not in hf[grid]
        assert "data" not in hf

    # Geometry matches the processed raster it was made from
    epsg, bbox = get_nisar_frame_bbox(out)
    assert epsg == io.get_raster_crs(TEST_OUTPUT_CGSLC_FILE).to_epsg()
    left, bottom, right, top = io.get_raster_bounds(TEST_OUTPUT_CGSLC_FILE)
    np.testing.assert_allclose(
        [bbox.left, bbox.bottom, bbox.right, bbox.top],
        [left, bottom, right, top],
    )
    # Wavelength is read from the GSLC-location centerFrequency, copied from the GSLC
    assert _frequency_to_wavelength("frequencyA", out) == _frequency_to_wavelength(
        "frequencyA", TEST_GSLC_FILE
    )

    # And the dispersion/SLC are found where `run_combine` looks for them
    shape = io.get_raster_xysize(TEST_OUTPUT_CGSLC_FILE)[::-1]
    mean = tmp_path / "mean.tif"
    disp = tmp_path / "disp.tif"
    for f in (mean, disp):
        io.write_arr(
            arr=np.ones(shape, dtype="float32"),
            output_name=f,
            like_filename=TEST_OUTPUT_CGSLC_FILE,
        )
    out_disp, out_mean = run_combine(
        mean, disp, [out], num_slc=2, subdataset=NISAR_DATASET_NAME
    )
    assert out_disp.exists() and out_mean.exists()


def test_nodata_mask_keeps_exact_zero_pixels():
    """The reference pixel is exactly 0.0 and must not be treated as nodata."""
    import numpy as np

    arr = np.ma.masked_array(
        [[0.0, 1.0], [2.0, 3.0]], mask=[[False, False], [True, False]]
    )
    mask = product.nodata_mask(arr)
    assert mask.tolist() == [[False, False], [True, False]]
    # No mask at all -> nothing is nodata, even where the values are zero
    assert not product.nodata_mask(np.ma.masked_array(np.zeros((2, 2)))).any()


def test_compressed_slc_stores_combined_mask(tmp_path):
    """The compressed SLC stores the combined input mask as a GSLC-style `mask`."""
    import numpy as np
    from dolphin import io

    from disp_nisar._masking import combine_gslc_masks, read_gslc_validity_mask

    xsize, ysize = io.get_raster_xysize(TEST_OUTPUT_CGSLC_FILE)
    gt = io.get_raster_gt(TEST_OUTPUT_CGSLC_FILE)
    epsg = io.get_raster_crs(TEST_OUTPUT_CGSLC_FILE).to_epsg()

    def make_input(name, mask):
        path = tmp_path / name
        with h5py.File(path, "w") as hf:
            g = hf.create_group("science/LSAR/GSLC/grids/frequencyA")
            g["mask"] = mask
            g["xCoordinates"] = gt[0] + gt[1] * (np.arange(xsize) + 0.5)
            g["yCoordinates"] = gt[3] + gt[5] * (np.arange(ysize) + 0.5)
            g["projection"] = np.int32(epsg)
        return path

    m1 = np.ones((ysize, xsize), dtype="uint8")
    m1[:5, :] = 0
    m2 = np.ones((ysize, xsize), dtype="uint8")
    m2[:, :7] = 255
    # The first input has the date of the compressed SLC, so its metadata is copied
    in1 = make_input("NISAR_L2_GSLC_NI_F150_20060630T061920Z_20240504T181714Z.h5", m1)
    in2 = make_input("NISAR_L2_GSLC_NI_F150_20060712T061920Z_20240504T181714Z.h5", m2)

    [out] = product.create_compressed_products(
        {"": [TEST_OUTPUT_CGSLC_FILE]},
        cslc_file_list=[in1, in2],
        output_dir=tmp_path,
    )
    # Not left behind in the output directory
    assert not list(tmp_path.glob(".mask_*"))

    with h5py.File(out) as hf:
        stored = hf["/science/LSAR/GSLC/grids/frequencyA/mask"][:]
    expected = np.ones((ysize, xsize), dtype="uint8")
    expected[:5, :] = 0
    expected[:, :7] = 0
    np.testing.assert_array_equal(stored >= 1, expected.astype(bool))
    assert stored.dtype == np.uint8

    # Read back like a GSLC mask, and combine it with a new acquisition in the next run
    np.testing.assert_array_equal(read_gslc_validity_mask(out).values, expected)
    m3 = np.ones((ysize, xsize), dtype="uint8")
    m3[-1, :] = 0
    in3 = make_input("NISAR_L2_GSLC_NI_F150_20060724T061920Z_20240504T181714Z.h5", m3)
    nxt = combine_gslc_masks([in3, out]).values
    expected_next = expected.copy()
    expected_next[-1, :] = 0
    np.testing.assert_array_equal(nxt, expected_next)

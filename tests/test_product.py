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

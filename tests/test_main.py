from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dolphin import io

from disp_nisar import main
from disp_nisar.cli.run import run_main
from disp_nisar.ionosphere.options import IonosphereOptions

TEST_DATA_DIR = Path(__file__).parent / "data/delivery_data_small"


def test_run_main():
    config_file = TEST_DATA_DIR / "config_files/runconfig_historical.yaml"
    if not config_file.exists():
        pytest.skip(f"Test data does not exist at {TEST_DATA_DIR}")

    with chdir(TEST_DATA_DIR):
        run_main(config_file="config_files/runconfig_historical.yaml")


def _main_diff_inputs(
    method="main_diff",
    product_type="DISP_NISAR_HISTORICAL",
    gunw_files=None,
    iono_parameters=Path("algorithm_parameters_freqB.yaml"),
    frequency="frequencyA",
    gslc_name="NISAR_L2_GSLC_20260101.h5",
):
    pge_runconfig = SimpleNamespace(
        get_ionosphere_options=lambda: IonosphereOptions(method=method),
        dynamic_ancillary_file_group=SimpleNamespace(
            gunw_files=gunw_files,
            ionosphere_algorithm_parameters_file=iono_parameters,
        ),
        input_file_group=SimpleNamespace(frequency=frequency),
        primary_executable=SimpleNamespace(product_type=product_type),
    )
    return SimpleNamespace(cslc_file_list=[Path(gslc_name)]), pge_runconfig


@pytest.mark.parametrize(
    "kwargs",
    [
        # Each of these ran the split-spectrum workflow before main_diff was added
        {"method": "main_side"},
        {"method": "main_side", "product_type": "DISP_NISAR_FORWARD"},
        {"method": "main_side", "iono_parameters": None},
        # GUNW inputs always win, whatever the method
        {"gunw_files": [Path("gunw.h5")]},
    ],
)
def test_main_diff_is_opt_in(kwargs):
    cfg, pge_runconfig = _main_diff_inputs(**kwargs)
    assert main._get_main_diff_options(cfg, pge_runconfig) is None


def test_default_method_keeps_the_split_spectrum_workflow():
    assert IonosphereOptions().method == "main_side"


def test_main_diff_selected():
    cfg, pge_runconfig = _main_diff_inputs()
    assert main._get_main_diff_options(cfg, pge_runconfig).method == "main_diff"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frequency": "frequencyB"}, "nominal frequencyA"),
        ({"product_type": "DISP_NISAR_FORWARD"}, "historical GSLC stacks only"),
        ({"gslc_name": "compressed_20260101.h5"}, "historical GSLC stacks only"),
        ({"iono_parameters": None}, "explicit frequencyB algorithm file"),
    ],
)
def test_main_diff_rejects_unsupported_runs(kwargs, message):
    cfg, pge_runconfig = _main_diff_inputs(**kwargs)
    with pytest.raises(ValueError, match=message):
        main._get_main_diff_options(cfg, pge_runconfig)


def test_nodata_mask_keeps_valid_zeros(tmp_path):
    """The reference pixel of a displacement raster is a valid zero."""
    image = tmp_path / "20260101_20260113.tif"
    data = np.array([[0.0, 1.0], [np.nan, -2.0]], dtype=np.float32)
    io.write_arr(arr=data, output_name=image, nodata=np.nan)

    output = tmp_path / "mask.tif"
    main._create_nodata_mask(image, output)

    np.testing.assert_array_equal(io.load_gdal(output), [[1, 1], [0, 1]])


def _write_iono(path, units, metadata=None):
    io.write_arr(arr=np.ones((2, 2), dtype=np.float32), output_name=path, units=units)
    if metadata:
        io.set_raster_metadata(path, metadata)
    return path


def test_correction_in_meters_is_not_scaled_again(tmp_path):
    """The split-spectrum workflow writes meters, which were scaled by lambda/4pi.

    Regression: that left corrections about 52 times too small at L-band.
    """
    iono = np.array([[0.5]], dtype=np.float32)
    path = _write_iono(tmp_path / "iono.tif", "meters")
    np.testing.assert_array_equal(main._ionosphere_to_meters(path, iono, 0.24), iono)


def test_correction_in_radians_is_converted(tmp_path):
    iono = np.array([[4 * np.pi]], dtype=np.float32)
    path = _write_iono(tmp_path / "iono.tif", "radians")
    np.testing.assert_allclose(main._ionosphere_to_meters(path, iono, 0.24), 0.24)


def test_main_diff_correction_must_declare_meters(tmp_path):
    path = _write_iono(
        tmp_path / "iono.tif", "radians", {"ionosphere_method": "main_diff"}
    )
    with pytest.raises(ValueError, match="must declare meters"):
        main._ionosphere_to_meters(path, np.ones((1, 1)), 0.24)


class _Workflow(SimpleNamespace):
    """The parts of a frequency-B DisplacementWorkflow the main_diff path sets."""

    def __init__(self, method="whirlwind"):
        super().__init__(
            output_options=SimpleNamespace(bounds=None, bounds_epsg=None, epsg=None),
            unwrap_options=SimpleNamespace(unwrap_method=method, run_unwrap=True),
            timeseries_options=SimpleNamespace(run_inversion=True, run_velocity=True),
            mask_file="raw_water_distance.tif",
            work_directory=Path("scratch_freqB"),
        )


@pytest.fixture
def main_diff_run(monkeypatch):
    """Stub the expensive stages, recording what the workflow hands them."""
    seen = SimpleNamespace(cfg_b=_Workflow(), estimation=None, displacement=None)
    outputs_b = SimpleNamespace(
        stitched_ifg_paths=["b.int.tif"],
        stitched_cor_paths=["b.cor.tif"],
        stitched_similarity_file="b_sim.tif",
    )

    def run_displacement(cfg, debug, raise_on_empty):
        seen.displacement = (cfg, debug, raise_on_empty)
        return outputs_b

    def estimation(**kwargs):
        seen.estimation = kwargs
        return ["correction.tif"]

    monkeypatch.setattr(main, "run_displacement", run_displacement)
    monkeypatch.setattr(main, "_prune_block_dirs", lambda _path: 0)
    monkeypatch.setattr(
        main, "read_reference_point", lambda _path: SimpleNamespace(row=7, col=9)
    )
    monkeypatch.setattr(
        "disp_nisar.ionosphere.gslc.get_center_frequencies",
        lambda _path: (1.2395e9, 1.2935e9),
    )
    monkeypatch.setattr(
        "disp_nisar.ionosphere.main_diff.run_main_diff_estimation", estimation
    )
    return seen


def _run_workflow(seen, gslcs=("g_20260101.h5", "g_20260113.h5"), products=None):
    cfg = SimpleNamespace(
        cslc_file_list=[Path(g) for g in gslcs],
        output_options=SimpleNamespace(
            bounds=(1, 2, 3, 4), bounds_epsg=4326, epsg=32611
        ),
        mask_file="prepared_mask.tif",
        work_directory=Path("scratch"),
    )
    pge_runconfig = SimpleNamespace(to_workflow=lambda **_kwargs: seen.cfg_b)
    full_outputs = SimpleNamespace(
        stitched_ifg_paths=["a.int.tif"],
        stitched_cor_paths=["a.cor.tif"],
        stitched_similarity_file="a_sim.tif",
    )
    return main._run_main_diff_workflow(
        cfg,
        pge_runconfig,
        full_outputs,
        [Path("out/20260101_20260113.tif")] if products is None else products,
        IonosphereOptions(method="main_diff"),
        False,
    )


def test_main_diff_workflow_stops_frequency_b_at_stitched_ifgs(main_diff_run):
    assert _run_workflow(main_diff_run) == ["correction.tif"]

    cfg_b = main_diff_run.cfg_b
    assert main_diff_run.displacement == (cfg_b, False, False)
    assert cfg_b.unwrap_options.run_unwrap is False
    assert cfg_b.timeseries_options.run_inversion is False
    assert cfg_b.timeseries_options.run_velocity is False
    # The prepared mask, since the raw ancillary is a water distance map
    assert cfg_b.mask_file == "prepared_mask.tif"
    assert cfg_b.output_options.bounds == (1, 2, 3, 4)
    assert cfg_b.output_options.epsg == 32611


def test_main_diff_workflow_passes_both_bands_and_the_reference(main_diff_run):
    _run_workflow(main_diff_run)

    kwargs = main_diff_run.estimation
    assert kwargs["band_a"].interferograms == ["a.int.tif"]
    assert kwargs["band_b"].interferograms == ["b.int.tif"]
    assert kwargs["band_a"].similarity == "a_sim.tif"
    assert kwargs["reference_a"] == (7, 9)
    assert (kwargs["f_A"], kwargs["f_B"]) == (1.2395e9, 1.2935e9)
    assert kwargs["out_dir"] == Path("scratch/ionosphere_main_diff")
    assert [d.strftime("%Y%m%d") for d in kwargs["gslc_by_date"]] == [
        "20260101",
        "20260113",
    ]


def test_main_diff_workflow_without_products_does_nothing(main_diff_run):
    assert _run_workflow(main_diff_run, products=[]) == []
    assert main_diff_run.displacement is None


def test_main_diff_workflow_rejects_other_unwrappers(main_diff_run):
    main_diff_run.cfg_b = _Workflow(method="spurt")
    with pytest.raises(ValueError, match="requires snaphu or whirlwind"):
        _run_workflow(main_diff_run)


def test_main_diff_workflow_rejects_duplicate_acquisitions(main_diff_run):
    with pytest.raises(ValueError, match="Duplicate GSLC acquisition"):
        _run_workflow(main_diff_run, gslcs=("g_20260101.h5", "h_20260101.h5"))


def test_main_diff_workflow_rejects_changing_frequencies(main_diff_run, monkeypatch):
    frequencies = iter([(1.2395e9, 1.2935e9), (1.2395e9, 1.2945e9)])
    monkeypatch.setattr(
        "disp_nisar.ionosphere.gslc.get_center_frequencies",
        lambda _path: next(frequencies),
    )
    with pytest.raises(ValueError, match="change within the GSLC stack"):
        _run_workflow(main_diff_run)

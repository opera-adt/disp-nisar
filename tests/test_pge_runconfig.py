import datetime
import random
import sys
import warnings
from pathlib import Path

import pytest
from dolphin.stack import CompressedSlcPlan

from disp_nisar import pge_runconfig
from disp_nisar.pge_runconfig import (
    AlgorithmParameters,
    DynamicAncillaryFileGroup,
    InputFileGroup,
    PrimaryExecutable,
    ProductPathGroup,
    RunConfig,
    StaticAncillaryFileGroup,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*utcnow.*:DeprecationWarning",
)


def test_algorithm_parameters_schema():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        AlgorithmParameters.print_yaml_schema()


def test_run_config_schema():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        RunConfig.print_yaml_schema()


@pytest.fixture
def input_file_group(slc_file_list_nc_with_sds):
    file_list, subdataset = slc_file_list_nc_with_sds
    return InputFileGroup(gslc_file_list=file_list, frame_id=11114)


@pytest.fixture
def algorithm_parameters_file(tmp_path):
    f = tmp_path / "test.yaml"
    AlgorithmParameters().to_yaml(f)
    return f


@pytest.fixture
def dynamic_ancillary_file_group(algorithm_parameters_file, sample_gunw_list):
    return DynamicAncillaryFileGroup(
        algorithm_parameters_file=algorithm_parameters_file,
        gunw_files=sample_gunw_list,
    )


@pytest.fixture
def frame_to_bounds_json_file():
    return Path(__file__).parent / "data/Frame_to_bounds_DISP-NI_v0.1.json"


@pytest.fixture
def static_ancillary_file_group(frame_to_bounds_json_file, algorithm_parameters_file):
    return StaticAncillaryFileGroup(
        frame_to_bounds_json=frame_to_bounds_json_file,
        algorithm_parameters_file=algorithm_parameters_file,
    )


@pytest.fixture
def reference_date_json_file():
    return Path(__file__).parent / "data/reference-dates.json"


@pytest.fixture
def product_path_group(tmp_path):
    product_path = tmp_path / "product_path"
    product_path.mkdir()
    return ProductPathGroup(product_path=product_path)


@pytest.fixture
def runconfig_minimum(
    input_file_group,
    dynamic_ancillary_file_group,
    static_ancillary_file_group,
    product_path_group,
):
    c = RunConfig(
        input_file_group=input_file_group,
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    return c


def test_algorithm_parameters_defaults():
    """Test that AlgorithmParameters has the expected default values."""
    from dolphin.workflows import (
        InterferogramNetwork,
        OutputOptions,
        PhaseLinkingOptions,
        PsOptions,
        TimeseriesOptions,
        UnwrapOptions,
    )

    params = AlgorithmParameters()

    # Check direct attributes
    assert params.algorithm_parameters_overrides_json is None
    assert params.subdataset == "/science/LSAR/GSLC/grids/frequencyA/HH"
    assert params.recommended_temporal_coherence_threshold == 0.6
    assert params.recommended_similarity_threshold == 0.5
    assert params.spatial_wavelength_cutoff == 25_000
    assert params.browse_image_vmin_vmax == (-0.10, 0.10)
    assert params.num_parallel_products == 3

    # Check that nested objects are created with their default factories
    assert isinstance(params.ps_options, PsOptions)
    assert isinstance(params.phase_linking, PhaseLinkingOptions)
    assert isinstance(params.interferogram_network, InterferogramNetwork)
    assert isinstance(params.unwrap_options, UnwrapOptions)
    assert isinstance(params.timeseries_options, TimeseriesOptions)
    assert isinstance(params.output_options, OutputOptions)


def test_runconfig_to_yaml(runconfig_minimum):
    print(runconfig_minimum.to_yaml(sys.stdout))


def test_runconfig_to_workflow(runconfig_minimum):
    print(runconfig_minimum.to_workflow())


def test_runconfig_from_workflow(
    tmp_path, frame_to_bounds_json_file, reference_date_json_file, runconfig_minimum
):
    w1 = runconfig_minimum.to_workflow()
    frame_id = runconfig_minimum.input_file_group.frame_id
    frequency = runconfig_minimum.input_file_group.frequency
    polarization = runconfig_minimum.input_file_group.polarization
    algo_file = tmp_path / "algo_params.yaml"
    proc_mode = "forward"
    w2 = RunConfig.from_workflow(
        w1,
        frame_id=frame_id,
        frequency=frequency,
        polarization=polarization,
        frame_to_bounds_json=frame_to_bounds_json_file,
        reference_date_json=reference_date_json_file,
        processing_mode=proc_mode,
        algorithm_parameters_file=algo_file,
    ).to_workflow()

    # these will be slightly different
    w2.creation_time_utc = w1.creation_time_utc
    assert w1 == w2


def test_runconfig_yaml_roundtrip(tmp_path, runconfig_minimum):
    f = tmp_path / "test.yaml"
    runconfig_minimum.to_yaml(f)
    c = RunConfig.from_yaml(f)
    assert c == runconfig_minimum


@pytest.fixture
def sample_slc_list():
    # "23210": [
    #   "2016-07-08T16:15:44",
    #   "2017-07-09T16:15:07",
    #   "2018-07-16T16:15:14",
    #   "2019-07-11T16:15:20",
    #   "2020-07-17T16:15:27",
    #   "2021-07-12T16:15:32",
    #   "2022-07-13T16:16:21",
    #   "2023-07-08T16:16:25",
    #   "2024-07-14T16:16:24"
    # ],
    dates = "20170610T000000Z_20170610T000000Z_20171001T000000Z_20240429T000000Z"
    return [
        Path(__file__).parent
        / f"data/COMPRESSED_NISAR_L2_GSLC_NI_F150_{dates}_NI_HH_v0.1.h5",
        # This is the compressed SLC where the "base phase" date within a later
        # reference date. So we expected output index of 1:
        "COMPRESSED_NISAR_L2_GSLC_NI_F150_20170709T000000Z_20170709T000000Z_20171201T000000Z_20240429T000000Z_NI_HH_v0.1.h5",
        "COMPRESSED_NISAR_L2_GSLC_NI_F150_20170709T000000Z_20171210T000000Z_20180604T000000Z_20240429T000000Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180610T161531Z_20240429T233903Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180622T161532Z_20240430T025857Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180628T161614Z_20240430T043443Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180710T161615Z_20240428T043529Z_NI_HH_v0.1.h5",
        # This one should become the "extra reference":
        "NISAR_L2_GSLC_NI_F150_20180716T161534Z_20240428T062045Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180722T161616Z_20240428T075037Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180728T161534Z_20240428T093746Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180803T161616Z_20240428T110636Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180809T161535Z_20240428T125546Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180815T161617Z_20240428T143339Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20180827T161618Z_20240428T175537Z_NI_HH_v0.1.h5",
    ]


@pytest.fixture
def sample_gunw_list():
    # "23210": [
    #   "2016-07-08T16:15:44",
    #   "2017-07-09T16:15:07",
    #   "2018-07-16T16:15:14",
    #   "2019-07-11T16:15:20",
    #   "2020-07-17T16:15:27",
    #   "2021-07-12T16:15:32",
    #   "2022-07-13T16:16:21",
    #   "2023-07-08T16:16:25",
    #   "2024-07-14T16:16:24"
    # ],
    return [
        "NISAR_L2_PR_GUNW_F150_HH_20170610T000000Z_20170610T000000Z_20171001T000000Z_20240429T000000Z_NI_HH_v0.1.h5",
        # This is the compressed SLC where the "base phase" date within a later
        # reference date. So we expected output index of 1:
        "NISAR_L2_PR_GUNW_F150_HH_20170709T000000Z_20170709T000000Z_20171201T000000Z_20240429T000000Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20170709T000000Z_20171210T000000Z_20180604T000000Z_20240429T000000Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180610T161531Z_20240429T233903Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180622T161532Z_20240430T025857Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180628T161614Z_20240430T043443Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180710T161615Z_20240428T043529Z_NI_HH_v0.1.h5",
        # This one should become the "extra reference":
        "NISAR_L2_PR_GUNW_F150_HH_20180716T161534Z_20240428T062045Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180722T161616Z_20240428T075037Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180728T161534Z_20240428T093746Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180803T161616Z_20240428T110636Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180809T161535Z_20240428T125546Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180815T161617Z_20240428T143339Z_NI_HH_v0.1.h5",
        "NISAR_L2_PR_GUNW_F150_HH_20180827T161618Z_20240428T175537Z_NI_HH_v0.1.h5",
    ]


def test_reference_changeover(
    dynamic_ancillary_file_group,
    static_ancillary_file_group,
    product_path_group,
    sample_slc_list,
):
    rc = RunConfig(
        input_file_group=InputFileGroup(gslc_file_list=sample_slc_list, frame_id=23210),
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    cfg = rc.to_workflow()
    # TODO: needs to be set in reference change json
    # assert cfg.output_options.extra_reference_date == datetime.datetime(2018, 7, 16)
    # assert cfg.phase_linking.output_reference_idx == 1
    assert cfg.output_options.extra_reference_date is None
    assert cfg.phase_linking.output_reference_idx == 0

    # Check a that non-exact match to the reference still works,
    # AFTER the reference request
    sample_slc_list[7] = (
        "NISAR_L2_GSLC_NI_F150_20180717T161534Z_20240428T062045Z_NI_HH_v0.1.h5"
    )
    rc = RunConfig(
        input_file_group=InputFileGroup(gslc_file_list=sample_slc_list, frame_id=23210),
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    cfg = rc.to_workflow()
    # TODO: needs to be set in reference change json
    # assert cfg.output_options.extra_reference_date == datetime.datetime(2018, 7, 17)
    assert cfg.output_options.extra_reference_date is None

    # ...but not BEFORE the reference request
    sample_slc_list[7] = (
        "NISAR_L2_GSLC_NI_F150_20180715T161534Z_20240428T062045Z_NI_HH_v0.1.h5"
    )
    rc = RunConfig(
        input_file_group=InputFileGroup(gslc_file_list=sample_slc_list, frame_id=23210),
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    cfg = rc.to_workflow()
    # TODO: needs to be set in reference change json
    # assert cfg.output_options.extra_reference_date == datetime.datetime(2018, 7, 17)
    assert cfg.output_options.extra_reference_date is None


def _make_frame_files(
    sensing_time_list: list[datetime.datetime], num_compressed: int
) -> list[str]:
    comp_slcs = [
        # Note the fake (start, end) datetimes since we dont use those for these testes
        f"COMPRESSED_NISAR_F150_{d.strftime('%Y%m%dT%H%M%S')}_20200101_20210101.h5"
        for d in sensing_time_list[:num_compressed]
    ]
    real_slcs = [
        f"NISAR_F150_{d.strftime('%Y%m%dT%H%M%S')}.h5"
        for d in sensing_time_list[num_compressed:]
    ]
    return comp_slcs + real_slcs


def test_reference_date_computation():
    # Test from a sample alaska frame after dropping winter
    # it spans multiple years
    sensing_time_list = [
        datetime.datetime(2018, 8, 8, 16, 12, 51),
        datetime.datetime(2018, 8, 20, 16, 12, 51),
        datetime.datetime(2018, 9, 1, 16, 12, 52),
        datetime.datetime(2018, 9, 25, 16, 12, 53),
        datetime.datetime(2019, 8, 15, 16, 12, 57),
        datetime.datetime(2019, 8, 27, 16, 12, 58),
        datetime.datetime(2019, 9, 8, 16, 12, 58),
        datetime.datetime(2019, 9, 20, 16, 12, 59),
        datetime.datetime(2020, 6, 10, 16, 13),
        datetime.datetime(2020, 6, 22, 16, 13, 1),
        datetime.datetime(2020, 7, 4, 16, 13, 1),
        datetime.datetime(2020, 7, 16, 16, 13, 2),
        datetime.datetime(2020, 8, 9, 16, 13, 3),
        datetime.datetime(2020, 8, 21, 16, 13, 4),
        datetime.datetime(2020, 9, 2, 16, 13, 5),
    ]
    gslc_file_list = _make_frame_files(sensing_time_list, 0)
    random.shuffle(gslc_file_list)

    # Assume we nominally will reset the reference each august
    reference_datetimes = [datetime.datetime(y, 8, 1) for y in range(2018, 2025)]

    # We expect that
    # - The "output index" should be 0, since there's no compressed SLCs
    # - the "extra reference date" will be the *latest* one from the reference list
    # this is be 2020-08-09
    output_reference_idx, extra_reference_date = pge_runconfig._compute_reference_dates(
        reference_datetimes,
        gslc_file_list,
        compressed_slc_plan=CompressedSlcPlan.ALWAYS_FIRST,
    )
    assert output_reference_idx == 0
    assert extra_reference_date == datetime.date(2020, 8, 9)


def test_reference_first_in_stack():
    sensing_time_list = [
        datetime.datetime(2016, 8, 10, 0, 0),
        datetime.datetime(2016, 9, 3, 0, 0),
        datetime.datetime(2016, 9, 27, 0, 0),
        datetime.datetime(2016, 10, 21, 0, 0),
        datetime.datetime(2016, 11, 14, 0, 0),
        datetime.datetime(2016, 12, 8, 0, 0),
        datetime.datetime(2017, 1, 1, 0, 0),
        datetime.datetime(2017, 1, 13, 0, 0),
        datetime.datetime(2017, 1, 19, 0, 0),
        datetime.datetime(2017, 1, 25, 0, 0),
        datetime.datetime(2017, 2, 18, 0, 0),
        datetime.datetime(2017, 3, 2, 0, 0),
        datetime.datetime(2017, 3, 14, 0, 0),
        datetime.datetime(2017, 3, 26, 0, 0),
        datetime.datetime(2017, 4, 7, 0, 0),
    ]
    gslc_file_list = _make_frame_files(sensing_time_list, 0)
    random.shuffle(gslc_file_list)

    # Assume we nominally will reset the reference each august
    reference_datetimes = [datetime.datetime(y, 8, 1) for y in range(2018, 2025)]
    #   "11114": [
    reference_datetimes = [
        datetime.datetime.fromisoformat(s)
        for s in [
            "2016-08-10T14:07:13",
            "2017-08-17T14:07:19",
            "2018-08-12T14:07:25",
            "2019-08-13T14:06:50",
            "2020-08-13T14:07:38",
            "2021-08-14T14:07:02",
            "2022-08-15T14:07:50",
            "2023-08-10T14:07:54",
            "2024-08-16T14:07:51",
        ]
    ]

    # We expect that
    # - The "output index" should be 0: this is the first ministack
    # - the "extra reference date" will be the None
    output_reference_idx, extra_reference_date = pge_runconfig._compute_reference_dates(
        reference_datetimes,
        gslc_file_list,
        compressed_slc_plan=CompressedSlcPlan.ALWAYS_FIRST,
    )
    assert output_reference_idx == 0
    assert extra_reference_date is None


def test_repeated_compressed_dates():
    gslc_file_list = [
        "compressed_F150_20180722_20190412_20190705.h5",
        "compressed_F150_20190711_20190711_20191003.h5",
        "compressed_F150_20190711_20191009_20200107.h5",
        "compressed_F150_20190711_20200113_20200406.h5",
        "compressed_F150_20200717_20200412_20200729.h5",
        "NISAR_L2_GSLC_NI_F150_20200804T161629Z_20240501T010610Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20200810T161548Z_20240501T030849Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20200816T161630Z_20240501T042747Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20200822T161548Z_20240501T062744Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20200828T161631Z_20240501T075057Z_NI_HH_v0.1.h5",
        "NISAR_L2_GSLC_NI_F150_20200903T161549Z_20240501T094950Z_NI_HH_v0.1.h5",
    ]
    random.shuffle(gslc_file_list)

    reference_datetimes = [
        datetime.datetime(2017, 7, 9),
        datetime.datetime(2018, 7, 16),
        datetime.datetime(2019, 7, 11),
        datetime.datetime(2020, 7, 17),
        datetime.datetime(2021, 7, 12),
    ]

    output_reference_idx, extra_reference_date = pge_runconfig._compute_reference_dates(
        reference_datetimes,
        gslc_file_list,
        compressed_slc_plan=CompressedSlcPlan.ALWAYS_FIRST,
    )
    # Should be the latest one: the compressed slc with 20200717
    assert output_reference_idx == 4
    assert extra_reference_date is None


def test_reference_date_last_per_ministack():
    compressed_slc_plan = CompressedSlcPlan.LAST_PER_MINISTACK
    sensing_time_list = [
        datetime.datetime(2016, 8, 10, 0, 0),
        datetime.datetime(2016, 9, 3, 0, 0),
        datetime.datetime(2016, 9, 27, 0, 0),
        datetime.datetime(2016, 10, 21, 0, 0),
        datetime.datetime(2016, 11, 14, 0, 0),
        datetime.datetime(2016, 12, 8, 0, 0),
        datetime.datetime(2017, 1, 1, 0, 0),
        datetime.datetime(2017, 1, 13, 0, 0),
        datetime.datetime(2017, 1, 19, 0, 0),
        datetime.datetime(2017, 1, 25, 0, 0),
        datetime.datetime(2017, 2, 18, 0, 0),
        datetime.datetime(2017, 3, 2, 0, 0),
        datetime.datetime(2017, 3, 14, 0, 0),
        datetime.datetime(2017, 3, 26, 0, 0),
        datetime.datetime(2017, 4, 7, 0, 0),
    ]
    gslc_file_list = _make_frame_files(
        sensing_time_list=sensing_time_list, num_compressed=0
    )
    random.shuffle(gslc_file_list)

    # Assume we nominally will reset the reference each august
    reference_datetimes = []
    # We expect that
    # - The "output index" should be 0: this is the first ministack
    # - the "extra reference date" will be the None
    output_reference_idx, extra_reference_date = pge_runconfig._compute_reference_dates(
        reference_datetimes, gslc_file_list, compressed_slc_plan=compressed_slc_plan
    )
    assert output_reference_idx == 0
    assert extra_reference_date is None

    # Now add compressed, and not that output reference index should be the latest one
    for num_compressed in range(1, 6):
        gslc_file_list = _make_frame_files(
            sensing_time_list=sensing_time_list, num_compressed=num_compressed
        )
        random.shuffle(gslc_file_list)
        output_reference_idx, extra_reference_date = (
            pge_runconfig._compute_reference_dates(
                reference_datetimes,
                gslc_file_list,
                compressed_slc_plan=compressed_slc_plan,
            )
        )
        assert output_reference_idx == num_compressed - 1

        assert extra_reference_date is None


@pytest.fixture
def overrides_file():
    return (
        Path(__file__).parent
        / "data/opera-disp-nisar-algorithm-parameters-overrides-dummy.json"
    )


def test_algorithm_overrides_hawaii(overrides_file, algorithm_parameters_file):
    orig_params = AlgorithmParameters.from_yaml(algorithm_parameters_file)
    orig_params.algorithm_parameters_overrides_json = overrides_file

    p2 = pge_runconfig._override_parameters(orig_params, 23210)  # hawaii
    assert p2.unwrap_options.unwrap_method == "spurt"


def test_algorithm_overrides_empty_frame(overrides_file, algorithm_parameters_file):
    orig_params = AlgorithmParameters.from_yaml(algorithm_parameters_file)
    orig_params.algorithm_parameters_overrides_json = overrides_file

    # frame id with no override
    p4 = pge_runconfig._override_parameters(orig_params, 1234)
    assert p4.unwrap_options == orig_params.unwrap_options

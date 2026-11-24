#!/usr/bin/env python3
"""Create the delivery runconfig/algorithm-parameter files for DISP-NISAR.

Writes, into `config_files/`, for each processing mode (forward, historical) and
each option (A: with GUNW ionosphere files, B: without):
    runconfig_{mode}_option{A,B}.yaml
and for each mode:
    algorithm_parameters_{mode}.yaml
    algorithm_parameters_ionosphere_{mode}.yaml

The files are built directly from the `disp_nisar.pge_runconfig` models, so any
option that is added/removed there is picked up (missing options get their defaults).
"""

from __future__ import annotations

from pathlib import Path

from disp_nisar.enums import ProcessingMode
from disp_nisar.pge_runconfig import (
    AlgorithmParameters,
    DynamicAncillaryFileGroup,
    InputFileGroup,
    PrimaryExecutable,
    ProductPathGroup,
    RunConfig,
    StaticAncillaryFileGroup,
)

# Note on ionosphere/troposphere files:
# Option A: GUNW files are passed (ionosphere/geometry layers read from them).
# Option B: no GUNW files.
# The troposphere corrections are to be created separately.

CFG_DIR = Path("config_files")
FRAME_ID = 5826
FREQUENCY = "frequencyA"
POLARIZATION = "HH"
AZIMUTH_BLOCKS = 6
PRODUCT_VERSION = "0.4"

GSLC_URL = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GSLC_V1"
GUNW_URL = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/L2_L_GUNW"

# Historical: the 6 GSLCs 009-015; forward: compressed SLC + the newest GSLCs (010-016)
GSLC_IDS = [
    "NISAR_L2_PR_GSLC_015_034_A_019_4005_DHDH_A_20260312T130829_20260312T130904_P05036_P_F_J_001",
    "NISAR_L2_PR_GSLC_013_034_A_019_4005_DHDH_A_20260216T130828_20260216T130903_P05036_P_F_J_001",
    "NISAR_L2_PR_GSLC_012_034_A_019_4005_DHDH_A_20260204T130828_20260204T130903_P05036_N_F_J_001",
    "NISAR_L2_PR_GSLC_011_034_A_019_4005_DHDH_A_20260123T130827_20260123T130902_P05036_P_F_J_001",
    "NISAR_L2_PR_GSLC_010_034_A_019_4005_DHDH_A_20260111T130827_20260111T130902_P05036_P_F_J_001",
    "NISAR_L2_PR_GSLC_009_034_A_019_4005_DHDH_A_20251230T130826_20251230T130901_P05036_P_F_J_001",
]

GUNW_IDS = [
    "NISAR_L2_PR_GUNW_013_034_A_019_015_4000_SH_20260216T130828_20260216T130903_20260312T130829_20260312T130904_P05036_P_F_J_001",
    "NISAR_L2_PR_GUNW_012_034_A_019_013_4000_SH_20260204T130828_20260204T130903_20260216T130828_20260216T130903_P05036_P_F_J_001",
    "NISAR_L2_PR_GUNW_011_034_A_019_012_4000_SH_20260123T130827_20260123T130902_20260204T130828_20260204T130903_P05036_N_F_J_001",
    "NISAR_L2_PR_GUNW_010_034_A_019_011_4000_SH_20260111T130827_20260111T130902_20260123T130827_20260123T130902_P05036_P_F_J_001",
    "NISAR_L2_PR_GUNW_009_034_A_019_010_4000_SH_20251230T130826_20251230T130901_20260111T130827_20260111T130902_P05036_P_F_J_001",
]

# Forward mode only: the newest GSLC and the GUNW linking it to the previous one
FORWARD_GSLC_IDS = [
    "NISAR_L2_PR_GSLC_016_034_A_019_4005_DHDH_A_20260324T130829_20260324T130904_P05036_P_F_J_001",
]
FORWARD_GUNW_IDS = [
    "NISAR_L2_PR_GUNW_015_034_A_019_016_4000_SH_20260312T130829_20260312T130904_20260324T130829_20260324T130904_P05036_P_F_J_001",
]
COMPRESSED_SLC = "input_gslcs/compressed_20260312_20251230_20260312.h5"


def _url(base: str, ident: str) -> str:
    return f"{base}/{ident}/{ident}.h5"


GSLC_FILES = {
    "historical": [_url(GSLC_URL, i) for i in GSLC_IDS],
    "forward": [
        COMPRESSED_SLC,
        *(_url(GSLC_URL, i) for i in [*FORWARD_GSLC_IDS, *GSLC_IDS[:-1]]),
    ],
}
GUNW_FILES = {
    "historical": [_url(GUNW_URL, i) for i in GUNW_IDS],
    "forward": [_url(GUNW_URL, i) for i in [*FORWARD_GUNW_IDS, *GUNW_IDS]],
}

DYNAMIC_ANCILLARY_DIR = "dynamic_ancillary_files"
STATIC_ANCILLARY_DIR = "static_ancillary_files"
MASK_FILE = f"{DYNAMIC_ANCILLARY_DIR}/water_mask.vrt"
DEM_FILE = f"{DYNAMIC_ANCILLARY_DIR}/dem.vrt"
OVERRIDES_JSON = (
    f"{STATIC_ANCILLARY_DIR}/opera-disp-nisar-algorithm-parameters-overrides.json"
)
FRAME_TO_BOUNDS_JSON = (
    f"{STATIC_ANCILLARY_DIR}/opera-nisar-disp-0.1.0-frame-to-bounds.json"
)
REFERENCE_DATES_JSON = (
    f"{STATIC_ANCILLARY_DIR}/opera-disp-nisar-reference-dates-dummy.json"
)

WORKER_SETTINGS = {
    "gpu_enabled": False,
    "threads_per_worker": 2,
    "n_parallel_bursts": 6,
    "block_shape": [1024, 1024],
}
HISTORICAL_BLOCK_SHAPE = [512, 512]

# Values that differ from the `AlgorithmParameters` defaults, per file.
ALGORITHM_PARAMETERS = {
    ("freqA", ProcessingMode.FORWARD): {
        "ps_options": {"amp_dispersion_threshold": 0.2},
        "phase_linking": {
            "ministack_size": 100,
            "max_num_compressed": 100,
            "output_reference_idx": 0,
            "half_window": {"x": 7},
            "compressed_slc_plan": "last_per_ministack",
            "write_crlb": False,
        },
        "interferogram_network": {"max_bandwidth": 4},
        "unwrap_options": {
            "run_interpolation": True,
            "n_parallel_jobs": 4,
            "unwrap_method": "whirlwind",
            "preprocess_options": {
                "max_radius": 150,
                "interpolation_cor_threshold": 0.001,
                "interpolation_similarity_threshold": 0.4,
            },
            "snaphu_options": {
                "ntiles": [5, 5],
                "tile_overlap": [300, 300],
                "n_parallel_tiles": 5,
                "single_tile_reoptimize": True,
            },
            "whirlwind_options": {
                "interpolate": True,
            },
            "spurt_options": {
                "tiler_settings": {
                    "max_tiles": 25,
                    "target_points_per_tile": 1400000,
                    "dilation_factor": 0.04,
                },
                "solver_settings": {
                    "t_worker_count": 16,
                    "s_worker_count": 3,
                    "t_cost_type": "distance",
                },
                "merger_settings": {"num_parallel_ifgs": 21},
            },
        },
        "timeseries_options": {"run_velocity": False, "correlation_threshold": 0.0},
        "output_options": {
            "strides": {"x": 12, "y": 12},
            "bounds": [-156.26, 18.87, -154.71, 20.16],
            "add_overviews": False,
        },
        "recommended_temporal_coherence_threshold": 0.7,
        "recommended_similarity_threshold": 0.6,
        "spatial_wavelength_cutoff": 30000.0,
        "browse_image_vmin_vmax": [-0.05, 0.05],
        "forward_mode_network_size": 4,
        "num_parallel_products": 4,
    },
    ("freqA", ProcessingMode.HISTORICAL): {
        "ps_options": {"amp_dispersion_threshold": 0.2},
        "phase_linking": {
            "ministack_size": 100,
            "max_num_compressed": 100,
            "output_reference_idx": 0,
            "half_window": {"x": 7},
            "compressed_slc_plan": "last_per_ministack",
            "write_crlb": False,
        },
        "interferogram_network": {"max_bandwidth": 3},
        "unwrap_options": {
            "run_interpolation": True,
            "n_parallel_jobs": 4,
            "unwrap_method": "whirlwind",
            "preprocess_options": {
                "max_radius": 150,
                "interpolation_cor_threshold": 0.001,
                "interpolation_similarity_threshold": 0.4,
            },
            "snaphu_options": {
                "ntiles": [5, 5],
                "tile_overlap": [300, 300],
                "n_parallel_tiles": 7,
                "single_tile_reoptimize": True,
            },
            "whirlwind_options": {
                "interpolate": True,
            },
            "spurt_options": {
                "tiler_settings": {
                    "max_tiles": 25,
                    "target_points_per_tile": 1400000,
                    "dilation_factor": 0.04,
                },
                "solver_settings": {
                    "t_worker_count": 16,
                    "s_worker_count": 3,
                    "t_cost_type": "distance",
                },
                "merger_settings": {"num_parallel_ifgs": 21},
            },
        },
        "timeseries_options": {"run_velocity": False, "correlation_threshold": 0.0},
        "output_options": {
            "strides": {"x": 12, "y": 12},
            "bounds": [-156.26, 18.87, -154.71, 20.16],
            "add_overviews": False,
        },
        "recommended_temporal_coherence_threshold": 0.7,
        "recommended_similarity_threshold": 0.6,
        "spatial_wavelength_cutoff": 30000.0,
        "browse_image_vmin_vmax": [-0.05, 0.05],
        "num_parallel_products": 4,
    },
    ("ionosphere", ProcessingMode.FORWARD): {
        "ps_options": {"amp_dispersion_threshold": 0.2},
        "phase_linking": {
            "ministack_size": 100,
            "max_num_compressed": 100,
            "output_reference_idx": 0,
            "half_window": {"x": 15, "y": 40},
            "compressed_slc_plan": "last_per_ministack",
            "write_crlb": False,
        },
        "interferogram_network": {"max_bandwidth": 3},
        "unwrap_options": {
            "run_interpolation": True,
            "n_parallel_jobs": 4,
            "unwrap_method": "whirlwind",
            "preprocess_options": {
                "max_radius": 150,
                "interpolation_cor_threshold": 0.001,
                "interpolation_similarity_threshold": 0.4,
            },
            "snaphu_options": {
                "tile_overlap": [300, 300],
                "n_parallel_tiles": 5,
                "single_tile_reoptimize": True,
            },
            "whirlwind_options": {
                "interpolate": True,
            },
            "spurt_options": {
                "tiler_settings": {
                    "max_tiles": 25,
                    "target_points_per_tile": 1400000,
                    "dilation_factor": 0.04,
                },
                "solver_settings": {
                    "t_worker_count": 16,
                    "s_worker_count": 3,
                    "t_cost_type": "distance",
                },
                "merger_settings": {"num_parallel_ifgs": 21},
            },
        },
        "timeseries_options": {
            "run_velocity": False,
            "correlation_threshold": 0.0,
            "num_parallel_blocks": 8,
        },
        "output_options": {
            "strides": {"x": 2, "y": 16},
            "bounds": [-156.26, 18.87, -154.71, 20.16],
            "add_overviews": False,
        },
        "recommended_temporal_coherence_threshold": 0.7,
        "recommended_similarity_threshold": 0.6,
        "spatial_wavelength_cutoff": 30000.0,
        "browse_image_vmin_vmax": [-0.05, 0.05],
        "forward_mode_network_size": 4,
        "num_parallel_products": 4,
    },
    ("ionosphere", ProcessingMode.HISTORICAL): {
        "ps_options": {"amp_dispersion_threshold": 0.2},
        "phase_linking": {
            "ministack_size": 100,
            "max_num_compressed": 100,
            "output_reference_idx": 0,
            "half_window": {"x": 15, "y": 40},
            "compressed_slc_plan": "last_per_ministack",
            "write_crlb": False,
        },
        "interferogram_network": {"max_bandwidth": 3},
        "unwrap_options": {
            "run_interpolation": True,
            "n_parallel_jobs": 4,
            "unwrap_method": "whirlwind",
            "preprocess_options": {
                "max_radius": 150,
                "interpolation_cor_threshold": 0.001,
                "interpolation_similarity_threshold": 0.4,
            },
            "snaphu_options": {
                "tile_overlap": [300, 300],
                "n_parallel_tiles": 5,
                "single_tile_reoptimize": True,
            },
            "whirlwind_options": {
                "interpolate": True,
            },
            "spurt_options": {
                "tiler_settings": {
                    "max_tiles": 25,
                    "target_points_per_tile": 1400000,
                    "dilation_factor": 0.04,
                },
                "solver_settings": {
                    "t_worker_count": 16,
                    "s_worker_count": 3,
                    "t_cost_type": "distance",
                },
                "merger_settings": {"num_parallel_ifgs": 21},
            },
        },
        "timeseries_options": {
            "run_velocity": False,
            "correlation_threshold": 0.0,
            "num_parallel_blocks": 8,
        },
        "output_options": {
            "strides": {"x": 2, "y": 16},
            "bounds": [-156.26, 18.87, -154.71, 20.16],
            "add_overviews": False,
        },
        "recommended_temporal_coherence_threshold": 0.7,
        "recommended_similarity_threshold": 0.6,
        "spatial_wavelength_cutoff": 30000.0,
        "browse_image_vmin_vmax": [-0.05, 0.05],
        "num_parallel_products": 4,
    },
}


def _merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base`, raising on unknown keys."""
    for key, value in overrides.items():
        if key not in base:
            msg = f"{key!r} is not an AlgorithmParameters option"
            raise KeyError(msg)
        if isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def make_algorithm_parameters(kind: str, mode: ProcessingMode) -> Path:
    """Write the algorithm parameters file for `kind` (freqA/ionosphere) and `mode`."""
    base = AlgorithmParameters().model_dump(mode="json")
    params = AlgorithmParameters.model_validate(
        _merge(base, ALGORITHM_PARAMETERS[(kind, mode)])
    )
    name = "algorithm_parameters" if kind == "freqA" else f"algorithm_parameters_{kind}"
    outfile = CFG_DIR / f"{name}_{mode.value}.yaml"
    params.to_yaml(outfile)
    return outfile


def make_runconfig(mode: ProcessingMode, option: str) -> Path:
    """Write the runconfig for `mode` and `option` ("A": with GUNW files, "B": none)."""
    tag = f"{mode.value}_option{option}"
    output = Path(f"output/{tag}")
    runconfig = RunConfig(
        input_file_group=InputFileGroup(
            gslc_file_list=GSLC_FILES[mode.value],
            frame_id=FRAME_ID,
            frequency=FREQUENCY,
            polarization=POLARIZATION,
            azimuth_blocks=AZIMUTH_BLOCKS,
        ),
        dynamic_ancillary_file_group=DynamicAncillaryFileGroup(
            algorithm_parameters_file=CFG_DIR
            / f"algorithm_parameters_{mode.value}.yaml",
            ionosphere_algorithm_parameters_file=(
                CFG_DIR / f"algorithm_parameters_ionosphere_{mode.value}.yaml"
            ),
            mask_file=MASK_FILE,
            dem_file=DEM_FILE,
            gunw_files=GUNW_FILES[mode.value] if option == "A" else [],
        ),
        static_ancillary_file_group=StaticAncillaryFileGroup(
            algorithm_parameters_overrides_json=OVERRIDES_JSON,
            frame_to_bounds_json=FRAME_TO_BOUNDS_JSON,
            reference_date_database_json=REFERENCE_DATES_JSON,
        ),
        primary_executable=PrimaryExecutable(
            product_type=f"DISP_NISAR_{mode.value.upper()}"
        ),
        product_path_group=ProductPathGroup(
            product_path=output,
            scratch_path=Path(f"scratch/{tag}"),
            sas_output_path=output,
            product_version=PRODUCT_VERSION,
            save_compressed_slc=True,
        ),
        worker_settings={
            **WORKER_SETTINGS,
            **(
                {"block_shape": HISTORICAL_BLOCK_SHAPE}
                if mode == ProcessingMode.HISTORICAL
                else {}
            ),
        },
        log_file=Path(f"scratch/{tag}/log_sas.log"),
    )
    outfile = CFG_DIR / f"runconfig_{tag}.yaml"
    runconfig.to_yaml(outfile)
    return outfile


if __name__ == "__main__":
    CFG_DIR.mkdir(exist_ok=True)
    for mode in ProcessingMode:
        for kind in ("freqA", "ionosphere"):
            print(make_algorithm_parameters(kind, mode))
        for option in ("A", "B"):
            print(make_runconfig(mode, option))

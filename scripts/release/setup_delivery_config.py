#!/usr/bin/env python3
import subprocess
from pathlib import Path

from disp_nisar.enums import ProcessingMode

# Note on ionosphere/troposphere files:
# The ionosphere corrections are to be read from gunw files which is missing.
# The troposphere corrections are to be created separately.


def setup_delivery(cfg_dir: Path, mode: ProcessingMode):
    """Set up the dolphin config file for the delivery for one mode."""
    cfg_dir.mkdir(exist_ok=True)
    single_flag = "--single" if mode == ProcessingMode.FORWARD else ""
    outfile = f"{cfg_dir}/dolphin_config_{mode.value}.yaml"
    cmd = (
        "dolphin config "
        f" --keep-paths-relative --work scratch/{mode.value} --strides 6 6"
        # Inputs:
        " --slc-files ./input_slcs/*h5 --subdataset"
        " /science/LSAR/GSLC/grids/frequencyA/HH"
        # Phase linking stuff
        f" --ministack-size 15 {single_flag}"
        # Dynamic ancillary files #
        ###########################
        # TODO # seasonal coherence averages
        # Ionosphere files:
        # " --ionosphere-files ./dynamic_ancillary_files/ionosphere_files/*"
        # Geometry files/static layers
        " --geometry-files ./dynamic_ancillary_files/gunw_files/*.h5"
        " --mask-file ./dynamic_ancillary_files/water_mask.tif"
        #
        # Unwrapping stuff
        " --unwrap-method snaphu --ntiles 2 2 --downsample 5 5 --run-interpolation"
        # Worker stuff
        " --threads-per-worker 16 --n-parallel-bursts 4 --n-parallel-unwrap 4 "
        f" --log-file scratch/{mode.value}/log_sas.log"
        f" -o {outfile}"
    )
    print(cmd)
    subprocess.run(cmd, shell=True, check=False)
    return outfile


if __name__ == "__main__":
    cfg_dir = Path("config_files")
    static_ancillary_dir = Path("static_ancillary_files")
    # Rosamond, track 01:
    frame_id = 150
    frequency = "frequencyA"
    polarization = "HH"
    reference_json = "opera-disp-nisar-reference-dates-dummy.json"
    frame_to_bounds_json = "Frame_to_bounds_DISP-NI_v0.1.json"
    # Creates one file for the forward mode and one for the historical mode.
    for mode in ProcessingMode:
        output_directory = Path(f"output/{mode.value}")
        # TODO: adjust the number of
        # ionosphere files
        # troposphere files
        dolphin_cfg_file = setup_delivery(cfg_dir=cfg_dir, mode=mode)
        # Run the "convert_config.py" script in the same directory
        # as this script.
        this_dir = Path(__file__).parent
        convert_config = this_dir / "convert_config.py"
        arg_string = (
            f" --frame-id {frame_id} "
            f" --frequency {frequency} "
            f" --polarization {polarization} "
            f" --reference_date_database_json {static_ancillary_dir}/{reference_json} "
            f" --frame-to-bounds-json {static_ancillary_dir}/{frame_to_bounds_json} "
            f" --output-directory {output_directory}"
            f" --processing-mode {mode.value} --save-compressed-slc -o"
            f" {cfg_dir}/runconfig_{mode.value}.yaml  -a"
            f" {cfg_dir}/algorithm_parameters_{mode.value}.yaml"
        )
        cmd = f"python {convert_config} {dolphin_cfg_file} {arg_string}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=False)
        # Remove the `dolphin` yamls
        for f in cfg_dir.glob("dolphin_config*.yaml"):
            f.unlink()

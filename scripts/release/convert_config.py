#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from dolphin.workflows.config import DisplacementWorkflow

from disp_nisar.enums import ProcessingMode
from disp_nisar.pge_runconfig import RunConfig


def convert_to_runconfig(
    dolphin_config_file: str,
    frame_id: int,
    frequency: str,
    polarization: str,
    processing_mode: ProcessingMode,
    frame_to_bounds_json: Path | None = None,
    reference_date_database_json: Path | None = None,
    algorithm_parameters_file: Path = Path("algorithm_parameters.yaml"),
    output_directory: Path | None = None,
    save_compressed_slc: bool = True,
    outfile: str | Path = "runconfig.yaml",
):
    """Run the conversion CLI."""
    workflow = DisplacementWorkflow.from_yaml(dolphin_config_file)
    rc = RunConfig.from_workflow(
        workflow,
        frame_id=frame_id,
        frequency=frequency,
        polarization=polarization,
        frame_to_bounds_json=frame_to_bounds_json,
        reference_date_json=reference_date_database_json,
        algorithm_parameters_file=algorithm_parameters_file,
        processing_mode=processing_mode,
        save_compressed_slc=save_compressed_slc,
        output_directory=output_directory,
    )
    rc.to_yaml(outfile)


def main():
    """Run the conversion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert a `dolphin_config.yaml` to `runconfig.yaml` for SDS"
    )

    parser.add_argument(
        "dolphin_config_file", type=str, help="Path to dolphin configuration YAML file."
    )
    parser.add_argument("--frame-id", required=True, type=int, help="Frame ID.")
    parser.add_argument(
        "--frequency",
        required=True,
        type=str,
        help="Frequency (frequencyA, frequencyB).",
    )
    parser.add_argument(
        "--polarization", required=True, type=str, help="Polarization (HH, VV, HV, VH)."
    )
    parser.add_argument(
        "--processing-mode",
        required=True,
        type=ProcessingMode,
        choices=list(ProcessingMode),
        help="Processing mode.",
    )
    parser.add_argument(
        "-od",
        "--output-directory",
        type=Path,
        help=(
            "Name of output directory for final products. If not specified, will use"
            " `outputs` one level up from the dolphin working directory."
        ),
    )
    parser.add_argument(
        "-o", "--outfile", type=Path, default="runconfig.yaml", help="Output file path."
    )
    parser.add_argument(
        "--frame-to-bounds-json",
        type=Path,
        help=(
            "Path to frame-to-bounds mapping JSON file, summarizing DISP frame"
            " database."
        ),
    )
    parser.add_argument(
        "--reference_date_database_json",
        type=Path,
        help="JSON file containing list of reference date changes for each frame.",
    )
    parser.add_argument(
        "-a",
        "--algorithm-parameters-file",
        type=Path,
        default="algorithm_parameters.yaml",
        help="Path to algorithm parameters file.",
    )
    parser.add_argument(
        "--save-compressed-slc",
        action="store_true",
        help=(
            "Indicate in the runconfig that the compressed SLCs should be saved in the"
            " output. Note that for the historical mode, this will only save the"
        ),
    )

    args = parser.parse_args()

    convert_to_runconfig(
        dolphin_config_file=args.dolphin_config_file,
        frame_id=args.frame_id,
        frequency=args.frequency,
        polarization=args.polarization,
        processing_mode=args.processing_mode,
        frame_to_bounds_json=args.frame_to_bounds_json,
        reference_date_database_json=args.reference_date_database_json,
        algorithm_parameters_file=args.algorithm_parameters_file,
        save_compressed_slc=args.save_compressed_slc,
        output_directory=args.output_directory,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Standalone script to run static layers workflow with data staging."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from disp_nisar.pge_runconfig import StaticLayersRunConfig

logger = logging.getLogger("disp_nisar")

NAME_TEMPLATE = "OPERA_L3_DISP-NI-STATIC_F{frame_id:05d}_20250101_v1.0"


def get_frame_info(frame_id: int, gpkg_file: Path) -> dict:
    """Get frame information from GeoPackage.

    Parameters
    ----------
    frame_id : int
        Frame ID
    gpkg_file : Path
        Path to GeoPackage file

    Returns
    -------
    dict
        Frame information with keys: epsg, track, frame, bounds

    """
    import geopandas as gpd
    from dolphin import Bbox

    gdf = gpd.read_file(gpkg_file)
    frame_data = gdf[gdf["frame_idx"] == frame_id]

    if len(frame_data) == 0:
        raise ValueError(f"Frame {frame_id} not found in {gpkg_file}")

    row = frame_data.iloc[0]
    return {
        "epsg": int(row["epsg"]),
        "track": int(row["track"]),
        "frame": int(row["frame"]),
        "bounds": Bbox(
            left=float(row["mapTopLeftX"]),
            bottom=float(row["mapBottomRightY"]),
            right=float(row["mapBottomRightX"]),
            top=float(row["mapTopLeftY"]),
        ),
        "pass_direction": row["passDirection"],
    }


def validate_frame_id_matches_gslc(
    gslc_file: Path,
    frame_info: dict,
    frame_id: int,
) -> None:
    """Confirm a user-supplied ``--frame-id`` matches the GSLC's own track/frame.

    Parameters
    ----------
    gslc_file : Path
        Path to the NISAR GSLC HDF5 file to check.
    frame_info : dict
        Frame information from get_frame_info() for the given frame_id.
    frame_id : int
        The frame ID being validated (for error messages only).

    Raises
    ------
    ValueError
        If the GSLC's track/frame number don't match frame_info.

    """
    import h5py

    with h5py.File(gslc_file, "r") as hf:
        gslc_track = int(hf["/science/LSAR/identification/trackNumber"][()])
        gslc_frame = int(hf["/science/LSAR/identification/frameNumber"][()])

    if gslc_track != frame_info["track"] or gslc_frame != frame_info["frame"]:
        raise ValueError(
            f"--frame-id {frame_id} corresponds to track {frame_info['track']}, "
            f"frame {frame_info['frame']} in the frame database, but "
            f"{gslc_file} is track {gslc_track}, frame {gslc_frame}. "
            "Double-check --frame-id against this GSLC, or look up the correct "
            "frame_idx for this track/frame in the frame GeoPackage."
        )


def download_gslc_for_frame(
    frame_id: int,
    frame_info: dict,
    output_dir: Path,
) -> Path:
    """Download a GSLC file for the given frame using opera_utils.nisar.

    Parameters
    ----------
    frame_id : int
        Frame ID
    frame_info : dict
        Frame information from get_frame_info()
    output_dir : Path
        Directory to save downloaded GSLC

    Returns
    -------
    Path
        Path to downloaded GSLC file

    """
    from opera_utils.nisar import download_gslcs, search

    logger.info(
        f"Searching for GSLC data for frame {frame_id}, track {frame_info['track']}"
    )

    # Get frame bounds in WGS84
    import pyproj

    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{frame_info['epsg']}",
        "EPSG:4326",
        always_xy=True,
    )

    bounds_utm = frame_info["bounds"]
    # Transform corners to get WGS84 bbox
    lon1, lat1 = transformer.transform(bounds_utm.left, bounds_utm.bottom)
    lon2, lat2 = transformer.transform(bounds_utm.right, bounds_utm.top)

    wgs84_bbox = (
        min(lon1, lon2),
        min(lat1, lat2),
        max(lon1, lon2),
        max(lat1, lat2),
    )

    # Determine orbit direction (A=ascending, D=descending)
    orbit_dir = "A" if frame_info["pass_direction"].lower().startswith("asc") else "D"

    logger.info(f"  Bounds (WGS84): {wgs84_bbox}")
    logger.info(f"  Track: {frame_info['track']}")
    logger.info(f"  Frame number: {frame_info['frame']}")
    logger.info(f"  Orbit direction: {orbit_dir}")

    # Try searching with different constraints, starting broad then narrowing
    results = []

    # Try 1: Search with all constraints but no date range
    logger.info("  Searching with track/frame/orbit constraints...")
    try:
        results = search(
            bbox=wgs84_bbox,
            track_frame_number=frame_info["frame"],
            relative_orbit_number=frame_info["track"],
            orbit_direction=orbit_dir,
        )
    except Exception as e:
        logger.warning(f"  Search with all constraints failed: {e}")

    # Try 2: If no results, try with just track and bounds
    if len(results) == 0:
        logger.info("  No results. Trying with just track and bounds...")
        try:
            results = search(
                bbox=wgs84_bbox,
                relative_orbit_number=frame_info["track"],
            )
        except Exception as e:
            logger.warning(f"  Search with track/bounds failed: {e}")

    # Try 3: If still no results, try with just bounds
    if len(results) == 0:
        logger.info("  No results. Trying with just bounds...")
        try:
            results = search(bbox=wgs84_bbox)
        except Exception as e:
            logger.warning(f"  Search with bounds only failed: {e}")

    if len(results) == 0:
        raise ValueError(
            f"No GSLC data found for frame {frame_id}.\n"
            "Search criteria tried:\n"
            f"  Track: {frame_info['track']}\n"
            f"  Frame number: {frame_info['frame']}\n"
            f"  Orbit: {orbit_dir}\n"
            f"  Bounds: {wgs84_bbox}\n"
            "\nPlease provide a GSLC file using --gslc-file option."
        )

    logger.info(f"Found {len(results)} GSLC products")

    # Check if we already have one downloaded
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_gslcs = list(output_dir.glob("NISAR_*.h5"))
    if existing_gslcs:
        logger.info(f"Using existing GSLC: {existing_gslcs[0]}")
        return existing_gslcs[0]

    # Download the most recent one
    most_recent = results[0]
    logger.info("Downloading most recent GSLC:")
    logger.info(f"  Filename: {most_recent.filename}")
    logger.info(f"  Date: {most_recent.start_datetime}")

    # Use more flexible download parameters
    try:
        downloaded = download_gslcs(
            output_dir=output_dir,
            bbox=wgs84_bbox,
            relative_orbit_number=frame_info["track"],
            start_datetime=most_recent.start_datetime,
            end_datetime=most_recent.end_datetime,
            max_jobs=1,
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Trying download with just bbox and date range...")
        downloaded = download_gslcs(
            output_dir=output_dir,
            bbox=wgs84_bbox,
            start_datetime=most_recent.start_datetime,
            end_datetime=most_recent.end_datetime,
            max_jobs=1,
        )

    if len(downloaded) == 0:
        raise RuntimeError("Failed to download GSLC file")

    logger.info(f"GSLC downloaded to: {downloaded[0]}")
    return downloaded[0]


def download_dem(
    gslc_file: Path,
    frequency: str,
    polarization: str,
    output: Path,
) -> Path:
    """Stage a DEM for the NISAR frame.

    Parameters
    ----------
    gslc_file : Path
        Path to GSLC file to extract bounds from
    frequency : str
        Frequency band (frequencyA or frequencyB)
    polarization : str
        Polarization (HH, HV, VH, VV)
    output : Path
        Output path for the DEM

    Returns
    -------
    Path
        Path to staged DEM

    """
    logger.info(f"Extracting bounds from GSLC file: {gslc_file}")

    # Extract bounds from GSLC
    from disp_nisar._utils import get_nisar_frame_bbox

    epsg_utm, bounds_utm = get_nisar_frame_bbox(gslc_file, frequency, polarization)

    # Convert UTM bounds to WGS84 for DEM download
    import pyproj
    from dolphin import Bbox

    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{epsg_utm}",
        "EPSG:4326",
        always_xy=True,
    )

    left, bottom, right, top = bounds_utm
    # Transform corners
    lon1, lat1 = transformer.transform(left, bottom)
    lon2, lat2 = transformer.transform(right, top)

    bbox_wgs84 = Bbox(
        left=min(lon1, lon2),
        bottom=min(lat1, lat2),
        right=max(lon1, lon2),
        top=max(lat1, lat2),
    )

    logger.info(f"Frame bounds (WGS84): {bbox_wgs84}")
    logger.info(f"Downloading DEM to {output}")

    output.parent.mkdir(parents=True, exist_ok=True)

    # Download DEM directly from S3 using GDAL Translate
    # This avoids the issue with stage_dem() creating a broken VRT
    from osgeo import gdal

    s3_dem_vrt = "/vsis3/opera-dem/EPSG4326/EPSG4326.vrt"

    # Add margin to bbox (5 km)
    margin_deg = 5 / 111  # Approximately 5 km in degrees
    proj_win = (
        bbox_wgs84.left - margin_deg,
        bbox_wgs84.top + margin_deg,
        bbox_wgs84.right + margin_deg,
        bbox_wgs84.bottom - margin_deg,
    )

    translate_options = gdal.TranslateOptions(
        format="GTiff",
        projWin=proj_win,
        projWinSRS="EPSG:4326",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )

    logger.info(f"Translating DEM from {s3_dem_vrt}")
    ds = gdal.Translate(str(output), s3_dem_vrt, options=translate_options)
    if ds is None:
        raise RuntimeError(f"Failed to download DEM to {output}")
    ds = None  # Close the dataset

    logger.info(f"DEM saved to {output}")

    # Return absolute path to avoid relative path issues
    return output.resolve()


def create_runconfig(
    frame_id: int,
    gslc_file: Path,
    dem_file: Path,
    output_dir: Path,
    scratch_dir: Path,
    frame_gpkg: Path,
    frequency: str = "frequencyA",
    polarization: str = "HH",
    mask_file: Path | None = None,
    product_version: str = "0.4",
    create_3band_los: bool = True,
    product_spacing_m: int | None = None,
) -> Path:
    """Create a runconfig for static layers processing.

    Parameters
    ----------
    frame_id : int
        NISAR frame ID
    gslc_file : Path
        Path to GSLC file
    dem_file : Path
        Path to DEM file
    output_dir : Path
        Output directory
    scratch_dir : Path
        Scratch directory
    frame_gpkg : Path
        Path to frame GeoPackage file
    frequency : str
        Frequency band
    polarization : str
        Polarization
    mask_file : Path, optional
        Optional mask file
    product_version : str
        Product version
    create_3band_los : bool
        Whether to create 3-band LOS output
    product_spacing_m : int, optional
        Product spacing in meters

    Returns
    -------
    Path
        Path to created runconfig file

    """
    from disp_nisar.pge_runconfig import (
        InputFileGroup,
        PrimaryExecutable,
        ProductPathGroup,
        StaticAncillaryFileGroup,
        StaticLayersDynamicAncillaryFileGroup,
        StaticLayersRunConfig,
        WorkerSettings,
    )

    # Ensure all paths are absolute to avoid relative path issues
    runconfig = StaticLayersRunConfig(
        input_file_group=InputFileGroup(
            frame_id=frame_id,
            frequency=frequency,
            polarization=polarization,
        ),
        dynamic_ancillary_file_group=StaticLayersDynamicAncillaryFileGroup(
            gslc_file=gslc_file.resolve(),
            dem_file=dem_file.resolve(),
            mask_file=mask_file.resolve() if mask_file else None,
        ),
        static_ancillary_file_group=StaticAncillaryFileGroup(
            frame_to_bounds_json=frame_gpkg.resolve(),
        ),
        primary_executable=PrimaryExecutable(product_type="DISP_NISAR_STATIC"),
        product_path_group=ProductPathGroup(
            product_path=output_dir.resolve(),
            scratch_path=scratch_dir.resolve(),
            sas_output_path=output_dir.resolve(),
            product_version=product_version,
        ),
        worker_settings=WorkerSettings(threads_per_worker=2),
        log_file=scratch_dir.resolve() / "log_sas.log",
        create_3band_los=create_3band_los,
        product_spacing_m=product_spacing_m,
    )

    runconfig_path = output_dir / "runconfig_static.yaml"
    runconfig.to_yaml(runconfig_path)
    logger.info(f"Created runconfig at {runconfig_path}")

    return runconfig_path


def _rename_outputs(runconfig: "StaticLayersRunConfig") -> None:
    """Rename generated static layer files to the OPERA product naming convention."""
    frame_id = runconfig.input_file_group.frame_id
    template = NAME_TEMPLATE.format(frame_id=frame_id)

    los_files = list(runconfig.product_path_group.output_directory.glob("los_enu.tif"))
    if los_files:
        los_files[0].rename(los_files[0].parent / f"{template}_los_enu.tif")

    dem_file = next(iter(runconfig.product_path_group.output_directory.glob("dem*tif")))
    dem_file.rename(dem_file.parent / f"{template}_dem.tif")

    mask_file = next(
        iter(runconfig.product_path_group.output_directory.glob("layover*tif"))
    )
    mask_file.rename(mask_file.parent / f"{template}_layover_shadow_mask.tif")


@click.command()
@click.option("--frame-id", required=True, type=int, help="NISAR frame ID")
@click.option(
    "--gslc-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to NISAR GSLC HDF5 file (if not provided, will download one)",
)
@click.option(
    "--frame-gpkg",
    type=click.Path(exists=True, path_type=Path),
    default="/u/aurora-r0/smirzaee/scratch/NISAR/static_ancillary_files/opera-nisar-disp-frames.gpkg",
    help="Path to frame GeoPackage file",
)
@click.option(
    "--frequency",
    default="frequencyA",
    type=click.Choice(["frequencyA", "frequencyB"]),
    help="Frequency band (default: frequencyA)",
)
@click.option(
    "--polarization",
    default="HH",
    type=click.Choice(["HH", "HV", "VH", "VV"]),
    help="Polarization (default: HH)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: F{frame_id:05d})",
)
@click.option(
    "--dem-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to DEM file (if not provided, will download)",
)
@click.option(
    "--mask-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional water/land mask file",
)
@click.option(
    "--create-3band-los/--no-create-3band-los",
    default=True,
    help="Create 3-band LOS output (East/North/Up)",
)
@click.option(
    "--product-spacing-m",
    type=int,
    default=30,
    help="Resample geometry to this spacing in meters (e.g., 90)",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(
    frame_id: int,
    gslc_file: Path | None,
    frame_gpkg: Path,
    frequency: str,
    polarization: str,
    output_dir: Path | None,
    dem_file: Path | None,
    mask_file: Path | None,
    create_3band_los: bool,
    product_spacing_m: int | None,
    debug: bool,
) -> None:
    """Run the full static layers setup and processing workflow.

    This script orchestrates:
    1. Get frame information from GeoPackage
    2. Download GSLC if not provided
    3. Download DEM (if not provided)
    4. Create runconfig
    5. Generate static layers
    6. Report outputs
    """
    from dolphin._log import setup_logging

    setup_logging(logger_name="disp_nisar", debug=debug)

    # Default output directory (use absolute paths to avoid issues)
    if output_dir is None:
        output_dir = Path(f"F{frame_id:05d}").resolve()
    else:
        output_dir = output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting static layers workflow for frame {frame_id}")
    logger.info(f"Output directory: {output_dir}")

    # Step 0: Get frame information from GeoPackage
    logger.info(f"Reading frame information from {frame_gpkg}")
    frame_info = get_frame_info(frame_id, frame_gpkg)
    logger.info(
        f"Frame {frame_id} info: Track {frame_info['track']}, "
        f"EPSG {frame_info['epsg']}, {frame_info['pass_direction']}"
    )

    # Step 1: Download GSLC if not provided
    if gslc_file is None:
        logger.info("No GSLC file provided, downloading one...")
        gslc_file = download_gslc_for_frame(
            frame_id=frame_id,
            frame_info=frame_info,
            output_dir=scratch_dir / "gslc",
        )
    else:
        # Ensure gslc_file is absolute
        gslc_file = gslc_file.resolve()
        logger.info(f"Using provided GSLC: {gslc_file}")
        validate_frame_id_matches_gslc(gslc_file, frame_info, frame_id)

    # Step 2: Download DEM if not provided
    if dem_file is None:
        logger.info("No DEM file provided, downloading...")
        dem_file = download_dem(
            gslc_file=gslc_file,
            frequency=frequency,
            polarization=polarization,
            output=scratch_dir / "dem.tif",
        )
    else:
        logger.info(f"Using provided DEM: {dem_file}")

    # Step 3: Create runconfig
    logger.info("Creating runconfig...")
    runconfig_path = create_runconfig(
        frame_id=frame_id,
        gslc_file=gslc_file,
        dem_file=dem_file,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        frame_gpkg=frame_gpkg,
        frequency=frequency,
        polarization=polarization,
        mask_file=mask_file,
        create_3band_los=create_3band_los,
        product_spacing_m=product_spacing_m,
    )

    # Step 4: Run static layers workflow
    logger.info("Running static layers workflow...")
    from disp_nisar.main_static_layers import run_static_layers
    from disp_nisar.pge_runconfig import StaticLayersRunConfig

    rc = StaticLayersRunConfig.from_yaml(runconfig_path)
    outputs = run_static_layers(rc)

    # Step 5: Report outputs
    logger.info("\n" + "=" * 60)
    logger.info("Static layers generation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info(f"  - Incidence angle: {outputs.incidence_angle_path.name}")
    logger.info(f"  - LOS East: {outputs.los_east_path.name}")
    logger.info(f"  - LOS North: {outputs.los_north_path.name}")
    logger.info(f"  - Layover/Shadow mask: {outputs.layover_shadow_mask_path.name}")
    logger.info(f"  - DEM (UTM): {outputs.dem_warped_path.name}")
    if outputs.los_combined_path:
        logger.info(f"  - LOS combined (3-band): {outputs.los_combined_path.name}")
    logger.info("=" * 60)

    # Step 6: Rename outputs to the OPERA product naming convention
    _rename_outputs(rc)


if __name__ == "__main__":
    main()

"""Main workflow module for generating static geometry layers."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import rasterio as rio
from dolphin import Bbox, PathOrStr, io
from dolphin._log import log_runtime, setup_logging
from dolphin._overviews import Resampling, create_overviews
from dolphin.utils import get_max_memory_usage
from opera_utils import CslcParseError, parse_filename
from osgeo import gdal, osr

from disp_nisar import __version__
from disp_nisar._geometry import (
    prepare_geometry_layers,
)
from disp_nisar.browse_image import make_browse_image_from_arr
from disp_nisar.pge_runconfig import StaticLayersRunConfig

logger = logging.getLogger(__name__)

gdal.UseExceptions()

__all__ = ["run_static_layers", "StaticLayersOutputs"]

DATE_TIME_METADATA_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
PRODUCT_SPECIFICATION_VERSION = "0.4"

DEM_METADATA = {
    "dem_interpolation_algorithm": "bilinear",
    "dem_egm_model": "Earth Gravitational Model 2008 (EGM2008)",
    "input_dem_source": "Copernicus GLO-30 DEM",
}

# NOTE compare layover shadow mask with NISAR offical one when is available
MASK_DESCRIPTION = (
    "Layover/Shadow Mask. Values: 0: masked (layover/shadow); 1: good data"
)


class StaticLayersOutputs(NamedTuple):
    """Output paths from static layers workflow."""

    incidence_angle_path: Path
    los_east_path: Path
    los_north_path: Path
    layover_shadow_mask_path: Path
    dem_warped_path: Path
    los_combined_path: Path | None = None


@log_runtime
def run_static_layers(
    pge_runconfig: StaticLayersRunConfig,
) -> StaticLayersOutputs:
    """Run the Static Layers workflow for a NISAR frame.

    Parameters
    ----------
    pge_runconfig : StaticLayersRunConfig
        Configuration for static layers processing.

    Returns
    -------
    StaticLayersOutputs
        Paths to all output files.

    """
    processing_start_datetime = datetime.now(tz=timezone.utc)

    setup_logging(logger_name="disp_nisar", filename=pge_runconfig.log_file)

    logger.info("Starting Static Layers workflow")
    logger.info(f"Frame ID: {pge_runconfig.input_file_group.frame_id}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")

    scratch_dir = pge_runconfig.product_path_group.scratch_path
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Extract basic config parameters
    frame_id = pge_runconfig.input_file_group.frame_id
    if frame_id is None:
        raise ValueError("frame_id is required in input_file_group")
    frequency = pge_runconfig.input_file_group.frequency
    polarization = pge_runconfig.input_file_group.polarization
    gslc_file = pge_runconfig.dynamic_ancillary_file_group.gslc_file
    dem_file = pge_runconfig.dynamic_ancillary_file_group.dem_file

    # Convert empty paths to None
    if gslc_file and str(gslc_file).strip() == "":
        gslc_file = None
    if dem_file and str(dem_file).strip() == "":
        dem_file = None

    # Auto-download GSLC if not provided or doesn't exist,
    # and frame_to_bounds is a GeoPackage
    frame_to_bounds_json = (
        pge_runconfig.static_ancillary_file_group.frame_to_bounds_json
    )
    if (
        (gslc_file is None or (gslc_file and not Path(gslc_file).exists()))
        and frame_to_bounds_json
        and Path(frame_to_bounds_json).suffix == ".gpkg"
    ):
        if gslc_file:
            logger.warning(f"GSLC file specified but not found: {gslc_file}")
        logger.info("Attempting auto-download of GSLC...")
        # Get frame info from GeoPackage
        import geopandas as gpd
        import pyproj
        from opera_utils.nisar import download_gslcs, search

        gdf = gpd.read_file(frame_to_bounds_json)
        frame_data = gdf[gdf["frame_idx"] == frame_id]
        if len(frame_data) == 0:
            raise ValueError(f"Frame {frame_id} not found in {frame_to_bounds_json}")

        row = frame_data.iloc[0]
        epsg_frame = int(row["epsg"])
        bounds_utm = Bbox(
            left=float(row["mapTopLeftX"]),
            bottom=float(row["mapBottomRightY"]),
            right=float(row["mapBottomRightX"]),
            top=float(row["mapTopLeftY"]),
        )

        # Convert to WGS84
        transformer = pyproj.Transformer.from_crs(
            f"EPSG:{epsg_frame}", "EPSG:4326", always_xy=True
        )
        lon1, lat1 = transformer.transform(bounds_utm.left, bounds_utm.bottom)
        lon2, lat2 = transformer.transform(bounds_utm.right, bounds_utm.top)
        wgs84_bbox = (
            min(lon1, lon2),
            min(lat1, lat2),
            max(lon1, lon2),
            max(lat1, lat2),
        )

        # Search and download
        results = search(bbox=wgs84_bbox, relative_orbit_number=int(row["track"]))
        if len(results) == 0:
            raise ValueError(f"No GSLC data found for frame {frame_id}")

        gslc_dir = scratch_dir / "gslc_auto"
        gslc_dir.mkdir(exist_ok=True)
        downloaded = download_gslcs(
            output_dir=gslc_dir,
            bbox=wgs84_bbox,
            start_datetime=results[0].start_datetime,
            end_datetime=results[0].end_datetime,
            max_jobs=1,
        )
        gslc_file = downloaded[0]
        logger.info(f"Auto-downloaded GSLC: {gslc_file}")
    elif gslc_file is None:
        raise ValueError(
            "gslc_file is required when frame_to_bounds_json is not a GeoPackage"
        )

    # Auto-download DEM if not provided or doesn't exist
    if dem_file is None or (dem_file and not Path(dem_file).exists()):
        if dem_file:
            logger.warning(f"DEM file specified but not found: {dem_file}")
        logger.info("Attempting auto-download of DEM...")
        import pyproj
        from osgeo import gdal

        from disp_nisar._utils import get_nisar_frame_bbox

        epsg_utm, bounds_utm = get_nisar_frame_bbox(gslc_file, frequency, polarization)

        # Convert to WGS84
        transformer = pyproj.Transformer.from_crs(
            f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True
        )
        lon1, lat1 = transformer.transform(bounds_utm.left, bounds_utm.bottom)
        lon2, lat2 = transformer.transform(bounds_utm.right, bounds_utm.top)

        wgs84_bbox = Bbox(
            left=min(lon1, lon2),
            bottom=min(lat1, lat2),
            right=max(lon1, lon2),
            top=max(lat1, lat2),
        )

        # Download DEM directly from S3
        dem_file = scratch_dir / "dem_auto.tif"
        s3_dem_vrt = "/vsis3/opera-dem/EPSG4326/EPSG4326.vrt"
        margin_deg = 5 / 111
        proj_win = (
            wgs84_bbox.left - margin_deg,
            wgs84_bbox.top + margin_deg,
            wgs84_bbox.right + margin_deg,
            wgs84_bbox.bottom - margin_deg,
        )

        translate_options = gdal.TranslateOptions(
            format="GTiff",
            projWin=proj_win,
            projWinSRS="EPSG:4326",
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )

        logger.info(f"Downloading DEM to {dem_file}")
        ds = gdal.Translate(str(dem_file), s3_dem_vrt, options=translate_options)
        if ds is None:
            raise RuntimeError("Failed to download DEM")
        ds = None
        logger.info(f"Auto-downloaded DEM: {dem_file}")

    # Extract metadata from GSLC
    with h5py.File(gslc_file, "r") as hf:
        orbit_direction = hf["/science/LSAR/identification/orbitPassDirection"][()]
        if isinstance(orbit_direction, bytes):
            orbit_direction = orbit_direction.decode("utf-8")
        track_number = int(hf["/science/LSAR/identification/trackNumber"][()])

    # Acquisition mode is the 4-digit bandwidth code in the official NISAR SDS
    # filename (e.g. "4005"); older/simulated filenames don't carry it.
    try:
        acquisition_mode = parse_filename(gslc_file)["bandwidth"]
    except (CslcParseError, KeyError):
        acquisition_mode = "unknown"

    # Get frame bounds and EPSG
    # Try to load from frame_to_bounds_json (can be JSON or GeoPackage)
    frame_to_bounds_json = (
        pge_runconfig.static_ancillary_file_group.frame_to_bounds_json
    )

    if frame_to_bounds_json:
        frame_to_bounds_path = Path(frame_to_bounds_json)

        if frame_to_bounds_path.suffix == ".gpkg":
            # Load from GeoPackage
            import geopandas as gpd

            logger.info(f"Loading frame bounds from GeoPackage: {frame_to_bounds_path}")
            gdf = gpd.read_file(frame_to_bounds_path)
            frame_data = gdf[gdf["frame_idx"] == frame_id]

            if len(frame_data) == 0:
                raise ValueError(
                    f"Frame {frame_id} not found in {frame_to_bounds_path}"
                )

            row = frame_data.iloc[0]
            epsg = int(row["epsg"])
            bounds = Bbox(
                left=float(row["mapTopLeftX"]),
                bottom=float(row["mapBottomRightY"]),
                right=float(row["mapBottomRightX"]),
                top=float(row["mapTopLeftY"]),
            )
        else:
            # Load from JSON
            import json

            with open(frame_to_bounds_json) as f:
                frame_bounds_data = json.load(f)
            frame_info = frame_bounds_data[str(frame_id)]
            epsg = frame_info["epsg"]
            bounds = Bbox(*frame_info["bbox"])
    else:
        # Fall back to extracting from GSLC metadata
        logger.info("No frame_to_bounds_json provided, extracting bounds from GSLC")
        from disp_nisar._utils import get_nisar_frame_bbox

        epsg, bounds = get_nisar_frame_bbox(gslc_file, frequency, polarization)

    logger.info(f"Frame EPSG: {epsg}, Bounds: {bounds}")

    # Create a template raster for the frame
    # This is needed for prepare_geometry_layers to define output grid
    template_raster = scratch_dir / "template_frame.tif"
    _create_template_raster(template_raster, bounds, epsg)

    # Step 1: Generate geometry layers using existing prepare_geometry_layers
    # Check if outputs already exist (in scratch or output directory)
    output_dir_path = pge_runconfig.product_path_group.output_directory
    incidence_path = scratch_dir / "incidence_angle.tif"
    los_east_path = scratch_dir / "los_east.tif"
    los_north_path = scratch_dir / "los_north.tif"
    layover_shadow_mask_path = scratch_dir / "layover_shadow_mask.tif"

    # Check if files exist in scratch dir first, if not check output dir
    geometry_files_exist = all(
        p.exists()
        for p in [
            incidence_path,
            los_east_path,
            los_north_path,
            layover_shadow_mask_path,
        ]
    )

    if not geometry_files_exist:
        # Check if they exist in output directory and copy to scratch
        output_paths = [
            output_dir_path / "incidence_angle.tif",
            output_dir_path / "los_east.tif",
            output_dir_path / "los_north.tif",
            output_dir_path / "layover_shadow_mask.tif",
        ]
        if all(p.exists() for p in output_paths):
            logger.info("Geometry layers found in output directory, copying to scratch")
            shutil.copy2(output_paths[0], incidence_path)
            shutil.copy2(output_paths[1], los_east_path)
            shutil.copy2(output_paths[2], los_north_path)
            shutil.copy2(output_paths[3], layover_shadow_mask_path)
            geometry_files_exist = True

    if geometry_files_exist:
        logger.info("Geometry layers already exist, skipping generation")
    else:
        logger.info("Generating geometry layers (incidence, LOS, layover/shadow)")
        geometry_outputs = prepare_geometry_layers(
            gslc_path=gslc_file,
            dem_path=dem_file,
            output_dir=scratch_dir,
            template_raster=template_raster,
            incidence_output_name="incidence_angle.tif",
            los_east_output_name="los_east.tif",
            los_north_output_name="los_north.tif",
            layover_shadow_output_name="layover_shadow_mask.tif",
            chunk_size=200,
            n_workers=pge_runconfig.worker_settings.threads_per_worker,
        )

        incidence_path = geometry_outputs["incidence_angle"]
        los_east_path = geometry_outputs["los_east"]
        los_north_path = geometry_outputs["los_north"]
        layover_shadow_mask_path = geometry_outputs["layover_shadow_mask"]

    # Step 2: Warp DEM to UTM at 30m resolution
    dem_warped_path = scratch_dir / "dem_warped_utm.tif"
    if dem_warped_path.exists():
        logger.info("Warped DEM already exists, skipping")
    elif (output_dir_path / "dem_warped_utm.tif").exists():
        logger.info("Warped DEM found in output directory, copying to scratch")
        shutil.copy2(output_dir_path / "dem_warped_utm.tif", dem_warped_path)
    else:
        logger.info("Warping DEM to UTM grid at 30m resolution")
        dem_warped_path = warp_dem_to_utm(
            dem_file=dem_file,
            epsg=epsg,
            bounds=bounds,
            output_dir=scratch_dir,
            spacing=30.0,
        )

    # Step 3: Optionally create 3-band LOS
    los_combined_path = None
    if pge_runconfig.create_3band_los:
        los_combined_path = scratch_dir / "los_enu.tif"
        if los_combined_path.exists():
            logger.info("3-band LOS already exists, skipping")
        elif (output_dir_path / "los_enu.tif").exists():
            logger.info("3-band LOS found in output directory, copying to scratch")
            shutil.copy2(output_dir_path / "los_enu.tif", los_combined_path)
        else:
            logger.info("Creating 3-band LOS output")
            los_combined_path = _make_3band_los(
                los_east_path=los_east_path,
                los_north_path=los_north_path,
                output_dir=scratch_dir,
            )

    # Step 4: Optionally resample geometry to product spacing
    # Note: This is typically done later in the main workflow when creating products
    # For now, we skip this step as downsample_geometry_for_products requires
    # a reference raster with the target spacing
    if pge_runconfig.product_spacing_m:
        logger.warning(
            f"product_spacing_m={pge_runconfig.product_spacing_m} specified, "
            "but resampling is not implemented in static layers workflow. "
            "Geometry will remain at native DEM resolution."
        )

    # Step 5: Add metadata
    logger.info("Adding product metadata")
    static_layers_paths = StaticLayersOutputs(
        incidence_angle_path=incidence_path,
        los_east_path=los_east_path,
        los_north_path=los_north_path,
        layover_shadow_mask_path=layover_shadow_mask_path,
        dem_warped_path=dem_warped_path,
        los_combined_path=los_combined_path,
    )

    add_product_metadata(
        static_layers_paths=static_layers_paths,
        pge_runconfig=pge_runconfig,
        frame_id=frame_id,
        orbit_direction=orbit_direction,
        frequency=frequency,
        processing_datetime=processing_start_datetime,
        track_number=track_number,
        acquisition_mode=acquisition_mode,
    )

    # Step 6: Create outputs (overviews, browse image, move to output directory)
    logger.info("Creating outputs (overviews and browse images)")
    create_outputs(
        static_layers_paths=static_layers_paths,
        output_dir=pge_runconfig.product_path_group.output_directory,
    )

    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"disp-nisar version: {__version__}")
    logger.info("Static Layers workflow complete")

    # Cleanup scratch directory
    logger.info("Cleaning up scratch directory...")
    try:
        shutil.rmtree(scratch_dir)
        logger.info(f"Deleted scratch directory: {scratch_dir}")
    except Exception as e:
        logger.warning(f"Failed to delete scratch directory: {e}")

    return static_layers_paths


def _create_template_raster(
    output_path: Path,
    bounds: Bbox,
    epsg: int,
    spacing: float = 90.0,
) -> Path:
    """Create a template raster defining the frame grid.

    Parameters
    ----------
    output_path : Path
        Path for the template raster
    bounds : Bbox
        Bounding box (left, bottom, right, top)
    epsg : int
        EPSG code for the CRS
    spacing : float
        Pixel spacing in meters

    Returns
    -------
    Path
        Path to created template raster

    """
    left, bottom, right, top = bounds
    width = int((right - left) / spacing)
    height = int((top - bottom) / spacing)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(output_path),
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform([left, spacing, 0, top, 0, -spacing])
    ds.SetProjection(srs.ExportToWkt())
    # Fill with ones (good data)
    band = ds.GetRasterBand(1)
    band.Fill(1)
    band.FlushCache()
    ds = None

    return output_path


def warp_dem_to_utm(
    dem_file: PathOrStr,
    epsg: int,
    bounds: Bbox,
    output_dir: Path,
    spacing: float = 30.0,
) -> Path:
    """Warp DEM to UTM grid at specified spacing.

    Parameters
    ----------
    dem_file : PathOrStr
        Path to input DEM (typically EPSG:4326)
    epsg : int
        Target EPSG code (UTM zone)
    bounds : Bbox
        Target bounds (left, bottom, right, top) in target CRS
    output_dir : Path
        Output directory
    spacing : float
        Output pixel spacing in meters

    Returns
    -------
    Path
        Path to warped DEM

    """
    output_path = Path(output_dir) / "dem_warped_utm.tif"

    left, bottom, right, top = bounds
    width = int((right - left) / spacing)
    height = int((top - bottom) / spacing)

    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=f"EPSG:{epsg}",
        outputBounds=bounds,
        width=width,
        height=height,
        resampleAlg="cubic",
        dstNodata=np.nan,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"],
    )

    logger.info(f"Warping DEM to {epsg} at {spacing}m spacing")
    gdal.Warp(str(output_path), str(dem_file), options=warp_options)

    return output_path


def _make_3band_los(
    los_east_path: Path,
    los_north_path: Path,
    output_dir: Path,
) -> Path:
    """Create 3-band LOS GeoTIFF (East, North, Up components).

    Parameters
    ----------
    los_east_path : Path
        Path to LOS east component
    los_north_path : Path
        Path to LOS north component
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to 3-band LOS file

    """
    output_path = Path(output_dir) / "los_enu.tif"

    # Read los_east and los_north
    with rio.open(los_east_path) as src:
        los_east = src.read(1)
        profile = src.profile.copy()

    with rio.open(los_north_path) as src:
        los_north = src.read(1)

    # Compute los_up = sqrt(1 - east^2 - north^2)
    los_up = np.sqrt(1 - los_east**2 - los_north**2)

    # Apply mantissa rounding for compression
    from dolphin.io import round_mantissa

    round_mantissa(los_east, keep_bits=9)
    round_mantissa(los_north, keep_bits=9)
    round_mantissa(los_up, keep_bits=9)

    # Write 3-band GeoTIFF
    profile.update(
        count=3,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        interleave="pixel",
    )

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(los_east, 1)
        dst.write(los_north, 2)
        dst.write(los_up, 3)
        dst.set_band_description(1, "LOS East")
        dst.set_band_description(2, "LOS North")
        dst.set_band_description(3, "LOS Up")

    return output_path


def add_product_metadata(
    static_layers_paths: StaticLayersOutputs,
    pge_runconfig: StaticLayersRunConfig,
    frame_id: int,
    orbit_direction: str,
    frequency: str,
    processing_datetime: datetime,
    track_number: int,
    acquisition_mode: str,
):
    """Add comprehensive metadata to all static layer products.

    Parameters
    ----------
    static_layers_paths : StaticLayersOutputs
        Paths to all output files
    pge_runconfig : StaticLayersRunConfig
        Run configuration
    frame_id : int
        NISAR frame ID
    orbit_direction : str
        Orbit direction (ascending/descending)
    frequency : str
        Frequency band (frequencyA/frequencyB)
    processing_datetime : datetime
        Processing start time
    track_number : int
        Track/relative orbit number of the source GSLC
    acquisition_mode : str
        Radar acquisition (bandwidth) mode code parsed from the GSLC filename

    """
    metadata = {
        "platform": "NISAR",
        "instrument_name": "NISAR L-SAR",
        "project": "OPERA",
        "institution": "NASA JPL",
        "contact_information": "opera-sds-ops@jpl.nasa.gov",
        "radar_band": "L" if frequency == "frequencyA" else "S",
        "frequency": frequency,
        "product_type": "DISP_NISAR_STATIC",
        "product_version": pge_runconfig.product_path_group.product_version,
        "product_specification_version": PRODUCT_SPECIFICATION_VERSION,
        "processing_facility": "NASA JPL",
        "ceos_analysis_ready_data_document_identifier": (
            "https://ceos.org/ard/files/PFS/SAR/v1.2/"
            "CEOS-ARD_PFS_Synthetic_Aperture_Radar_v1.2.pdf"
        ),
        "frame_id": str(frame_id),
        "track_number": str(track_number),
        "orbit_direction": orbit_direction,
        "acquisition_mode": acquisition_mode,
        "look_direction": "left",
        "processing_datetime": processing_datetime.strftime(DATE_TIME_METADATA_FORMAT),
        "disp_nisar_software_version": __version__,
        "imaging_geometry": "Geocoded",
        "product_sample_spacing": "30",
        "source_data_original_institution": "NASA TBC",
    }

    # Add DEM-specific metadata to dem_warped
    dem_metadata = metadata.copy()
    dem_metadata.update(DEM_METADATA)

    # Add metadata to all files
    for path in [
        static_layers_paths.incidence_angle_path,
        static_layers_paths.los_east_path,
        static_layers_paths.los_north_path,
        static_layers_paths.layover_shadow_mask_path,
    ]:
        _add_metadata_to_file(path, metadata)

    _add_metadata_to_file(static_layers_paths.dem_warped_path, dem_metadata)

    if static_layers_paths.los_combined_path:
        _add_metadata_to_file(static_layers_paths.los_combined_path, metadata)

    # Set descriptions
    io.set_raster_description(
        static_layers_paths.incidence_angle_path,
        description="Incidence angle at surface (degrees)",
    )
    io.set_raster_description(
        static_layers_paths.los_east_path,
        description="Line-of-sight unit vector East component",
    )
    io.set_raster_description(
        static_layers_paths.los_north_path,
        description="Line-of-sight unit vector North component",
    )
    io.set_raster_description(
        static_layers_paths.layover_shadow_mask_path,
        description=MASK_DESCRIPTION,
    )
    io.set_raster_description(
        static_layers_paths.dem_warped_path,
        description="Digital Elevation Model (meters above WGS84 ellipsoid)",
    )


def _add_metadata_to_file(file_path: Path, metadata: dict):
    """Add metadata to a GeoTIFF file using GDAL.

    Parameters
    ----------
    file_path : Path
        Path to GeoTIFF file
    metadata : dict
        Metadata key-value pairs

    """
    ds = gdal.Open(str(file_path), gdal.GA_Update)
    if ds is None:
        logger.warning(f"Could not open {file_path} for metadata writing")
        return

    ds.SetMetadata(metadata)
    ds.FlushCache()
    ds = None


def create_outputs(
    static_layers_paths: StaticLayersOutputs,
    output_dir: Path,
):
    """Create formatted outputs with overviews and browse images.

    Parameters
    ----------
    static_layers_paths : StaticLayersOutputs
        Paths to all output files
    output_dir : Path
        Final output directory

    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create overviews for all files
    file_list = [
        static_layers_paths.incidence_angle_path,
        static_layers_paths.los_east_path,
        static_layers_paths.los_north_path,
        static_layers_paths.layover_shadow_mask_path,
        static_layers_paths.dem_warped_path,
    ]
    if static_layers_paths.los_combined_path:
        file_list.append(static_layers_paths.los_combined_path)

    # Filter to only files that need processing (not already in output dir)
    files_to_process = []
    for path in file_list:
        dest_path = output_dir / path.name
        if dest_path.exists():
            logger.info(f"{path.name} already exists in output directory, skipping")
        else:
            files_to_process.append(path)

    if not files_to_process:
        logger.info("All output files already exist, skipping processing")
        return

    create_overviews(
        file_paths=files_to_process,
        levels=[4, 8, 16, 32, 64],
        resampling=Resampling.NEAREST,
    )

    # Create browse image from los_east (or 3-band if available)
    browse_filename = None
    if static_layers_paths.los_combined_path:
        browse_filename = "los_enu.browse.png"
        if not (output_dir / browse_filename).exists():
            # Use the up component (band 3) for browse
            arr = io.load_gdal(
                static_layers_paths.los_combined_path, band=3, masked=True
            )
        else:
            browse_filename = None  # Already exists
    else:
        browse_filename = "los_east.browse.png"
        if not (output_dir / browse_filename).exists():
            arr = io.load_gdal(static_layers_paths.los_east_path, masked=True)
        else:
            browse_filename = None  # Already exists

    if browse_filename:
        # Create a simple mask (1=good where data exists, 0=bad where NaN)
        mask = (~np.isnan(arr)).astype(np.uint8)

        make_browse_image_from_arr(
            output_filename=output_dir / browse_filename,
            arr=arr,
            mask=mask,
            vmin=0.5,
            vmax=1.0,
            cmap="gray",
        )

    # Move all files to output directory
    for path in files_to_process:
        dest_path = output_dir / path.name
        shutil.move(str(path), dest_path)
        logger.info(f"Moved {path.name} to {output_dir}")

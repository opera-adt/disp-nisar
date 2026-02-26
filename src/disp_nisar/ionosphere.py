# Read Ionosphere correction layers from GUNW products and invert to timeseries

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import h5py
import numpy as np
from dolphin import io
from dolphin.utils import full_suffix
from opera_utils import get_dates
from opera_utils.stitching import warp_to_match
from osgeo import gdal, osr

logger = logging.getLogger(__name__)

# NISAR GUNW HDF5 paths
GUNW_IONO_PATH_TEMPLATE = (
    "/science/LSAR/GUNW/grids/{frequency}/unwrappedInterferogram/"
    "{polarization}/ionospherePhaseScreen"
)
GUNW_IDENTIFICATION_PATH = "/science/LSAR/identification"
GUNW_GRID_PATH_TEMPLATE = "/science/LSAR/GUNW/grids/{frequency}"


def get_gunw_dates(gunw_file: Path) -> tuple:
    """Extract reference and secondary dates from a GUNW file.

    Parameters
    ----------
    gunw_file : Path
        Path to a GUNW HDF5 file.

    Returns
    -------
    tuple
        (reference_datetime, secondary_datetime) tuple.

    """
    with h5py.File(gunw_file, "r") as f:
        id_group = f[GUNW_IDENTIFICATION_PATH]
        # NISAR GUNW stores dates in identification group
        ref_date = id_group["referenceZeroDopplerStartTime"][()].decode()
        sec_date = id_group["secondaryZeroDopplerStartTime"][()].decode()
    return date.fromisoformat(ref_date[:10]), date.fromisoformat(sec_date[:10])


def _get_gunw_spatial_ref(
    gunw_file: Path,
    frequency: str,
    polarization: str,
) -> tuple[int, tuple[float, float, float, float, float, float]]:
    """Read EPSG code and GDAL geotransform from a GUNW HDF5 grid.

    Parameters
    ----------
    gunw_file : Path
        Path to a GUNW HDF5 file.
    frequency : str
        Frequency band (e.g., "frequencyA").
    polarization : str
        Polarization (e.g., "HH", "VV", "HV", "VH").

    Returns
    -------
    tuple[int, tuple]
        (epsg, gdal_geotransform) where gdal_geotransform is
        (x_origin, x_res, 0, y_origin, 0, -y_res) for a north-up grid.

    """
    grid_path = GUNW_IONO_PATH_TEMPLATE.format(
        frequency=frequency, polarization=polarization
    ).rsplit("/", 1)[0]
    with h5py.File(gunw_file, "r") as f:
        grid = f[grid_path]
        epsg = int(grid["projection"][()])
        x_spacing = float(grid["xCoordinateSpacing"][()])
        y_spacing = float(grid["yCoordinateSpacing"][()])
        x_coords = grid["xCoordinates"][:]
        y_coords = grid["yCoordinates"][:]
    # Pixel-center coords → pixel-corner origin for GDAL geotransform
    x_origin = float(x_coords.min()) - abs(x_spacing) / 2.0
    y_origin = float(y_coords.max()) + abs(y_spacing) / 2.0
    gt = (x_origin, abs(x_spacing), 0.0, y_origin, 0.0, -abs(y_spacing))
    return epsg, gt


def _create_gunw_grid_tif(
    path: Path, rows: int, cols: int, epsg: int, gt: tuple
) -> None:
    """Create a NaN-filled float32 GeoTIFF on the GUNW spatial grid."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float("nan"))
    band.Fill(float("nan"))
    ds.FlushCache()
    ds = None


def read_ionosphere_from_gunw(
    gunw_file: Path,
    frequency: str = "frequencyA",
    polarization: str = "HH",
    row_slice: slice | None = None,
) -> np.ndarray | None:
    """Read ionosphere phase screen from a single GUNW file.

    Parameters
    ----------
    gunw_file : Path
        Path to GUNW HDF5 file.
    frequency : str
        Frequency band (e.g., "frequencyA", "frequencyB").
    polarization : str
        Polarization (e.g., "HH", "VV", "HV", "VH").
    row_slice : slice, optional
        Row slice for block reading. If None, reads the full array.

    Returns
    -------
    np.ndarray or None
        2D array of ionospheric phase screen in radians, or None if not found.

    """
    iono_path = GUNW_IONO_PATH_TEMPLATE.format(
        frequency=frequency, polarization=polarization
    )
    with h5py.File(gunw_file, "r") as f:
        if iono_path not in f:
            logger.warning(
                f"Ionosphere dataset not found at {iono_path} in {gunw_file}"
            )
            return None
        if row_slice is not None:
            iono_data = f[iono_path][row_slice, :].astype(np.float32)
        else:
            iono_data = f[iono_path][()].astype(np.float32)
    return iono_data


def build_design_matrix(ifg_date_pairs: list[tuple], unique_dates: list) -> np.ndarray:
    """Build design matrix for inverting interferograms to timeseries.

    Parameters
    ----------
    ifg_date_pairs : list[tuple]
        List of (reference_date, secondary_date) tuples for each interferogram.
    unique_dates : list
        Sorted list of unique dates in the stack.

    Returns
    -------
    np.ndarray
        Design matrix A of shape (num_ifgs, num_dates - 1).
        First date is the reference, so we have num_dates - 1 unknowns.

    """
    num_ifgs = len(ifg_date_pairs)
    num_dates = len(unique_dates)

    # We solve for displacement relative to first date
    # So we have num_dates - 1 unknowns
    A = np.zeros((num_ifgs, num_dates - 1), dtype=np.float32)

    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    for i, (ref_date, sec_date) in enumerate(ifg_date_pairs):
        ref_idx = date_to_idx[ref_date]
        sec_idx = date_to_idx[sec_date]

        # Interferogram = secondary - reference
        # If reference is first date (idx=0), we skip it (implicit zero)
        if ref_idx > 0:
            A[i, ref_idx - 1] = -1
        if sec_idx > 0:
            A[i, sec_idx - 1] = 1

    return A


def invert_ifg_to_timeseries(
    ifg_stack: np.ndarray,
    design_matrix: np.ndarray,
    AtA_inv: np.ndarray | None = None,
) -> np.ndarray | None:
    """Invert interferogram stack to timeseries using least squares.

    Parameters
    ----------
    ifg_stack : np.ndarray
        Stack of interferograms, shape (num_ifgs, rows, cols).
    design_matrix : np.ndarray
        Design matrix, shape (num_ifgs, num_dates - 1).
    AtA_inv : np.ndarray, optional
        Precomputed pseudo-inverse of (A^T A), shape (num_dates-1, num_dates-1).
        If provided, skips the pinv computation (use when calling per block).

    Returns
    -------
    np.ndarray or None
        Timeseries stack, shape (num_dates - 1, rows, cols).
        Note: First date (reference) is implicitly zero.

    """
    num_ifgs, rows, cols = ifg_stack.shape
    num_unknowns = design_matrix.shape[1]

    # Reshape for vectorized least squares
    ifg_flat = ifg_stack.astype(np.float32).reshape(num_ifgs, -1)

    # Track pixels that are NaN in every interferogram — truly no-data
    all_nan_mask = ~np.isfinite(ifg_flat).any(axis=0)
    # Replace NaN with 0 so NaNs don't propagate through matrix ops
    nan_mask = ~np.isfinite(ifg_flat)
    if nan_mask.any():
        ifg_flat = ifg_flat.copy()
        ifg_flat[nan_mask] = 0.0

    # Solve least squares: A @ x = b  using normal equations: x = (A^T A)^-1 A^T b
    if AtA_inv is None:
        AtA = design_matrix.T @ design_matrix
        rank = np.linalg.matrix_rank(AtA)
        if rank < num_unknowns:
            logger.warning(
                f"Design matrix is rank-deficient (rank={rank},"
                f" expected={num_unknowns}). Inversion may be unstable."
            )
        try:
            AtA_inv = np.linalg.pinv(AtA)
        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to invert design matrix: {e}")
            return None

    Atb = design_matrix.T @ ifg_flat
    timeseries_flat = AtA_inv @ Atb

    # Restore NaN for pixels that had no valid data in any interferogram
    timeseries_flat[:, all_nan_mask] = np.nan

    # Reshape back to (num_dates - 1, rows, cols)
    timeseries = timeseries_flat.reshape(num_unknowns, rows, cols)

    return timeseries


def apply_ionosphere_corrections(
    timeseries_paths: list[Path],
    iono_correction_paths: list[Path | None],
    wavelength: float,
) -> None:
    """Subtract ionosphere corrections from displacement timeseries files in-place.

    Parameters
    ----------
    timeseries_paths : list[Path]
        Paths to displacement timeseries rasters (in meters).
    iono_correction_paths : list[Path | None]
        Paths to ionosphere correction rasters (in radians), one per timeseries date.
        None entries are skipped.
    wavelength : float
        Radar wavelength in meters, used to convert iono phase (radians) to meters.

    """
    phase_to_meters = wavelength / (4.0 * np.pi)
    for ts_file, iono_file in zip(timeseries_paths, iono_correction_paths):
        if iono_file is None:
            logger.warning("No ionosphere correction for %s, skipping", ts_file)
            continue
        units = io.get_raster_units(ts_file)
        disp = io.load_gdal(ts_file, masked=True).astype(np.float32)
        iono = io.load_gdal(iono_file, masked=True).astype(np.float32)
        corrected = np.ma.filled(disp - phase_to_meters * iono, np.nan)
        tmp_file = ts_file.with_suffix(".iono_tmp.tif")
        io.write_arr(
            arr=corrected,
            like_filename=ts_file,
            output_name=tmp_file,
            nodata=np.nan,
            units=units,
        )
        tmp_file.replace(ts_file)
        logger.debug("Applied ionosphere correction to %s", ts_file)


def read_ionosphere_phase_screen(
    gunw_files: list[Path] | None,
    output_timeseries_files: list[Path] | None,
    frequency: str = "frequencyA",
    polarization: str = "HH",
    wavelength: float | None = None,
    block_size: int = 512,
) -> list[Path] | None:
    """Read ionosphere correction layers from GUNW products and invert to timeseries.

    Read ionosphere correction layers from interferograms and
    invert them to a time series for each date, processing in spatial blocks
    to limit peak memory usage.

    Parameters
    ----------
    gunw_files : list[Path]
        List of paths to GUNW products containing interferograms.
    output_timeseries_files : list[Path]
        List of paths to output timeseries files for which ionosphere
        corrections are needed (used to match output dates and bounds).
    frequency : str
        Frequency band (e.g., "frequencyA", "frequencyB").
    polarization : str
        Polarization (e.g., "HH", "VV", "HV", "VH").
    wavelength : float, optional
        Radar wavelength in meters. If provided, the ionosphere corrections are
        applied (subtracted) to the displacement timeseries files in-place after
        the correction rasters are written.
    block_size : int, optional
        Number of rows to process per block. Default is 512.

    Returns
    -------
    list[Path] | None
        List of paths to ionosphere correction files (one per timeseries date),
        or None if ionosphere data is not available.

    """
    if gunw_files is None or len(gunw_files) == 0:
        logger.warning("No GUNW files provided for ionosphere correction")
        return None

    if output_timeseries_files is None or len(output_timeseries_files) == 0:
        logger.warning("No output timeseries files provided")
        return None

    logger.info(f"Reading ionosphere phase screens from {len(gunw_files)} GUNW files")

    iono_path = GUNW_IONO_PATH_TEMPLATE.format(
        frequency=frequency, polarization=polarization
    )

    # Pass 1: validate each GUNW file, collect date pairs and shape — no data loaded
    valid_gunw_files: list[Path] = []
    ifg_date_pairs: list[tuple] = []
    iono_shape: tuple[int, int] | None = None

    for gunw_file in sorted(gunw_files):
        try:
            ref_date, sec_date = get_gunw_dates(gunw_file)
            with h5py.File(gunw_file, "r") as f:
                if iono_path not in f:
                    logger.warning(
                        f"Ionosphere path not found in {gunw_file}, skipping"
                    )
                    continue
                shape = f[iono_path].shape
            if iono_shape is None:
                iono_shape = shape
            elif shape != iono_shape:
                logger.warning(
                    f"Inconsistent iono shape {shape} vs {iono_shape} in"
                    f" {gunw_file}, skipping"
                )
                continue
            valid_gunw_files.append(gunw_file)
            ifg_date_pairs.append((ref_date, sec_date))
            logger.debug(f"Validated {gunw_file}: {ref_date} -> {sec_date}")
        except Exception as e:
            logger.warning(f"Failed to validate {gunw_file}: {e}")
            continue

    if not valid_gunw_files:
        logger.warning("No valid ionosphere data found in GUNW files")
        return None

    assert iono_shape is not None
    rows, cols = iono_shape

    # Get unique dates and build design matrix
    all_dates: set = set()
    for ref, sec in ifg_date_pairs:
        all_dates.add(ref)
        all_dates.add(sec)
    unique_dates = sorted(all_dates)
    logger.info(f"Found {len(unique_dates)} unique dates in interferogram network")

    design_matrix = build_design_matrix(ifg_date_pairs, unique_dates)

    # Precompute AtA_inv once — it is pixel-independent (small matrix)
    AtA = design_matrix.T @ design_matrix
    num_unknowns = design_matrix.shape[1]
    rank = np.linalg.matrix_rank(AtA)
    if rank < num_unknowns:
        logger.warning(
            f"Design matrix is rank-deficient (rank={rank}, expected={num_unknowns}). "
            "Inversion may be unstable."
        )
    AtA_inv = np.linalg.pinv(AtA)

    # Resolve output date indices and output paths before opening any file handles
    output_dir = output_timeseries_files[0].parent / "ionosphere"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the GUNW spatial reference once — used to georeference intermediate files
    gunw_epsg, gunw_gt = _get_gunw_spatial_ref(
        valid_gunw_files[0], frequency, polarization
    )

    # Build (out_file, tmp_path, out_path, ts_idx) for every requested output date.
    # tmp_path is a GeoTIFF on the GUNW grid; out_path is the final
    # timeseries-grid file.
    file_triples: list[tuple[Path, Path, Path, int | None]] = []
    for out_file in output_timeseries_files:
        out_dates = get_dates(out_file)
        sec_date = out_dates[1] if len(out_dates) >= 2 else out_dates[0]
        sec_date_only = sec_date.date() if hasattr(sec_date, "date") else sec_date
        ts_idx = next(
            (i for i, ud in enumerate(unique_dates) if sec_date_only == ud), None
        )
        out_name = out_file.name.replace(full_suffix(out_file), "_ionosphere.tif")
        tmp_name = out_file.name.replace(full_suffix(out_file), "_ionosphere_gunw.tif")
        file_triples.append(
            (out_file, output_dir / tmp_name, output_dir / out_name, ts_idx)
        )

    # Create temporary GeoTIFFs on the GUNW grid (NaN-filled, correct GUNW georef)
    logger.info("Inverting interferogram ionosphere to timeseries block by block")
    created_tmp_paths: set[Path] = set()
    for out_file, tmp_path, _out_path, ts_idx in file_triples:
        if ts_idx is None:
            logger.warning(
                "No matching ionosphere date for %s, skipping", out_file.name
            )
            continue
        _create_gunw_grid_tif(tmp_path, rows, cols, gunw_epsg, gunw_gt)
        created_tmp_paths.add(tmp_path)

    # Process in row-blocks: read → invert → write to GUNW-grid temp files
    for row_start in range(0, rows, block_size):
        row_end = min(row_start + block_size, rows)
        blk_rows = row_end - row_start
        row_sl = slice(row_start, row_end)

        # Read this block from every valid GUNW file
        iono_blocks = []
        skip_block = False
        for gf in valid_gunw_files:
            block = read_ionosphere_from_gunw(
                gf, frequency, polarization, row_slice=row_sl
            )
            if block is None:
                logger.error(
                    "Failed to read ionosphere block rows [%d:%d] from %s,"
                    " skipping block",
                    row_start,
                    row_end,
                    gf,
                )
                skip_block = True
                break
            iono_blocks.append(block)
        if skip_block:
            continue
        ifg_block = np.stack(iono_blocks, axis=0)  # (num_ifgs, blk_rows, cols)

        # Invert with precomputed AtA_inv — no matrix decomposition per block
        ts_block = invert_ifg_to_timeseries(ifg_block, design_matrix, AtA_inv=AtA_inv)
        if ts_block is None:
            logger.error(
                "Inversion failed for block [%d:%d], skipping", row_start, row_end
            )
            continue

        # Prepend zeros for the reference date → (num_dates, blk_rows, cols)
        zeros = np.zeros((1, blk_rows, cols), dtype=ts_block.dtype)
        ts_full_block = np.concatenate([zeros, ts_block], axis=0)

        for _out_file, tmp_path, _out_path, ts_idx in file_triples:
            if ts_idx is None or tmp_path not in created_tmp_paths:
                continue
            io.write_block(
                ts_full_block[ts_idx].astype(np.float32),
                tmp_path,
                row_start=row_start,
                col_start=0,
            )

    # Reproject each GUNW-grid temp file onto the timeseries grid, then remove temp
    logger.info("Reprojecting ionosphere corrections to timeseries grid")
    created_out_paths: set[Path] = set()
    for out_file, tmp_path, out_path, ts_idx in file_triples:
        if ts_idx is None or tmp_path not in created_tmp_paths:
            continue
        warp_to_match(
            input_file=tmp_path,
            match_file=out_file,
            output_file=out_path,
            resample_alg="bilinear",
        )
        tmp_path.unlink(missing_ok=True)
        created_out_paths.add(out_path)
        logger.debug(
            "Reprojected ionosphere correction to timeseries grid: %s", out_path
        )

    # Collect results preserving 1-to-1 correspondence with output_timeseries_files
    output_paths: list[Path | None] = []
    for _out_file, _tmp_path, out_path, ts_idx in file_triples:
        if ts_idx is None:
            output_paths.append(None)
        else:
            output_paths.append(out_path)
            logger.debug("Wrote ionosphere correction: %s", out_path)

    valid_paths = [p for p in output_paths if p is not None]
    if not valid_paths:
        logger.warning("No ionosphere correction files were created")
        return None

    logger.info(f"Created {len(valid_paths)} ionosphere correction files")

    if wavelength is not None:
        logger.info("Applying ionosphere corrections to displacement timeseries")
        apply_ionosphere_corrections(output_timeseries_files, output_paths, wavelength)

    return valid_paths

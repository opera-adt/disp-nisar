# Read Ionosphere correction layers from GUNW products and invert to timeseries

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
from dolphin import io
from dolphin.utils import full_suffix
from opera_utils import get_dates

logger = logging.getLogger(__name__)

# NISAR GUNW HDF5 paths
GUNW_IONO_PATH_TEMPLATE = (
    "/science/LSAR/GUNW/grids/{frequency}/unwrappedInterferogram/"
    "{polarization}/ionospherePhaseScreen"
)
GUNW_IDENTIFICATION_PATH = "/science/LSAR/identification"


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
    return ref_date, sec_date


def read_ionosphere_from_gunw(
    gunw_file: Path,
    frequency: str = "frequencyA",
    polarization: str = "HH",
) -> np.ndarray:
    """Read ionosphere phase screen from a single GUNW file.

    Parameters
    ----------
    gunw_file : Path
        Path to GUNW HDF5 file.
    frequency : str
        Frequency band (e.g., "frequencyA", "frequencyB").
    polarization : str
        Polarization (e.g., "HH", "VV", "HV", "VH").

    Returns
    -------
    np.ndarray
        2D array of ionospheric phase screen in radians.

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
        iono_data = f[iono_path][()]
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
    ifg_stack: np.ndarray, design_matrix: np.ndarray
) -> np.ndarray:
    """Invert interferogram stack to timeseries using least squares.

    Parameters
    ----------
    ifg_stack : np.ndarray
        Stack of interferograms, shape (num_ifgs, rows, cols).
    design_matrix : np.ndarray
        Design matrix, shape (num_ifgs, num_dates - 1).

    Returns
    -------
    np.ndarray
        Timeseries stack, shape (num_dates - 1, rows, cols).
        Note: First date (reference) is implicitly zero.

    """
    num_ifgs, rows, cols = ifg_stack.shape
    num_unknowns = design_matrix.shape[1]

    # Reshape for vectorized least squares
    ifg_flat = ifg_stack.reshape(num_ifgs, -1)

    # Solve least squares: A @ x = b
    # Using normal equations: x = (A^T A)^-1 A^T b
    AtA = design_matrix.T @ design_matrix
    Atb = design_matrix.T @ ifg_flat

    # Check if the design matrix is full rank
    rank = np.linalg.matrix_rank(AtA)
    if rank < num_unknowns:
        logger.warning(
            f"Design matrix is rank-deficient (rank={rank}, expected={num_unknowns}). "
            "Inversion may be unstable."
        )

    try:
        # Use pseudo-inverse for stability
        AtA_inv = np.linalg.pinv(AtA)
        timeseries_flat = AtA_inv @ Atb
    except np.linalg.LinAlgError as e:
        logger.error(f"Failed to invert design matrix: {e}")
        return None

    # Reshape back to (num_dates - 1, rows, cols)
    timeseries = timeseries_flat.reshape(num_unknowns, rows, cols)

    return timeseries


def read_ionosphere_phase_screen(
    gunw_files: list[Path] | None,
    output_timeseries_files: list[Path] | None,
    frequency: str = "frequencyA",
    polarization: str = "HH",
) -> list[Path] | None:
    """Read ionosphere correction layers from GUNW products and invert to timeseries.

    Read ionosphere correction layers from interferograms and
    invert them to a time series for each date.

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

    # Extract dates from each GUNW file
    ifg_date_pairs = []
    iono_data_list = []

    for gunw_file in sorted(gunw_files):
        try:
            ref_date, sec_date = get_gunw_dates(gunw_file)
            iono_data = read_ionosphere_from_gunw(gunw_file, frequency, polarization)

            if iono_data is None:
                logger.warning(f"Skipping {gunw_file} - no ionosphere data")
                continue

            ifg_date_pairs.append((ref_date, sec_date))
            iono_data_list.append(iono_data)
            logger.debug(f"Read ionosphere from {gunw_file}: {ref_date} -> {sec_date}")

        except Exception as e:
            logger.warning(f"Failed to read ionosphere from {gunw_file}: {e}")
            continue

    if len(iono_data_list) == 0:
        logger.warning("No valid ionosphere data found in GUNW files")
        return None

    # Get unique dates and sort them
    all_dates = set()
    for ref, sec in ifg_date_pairs:
        all_dates.add(ref)
        all_dates.add(sec)
    unique_dates = sorted(all_dates)

    logger.info(f"Found {len(unique_dates)} unique dates in interferogram network")

    # Build design matrix
    design_matrix = build_design_matrix(ifg_date_pairs, unique_dates)

    # Stack ionosphere data
    ifg_stack = np.stack(iono_data_list, axis=0)

    # Invert to timeseries
    logger.info("Inverting interferogram ionosphere to timeseries")
    timeseries = invert_ifg_to_timeseries(ifg_stack, design_matrix)

    if timeseries is None:
        logger.error("Failed to invert ionosphere to timeseries")
        return None

    # Prepend zeros for first date (reference)
    zeros = np.zeros((1,) + timeseries.shape[1:], dtype=timeseries.dtype)
    timeseries_full = np.concatenate([zeros, timeseries], axis=0)

    # Write output files matching timeseries dates
    output_dir = output_timeseries_files[0].parent / "ionosphere"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path | None] = []
    [get_dates(f) for f in output_timeseries_files]

    # Match timeseries dates to output dates and write files
    for out_file in output_timeseries_files:
        out_dates = get_dates(out_file)
        # Get the secondary date (index 1) for matching
        if len(out_dates) >= 2:
            sec_date = out_dates[1]
        else:
            sec_date = out_dates[0]

        # Find matching timeseries index
        try:
            # Convert to string for comparison if needed
            sec_date_str = str(sec_date)
            ts_idx = None
            for i, ud in enumerate(unique_dates):
                if sec_date_str in str(ud):
                    ts_idx = i
                    break

            if ts_idx is None:
                logger.warning(f"No matching ionosphere for date {sec_date}")
                output_paths.append(None)
                continue

            # Get ionosphere data for this date
            iono_ts = timeseries_full[ts_idx]

            # Write output file
            out_name = out_file.name.replace(full_suffix(out_file), "_ionosphere.tif")
            out_path = output_dir / out_name

            # Write using the timeseries file as template for georeferencing
            io.write_arr(
                arr=iono_ts.astype(np.float32),
                like_filename=out_file,
                output_name=out_path,
                nodata=np.nan,
            )

            output_paths.append(out_path)
            logger.debug(f"Wrote ionosphere correction: {out_path}")

        except Exception as e:
            logger.warning(f"Failed to write ionosphere for {out_file}: {e}")
            output_paths.append(None)

    # Filter out None values
    valid_paths = [p for p in output_paths if p is not None]

    if len(valid_paths) == 0:
        logger.warning("No ionosphere correction files were created")
        return None

    logger.info(f"Created {len(valid_paths)} ionosphere correction files")
    return valid_paths

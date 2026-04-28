"""GUNW-based ionosphere correction workflow.

Reads ionosphere phase screens from NISAR GUNW HDF5 products, inverts them
to a timeseries via least squares, reprojects to the displacement grid, and
optionally applies corrections to displacement timeseries files.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path

import h5py
import numpy as np
from dolphin import io
from dolphin.io import round_mantissa
from dolphin.utils import full_suffix
from opera_utils import get_dates
from opera_utils._utils import format_nc_filename
from opera_utils.stitching import warp_to_match

from .inversion import build_design_matrix, invert_ifg_to_timeseries

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NISAR GUNW HDF5 path templates
# ---------------------------------------------------------------------------

GUNW_IONO_PATH_TEMPLATE = (
    "/science/LSAR/GUNW/grids/{frequency}/unwrappedInterferogram/"
    "{polarization}/ionospherePhaseScreen"
)
GUNW_IDENTIFICATION_PATH = "/science/LSAR/identification"


# ---------------------------------------------------------------------------
# GUNW I/O helpers
# ---------------------------------------------------------------------------


def get_gunw_dates(gunw_file: Path) -> tuple[date, date]:
    """Extract reference and secondary dates from a GUNW HDF5 file.

    Parameters
    ----------
    gunw_file : Path
        Path to a NISAR GUNW HDF5 file.

    Returns
    -------
    tuple[date, date]
        ``(reference_date, secondary_date)`` as ``datetime.date`` objects.

    """
    with h5py.File(gunw_file, "r") as f:
        id_group = f[GUNW_IDENTIFICATION_PATH]
        ref_str = id_group["referenceZeroDopplerStartTime"][()].decode()
        sec_str = id_group["secondaryZeroDopplerStartTime"][()].decode()
    return date.fromisoformat(ref_str[:10]), date.fromisoformat(sec_str[:10])


def read_ionosphere_from_gunw(
    gunw_file: Path,
    frequency: str = "frequencyA",
    polarization: str = "HH",
    row_slice: slice | None = None,
) -> np.ndarray | None:
    """Read the ionosphere phase screen from a single GUNW HDF5 file.

    Parameters
    ----------
    gunw_file : Path
        Path to a NISAR GUNW HDF5 file.
    frequency : str
        Frequency band, e.g. ``"frequencyA"`` or ``"frequencyB"``.
    polarization : str
        Polarization, e.g. ``"HH"``, ``"VV"``, ``"HV"``, ``"VH"``.
    row_slice : slice, optional
        Row slice for block reading.  If None, reads the full 2D array.

    Returns
    -------
    np.ndarray or None
        2D float32 array of ionospheric phase screen in radians,
        or None if the dataset is not found in the file.

    """
    iono_path = GUNW_IONO_PATH_TEMPLATE.format(
        frequency=frequency, polarization=polarization
    )
    with h5py.File(gunw_file, "r") as f:
        if iono_path not in f:
            logger.warning(
                "Ionosphere dataset not found at %s in %s", iono_path, gunw_file
            )
            return None
        data = f[iono_path][row_slice, :] if row_slice is not None else f[iono_path][()]
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Apply correction helpers
# ---------------------------------------------------------------------------


def _apply_one(
    ts_file: Path,
    iono_file: Path,
    suffix: str = "iono_corrected",
    output_dir: Path | None = None,
) -> Path:
    """Apply ionosphere correction to a single displacement timeseries file.

    Parameters
    ----------
    ts_file : Path
        Path to a displacement timeseries raster.
    iono_file : Path
        Path to an ionosphere correction raster (same units as ``ts_file``).
    suffix : str, optional
        Suffix appended to the ``ts_file`` stem for the output filename.
        Default is ``"iono_corrected"``.
    output_dir : Path, optional
        Directory for the corrected output file.  Defaults to the same
        directory as ``ts_file``.

    Returns
    -------
    Path
        Path to the output corrected raster
        (``<output_dir>/<stem>.<suffix>.tif``).

    """
    units = io.get_raster_units(ts_file)
    disp = io.load_gdal(ts_file, masked=True).astype(np.float32)
    iono = io.load_gdal(iono_file, masked=True).astype(np.float32)
    corrected = np.ma.filled(disp - iono, np.nan)
    round_mantissa(corrected, keep_bits=12)
    stem = ts_file.with_suffix("").name
    out_dir = output_dir if output_dir is not None else ts_file.parent
    out_file = out_dir / f"{stem}.{suffix}.tif"
    io.write_arr(
        arr=corrected,
        like_filename=ts_file,
        output_name=out_file,
        nodata=np.nan,
        units=units,
    )
    logger.debug("Applied ionosphere correction → %s", out_file.name)
    return out_file


def apply_ionosphere_corrections(
    timeseries_paths: list[Path],
    iono_correction_paths: list[Path | None],
    output_dir: Path | None = None,
    n_workers: int = 4,
) -> list[Path]:
    """Subtract ionosphere corrections from displacement timeseries files.

    Each pair is processed in parallel.  Files without a matching correction
    (``None`` entries) are skipped with a warning.

    Parameters
    ----------
    timeseries_paths : list[Path]
        Paths to displacement timeseries rasters.
    iono_correction_paths : list[Path | None]
        Paths to ionosphere correction rasters (same units as timeseries),
        one per date.  ``None`` entries are skipped.
    output_dir : Path, optional
        Directory for corrected output files.  Defaults to each file's own
        directory.
    n_workers : int
        Number of parallel threads.  Default 4.

    Returns
    -------
    list[Path]
        Paths to corrected output files (``<stem>.iono_corrected.tif``),
        sorted.  Input files are not modified.

    """
    pairs = [
        (ts, iono)
        for ts, iono in zip(timeseries_paths, iono_correction_paths)
        if iono is not None
    ]
    skipped = len(timeseries_paths) - len(pairs)
    if skipped:
        logger.warning("Skipping %d files with no ionosphere correction", skipped)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = {
            exe.submit(_apply_one, ts, iono, output_dir=output_dir): ts
            for ts, iono in pairs
        }
        for fut in futures:
            try:
                output_paths.append(fut.result())
            except Exception as e:
                logger.error("Failed for %s: %s", futures[fut].name, e)

    return sorted(output_paths)


# ---------------------------------------------------------------------------
# Main GUNW workflow
# ---------------------------------------------------------------------------


def get_ionosphere_phase_screen(
    gunw_files: list[Path] | None,
    output_timeseries_files: list[Path] | None,
    output_dir: Path | None = None,
    frequency: str = "frequencyA",
    polarization: str = "HH",
    wavelength: float | None = None,
    block_size: int = 512,
    reference_point_file: Path | None = None,
    n_workers: int = 4,
    mask_file: Path | None = None,
) -> list[Path] | None:
    """Read ionosphere phase screens from GUNW products and invert to timeseries.

    Processes in spatial blocks to limit peak memory usage.  Outputs one
    ionosphere correction raster per timeseries date, reprojected to the
    timeseries grid.

    Parameters
    ----------
    gunw_files : list[Path] or None
        Paths to NISAR GUNW HDF5 products.
    output_timeseries_files : list[Path] or None
        Paths to output displacement timeseries rasters.  Files with
        ``"ionosphere"`` in their parent path are automatically excluded.
    output_dir : Path, optional
        Directory for ionosphere correction output files.  Defaults to an
        ``ionosphere/`` subdirectory next to the first timeseries file.
    frequency : str
        Frequency band, e.g. ``"frequencyA"`` or ``"frequencyB"``.
    polarization : str
        Polarization, e.g. ``"HH"``, ``"VV"``, ``"HV"``, ``"VH"``.
    wavelength : float, optional
        Radar wavelength in metres.  If provided, each ionosphere layer is
        converted from radians to metres after inversion:
        ``metres = -1 * radians * wavelength / (4π)``.
    block_size : int
        Number of rows to process per block.  Default 512.
    reference_point_file : Path, optional
        Path to ``reference_point.txt`` containing ``row,col`` in timeseries
        grid coordinates.  If provided, each ionosphere layer is re-referenced
        to that pixel (``iono[ref_row, ref_col]`` is subtracted).
    n_workers : int
        Number of parallel threads for reading and warping.  Default 4.
    mask_file : Path, optional
        Raster mask (1 = valid, 0 = nodata) on the timeseries grid.
        Reprojected to the GUNW grid; used to skip fully-masked blocks and
        to zero out invalid pixels before inversion.

    Returns
    -------
    list[Path] or None
        Paths to ionosphere correction files (one per timeseries date),
        or None if no ionosphere data is available.

    """
    if not gunw_files:
        logger.warning("No GUNW files provided for ionosphere correction")
        return None

    if not output_timeseries_files:
        logger.warning("No output timeseries files provided")
        return None

    ts_files = [
        f for f in output_timeseries_files if "ionosphere" not in list(Path(f).parts)
    ]
    if not ts_files:
        logger.warning("All timeseries files were filtered out (ionosphere paths)")
        return None

    logger.info("Reading ionosphere phase screens from %d GUNW files", len(gunw_files))

    iono_path = GUNW_IONO_PATH_TEMPLATE.format(
        frequency=frequency, polarization=polarization
    )

    # Pass 1: validate GUNW files — no data loaded yet
    valid_gunw: list[Path] = []
    ifg_date_pairs: list[tuple] = []
    iono_shape: tuple[int, int] | None = None

    for gf in sorted(gunw_files):
        try:
            ref_date, sec_date = get_gunw_dates(gf)
            with h5py.File(gf, "r") as f:
                if iono_path not in f:
                    logger.warning("Ionosphere path not found in %s, skipping", gf)
                    continue
                shape = f[iono_path].shape
            if iono_shape is None:
                iono_shape = shape
            elif shape != iono_shape:
                logger.warning(
                    "Inconsistent iono shape %s vs %s in %s, skipping",
                    shape,
                    iono_shape,
                    gf,
                )
                continue
            valid_gunw.append(gf)
            ifg_date_pairs.append((ref_date, sec_date))
        except Exception as e:
            logger.warning("Failed to validate %s: %s", gf, e)

    if not valid_gunw:
        logger.warning("No valid ionosphere data found in GUNW files")
        return None

    assert iono_shape is not None
    rows, cols = iono_shape

    unique_dates = sorted({d for pair in ifg_date_pairs for d in pair})
    logger.info("Found %d unique dates in interferogram network", len(unique_dates))

    # Check that GUNW network covers all timeseries dates
    ts_dates: set[date] = set()
    for f in ts_files:
        out_dates = get_dates(f)
        sec = out_dates[1] if len(out_dates) >= 2 else out_dates[0]
        ts_dates.add(sec.date() if hasattr(sec, "date") else sec)
    missing_dates = ts_dates - set(unique_dates)
    if missing_dates:
        raise ValueError(
            "GUNW files do not cover all timeseries dates. "
            f"Missing: {sorted(missing_dates)}"
        )

    # Check that the interferogram network is connected (union-find)
    parent = {d: d for d in unique_dates}

    def _find(x: date) -> date:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ref, sec in ifg_date_pairs:
        parent[_find(ref)] = _find(sec)

    roots = {_find(d) for d in unique_dates}
    if len(roots) > 1:
        components: dict[date, list[date]] = {}
        for d in unique_dates:
            components.setdefault(_find(d), []).append(d)
        raise ValueError(
            f"Interferogram network is disconnected ({len(roots)} components). "
            f"Isolated groups: {sorted(components.values(), key=len)}. "
            "Add bridging interferograms."
        )

    design_matrix = build_design_matrix(ifg_date_pairs, unique_dates)
    AtA_inv = np.linalg.pinv(design_matrix.T @ design_matrix)

    if not output_dir:
        output_dir = ts_files[0].parent / "ionosphere"
    output_dir.mkdir(parents=True, exist_ok=True)

    gunw_iono_gdal_path = format_nc_filename(valid_gunw[0], iono_path)

    # Reproject mask to GUNW grid for block-level skipping and masking
    gunw_mask: np.ndarray | None = None
    if mask_file is not None:
        try:
            tmp_mask = output_dir / "_tmp_mask_gunw.tif"
            warp_to_match(
                input_file=mask_file,
                match_file=gunw_iono_gdal_path,
                output_file=tmp_mask,
                resample_alg="nearest",
            )
            gunw_mask = io.load_gdal(tmp_mask).astype(bool)
            tmp_mask.unlink(missing_ok=True)
            logger.info("Loaded GUNW-grid mask from %s", mask_file.name)
        except Exception as e:
            logger.warning("Failed to load mask_file: %s", e)

    # Reference point for re-referencing
    ref_row: int | None = None
    ref_col: int | None = None
    if reference_point_file is not None:
        try:
            row_col = Path(reference_point_file).read_text().strip().split(",")
            ref_row, ref_col = int(row_col[0]), int(row_col[1])
            logger.info("Re-referencing ionosphere to pixel (%d, %d)", ref_row, ref_col)
        except Exception as e:
            logger.warning("Failed to read reference_point_file: %s", e)

    # Map each timeseries file → (out_file, tmp_path, out_path, ts_idx)
    file_triples: list[tuple[Path, Path, Path, int | None]] = []
    for idx, out_file in enumerate(ts_files):
        out_dates = get_dates(out_file)
        sec = out_dates[1] if len(out_dates) >= 2 else out_dates[0]
        sec_d = sec.date() if hasattr(sec, "date") else sec
        ts_idx = next((i for i, ud in enumerate(unique_dates) if sec_d == ud), None)
        tmp_path = output_dir / f"_tmp_ionosphere_{idx:04d}.tif"
        out_name = out_file.name.replace(full_suffix(out_file), "_ionosphere.tif")
        file_triples.append((out_file, tmp_path, output_dir / out_name, ts_idx))

    # Create empty GUNW-grid temp files
    created_tmp: set[Path] = set()
    for out_file, tmp_path, _out_path, ts_idx in file_triples:
        if ts_idx is None:
            logger.warning(
                "No matching ionosphere date for %s, skipping", out_file.name
            )
            continue
        io.write_arr(
            arr=None,
            like_filename=gunw_iono_gdal_path,
            output_name=tmp_path,
            dtype="float32",
            nodata=np.nan,
            units="radians",
        )
        created_tmp.add(tmp_path)

    # Block-by-block: parallel read → mask → invert → parallel write
    logger.info("Inverting interferogram ionosphere to timeseries block by block")
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        for row_start in range(0, rows, block_size):
            row_end = min(row_start + block_size, rows)
            blk_rows = row_end - row_start
            row_sl = slice(row_start, row_end)

            if gunw_mask is not None and not gunw_mask[row_start:row_end, :].any():
                logger.debug("Skipping fully masked block [%d:%d]", row_start, row_end)
                continue

            futs = [
                exe.submit(
                    read_ionosphere_from_gunw, gf, frequency, polarization, row_sl
                )
                for gf in valid_gunw
            ]
            iono_blocks = []
            skip = False
            for gf, fut in zip(valid_gunw, futs):
                blk = fut.result()
                if blk is None:
                    logger.error(
                        "Failed to read block [%d:%d] from %s",
                        row_start,
                        row_end,
                        gf,
                    )
                    skip = True
                    break
                iono_blocks.append(blk)
            if skip:
                continue

            ifg_block = np.stack(iono_blocks, axis=0)

            if gunw_mask is not None:
                ifg_block[:, ~gunw_mask[row_start:row_end, :]] = np.nan

            ts_block = invert_ifg_to_timeseries(
                ifg_block, design_matrix, AtA_inv=AtA_inv
            )
            if ts_block is None:
                logger.error("Inversion failed for block [%d:%d]", row_start, row_end)
                continue

            zeros = np.zeros((1, blk_rows, cols), dtype=ts_block.dtype)
            ts_full = np.concatenate([zeros, ts_block], axis=0)

            write_futs = [
                exe.submit(
                    io.write_block,
                    ts_full[ts_idx].astype(np.float32),
                    tmp_path,
                    row_start,
                    0,
                )
                for _out_file, tmp_path, _out_path, ts_idx in file_triples
                if ts_idx is not None and tmp_path in created_tmp
            ]
            for wf in write_futs:
                wf.result()

    def _warp_and_postprocess(out_file: Path, tmp_path: Path, out_path: Path) -> Path:
        """Reproject, mask, re-reference, and unit-convert one ionosphere layer.

        Parameters
        ----------
        out_file : Path
            Timeseries raster on the output grid, used as the warp target and
            to derive the valid-pixel mask.
        tmp_path : Path
            Temporary ionosphere raster on the GUNW grid (deleted after warp).
        out_path : Path
            Destination path for the final ionosphere correction raster.

        Returns
        -------
        Path
            Path to the written ionosphere correction file (``out_path``).

        Notes
        -----
        Captures ``ref_row``, ``ref_col``, and ``wavelength`` from the enclosing
        ``read_ionosphere_phase_screen`` scope.  Re-referencing is performed
        before unit conversion so the reference-pixel value is read from the
        radians array.

        """
        out_path.unlink(missing_ok=True)
        warp_to_match(
            input_file=tmp_path,
            match_file=out_file,
            output_file=out_path,
            resample_alg="bilinear",
        )
        tmp_path.unlink(missing_ok=True)

        iono = io.load_gdal(out_path).astype(np.float32)

        # Read reference value BEFORE masking — the timeseries ref pixel is
        # always 0 by construction, so masking (ts == 0) would NaN it out.
        ref_val = (
            iono[ref_row, ref_col]
            if ref_row is not None and ref_col is not None
            else np.nan
        )

        ts_data = io.load_gdal(out_file)
        iono[~np.isfinite(ts_data)] = np.nan

        if ref_row is not None and ref_col is not None:
            if np.isfinite(ref_val):
                iono -= ref_val
            else:
                logger.warning(
                    "Reference pixel (%d, %d) is NaN in %s, skipping re-referencing",
                    ref_row,
                    ref_col,
                    out_path.name,
                )

        if wavelength is not None:
            iono *= -1 * wavelength / (4.0 * np.pi)

        round_mantissa(iono, keep_bits=12)
        io.write_arr(
            arr=iono,
            like_filename=out_path,
            output_name=out_path,
            nodata=np.nan,
            units="meters" if wavelength is not None else "radians",
        )
        logger.debug("Wrote ionosphere correction: %s", out_path.name)
        return out_path

    logger.info("Reprojecting ionosphere corrections to timeseries grid")
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        warp_futs: dict[Future[Path], Path] = {}
        for out_file, tmp_path, out_path, ts_idx in file_triples:
            if ts_idx is not None and tmp_path in created_tmp:
                wpf: Future[Path] = exe.submit(  # type: ignore[assignment]
                    _warp_and_postprocess, out_file, tmp_path, out_path
                )
                warp_futs[wpf] = out_path
        for wpf in warp_futs:
            try:
                wpf.result()
            except Exception as e:
                logger.error("Warp failed for %s: %s", warp_futs[wpf].name, e)

    output_paths: list[Path | None] = [
        out_path if ts_idx is not None else None
        for _out_file, _tmp_path, out_path, ts_idx in file_triples
    ]
    valid_paths = [p for p in output_paths if p is not None]
    if not valid_paths:
        logger.warning("No ionosphere correction files were created")
        return None

    logger.info("Created %d ionosphere correction files", len(valid_paths))
    return valid_paths

"""Shared utilities for ionosphere estimation.

Covers unit conversions, smoothing, spatial I/O helpers, and grid alignment.
Used by both the GUNW and GSLC split-spectrum workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from dolphin import io
from rasterio.enums import Resampling
from rasterio.warp import reproject as rio_reproject
from rasterio.windows import Window, bounds, from_bounds, transform
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_LIGHT = 299_792_458.0  # speed of light, m/s


# ---------------------------------------------------------------------------
# Tiff options
# ---------------------------------------------------------------------------

WORKING_TIFF_OPTIONS = [
    "COMPRESS=LZW",
    "TILED=YES",
    "BLOCKXSIZE=256",
    "BLOCKYSIZE=256",
    "BIGTIFF=IF_SAFER",
]

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def freq_to_wavelength(freq_hz: float) -> float:
    """Convert center frequency (Hz) to wavelength (m).

    Parameters
    ----------
    freq_hz : float
        Center frequency in Hz.

    Returns
    -------
    float
        Wavelength in metres.

    """
    return C_LIGHT / freq_hz


def disp_to_phase(disp_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Convert LOS displacement (m) to phase (rad).

    Sign convention: positive displacement away from sensor → negative phase.

    Parameters
    ----------
    disp_m : np.ndarray
        LOS displacement in metres.
    wavelength_m : float
        Radar wavelength in metres.

    Returns
    -------
    np.ndarray
        Phase in radians, same shape as ``disp_m``.

    """
    return (-4.0 * np.pi / wavelength_m) * disp_m


def phase_to_disp(phase_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Convert phase (rad) to LOS displacement (m).

    Parameters
    ----------
    phase_rad : np.ndarray
        Phase in radians.
    wavelength_m : float
        Radar wavelength in metres.

    Returns
    -------
    np.ndarray
        LOS displacement in metres, same shape as ``phase_rad``.

    """
    return (-wavelength_m / (4.0 * np.pi)) * phase_rad


def coh_to_phase_sigma(coherence: np.ndarray, n_looks: int = 1) -> np.ndarray:
    """Convert coherence to phase standard deviation.

    Uses the Cramer-Rao approximation:
    ``sigma = sqrt(1 - coh^2) / (coh * sqrt(2 * n_looks))``.
    Safe against ``coh == 0`` (returns 0 there).

    Parameters
    ----------
    coherence : np.ndarray
        Coherence values in [0, 1].
    n_looks : int
        Number of looks used in coherence estimation.

    Returns
    -------
    np.ndarray
        Phase standard deviation in radians, same shape as ``coherence``.

    """
    coh = np.clip(coherence, 0, 1).astype(np.float32)
    return np.divide(
        np.sqrt(1.0 - coh**2),
        coh * np.sqrt(2.0 * n_looks),
        out=np.zeros_like(coh),
        where=coh > 0,
    )


def sigma_to_spatial(sigma_px: float, pixel_size_m: float) -> dict:
    """Convert Gaussian sigma (pixels) to spatial smoothing metrics.

    Parameters
    ----------
    sigma_px : float
        Gaussian sigma in pixels (as passed to ``smooth_iono`` or
        ``scipy.ndimage.gaussian_filter``).
    pixel_size_m : float
        Pixel size in metres (use the smaller of x/y resolution if they differ).

    Returns
    -------
    dict
        Keys: ``sigma_m``, ``fwhm_m``, ``cutoff_wavelength_m``.

    """
    sigma_m = sigma_px * pixel_size_m
    fwhm_m = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_m  # ≈ 2.355 × σ
    cutoff_m = 2.0 * np.pi * sigma_m / np.sqrt(np.log(2.0))  # ≈ 7.54 × σ

    print(f"sigma    = {sigma_px} px  →  {sigma_m:.0f} m")
    print(f"FWHM     = {fwhm_m:.0f} m  ({fwhm_m / 1000:.2f} km)  — spatial resolution")
    print(
        f"λ cutoff = {cutoff_m:.0f} m  ({cutoff_m / 1000:.2f} km)"
        "  — features longer than this pass through"
    )
    return {"sigma_m": sigma_m, "fwhm_m": fwhm_m, "cutoff_wavelength_m": cutoff_m}


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------


def _nn_fill(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of invalid pixels.

    Parameters
    ----------
    arr : np.ndarray
        Input array; may contain NaN or arbitrary values at invalid pixels.
    mask : np.ndarray of bool
        True = valid pixel to copy values FROM.

    Returns
    -------
    np.ndarray
        Array with invalid pixels replaced by the value of the nearest valid
        neighbour.  Guaranteed finite (residual NaN → 0.0).

    """
    filled = arr.copy()
    if not mask.all():
        _, idx = distance_transform_edt(~mask, return_indices=True)
        filled[~mask] = arr[idx[0][~mask], idx[1][~mask]]
    filled = np.where(np.isfinite(filled), filled, 0.0)
    return filled


def smooth_iono(
    arr: np.ndarray,
    sigma: float | tuple[float, float],
    mask: np.ndarray | None = None,
    *,
    restore_mask: bool = True,
) -> np.ndarray:
    """Nearest-fill gaps, Gaussian smooth, optionally restore masked pixels."""
    valid = np.isfinite(arr)

    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    if not valid.any():
        raise ValueError("No finite donor pixels available for smoothing")

    filled = _nn_fill(arr, valid)
    smoothed = gaussian_filter(
        filled.astype(np.float64),
        sigma=sigma,
    ).astype(np.float32)

    if restore_mask:
        smoothed[~valid] = np.nan

    return smoothed


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


def resample_to_match(
    src_path: str | Path,
    match_path: str | Path,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Read and reproject ``src_path`` onto the grid of ``match_path``.

    Nodata values are converted to NaN *before* reprojection so sentinel
    values (e.g. 0.0) do not bleed into valid pixels via interpolation.

    Parameters
    ----------
    src_path : str or Path
        Source GeoTIFF to resample (e.g., freqB timeseries file).
    match_path : str or Path
        Reference GeoTIFF whose CRS / transform / shape to match (freqA).
    resampling : rasterio.enums.Resampling
        Resampling algorithm (default: bilinear).

    Returns
    -------
    np.ndarray
        Resampled array on the ``match_path`` grid, shape (rows, cols) float32.
        NaN where no valid data after reprojection.

    """
    with rasterio.open(match_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)

    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_data[src_data == src_nodata] = np.nan

        dst_data = np.full(dst_shape, np.nan, dtype=np.float32)
        rio_reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    return dst_data


def check_reduction(source: Grid, target: Grid) -> None:
    """Validate source and target grids for spatial reduction.

    Require north-up, unrotated grids with matching coordinate reference
    systems. Target pixel sizes must be equal to or larger than source
    pixel sizes in both axes, allowing a relative tolerance of 1e-8.

    This function only validates grid compatibility. It does not resample
    data or require matching grid origins or integer pixel-size ratios.

    Parameters
    ----------
    source : Grid
        Source raster grid with a defined CRS, positive x pixel spacing,
        negative y pixel spacing, and no rotation or shear.
    target : Grid
        Target raster grid satisfying the same requirements.
        Must not be finer than ``source`` in either axis.

    Returns
    -------
    None
        Return normally if both grids satisfy the requirements.

    Raises
    ------
    ValueError
        If either grid has no CRS, is not north-up, or contains rotation
        or shear; if the coordinate reference systems differ; or if the
        target grid is finer than the source grid in either axis.

    """
    for grid in (source, target):
        t = grid.transform
        if (
            grid.crs is None
            or t.a <= 0
            or t.e >= 0
            or t.b != 0
            or t.d != 0
        ):
            raise ValueError(
                "Spatial reduction requires north-up georeferenced grids"
            )

    if source.crs != target.crs:
        raise ValueError(
            "Spatial reduction requires matching source/target CRS"
        )

    if (
        target.transform.a < source.transform.a * (1 - 1e-8)
        or abs(target.transform.e)
        < abs(source.transform.e) * (1 - 1e-8)
    ):
        raise ValueError(
            "Target must not be finer than source in either axis"
        )
    

def source_window(
    source: Grid,
    target: Grid,
    window: Window,
) -> Window:
    """Find the source window covering a target window.

    Convert the target window bounds to source pixel coordinates, round
    outward, and add one source pixel of padding on each side.

    Parameters
    ----------
    source : Grid
        Source raster grid.
    target : Grid
        Target raster grid. Must have the same CRS as ``source`` and
        must not be finer in either axis.
    window : Window
        Window in target pixel coordinates.

    Returns
    -------
    Window
        Window in source pixel coordinates, including padding.
        May extend beyond the source raster bounds; the caller must
        handle out-of-bounds reads.

    Raises
    ------
    ValueError
        If the source and target grids do not support reduction.

    """
    check_reduction(source, target)

    src_window = from_bounds(
        *bounds(window, target.transform),
        transform=source.transform,
    )

    col_start = int(np.floor(src_window.col_off)) - 1
    row_start = int(np.floor(src_window.row_off)) - 1
    col_stop = int(np.ceil(src_window.col_off + src_window.width)) + 1
    row_stop = int(np.ceil(src_window.row_off + src_window.height)) + 1

    return Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    )


def average_to_window(
    a: np.ndarray,
    source: Grid,
    src_window: Window,
    target: Grid,
    dst_window: Window,
) -> np.ndarray:
    """Area-average a source array onto a target window.

    Use pixel-overlap weights to average source values onto the target
    grid. Zeros are included in the average; no nodata value is declared.
    The caller must prepare finite input values before calling.

    Parameters
    ----------
    a : np.ndarray
        Two-dimensional source array corresponding to ``src_window``.
        Invalid contributions should be represented by zero when
        computing masked sums or valid-area fractions.
    source : Grid
        Source raster grid.
    src_window : Window
        Window defining the position of ``a`` in the source grid.
        Must cover the target window's spatial support.
    target : Grid
        Target raster grid.
    dst_window : Window
        Window defining the output position in the target grid.

    Returns
    -------
    np.ndarray
        Area-averaged float32 array with shape
        ``(dst_window.height, dst_window.width)``.
        Pixels without source coverage remain zero.

    """
    dst = np.zeros(
        (int(dst_window.height), int(dst_window.width)),
        dtype=np.float32,
    )

    rio_reproject(
        source=np.asarray(a, dtype=np.float32),
        destination=dst,
        src_transform=transform(src_window, source.transform),
        src_crs=source.crs,
        dst_transform=transform(dst_window, target.transform),
        dst_crs=target.crs,
        resampling=Resampling.average,
        src_nodata=None,
        dst_nodata=None,
        num_threads=1,
    )

    return dst


def resample_phasor_to_match(
    cifg: np.ndarray,
    valid: np.ndarray,
    source: Grid,
    src_window: Window,
    target: Grid,
    dst_window: Window,
    min_valid_fraction: float = 1.0,
    min_magnitude: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample wrapped phase onto a target window using complex averaging.

    Normalize valid complex samples to unit phasors, area-average their
    real and imaginary components, and divide by the valid-area fraction.
    Reject pixels with insufficient valid support or excessive phase
    cancellation, then normalize accepted averages to unit magnitude.

    Parameters
    ----------
    cifg : np.ndarray
        Two-dimensional complex interferogram corresponding to
        ``src_window``. Original amplitudes are discarded.
    valid : np.ndarray
        Boolean source mask with the same shape as ``z``.
        Nonfinite and zero-magnitude samples are also excluded.
    source : Grid
        Source interferogram grid.
    src_window : Window
        Window defining the position of ``z`` in the source grid.
        Must cover the target support, including invalid padding
        where the window extends beyond the source raster.
    target : Grid
        Target interferogram grid. Must have the same CRS as ``source``
        and must not be finer in either axis.
    dst_window : Window
        Window defining the output position in the target grid.
    min_valid_fraction : float, optional
        Minimum area-weighted valid fraction required to accept an
        output pixel. Default is 1.0, with a numerical tolerance of 1e-6.
    min_magnitude : float, optional
        Minimum magnitude of the mean phasor before final normalization.
        Small values indicate phase cancellation. Default is 0.1.

    Returns
    -------
    out : np.ndarray
        Complex64 unit phasors on the target window.
        Rejected pixels are zero; use ``good`` to identify valid output.
    good : np.ndarray
        Boolean mask identifying accepted output pixels.
    support : np.ndarray
        Float32 area-weighted valid fraction on the target window.

    Raises
    ------
    ValueError
        If the source and target grids do not support reduction.

    """
    check_reduction(source, target)

    valid = valid & np.isfinite(cifg) & (np.abs(cifg) > 0)

    unit = np.zeros(cifg.shape, dtype=np.complex64)
    np.divide(cifg, np.abs(cifg), out=unit, where=valid)

    support = average_to_window(
        valid, source, src_window, target, dst_window
    )

    mean = average_to_window(
        unit.real, source, src_window, target, dst_window
        ) + 1j * average_to_window(
            unit.imag, source, src_window, target, dst_window
        )
    mean /= np.maximum(support, 1e-12)

    magnitude = np.abs(mean)
    good = (
        (support >= min_valid_fraction - 1e-6)
        & (magnitude >= min_magnitude)
    )

    out = np.zeros(mean.shape, dtype=np.complex64)
    np.divide(mean, magnitude, out=out, where=good)

    return out, good, support

# ---------------------------------------------------------------------------
# GeoTIFF I/O  (backed by dolphin.io for consistency with the rest of the repo)
# ---------------------------------------------------------------------------


def read_tif(path: str | Path) -> np.ndarray:
    """Read a single-band GeoTIFF as a float32 array with NaN for nodata.

    Uses ``dolphin.io.load_gdal`` for consistency with the rest of the repo.

    Parameters
    ----------
    path : str or Path
        Path to the GeoTIFF file.

    Returns
    -------
    np.ndarray
        Float32 array with nodata values replaced by NaN, shape (rows, cols).

    """
    return io.load_gdal(path, masked=True).astype(np.float32).filled(np.nan)


def write_tif(
    path: str | Path,
    data: np.ndarray,
    like_filename: str | Path,
    units: str = "meters",
    nodata: float = np.nan,
) -> None:
    """Write a float32 GeoTIFF, copying georef from ``like_filename``.

    Uses ``dolphin.io.write_arr`` for consistency with the rest of the repo.

    Parameters
    ----------
    path : str or Path
        Output file path (parent directories created if absent).
    data : np.ndarray
        2D array to write.
    like_filename : str or Path
        Reference raster to copy CRS, transform, and shape from.
    units : str
        Units tag written to the file metadata.  Default ``"meters"``.
    nodata : float
        Nodata value.  Default NaN.

    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    io.write_arr(
        arr=data,
        like_filename=like_filename,
        output_name=path,
        nodata=nodata,
        units=units,
    )


def read_reference_point(ref_point_file: str | Path) -> tuple[int, int]:
    """Read row, col from a dolphin ``reference_point.txt`` file.

    Parameters
    ----------
    ref_point_file : str or Path
        Path to a plain-text file containing ``row,col`` on one line.

    Returns
    -------
    tuple[int, int]
        ``(row, col)`` pixel coordinates.

    """
    row, col = Path(ref_point_file).read_text().strip().split(",")
    return int(row), int(col)


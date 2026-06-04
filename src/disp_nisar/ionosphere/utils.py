"""Shared utilities for ionosphere estimation.

Covers unit conversions, smoothing, spatial I/O helpers, and grid alignment.
Used by both the GUNW and GSLC split-spectrum workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio  # used only by resample_to_match (in-memory reproject)
from dolphin import io
from rasterio.enums import Resampling
from rasterio.warp import reproject as rio_reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_LIGHT = 299_792_458.0  # speed of light, m/s


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
) -> np.ndarray:
    """Gaussian low-pass filter with nearest-neighbour gap-filling.

    Gaps (NaN or ``mask == False``) are filled via nearest-neighbour before
    filtering so they do not bleed into valid regions, then restored to NaN.

    Parameters
    ----------
    arr : np.ndarray
        Ionosphere phase or displacement array.  NaN = invalid.
    sigma : float or (sig_row, sig_col)
        Gaussian standard deviation in pixels.
    mask : np.ndarray of bool, optional
        Valid-pixel mask.  If None, uses ``np.isfinite(arr)``.

    Returns
    -------
    np.ndarray
        Smoothed array, same shape.  NaN where ``mask`` is False.

    """
    if mask is None:
        mask = np.isfinite(arr)

    filled = _nn_fill(arr, mask)
    smoothed = gaussian_filter(filled.astype(np.float64), sigma=sigma).astype(
        np.float32
    )
    smoothed[~mask] = np.nan
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

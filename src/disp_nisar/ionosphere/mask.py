"""Masking and gap-filling utilities for ionosphere correction.

All functions operate on 2D numpy arrays on a common spatial grid.
Used by both the GUNW and GSLC split-spectrum workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import _nn_fill, read_tif, resample_to_match, smooth_iono


def make_valid_mask(
    arr_A: np.ndarray,
    arr_B: np.ndarray,
    nodata_A: float | None = 0.0,
    nodata_B: float | None = 0.0,
) -> np.ndarray:
    """Boolean mask: True where both arrays have valid (finite, non-nodata) data.

    Parameters
    ----------
    arr_A, arr_B : np.ndarray
        FreqA and freqB arrays already on the same grid.
    nodata_A, nodata_B : float or None
        Nodata sentinel values (dolphin default = 0.0).  Pass None to skip.

    Returns
    -------
    np.ndarray of bool
        Same shape as ``arr_A``; True = valid in both arrays.

    """
    mask = np.isfinite(arr_A) & np.isfinite(arr_B)
    if nodata_A is not None:
        mask &= arr_A != nodata_A
    if nodata_B is not None:
        mask &= arr_B != nodata_B
    return mask


def fill_gaps(
    arr: np.ndarray,
    mask: np.ndarray,
    method: str = "gdal",
    max_search_dist: int = 500,
    smoothing_iterations: int = 0,
    sigma: float = 30.0,
) -> np.ndarray:
    """Fill invalid (masked) pixels using interpolation.

    Parameters
    ----------
    arr : np.ndarray
        Float array with NaN or arbitrary values at invalid pixels.
    mask : np.ndarray of bool
        True = valid pixel to interpolate FROM.
    method : str
        ``'gaussian'`` — Weighted Gaussian fill+smooth in one pass.
            ``gaussian_filter(data*mask, σ) / gaussian_filter(mask, σ)``
            Fills gaps AND smooths simultaneously; requires ``sigma`` to be set.
            O(n) separable filter — fastest option.
        ``'gdal'`` — GDAL FillNodata, inverse-distance weighted.
            Handles large gaps without smoothing; use when fill and smooth
            are needed as separate steps.
        ``'linear'`` — scipy griddata linear triangulation. Smooth across gaps
            but slow for large arrays (>1 M pixels).
        ``'nearest'`` — scipy griddata nearest neighbour. Fast but blocky.
    max_search_dist : int
        Maximum search distance in pixels (GDAL method only).
    smoothing_iterations : int
        Post-fill smoothing passes inside GDAL FillNodata (GDAL method only).
        0 = no extra smoothing.
    sigma : float
        Gaussian σ in pixels (``'gaussian'`` method only).

    Returns
    -------
    np.ndarray
        Same shape as ``arr``.  NaN only where no valid neighbours exist
        within ``max_search_dist`` (GDAL) or outside convex hull (linear).

    """
    filled = arr.copy().astype(np.float32)
    filled[~mask] = np.nan

    if method == "gaussian":
        w = mask.astype(np.float32)
        d = np.where(mask, arr, 0.0).astype(np.float32)
        from scipy.ndimage import gaussian_filter

        Gd = gaussian_filter(d, sigma=sigma)
        Gw = gaussian_filter(w, sigma=sigma)
        return np.where(Gw > 1e-4, Gd / Gw, np.nan).astype(np.float32)

    elif method == "gdal":
        from osgeo import gdal

        _NODATA = float(np.finfo(np.float32).min)
        rows, cols = arr.shape
        driver = gdal.GetDriverByName("MEM")

        ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
        band = ds.GetRasterBand(1)
        band.WriteArray(np.where(mask, arr, _NODATA).astype(np.float32))
        band.SetNoDataValue(_NODATA)

        mds = driver.Create("", cols, rows, 1, gdal.GDT_Byte)
        mband = mds.GetRasterBand(1)
        mband.WriteArray(mask.astype(np.uint8))

        gdal.FillNodata(band, mband, max_search_dist, smoothing_iterations)

        result = band.ReadAsArray().astype(np.float32)
        result[result == _NODATA] = np.nan
        return result

    elif method in ("linear", "nearest"):
        import warnings

        from scipy.interpolate import griddata

        rows, cols = arr.shape
        est_gb = rows * cols * 2 * 8 / 1e9
        if est_gb > 2.0:
            warnings.warn(
                f"fill_gaps method='{method}' will allocate ~{est_gb:.1f} GB for "
                f"a {rows}×{cols} array. Consider method='gdal' for large images.",
                ResourceWarning,
                stacklevel=2,
            )

        yy, xx = np.mgrid[0:rows, 0:cols]
        valid_pts = np.column_stack([yy[mask], xx[mask]])
        valid_vals = arr[mask].astype(np.float64)
        all_pts = np.column_stack([yy.ravel(), xx.ravel()])

        result = griddata(valid_pts, valid_vals, all_pts, method=method)
        return result.reshape(rows, cols).astype(np.float32)

    else:
        raise ValueError(
            f"method must be 'gaussian', 'gdal', 'linear', or 'nearest', got '{method}'"
        )


def make_similarity_mask(
    sim_A_path: str | Path,
    match_path: str | Path,
    sim_B_path: str | Path | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Build a quality mask from dolphin similarity files.

    Both similarity rasters are reprojected onto the ``match_path`` grid before
    thresholding so they are always spatially aligned with the ionosphere
    estimate regardless of their native resolution.

    Parameters
    ----------
    sim_A_path : str or Path
        FreqA similarity TIF (``linked_phase/similarity_*.tif``).
        Values in [-1, 1]; higher = better phase linking quality.
    match_path : str or Path
        Reference GeoTIFF defining the target grid (any freqA timeseries file).
    sim_B_path : str or Path or None
        FreqB similarity TIF reprojected to freqA grid before combining.
        If None, only freqA similarity is used.
    threshold : float
        Minimum similarity to keep (default 0.5).
        Pixels below threshold in either band → False.

    Returns
    -------
    np.ndarray of bool
        Shape matching ``match_path``; True = similarity ≥ threshold in all
        provided bands and finite.

    """
    sim_A = resample_to_match(sim_A_path, match_path)
    mask = np.isfinite(sim_A) & (sim_A >= threshold)

    if sim_B_path is not None:
        sim_B = resample_to_match(sim_B_path, match_path)
        mask &= np.isfinite(sim_B) & (sim_B >= threshold)

    return mask


def make_crlb_mask(
    pair: str,
    linked_phase_A: str | Path,
    linked_phase_B: str | Path,
    match_path: str | Path,
    threshold_rad: float = 1.0,
) -> np.ndarray:
    """Build a quality mask from dolphin CRLB (Cramér-Rao Lower Bound) files.

    CRLB is the theoretical minimum phase variance (rad) for each pixel.
    It spikes sharply at burst boundaries and stays near-zero in valid areas,
    making it a more physically meaningful mask than similarity.

    Takes the max CRLB of the two pair-endpoint dates in both bands, reprojects
    freqB onto the freqA grid, and returns True where both bands are below
    ``threshold_rad``.

    Parameters
    ----------
    pair : str
        Date-pair string, e.g. ``"20251028_20260120"``.
    linked_phase_A : str or Path
        FreqA linked_phase directory (``WORK_DIR / "linked_phase"``).
    linked_phase_B : str or Path
        FreqB linked_phase directory (``WORK_DIR / "freqB/linked_phase"``).
    match_path : str or Path
        Reference GeoTIFF defining the freqA output grid.
    threshold_rad : float
        Maximum CRLB to keep in radians.  Pixels above threshold → False.
        0.5 = strict, 1.0 = moderate (default), 2.0 = lenient.

    Returns
    -------
    np.ndarray of bool
        Shape matching ``match_path``; True = CRLB < threshold in both bands.

    """
    ref_date, sec_date = pair.split("_")
    crlb_dir_A = Path(linked_phase_A) / pair / "crlb"
    crlb_dir_B = Path(linked_phase_B) / pair / "crlb"

    crlb_A_ref, _ = read_tif(crlb_dir_A / f"crlb_{ref_date}.tif")
    crlb_A_sec, _ = read_tif(crlb_dir_A / f"crlb_{sec_date}.tif")
    crlb_A = np.fmax(crlb_A_ref, crlb_A_sec)

    crlb_B_ref = resample_to_match(crlb_dir_B / f"crlb_{ref_date}.tif", match_path)
    crlb_B_sec = resample_to_match(crlb_dir_B / f"crlb_{sec_date}.tif", match_path)
    crlb_B = np.fmax(crlb_B_ref, crlb_B_sec)

    return (
        np.isfinite(crlb_A)
        & (crlb_A < threshold_rad)
        & np.isfinite(crlb_B)
        & (crlb_B < threshold_rad)
    )


def apply_similarity_mask_and_fill(
    iono: np.ndarray,
    sim_A_path: str | Path | None = None,
    match_path: str | Path | None = None,
    sim_B_path: str | Path | None = None,
    sim_mask: np.ndarray | None = None,
    existing_mask: np.ndarray | None = None,
    threshold: float = 0.5,
    fill_method: str = "gdal",
    max_search_dist: int = 300,
    smooth_sigma: float = 30.0,
    mask_erosion_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask ionosphere with similarity, gap-fill, and smooth.

    A pre-computed ``sim_mask`` array can be passed directly to skip
    reprojection — recommended for batch processing where the similarity file
    is the same for all date pairs.

    Workflow
    --------
    1. Build quality mask from similarity (or use pre-computed ``sim_mask``).
    2. Quality mask = similarity ≥ ``threshold`` AND ``existing_mask``.
    3. Dilate stripe gaps by ``mask_erosion_px`` pixels to exclude contaminated
       edge pixels from fill sources (only stripe boundaries affected).
    4. Gap-fill masked regions via ``fill_method``.
    5. Restore NaN outside scene extent.
    6. Gaussian smooth the filled surface (skipped for ``'gaussian'`` fill).

    Parameters
    ----------
    iono : np.ndarray
        Ionosphere displacement (m) or phase (rad) on the freqA grid.
    sim_A_path : str or Path or None
        FreqA similarity TIF.  Ignored if ``sim_mask`` is provided.
    match_path : str or Path or None
        Reference GeoTIFF for grid alignment.  Ignored if ``sim_mask`` provided.
    sim_B_path : str or Path or None
        FreqB similarity TIF.  Ignored if ``sim_mask`` is provided.
    sim_mask : np.ndarray of bool or None
        Pre-computed similarity mask (from ``make_similarity_mask``).
    existing_mask : np.ndarray of bool or None
        Valid-data mask defining the scene extent.
        If None, uses ``np.isfinite(iono)``.
    threshold : float
        Minimum similarity to keep (default 0.5).  Ignored if ``sim_mask``
        is provided.
    fill_method : str
        Gap-fill algorithm: ``'gdal'``, ``'gaussian'``, ``'linear'``,
        or ``'nearest'``.
    max_search_dist : int
        Search radius in pixels for GDAL fill.  Default 300.
    smooth_sigma : float
        Gaussian σ in pixels applied after gap-fill.  0 = skip smoothing.
    mask_erosion_px : int
        Expand stripe gaps by this many pixels before using valid pixels as
        fill sources.  0 to disable.

    Returns
    -------
    iono_filled : np.ndarray
        Ionosphere with stripe gaps filled and smoothed.  NaN outside scene.
    quality_mask : np.ndarray of bool
        Combined mask used for gap-filling source pixels.

    """
    if existing_mask is None:
        existing_mask = np.isfinite(iono)

    if iono.shape != existing_mask.shape:
        raise ValueError(
            f"iono {iono.shape} and existing_mask {existing_mask.shape} shapes differ"
        )

    if sim_mask is None:
        if sim_A_path is None or match_path is None:
            raise ValueError(
                "Provide either sim_mask or both sim_A_path and match_path"
            )
        sim_mask = make_similarity_mask(
            sim_A_path=sim_A_path,
            match_path=match_path,
            sim_B_path=sim_B_path,
            threshold=threshold,
        )

    if sim_mask.shape != iono.shape:
        raise ValueError(
            f"sim_mask {sim_mask.shape} != iono {iono.shape}. "
            "Ensure sim_mask was built with match_path on the freqA grid."
        )

    quality_mask = existing_mask & sim_mask

    if mask_erosion_px > 0:
        from scipy.ndimage import maximum_filter1d

        stripe_mask = (~sim_mask).view(np.uint8)
        expanded_stripes = maximum_filter1d(
            stripe_mask, size=mask_erosion_px * 2 + 1, axis=1
        ).astype(bool)
        quality_mask = existing_mask & ~expanded_stripes

    iono_filled = fill_gaps(
        arr=iono,
        mask=quality_mask,
        method=fill_method,
        max_search_dist=max_search_dist,
        sigma=smooth_sigma,
    )

    iono_filled = np.where(existing_mask, iono_filled, np.nan)

    if smooth_sigma > 0 and fill_method != "gaussian":
        scene_mask = np.isfinite(iono_filled)
        iono_filled = smooth_iono(iono_filled, sigma=smooth_sigma, mask=scene_mask)

    return iono_filled, quality_mask


def mask_iono_outliers(
    iono: np.ndarray,
    threshold: float = 0.1,
    median_filter_size: int = 11,
) -> np.ndarray:
    """Flag ionosphere outliers using a median-filter residual test.

    A pixel is marked invalid (False) if it deviates from its local median
    by more than ``threshold`` (in the same units as ``iono``).  Isolated valid
    pixels are then removed via binary opening.

    Adapted from ``IonosphereEstimation.get_mask_median_filter`` +
    ``remove_single_pixels`` in ``isce3/atmosphere/ionosphere_estimation.py``.

    Parameters
    ----------
    iono : np.ndarray
        Ionosphere displacement (m) or phase (rad).  NaN = already invalid.
    threshold : float
        Maximum allowed deviation from local median (same units as ``iono``).
        Typical: 0.05 m for displacement, 0.3 rad for phase.
    median_filter_size : int
        Kernel size for the median filter (pixels, must be odd).

    Returns
    -------
    np.ndarray of bool
        True = valid pixel.

    """
    from scipy.ndimage import binary_opening, median_filter

    finite = np.isfinite(iono)
    filled = _nn_fill(iono, finite)

    residual = np.abs(filled - median_filter(filled, size=median_filter_size))
    mask = finite & (residual < threshold)

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return binary_opening(mask, structure=struct).astype(bool)

"""Selectable filtering for real, unwrapped ionospheric phase in radians.

The pipeline uses apply_ionosphere_filter() with validated algorithm-file
options. Available methods are guided (guide/filter, fill, restore original
observations, final filter) and smooth_iono (the existing smoother).

The guided method ports the sequence from ISCE3's IonosphereFilter.low_pass_filter.
It uses SciPy, not the ISCE3 package, and does not claim bitwise equivalence.
Source reviewed: isce-framework/isce3 at
043aa4b84f94f6cd99b84b03bb8001c2b8ddab20, under
python/packages/isce3/atmosphere/ionosphere_filter.py.

All arrays must share a grid, phase convention, and reference. Zero radians
is valid. A spatially coherent unwrap offset is not repaired by filtering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .options import IonosphereFilterOptions, IonosphereOptions
from .utils import _nn_fill, smooth_iono


def _float_array(value: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("Expected real unwrapped phase, not a complex interferogram")
    value = np.ma.asarray(value, dtype=np.float64).filled(np.nan)
    if value.ndim != 2 or (shape is not None and value.shape != shape):
        raise ValueError("All inputs must be 2-D arrays on the same grid")
    return value


def _mask_array(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = _float_array(value, shape)
    return np.isfinite(value) & (value > 0)


def _erode(valid: np.ndarray, erosion_px: tuple[int, int]) -> np.ndarray:
    """Drop samples within the (row, column) radii of an invalid pixel."""
    if not any(erosion_px):
        return valid
    invalid = ndi.maximum_filter(
        (~valid).astype(np.uint8),
        size=tuple(2 * n + 1 for n in erosion_px),
        mode="nearest",
    )
    return valid & (invalid == 0)


def _gaussian_kernel(sigma: float, truncate: float) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve(array: np.ndarray, kernels: tuple[np.ndarray, ...]) -> np.ndarray:
    for axis, kernel in enumerate(kernels):
        array = ndi.convolve1d(array, kernel, axis=axis, mode="constant", cval=0.0)
    return array


def _fill_from_samples(
    values: np.ndarray, valid: np.ndarray, method: str
) -> np.ndarray:
    """Fill missing values by nearest or linear interpolation with nearest fallback.

    Original samples are retained. Linear interpolation falls back to nearest
    for degenerate sample geometry and locations outside the convex hull.
    """
    if not valid.any():
        return np.full(values.shape, np.nan, dtype=np.float64)
    filled = _nn_fill(values, valid).astype(np.float64)
    if method == "linear" and (~valid).any():
        from scipy.interpolate import griddata
        from scipy.spatial import QhullError

        points = np.column_stack(np.nonzero(valid))
        targets = np.column_stack(np.nonzero(~valid))
        try:
            interpolated = griddata(points, values[valid], targets, method="linear")
        except (QhullError, ValueError):
            interpolated = np.full(len(targets), np.nan)
        candidate = filled[~valid]
        finite = np.isfinite(interpolated)
        candidate[finite] = interpolated[finite]
        filled[~valid] = candidate
    filled[valid] = values[valid]
    return filled


def _normalized_filter(
    values: np.ndarray, weights: np.ndarray, kernels: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Compute Gaussian weighted means without allowing NaN times zero."""
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    precision = np.where(good, weights, 0.0)
    numerator = _convolve(np.where(good, values, 0.0) * precision, kernels)
    denominator = _convolve(precision, kernels)
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape, np.nan),
        where=denominator > np.finfo(float).tiny,
    )


def _build_filter_guide(
    values: np.ndarray,
    settings: IonosphereFilterOptions,
    kernels: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Construct a NaN-aware median/Gaussian guide before filling.

    Recent SciPy supports a bounded-memory vectorized median. Fall back to
    generic_filter for older deployments without changing the statistic.
    """
    if settings.guide_filter_method == "none":
        return values.copy()
    guide = values.copy()
    if settings.guide_filter_method == "median_gaussian":
        if hasattr(ndi, "vectorized_filter"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                guide = ndi.vectorized_filter(
                    guide,
                    np.nanmedian,
                    size=settings.guide_median_size,
                    mode="reflect",
                    batch_memory=32 * 1024**2,
                )
        else:
            guide = ndi.generic_filter(
                guide,
                lambda a: (
                    np.median(a[np.isfinite(a)]) if np.isfinite(a).any() else np.nan
                ),
                size=settings.guide_median_size,
                mode="reflect",
            )
    return _normalized_filter(guide, np.isfinite(guide), kernels)


def filter_ionosphere_guided(
    phase: np.ndarray,
    *,
    donor_mask: np.ndarray,
    footprint: np.ndarray,
    pixel_size_m: tuple[float, float],
    settings: IonosphereFilterOptions,
    sigma_m: tuple[float, float],
    donor_erosion_px: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Apply the guide/fill/restore/filter sequence.

    Parameters
    ----------
    phase : np.ndarray
        Real unwrapped ionospheric phase, in radians, on a single grid.
    donor_mask, footprint : np.ndarray
        Original observation support and allowed output domain, respectively.
    pixel_size_m, sigma_m : tuple[float, float]
        Pixel spacing and Gaussian standard deviations in (row, column) order.
    settings : IonosphereFilterOptions
        Kernel, guide, filling, and outlier controls.
    donor_erosion_px : tuple[int, int], optional
        Additional donor exclusion radii along the row and column axes.

    Returns
    -------
    np.ndarray
        Smoothed phase as float32, NaN outside the footprint.

    Notes
    -----
    This is a NumPy/SciPy implementation of the filtering sequence, not a call
    to ISCE3. Rejected outliers are NOT restored as trusted samples, which
    deliberately differs from the reviewed ISCE3 implementation. No component
    labeling or unwrap-error correction is performed here.

    """
    data = _float_array(phase)
    domain = _mask_array(footprint, data.shape)
    valid = _mask_array(donor_mask, data.shape) & domain & np.isfinite(data)
    spacing = np.asarray(pixel_size_m, dtype=np.float64)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("pixel_size_m must contain positive row/column spacings")
    valid = _erode(valid, donor_erosion_px)
    if not valid.any():
        return np.full(data.shape, np.nan, np.float32)

    # ISCE3 exposes kernel lengths and sigmas separately. Permit the same
    # control, while accepting widths in meters on the geocoded grid.
    sigma_pixels = np.asarray(sigma_m, np.float64) / spacing
    kernels = []
    for axis, sigma in enumerate(sigma_pixels):
        if settings.kernel_size_px is None:
            kernels.append(_gaussian_kernel(sigma, settings.truncate))
        else:
            radius = settings.kernel_size_px[axis] // 2
            coords = np.arange(-radius, radius + 1, dtype=np.float64)
            kernel = np.exp(-0.5 * (coords / sigma) ** 2)
            kernels.append(kernel / kernel.sum())
    kernel_pair = tuple(kernels)

    working = np.where(valid, data, np.nan)
    guide = _build_filter_guide(working, settings, kernel_pair)

    # Compute a block-wide MAD, matching the structure of ISCE3's guide
    # residual test. Keep outlier rejection distinct from donor erosion.
    retained = valid.copy()
    if (
        settings.outlier_threshold is not None
        and settings.guide_filter_method != "none"
    ):
        residual = working - guide
        eligible = valid & np.isfinite(residual)
        if eligible.any():
            center = np.median(residual[eligible])
            scale = max(
                1.4826 * np.median(np.abs(residual[eligible] - center)),
                settings.outlier_min_scale_rad,
            )
            retained[eligible] &= (
                np.abs(residual[eligible] - center)
                <= settings.outlier_threshold * scale
            )
            if not np.array_equal(retained, valid):
                cleaned = np.where(retained, data, np.nan)
                guide = _build_filter_guide(cleaned, settings, kernel_pair)

    # Fill the guide first; these modeled samples contribute to the final
    # Gaussian filter, even in large gaps.
    guide_valid = np.isfinite(guide) & domain
    filled = _fill_from_samples(guide, guide_valid, settings.filling_method)
    # Restore ORIGINAL accepted observations before the final filtering.
    filled[retained] = data[retained]
    current = _normalized_filter(filled, domain.astype(np.float64), kernel_pair)
    current[~domain] = np.nan
    if settings.restore_mask:
        current[~retained] = np.nan
    return current.astype(np.float32)


def apply_ionosphere_filter(
    phase: np.ndarray,
    *,
    donor_mask: np.ndarray,
    footprint: np.ndarray,
    pixel_size_m: tuple[float, float],
    options: IonosphereOptions,
) -> np.ndarray:
    """Dispatch the configured method on an aligned unwrapped phase array.

    The main algorithm file owns these options. A null filtering.sigma_m
    falls back to the existing smooth_sigma_m scalar; a null erosion pair
    falls back to (0, mask_erosion_px). No coherence-to-sigma proxy is invented.
    """
    settings = options.filtering
    sigma_m = settings.sigma_m or (options.smooth_sigma_m,) * 2
    erosion = (
        settings.donor_erosion_px
        if settings.donor_erosion_px is not None
        else (0, options.mask_erosion_px)
    )
    if settings.method == "guided":
        return filter_ionosphere_guided(
            phase,
            donor_mask=donor_mask,
            footprint=footprint,
            pixel_size_m=pixel_size_m,
            settings=settings,
            sigma_m=sigma_m,
            donor_erosion_px=erosion,
        )
    # Reuse the existing implementation rather than copying its nearest-fill
    # and Gaussian operations. Its boundary/kernel behavior is preserved.
    data = _float_array(phase)
    domain = _mask_array(footprint, data.shape)
    valid = _mask_array(donor_mask, data.shape) & domain & np.isfinite(data)
    valid = _erode(valid, erosion)
    if not valid.any():
        return np.full(data.shape, np.nan, np.float32)
    sigma_px = (sigma_m[0] / pixel_size_m[0], sigma_m[1] / pixel_size_m[1])
    filtered = smooth_iono(
        data, sigma=sigma_px, mask=valid, restore_mask=settings.restore_mask
    )
    filtered[~domain] = np.nan
    return filtered


def filter_cache_signature(raw_path: Path, options: IonosphereOptions) -> str:
    """Identify raw input and all selected filter settings for cache reuse.

    Source modification time and size invalidate the cache after inversion.
    Version the implementation to invalidate earlier file-existence caches.
    """
    source = Path(raw_path).resolve()
    stat = source.stat()
    record = {
        "version": "selectable-filter-v1",
        "raw": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "filtering": options.filtering.model_dump(mode="json"),
        "smooth_sigma_m": options.smooth_sigma_m,
        "mask_erosion_px": options.mask_erosion_px,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

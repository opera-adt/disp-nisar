"""Mask-aware complex averaging onto the frequency-B interferogram grid."""

from __future__ import annotations

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, bounds, from_bounds, transform

from ._main_diff_io import Grid, check_reduction


def source_window(source: Grid, target: Grid, window: Window) -> Window:
    """Include all target support plus explicit padding beyond source edges."""
    check_reduction(source, target)
    w = from_bounds(*bounds(window, target.transform), transform=source.transform)
    c0, r0 = int(np.floor(w.col_off)) - 1, int(np.floor(w.row_off)) - 1
    c1, r1 = (
        int(np.ceil(w.col_off + w.width)) + 1,
        int(np.ceil(w.row_off + w.height)) + 1,
    )
    return Window(c0, r0, c1 - c0, r1 - r0)


def average_to_window(
    a: np.ndarray, source: Grid, src_window: Window, target: Grid, dst_window: Window
) -> np.ndarray:
    """Area-average finite values; zeros count in the support denominator."""
    dst: np.ndarray = np.zeros(
        (int(dst_window.height), int(dst_window.width)), np.float32
    )
    reproject(
        np.asarray(a, np.float32),
        dst,
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
    z: np.ndarray,
    valid: np.ndarray,
    source: Grid,
    src_window: Window,
    target: Grid,
    dst_window: Window,
    min_valid_fraction: float = 1.0,
    min_magnitude: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return averaged unit phasors, valid donors, and valid-area fraction.

    Average real/imaginary components, never wrapped angles. Reject insufficient
    support and destructive cancellation. Invalid source data contribute neither
    phase nor weight. The supplied source window must cover the target footprint.
    """
    check_reduction(source, target)
    valid = valid & np.isfinite(z) & (np.abs(z) > 0)
    unit = np.zeros(z.shape, np.complex64)
    np.divide(z, np.abs(z), out=unit, where=valid)
    support = average_to_window(valid, source, src_window, target, dst_window)
    mean = average_to_window(
        unit.real, source, src_window, target, dst_window
    ) + 1j * average_to_window(unit.imag, source, src_window, target, dst_window)
    mean /= np.maximum(support, 1e-12)
    mag = np.abs(mean)
    good = (support >= min_valid_fraction - 1e-6) & (mag >= min_magnitude)
    out = np.zeros(mean.shape, np.complex64)
    np.divide(mean, mag, out=out, where=good)
    return out, good, support

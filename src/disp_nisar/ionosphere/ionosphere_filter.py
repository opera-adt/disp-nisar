"""Selectable filtering for real, unwrapped ionospheric phase in radians.

The pipeline uses apply_ionosphere_filter() with validated algorithm-file
options. Available methods are isce3 (guide/filter, fill, restore original
observations, final weighted filter), smooth_iono (the existing smoother),
and donor_only (the original bounded-donor prototype retained below).

The ISCE3-style method ports the sequence from IonosphereFilter.low_pass_filter
and the inverse-variance principle in filter_data_with_sig. It uses SciPy,
not the compiled ISCE3 backend, and does not claim bitwise equivalence.
Source reviewed: isce-framework/isce3 at
043aa4b84f94f6cd99b84b03bb8001c2b8ddab20, under
python/packages/isce3/atmosphere/ionosphere_filter.py.

All arrays must share a grid, phase convention, and reference. Zero radians
is valid. A spatially coherent unwrap offset is not repaired by filtering.
Formal uncertainties assume independent errors and omit covariance introduced
by filling, reference uncertainty, model bias, and interpolation error.
The pipeline currently has no calibrated phase_std and uses equal precision.

filter_ionosphere_phase() and filter_ionosphere_raster() retain the previous
bounded donor-only API. The raster wrapper is not the selectable dispatcher.
"""

from __future__ import annotations

import json
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class FilterConfig:
    """Controls for the retained bounded, original-donor Gaussian filter."""

    sigma_m: tuple[float, float] = (300.0, 300.0)  # (row, column)
    max_fill_m: float = 600.0
    min_support: float = 0.10
    donor_erosion_px: tuple[int, int] = (0, 0)
    robust_passes: int = 1  # zero disables robust reweighting
    robust_cutoff: float = 4.0
    robust_floor_rad: float = 0.10
    std_floor_rad: float = 0.05  # caps maximum donor precision
    truncate: float = 4.0

    def validate(self) -> None:
        """Reject invalid widths, support thresholds, and robust-fit controls."""
        if len(self.sigma_m) != 2 or any(
            not np.isfinite(s) or s <= 0 for s in self.sigma_m
        ):
            raise ValueError("sigma_m must contain two positive finite widths")
        if not np.isfinite(self.max_fill_m) or self.max_fill_m < 0:
            raise ValueError("max_fill_m must be finite and nonnegative")
        if not 0 < self.min_support <= 1:
            raise ValueError("min_support must be in (0, 1]")
        if len(self.donor_erosion_px) != 2 or any(
            not isinstance(v, (int, np.integer)) or v < 0 for v in self.donor_erosion_px
        ):
            raise ValueError("donor_erosion_px must contain two nonnegative integers")
        if (
            not isinstance(self.robust_passes, (int, np.integer))
            or not 0 <= self.robust_passes <= 5
        ):
            raise ValueError("robust_passes must be an integer from zero to five")
        for value in (
            self.robust_cutoff,
            self.robust_floor_rad,
            self.std_floor_rad,
            self.truncate,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "cutoff, floors, and truncate must be positive and finite"
                )


@dataclass
class FilterResult:
    """Filtered phase, optional formal uncertainty, and support diagnostics."""

    phase: np.ndarray
    std: np.ndarray | None
    support: np.ndarray
    distance_m: np.ndarray  # exact within max_fill_m; inf beyond it
    robust_weight: np.ndarray  # original donors only; zero at excluded pixels


def _float_array(value, shape=None):
    if np.iscomplexobj(value):
        raise ValueError("Expected real unwrapped phase, not a complex interferogram")
    value = np.ma.asarray(value, dtype=np.float64).filled(np.nan)
    if value.ndim != 2 or (shape is not None and value.shape != shape):
        raise ValueError("All inputs must be 2-D arrays on the same grid")
    return value


def _mask_array(value, shape):
    value = _float_array(value, shape)
    return np.isfinite(value) & (value > 0)


def _gaussian_kernel(sigma, truncate):
    radius = int(truncate * sigma + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve(array, kernels, squared=False):
    for axis, kernel in enumerate(kernels):
        array = ndi.convolve1d(
            array,
            kernel**2 if squared else kernel,
            axis=axis,
            mode="constant",
            cval=0.0,
        )
    return array


def filter_ionosphere_phase(
    phase,
    *,
    donor_mask,
    footprint,
    pixel_size_m: tuple[float, float],
    phase_std=None,
    config: FilterConfig | None = None,
) -> FilterResult:
    """Smooth from original trusted samples and fill only supported nearby gaps.

    phase_std is ionosphere phase uncertainty in radians, not A/D coherence.
    If omitted, initial donor weights are equal and output std is None.
    donor_mask and footprint use True=allowed; footprint defines fill targets.
    Erosion expands donor gaps along the requested row/column axes.
    Robust reweighting is a spike-control heuristic, not an unwrap correction.
    """
    cfg = config or FilterConfig()
    cfg.validate()
    spacing = np.asarray(pixel_size_m, dtype=np.float64)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("pixel_size_m must contain positive (row, column) spacings")
    data = _float_array(phase)
    shape = data.shape
    domain = _mask_array(footprint, shape)
    valid = _mask_array(donor_mask, shape) & domain & np.isfinite(data)
    uncertainty = None
    if phase_std is not None:
        uncertainty = _float_array(phase_std, shape)
        valid &= np.isfinite(uncertainty) & (uncertainty >= 0)
        uncertainty = np.where(valid, np.maximum(uncertainty, cfg.std_floor_rad), 1.0)
    if any(cfg.donor_erosion_px):
        gaps = ndi.maximum_filter(
            (~valid).astype(np.uint8),
            size=tuple(2 * n + 1 for n in cfg.donor_erosion_px),
            mode="nearest",
        )
        valid &= gaps == 0

    nan = np.full(shape, np.nan, dtype=np.float32)
    zero = np.zeros(shape, dtype=np.float32)
    if not valid.any():
        return FilterResult(
            nan,
            None if uncertainty is None else nan.copy(),
            zero,
            np.full(shape, np.inf, dtype=np.float32),
            zero.copy(),
        )

    kernels = tuple(
        _gaussian_kernel(s / d, cfg.truncate) for s, d in zip(cfg.sigma_m, spacing)
    )
    distance, nearest = ndi.distance_transform_edt(
        ~valid,
        sampling=spacing,
        return_indices=True,
    )
    support_numerator = _convolve(valid.astype(np.float64), kernels)
    support_denominator = _convolve(domain.astype(np.float64), kernels)
    support = np.divide(
        support_numerator,
        support_denominator,
        out=np.zeros(shape),
        where=support_denominator > 0,
    )
    allowed = domain & (
        valid | ((distance <= cfg.max_fill_m) & (support >= cfg.min_support))
    )
    sample = np.where(valid, data, 0.0)  # never multiply NaN by zero
    base_weight = valid.astype(np.float64)
    if uncertainty is not None:
        base_weight /= uncertainty**2
    robust = valid.astype(np.float64)

    # Nearest values are used ONLY to construct a local spike-detection guide.
    # They never become observations or receive precision in the final fit.
    if cfg.robust_passes:
        guide = ndi.median_filter(data[tuple(nearest)], size=3, mode="nearest")
        residual_at_donors = np.where(valid, data - guide, 0.0)
        local_mad = ndi.median_filter(
            np.abs(residual_at_donors[tuple(nearest)]),
            size=7,
            mode="nearest",
        )
        scale = np.maximum(1.4826 * local_mad, cfg.robust_floor_rad)
        if uncertainty is not None:
            scale = np.maximum(scale, uncertainty)

    def fit(weights):
        denominator = _convolve(weights, kernels)
        numerator = _convolve(sample * weights, kernels)
        fitted = np.divide(
            numerator,
            denominator,
            out=np.full(shape, np.nan),
            where=denominator > np.finfo(float).tiny,
        )
        return fitted, denominator

    for _ in range(cfg.robust_passes):
        residual = np.where(valid, data - guide, 0.0)
        ratio = np.divide(
            cfg.robust_cutoff * scale,
            np.abs(residual),
            out=np.ones(shape),
            where=np.abs(residual) > 0,
        )
        # Non-increasing weights: rejected spikes cannot return at later passes.
        robust = np.minimum(robust, np.minimum(1.0, ratio) ** 2)
        guide, _ = fit(base_weight * robust)
    weights = base_weight * robust
    fitted, denominator = fit(weights)
    fitted[~allowed] = np.nan

    filtered_std = None
    if uncertainty is not None:
        # Squared kernel, NOT Gaussian smoothing of the std-dev image.
        variance_numerator = _convolve(
            weights**2 * uncertainty**2, kernels, squared=True
        )
        variance = np.divide(
            variance_numerator,
            denominator**2,
            out=np.full(shape, np.nan),
            where=denominator > 0,
        )
        filtered_std = np.sqrt(variance)
        filtered_std[~allowed] = np.nan
        filtered_std = filtered_std.astype(np.float32)
    distance = np.where(domain & (distance <= cfg.max_fill_m), distance, np.inf)
    support[~domain] = np.nan
    return FilterResult(
        fitted.astype(np.float32),
        filtered_std,
        support.astype(np.float32),
        distance.astype(np.float32),
        robust.astype(np.float32),
    )


def filter_ionosphere_raster(
    input_path,
    output_path,
    *,
    footprint_path,
    donor_mask_path=None,
    std_path=None,
    diagnostics_path=None,
    config: FilterConfig | None = None,
    block_size: int = 512,
    overwrite: bool = False,
):
    """Filter saved GeoTIFFs in overlapping blocks, without re-running inversion.

    Requires a projected CRS with orthogonal pixel axes. Every auxiliary raster
    must match the input grid; this function performs no reprojection.
    Reads honor input nodata metadata (including an explicit nodata=0).
    Diagnostics bands: std(rad), support, donor distance(m), robust weight.
    No std input => std diagnostic band is NaN, rather than an invented error.
    """
    import rasterio
    from rasterio.windows import Window

    cfg = config or FilterConfig()
    cfg.validate()
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    destination = Path(output_path).resolve()
    diagnostic = Path(diagnostics_path).resolve() if diagnostics_path else None
    inputs = [
        Path(p).resolve()
        for p in (input_path, footprint_path, donor_mask_path, std_path)
        if p is not None
    ]
    outputs = [p for p in (destination, diagnostic) if p is not None]
    if len(set(outputs)) != len(outputs) or any(p in inputs for p in outputs):
        raise ValueError("Outputs must differ from one another and from all inputs")
    for path in outputs:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = [
        p.with_name(p.stem + "." + uuid.uuid4().hex + ".partial.tif") for p in outputs
    ]
    try:
        with ExitStack() as stack:
            source = stack.enter_context(rasterio.open(input_path))
            if np.dtype(source.dtypes[0]).kind == "c":
                raise ValueError("Input must be real unwrapped ionosphere phase")
            footprint_ds = stack.enter_context(rasterio.open(footprint_path))
            donor_ds = (
                stack.enter_context(rasterio.open(donor_mask_path))
                if donor_mask_path
                else None
            )
            std_ds = stack.enter_context(rasterio.open(std_path)) if std_path else None
            for other in (footprint_ds, donor_ds, std_ds):
                if other and (
                    other.shape != source.shape
                    or other.transform != source.transform
                    or other.crs != source.crs
                ):
                    raise ValueError(
                        "Auxiliary raster does not match the ionosphere grid"
                    )
            if source.crs is None or not source.crs.is_projected:
                raise ValueError("Meter-based filtering requires a projected CRS")
            affine = source.transform
            column_vector = np.array([affine.a, affine.d])
            row_vector = np.array([affine.b, affine.e])
            if abs(np.dot(column_vector, row_vector)) > 1e-8 * np.linalg.norm(
                column_vector
            ) * np.linalg.norm(row_vector):
                raise ValueError("Sheared grids require resampling before filtering")
            meters_per_unit = source.crs.linear_units_factor[1]
            spacing = (
                np.array([np.linalg.norm(row_vector), np.linalg.norm(column_vector)])
                * meters_per_unit
            )
            if not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError("Invalid pixel spacing")
            radius = np.array(
                [int(cfg.truncate * s / d + 0.5) for s, d in zip(cfg.sigma_m, spacing)]
            )
            # Includes original-donor distance, median/MAD guide stencils,
            # erosion, and the dependency radius of each robust fit.
            halo = (
                (cfg.robust_passes + 1) * radius
                + np.ceil(cfg.max_fill_m / spacing).astype(int)
                + np.array(cfg.donor_erosion_px)
                + 8
            )
            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=np.nan,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
                predictor=3,
                BIGTIFF="IF_SAFER",
            )
            writer = stack.enter_context(rasterio.open(temporary[0], "w", **profile))
            writer.update_tags(**source.tags())
            writer.update_tags(
                ionosphere_filter="donor_only_weighted_gaussian",
                filter_config=json.dumps(cfg.__dict__),
            )
            writer.set_band_unit(1, "radians")
            diag_writer = None
            if diagnostic:
                profile.update(count=4)
                diag_writer = stack.enter_context(
                    rasterio.open(temporary[1], "w", **profile)
                )
                for band, name in enumerate(
                    (
                        "formal_std_rad",
                        "original_donor_support",
                        "bounded_donor_distance_m",
                        "robust_donor_weight",
                    ),
                    1,
                ):
                    diag_writer.set_band_description(band, name)
                diag_writer.update_tags(
                    uncertainty="independent_errors_conditional_on_robust_weights"
                )

            def read(ds, window):
                return (
                    ds.read(1, window=window, masked=True)
                    .astype(np.float64)
                    .filled(np.nan)
                )

            height, width = source.shape
            for row in range(0, height, block_size):
                for col in range(0, width, block_size):
                    end_row, end_col = min(row + block_size, height), min(
                        col + block_size, width
                    )
                    start_row, start_col = max(0, row - halo[0]), max(0, col - halo[1])
                    stop_row, stop_col = min(height, end_row + halo[0]), min(
                        width, end_col + halo[1]
                    )
                    read_window = Window(
                        int(start_col),
                        int(start_row),
                        int(stop_col - start_col),
                        int(stop_row - start_row),
                    )
                    values = read(source, read_window)
                    domain = read(footprint_ds, read_window)
                    donors = (
                        read(donor_ds, read_window) if donor_ds else np.isfinite(values)
                    )
                    result = filter_ionosphere_phase(
                        values,
                        donor_mask=donors,
                        footprint=domain,
                        pixel_size_m=tuple(spacing),
                        phase_std=read(std_ds, read_window) if std_ds else None,
                        config=cfg,
                    )
                    crop = np.s_[
                        int(row - start_row) : int(end_row - start_row),
                        int(col - start_col) : int(end_col - start_col),
                    ]
                    write_window = Window(col, row, end_col - col, end_row - row)
                    writer.write(result.phase[crop], 1, window=write_window)
                    if diag_writer:
                        std_band = (
                            result.std
                            if result.std is not None
                            else np.full(values.shape, np.nan, dtype=np.float32)
                        )
                        for band, values_out in enumerate(
                            (
                                std_band,
                                result.support,
                                result.distance_m,
                                result.robust_weight,
                            ),
                            1,
                        ):
                            diag_writer.write(
                                values_out[crop], band, window=write_window
                            )
        for partial, final in zip(temporary, outputs):
            partial.replace(final)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return str(destination)


def _fill_from_samples(values, valid, method):
    """Fill missing values by nearest or linear interpolation with nearest fallback.

    Original samples are retained. Linear interpolation falls back to nearest
    for degenerate sample geometry and locations outside the convex hull.
    """
    if not valid.any():
        return np.full(values.shape, np.nan, dtype=np.float64)
    nearest = ndi.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = np.asarray(values[tuple(nearest)], dtype=np.float64)
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


def _normalized_filter(values, weights, kernels):
    """Compute Gaussian weighted means without allowing NaN times zero."""
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    precision = np.where(good, weights, 0.0)
    numerator = _convolve(np.where(good, values, 0.0) * precision, kernels)
    denominator = _convolve(precision, kernels)
    return (
        np.divide(
            numerator,
            denominator,
            out=np.full(values.shape, np.nan),
            where=denominator > np.finfo(float).tiny,
        ),
        denominator,
    )


def _build_filter_guide(values, settings, kernels):
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
    guide, _ = _normalized_filter(guide, np.isfinite(guide), kernels)
    return guide


def filter_ionosphere_isce3(
    phase,
    *,
    donor_mask,
    footprint,
    pixel_size_m,
    settings,
    sigma_m,
    donor_erosion_px=(0, 0),
    phase_std=None,
):
    """Apply the ISCE3-style guide/fill/restore/weighted-filter sequence.

    Parameters
    ----------
    phase : np.ndarray
        Real unwrapped ionospheric phase, in radians, on a single grid.
    donor_mask, footprint : np.ndarray
        Original observation support and allowed output domain, respectively.
    pixel_size_m, sigma_m : tuple[float, float]
        Pixel spacing and Gaussian standard deviations in (row, column) order.
    settings : IonosphereFilterOptions
        Kernel, guide, filling, iteration, and outlier controls.
    donor_erosion_px : tuple[int, int], optional
        Additional donor exclusion radii along the row and column axes.
    phase_std : np.ndarray, optional
        Calibrated phase uncertainty in radians. Without it, filtering uses
        uniform precision and reports no formal output standard deviation.

    Returns
    -------
    FilterResult
        Smoothed phase and original-donor diagnostics. Filled phase/sigma
        participate in the final filter, unlike the donor_only method.

    Notes
    -----
    This is a NumPy/SciPy implementation of the filtering sequence, not a
    call to the compiled ISCE3 backend. Each iteration masks the working
    image to the original donor support and restores original samples.
    Consequently extra iterations need not change the result.
    Rejected outliers are NOT restored as trusted samples. This deliberately
    avoids reintroducing rejected samples at the restoration step in the
    reviewed ISCE3 implementation. No component labeling or unwrap-error
    correction is performed here. Zero radians is a valid observation.

    """
    data = _float_array(phase)
    domain = _mask_array(footprint, data.shape)
    valid = _mask_array(donor_mask, data.shape) & domain & np.isfinite(data)
    spacing = np.asarray(pixel_size_m, dtype=np.float64)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("pixel_size_m must contain positive row/column spacings")
    uncertainty = None
    if phase_std is not None:
        uncertainty = _float_array(phase_std, data.shape)
        valid &= np.isfinite(uncertainty) & (uncertainty >= 0)
        uncertainty = np.maximum(uncertainty, settings.std_floor_rad)
    if any(donor_erosion_px):
        invalid = ndi.maximum_filter(
            (~valid).astype(np.uint8),
            size=tuple(2 * n + 1 for n in donor_erosion_px),
            mode="nearest",
        )
        valid &= invalid == 0
    if not valid.any():
        empty = np.full(data.shape, np.nan, np.float32)
        return FilterResult(
            empty,
            None if uncertainty is None else empty.copy(),
            np.zeros(data.shape, np.float32),
            np.full(data.shape, np.inf, np.float32),
            np.zeros(data.shape, np.float32),
        )

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
    kernels = tuple(kernels)
    original = np.where(valid, data, np.nan)
    current = original.copy()
    output_std = None
    retained = valid.copy()

    for _ in range(settings.iterations):
        # Restore observations before guide construction, as in ISCE3.
        working = np.where(valid, current, np.nan)
        working[valid] = original[valid]
        guide = _build_filter_guide(working, settings, kernels)

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
                    guide = _build_filter_guide(cleaned, settings, kernels)

        # Fill the guide first. Unlike donor_only, these modeled samples
        # contribute to the final Gaussian filter, even in large gaps.
        guide_valid = np.isfinite(guide) & domain
        filled = _fill_from_samples(guide, guide_valid, settings.filling_method)
        # Restore ORIGINAL accepted observations before the final filtering.
        filled[retained] = data[retained]
        if uncertainty is None:
            precision = domain.astype(np.float64)
            filled_sigma = None
        else:
            filled_sigma = _fill_from_samples(
                uncertainty, retained, settings.filling_method
            )
            filled_sigma[retained] = uncertainty[retained]
            precision = np.where(domain, 1.0 / filled_sigma**2, 0.0)
        current, denominator = _normalized_filter(filled, precision, kernels)
        current[~domain] = np.nan
        if filled_sigma is not None:
            # Formal independent-error propagation; interpolation covariance
            # and uncertainty of filled/model values are not captured.
            variance_num = _convolve(
                np.where(domain, precision**2 * filled_sigma**2, 0.0),
                kernels,
                squared=True,
            )
            variance = np.divide(
                variance_num,
                denominator**2,
                out=np.full(data.shape, np.nan),
                where=denominator > 0,
            )
            output_std = np.sqrt(variance)
            output_std[~domain] = np.nan

    support, _ = _normalized_filter(valid.astype(float), domain, kernels)
    distance = ndi.distance_transform_edt(~retained, sampling=spacing)
    distance[~domain] = np.inf
    if settings.restore_mask:
        current[~retained] = np.nan
        if output_std is not None:
            output_std[~retained] = np.nan
    return FilterResult(
        current.astype(np.float32),
        None if output_std is None else output_std.astype(np.float32),
        support.astype(np.float32),
        distance.astype(np.float32),
        retained.astype(np.float32),
    )


def apply_ionosphere_filter(
    phase, *, donor_mask, footprint, pixel_size_m, options, phase_std=None
):
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
    if settings.method == "isce3":
        return filter_ionosphere_isce3(
            phase,
            donor_mask=donor_mask,
            footprint=footprint,
            pixel_size_m=pixel_size_m,
            settings=settings,
            sigma_m=sigma_m,
            donor_erosion_px=erosion,
            phase_std=phase_std,
        )
    if settings.method == "donor_only":
        return filter_ionosphere_phase(
            phase,
            donor_mask=donor_mask,
            footprint=footprint,
            pixel_size_m=pixel_size_m,
            phase_std=phase_std,
            config=FilterConfig(
                sigma_m=sigma_m,
                donor_erosion_px=erosion,
                max_fill_m=settings.max_fill_m,
                min_support=settings.min_support,
                robust_passes=settings.robust_passes,
                robust_cutoff=settings.robust_cutoff,
                robust_floor_rad=settings.robust_floor_rad,
                std_floor_rad=settings.std_floor_rad,
                truncate=settings.truncate,
            ),
        )
    # Reuse the existing implementation rather than copying its nearest-fill
    # and Gaussian operations. Its boundary/kernel behavior is preserved.
    from .utils import smooth_iono

    data = _float_array(phase)
    domain = _mask_array(footprint, data.shape)
    valid = _mask_array(donor_mask, data.shape) & domain & np.isfinite(data)
    if any(erosion):
        gaps = ndi.maximum_filter(
            (~valid).astype(np.uint8),
            size=tuple(2 * n + 1 for n in erosion),
            mode="nearest",
        )
        valid &= gaps == 0
    if not valid.any():
        empty = np.full(data.shape, np.nan, np.float32)
        return FilterResult(
            empty,
            None,
            np.zeros(data.shape, np.float32),
            np.full(data.shape, np.inf, np.float32),
            np.zeros(data.shape, np.float32),
        )
    sigma_px = tuple(s / d for s, d in zip(sigma_m, pixel_size_m))
    filtered = smooth_iono(
        data, sigma=sigma_px, mask=valid, restore_mask=settings.restore_mask
    )
    filtered[~domain] = np.nan
    return FilterResult(
        filtered,
        None,
        valid.astype(np.float32),
        ndi.distance_transform_edt(~valid, sampling=pixel_size_m).astype(np.float32),
        valid.astype(np.float32),
    )


def filter_cache_signature(raw_path, options):
    """Identify raw input and all selected filter settings for cache reuse.

    Source modification time and size invalidate the cache after inversion.
    Version the implementation to invalidate earlier file-existence caches.
    """
    import hashlib

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

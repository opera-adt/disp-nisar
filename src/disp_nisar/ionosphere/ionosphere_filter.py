"""Experimental donor-only filtering of unwrapped ionospheric phase (radians).

Dependencies: numpy, scipy; rasterio only for filter_ionosphere_raster().
Use the unfilled ionosphere estimate, never a corrected displacement/IFG.

To experiment on saved, aligned rasters without repeating phase linking,
unwrapping, or time-series inversion:

    filter_ionosphere_raster(
        "20260624_20260706_iono_B.rad.tif",
        "20260624_20260706_iono_B_filter_v2.rad.tif",
        footprint_path="common_footprint_B.tif",
        donor_mask_path="trusted_donors_B.tif",  # optional
        std_path=None,  # optional calibrated ionosphere std raster
        diagnostics_path="20260624_20260706_filter_diagnostics.tif",
        config=FilterConfig(sigma_m=(300, 300), max_fill_m=600),
    )

The footprint/donor names above are examples, not assumed pipeline outputs.
Use masks on the B grid and a new output name when comparing parameter sets.

Example inside the reviewed disp-nisar main_diff.py, in place of the
apply_similarity_mask_and_fill() call (keep the existing correction writer):

    result = filter_ionosphere_phase(
        iono, donor_mask=trusted, footprint=footprint,
        pixel_size_m=(dy_m, dx_m),
        config=FilterConfig(sigma_m=(300, 300), max_fill_m=600,
                            donor_erosion_px=(0, options.mask_erosion_px)),
        phase_std=None,  # optional calibrated ionosphere uncertainty, radians
    )
    iono_filled = result.phase

All arrays must already have the same grid, phase convention, and reference.
Zero phase is valid. Unreferenced unwrap components must be excluded upstream.
Widths are Gaussian standard deviations, NOT kernel radii or FWHM.
The output reference value can change; re-reference through the existing
correction writer after matching the frequency-A grid.

This original NumPy/SciPy prototype follows the inverse-variance filtering
principle in ISCE3 filter_data_with_sig, with disp-nisar-specific mask/distance
controls. It does not copy ISCE3's file-based low_pass_filter implementation:
https://github.com/isce-framework/isce3/blob/develop/python/packages/isce3/atmosphere/ionosphere_filter.py
https://github.com/isce-framework/isce3/blob/develop/python/packages/nisar/workflows/ionosphere.py
Source revision reviewed: ba0f9960359cebdba34b6bde099e306cf6db1eba.
Validation: 18 synthetic and GeoTIFF tests, including block/full-array equality.
Not validated with real NISAR frames or the compiled ISCE3 filter backend.

Formal output uncertainty assumes independent input errors and treats fitted
robust weights as fixed. It excludes spatial/temporal covariance, reference
uncertainty, filtering bias, and interpolation/model error. Similarity and
temporal coherence are not calibrated phase standard deviations.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import json
import uuid

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class FilterConfig:
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
        if len(self.sigma_m) != 2 or any(
            not np.isfinite(s) or s <= 0 for s in self.sigma_m
        ):
            raise ValueError("sigma_m must contain two positive finite widths")
        if not np.isfinite(self.max_fill_m) or self.max_fill_m < 0:
            raise ValueError("max_fill_m must be finite and nonnegative")
        if not 0 < self.min_support <= 1:
            raise ValueError("min_support must be in (0, 1]")
        if len(self.donor_erosion_px) != 2 or any(
            not isinstance(v, (int, np.integer)) or v < 0
            for v in self.donor_erosion_px
        ):
            raise ValueError("donor_erosion_px must contain two nonnegative integers")
        if not isinstance(self.robust_passes, (int, np.integer)) or not 0 <= self.robust_passes <= 5:
            raise ValueError("robust_passes must be an integer from zero to five")
        for value in (self.robust_cutoff, self.robust_floor_rad,
                      self.std_floor_rad, self.truncate):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("cutoff, floors, and truncate must be positive and finite")


@dataclass
class FilterResult:
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
            array, kernel ** 2 if squared else kernel,
            axis=axis, mode="constant", cval=0.0,
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
            size=tuple(2 * n + 1 for n in cfg.donor_erosion_px), mode="nearest",
        )
        valid &= gaps == 0

    nan = np.full(shape, np.nan, dtype=np.float32)
    zero = np.zeros(shape, dtype=np.float32)
    if not valid.any():
        return FilterResult(nan, None if uncertainty is None else nan.copy(),
                            zero, np.full(shape, np.inf, dtype=np.float32), zero.copy())

    kernels = tuple(_gaussian_kernel(s / d, cfg.truncate)
                    for s, d in zip(cfg.sigma_m, spacing))
    distance, nearest = ndi.distance_transform_edt(
        ~valid, sampling=spacing, return_indices=True,
    )
    support_numerator = _convolve(valid.astype(np.float64), kernels)
    support_denominator = _convolve(domain.astype(np.float64), kernels)
    support = np.divide(support_numerator, support_denominator,
                        out=np.zeros(shape), where=support_denominator > 0)
    allowed = domain & (
        valid
        | (
            (distance <= cfg.max_fill_m)
            & (support >= cfg.min_support)
        )
        )
    sample = np.where(valid, data, 0.0)  # never multiply NaN by zero
    base_weight = valid.astype(np.float64)
    if uncertainty is not None:
        base_weight /= uncertainty ** 2
    robust = valid.astype(np.float64)

    # Nearest values are used ONLY to construct a local spike-detection guide.
    # They never become observations or receive precision in the final fit.
    if cfg.robust_passes:
        guide = ndi.median_filter(data[tuple(nearest)], size=3, mode="nearest")
        residual_at_donors = np.where(valid, data - guide, 0.0)
        local_mad = ndi.median_filter(
            np.abs(residual_at_donors[tuple(nearest)]), size=7, mode="nearest",
        )
        scale = np.maximum(1.4826 * local_mad, cfg.robust_floor_rad)
        if uncertainty is not None:
            scale = np.maximum(scale, uncertainty)

    def fit(weights):
        denominator = _convolve(weights, kernels)
        numerator = _convolve(sample * weights, kernels)
        fitted = np.divide(numerator, denominator, out=np.full(shape, np.nan),
                           where=denominator > np.finfo(float).tiny)
        return fitted, denominator

    for _ in range(cfg.robust_passes):
        residual = np.where(valid, data - guide, 0.0)
        ratio = np.divide(cfg.robust_cutoff * scale, np.abs(residual),
                          out=np.ones(shape), where=np.abs(residual) > 0)
        # Non-increasing weights: rejected spikes cannot return at later passes.
        robust = np.minimum(robust, np.minimum(1.0, ratio) ** 2)
        guide, _ = fit(base_weight * robust)
    weights = base_weight * robust
    fitted, denominator = fit(weights)
    fitted[~allowed] = np.nan

    filtered_std = None
    if uncertainty is not None:
        # Squared kernel, NOT Gaussian smoothing of the std-dev image.
        variance_numerator = _convolve(weights ** 2 * uncertainty ** 2,
                                       kernels, squared=True)
        variance = np.divide(variance_numerator, denominator ** 2,
                             out=np.full(shape, np.nan), where=denominator > 0)
        filtered_std = np.sqrt(variance)
        filtered_std[~allowed] = np.nan
        filtered_std = filtered_std.astype(np.float32)
    distance = np.where(domain & (distance <= cfg.max_fill_m), distance, np.inf)
    support[~domain] = np.nan
    return FilterResult(fitted.astype(np.float32), filtered_std,
                        support.astype(np.float32), distance.astype(np.float32),
                        robust.astype(np.float32))


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
    inputs = [Path(p).resolve() for p in
              (input_path, footprint_path, donor_mask_path, std_path) if p is not None]
    outputs = [p for p in (destination, diagnostic) if p is not None]
    if len(set(outputs)) != len(outputs) or any(p in inputs for p in outputs):
        raise ValueError("Outputs must differ from one another and from all inputs")
    for path in outputs:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = [p.with_name(p.stem + "." + uuid.uuid4().hex + ".partial.tif") for p in outputs]
    try:
        with ExitStack() as stack:
            source = stack.enter_context(rasterio.open(input_path))
            if np.dtype(source.dtypes[0]).kind == "c":
                raise ValueError("Input must be real unwrapped ionosphere phase")
            footprint_ds = stack.enter_context(rasterio.open(footprint_path))
            donor_ds = stack.enter_context(rasterio.open(donor_mask_path)) if donor_mask_path else None
            std_ds = stack.enter_context(rasterio.open(std_path)) if std_path else None
            for other in (footprint_ds, donor_ds, std_ds):
                if other and (other.shape != source.shape or other.transform != source.transform
                              or other.crs != source.crs):
                    raise ValueError("Auxiliary raster does not match the ionosphere grid")
            if source.crs is None or not source.crs.is_projected:
                raise ValueError("Meter-based filtering requires a projected CRS")
            affine = source.transform
            column_vector = np.array([affine.a, affine.d])
            row_vector = np.array([affine.b, affine.e])
            if abs(np.dot(column_vector, row_vector)) > 1e-8 * np.linalg.norm(column_vector) * np.linalg.norm(row_vector):
                raise ValueError("Sheared grids require resampling before filtering")
            meters_per_unit = source.crs.linear_units_factor[1]
            spacing = np.array([np.linalg.norm(row_vector), np.linalg.norm(column_vector)]) * meters_per_unit
            if not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError("Invalid pixel spacing")
            radius = np.array([int(cfg.truncate * s / d + 0.5)
                               for s, d in zip(cfg.sigma_m, spacing)])
            # Includes original-donor distance, median/MAD guide stencils,
            # erosion, and the dependency radius of each robust fit.
            halo = ((cfg.robust_passes + 1) * radius
                    + np.ceil(cfg.max_fill_m / spacing).astype(int)
                    + np.array(cfg.donor_erosion_px) + 8)
            profile = source.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan,
                           tiled=True, blockxsize=256, blockysize=256,
                           compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
            writer = stack.enter_context(rasterio.open(temporary[0], "w", **profile))
            writer.update_tags(**source.tags())
            writer.update_tags(ionosphere_filter="donor_only_weighted_gaussian",
                               filter_config=json.dumps(cfg.__dict__))
            writer.set_band_unit(1, "radians")
            diag_writer = None
            if diagnostic:
                profile.update(count=4)
                diag_writer = stack.enter_context(rasterio.open(temporary[1], "w", **profile))
                for band, name in enumerate(("formal_std_rad", "original_donor_support",
                                             "bounded_donor_distance_m", "robust_donor_weight"), 1):
                    diag_writer.set_band_description(band, name)
                diag_writer.update_tags(uncertainty="independent_errors_conditional_on_robust_weights")
            def read(ds, window):
                return ds.read(1, window=window, masked=True).astype(np.float64).filled(np.nan)
            height, width = source.shape
            for row in range(0, height, block_size):
                for col in range(0, width, block_size):
                    end_row, end_col = min(row + block_size, height), min(col + block_size, width)
                    start_row, start_col = max(0, row - halo[0]), max(0, col - halo[1])
                    stop_row, stop_col = min(height, end_row + halo[0]), min(width, end_col + halo[1])
                    read_window = Window(int(start_col), int(start_row),
                                         int(stop_col - start_col), int(stop_row - start_row))
                    values = read(source, read_window)
                    domain = read(footprint_ds, read_window)
                    donors = read(donor_ds, read_window) if donor_ds else np.isfinite(values)
                    result = filter_ionosphere_phase(
                        values, donor_mask=donors, footprint=domain, pixel_size_m=tuple(spacing),
                        phase_std=read(std_ds, read_window) if std_ds else None, config=cfg,
                    )
                    crop = np.s_[int(row - start_row):int(end_row - start_row),
                                 int(col - start_col):int(end_col - start_col)]
                    write_window = Window(col, row, end_col - col, end_row - row)
                    writer.write(result.phase[crop], 1, window=write_window)
                    if diag_writer:
                        std_band = result.std if result.std is not None else np.full(values.shape, np.nan, dtype=np.float32)
                        for band, values_out in enumerate((std_band, result.support, result.distance_m,
                                                           result.robust_weight), 1):
                            diag_writer.write(values_out[crop], band, window=write_window)
        for partial, final in zip(temporary, outputs):
            partial.replace(final)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return str(destination)

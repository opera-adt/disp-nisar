"""DISP-specific settings for optional main-differential estimation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IonosphereFilterOptions(BaseModel):
    """Algorithm-file controls for filtering unwrapped ionospheric phase.

    Axis pairs are (row, column). Widths are Gaussian standard deviations
    in meters; kernel sizes, when supplied, are full odd lengths in pixels.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method: Literal["isce3", "smooth_iono", "donor_only"] = "isce3"
    sigma_m: tuple[float, float] | None = None
    kernel_size_px: tuple[int, int] | None = None
    truncate: float = Field(4.0, gt=0)
    iterations: int = Field(1, ge=1, le=10)
    filling_method: Literal["nearest", "linear"] = "nearest"
    guide_filter_method: Literal["median_gaussian", "gaussian", "none"] = (
        "median_gaussian"
    )
    guide_median_size: int = Field(3, ge=1)
    outlier_threshold: float | None = Field(3.5, gt=0)
    outlier_min_scale_rad: float = Field(0.1, gt=0)
    std_floor_rad: float = Field(0.05, gt=0)
    donor_erosion_px: tuple[int, int] | None = None
    restore_mask: bool = False
    # These controls apply only to the retained donor_only method.
    max_fill_m: float = Field(600.0, ge=0)
    min_support: float = Field(0.10, gt=0, le=1)
    robust_passes: int = Field(1, ge=0, le=5)
    robust_cutoff: float = Field(4.0, gt=0)
    robust_floor_rad: float = Field(0.10, gt=0)

    @field_validator("sigma_m")
    @classmethod
    def positive_widths(cls, value):
        """Require finite, positive Gaussian widths when explicitly supplied."""
        if value is not None and any(v <= 0 for v in value):
            raise ValueError("sigma_m widths must be positive")
        return value

    @field_validator("kernel_size_px")
    @classmethod
    def odd_kernel_sizes(cls, value):
        """Require positive odd full kernel lengths."""
        if value is not None and any(v < 1 or v % 2 != 1 for v in value):
            raise ValueError("kernel_size_px lengths must be positive and odd")
        return value

    @field_validator("guide_median_size")
    @classmethod
    def odd_median_size(cls, value):
        """Keep the median guide centered on each pixel."""
        if value % 2 != 1:
            raise ValueError("guide_median_size must be odd")
        return value

    @field_validator("donor_erosion_px")
    @classmethod
    def nonnegative_erosion(cls, value):
        """Require nonnegative erosion radii in pixels."""
        if value is not None and any(v < 0 for v in value):
            raise ValueError("donor_erosion_px must be nonnegative")
        return value


class IonosphereOptions(BaseModel):
    """Retain the existing estimator by default; main_diff is opt-in."""

    model_config = ConfigDict(extra="forbid")
    method: Literal["main_side", "main_diff"] = "main_side"
    mask_reduction: Literal["all_valid", "nearest"] = "all_valid"
    correlation_threshold: float = Field(0.2, gt=0, le=1)
    similarity_threshold: float = Field(0.4, gt=0, le=1)
    min_valid_fraction: float = Field(1.0, gt=0, le=1)
    min_phasor_magnitude: float = Field(0.1, gt=0, le=1)
    block_size: int = Field(256, ge=16, le=2048)
    mask_work_mb: int = Field(128, ge=16)
    smooth_sigma_m: float = Field(400.0, gt=0, allow_inf_nan=False)
    mask_erosion_px: int = Field(3, ge=0)
    filtering: IonosphereFilterOptions = Field(default_factory=IonosphereFilterOptions)
    nlooks: float = Field(
        5.0,
        gt=0,
        description=(
            "Effective looks for the correlation proxy; validate for the chosen PL"
            " settings."
        ),
    )
    resume: bool = True
    keep_intermediates: bool = False

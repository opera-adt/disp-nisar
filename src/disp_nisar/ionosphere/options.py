"""DISP-specific settings for optional main-differential estimation."""

from typing import Literal

from dolphin.workflows.config import YamlModel
from pydantic import ConfigDict, Field, field_validator


class IonosphereFilterOptions(YamlModel):
    """Algorithm-file controls for filtering unwrapped ionospheric phase.

    Axis pairs are (row, column). Widths are Gaussian standard deviations
    in meters; kernel sizes, when supplied, are full odd lengths in pixels.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method: Literal["guided", "smooth_iono"] = Field(
        "guided", description="Filter applied to the ionospheric phase"
    )
    sigma_m: tuple[float, float] | None = Field(
        None,
        description=(
            "Gaussian standard deviations (row, column) in meters. If None, uses"
            " smooth_sigma_m"
        ),
    )
    kernel_size_px: tuple[int, int] | None = Field(
        None,
        description=(
            "Full odd kernel lengths (row, column) in pixels. If None, derived from"
            " sigma_m and truncate"
        ),
    )
    truncate: float = Field(
        4.0, gt=0, description="Kernel half-width, in standard deviations"
    )
    filling_method: Literal["nearest", "linear"] = Field(
        "nearest", description="How gaps in the guide are filled before filtering"
    )
    guide_filter_method: Literal["median_gaussian", "gaussian", "none"] = Field(
        "median_gaussian", description="Filter used to build the guide image"
    )
    guide_median_size: int = Field(
        3, ge=1, description="Median window of the guide, in pixels. Must be odd"
    )
    outlier_threshold: float | None = Field(
        3.5,
        gt=0,
        description=(
            "Reject samples this many robust deviations from the guide. If None,"
            " disables rejection"
        ),
    )
    outlier_min_scale_rad: float = Field(
        0.1, gt=0, description="Lower limit on the robust deviation, in radians"
    )
    donor_erosion_px: tuple[int, int] | None = Field(
        None,
        description=(
            "Extra exclusion radii (row, column) around invalid donors, in pixels. If"
            " None, uses (0, mask_erosion_px)"
        ),
    )
    restore_mask: bool = Field(
        False, description="Set pixels outside the donors back to NaN after filtering"
    )

    @field_validator("sigma_m")
    @classmethod
    def positive_widths(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Require finite, positive Gaussian widths when explicitly supplied."""
        if value is not None and any(v <= 0 for v in value):
            raise ValueError("sigma_m widths must be positive")
        return value

    @field_validator("kernel_size_px")
    @classmethod
    def odd_kernel_sizes(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        """Require positive odd full kernel lengths."""
        if value is not None and any(v < 1 or v % 2 != 1 for v in value):
            raise ValueError("kernel_size_px lengths must be positive and odd")
        return value

    @field_validator("guide_median_size")
    @classmethod
    def odd_median_size(cls, value: int) -> int:
        """Keep the median guide centered on each pixel."""
        if value % 2 != 1:
            raise ValueError("guide_median_size must be odd")
        return value

    @field_validator("donor_erosion_px")
    @classmethod
    def nonnegative_erosion(
        cls, value: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        """Require nonnegative erosion radii in pixels."""
        if value is not None and any(v < 0 for v in value):
            raise ValueError("donor_erosion_px must be nonnegative")
        return value


class IonosphereOptions(YamlModel):
    """Retain the existing estimator by default; main_diff is opt-in."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["main_side", "main_diff"] = Field(
        "main_side",
        description=(
            "Dual-band estimator. main_side runs the split-spectrum workflow on"
            " separately unwrapped bands; main_diff unwraps A and A*conj(B)"
        ),
    )
    mask_reduction: Literal["all_valid", "nearest"] = Field(
        "all_valid", description="How GSLC masks are reduced to the IFG grids"
    )
    correlation_threshold: float = Field(
        0.2, gt=0, le=1, description="Smallest correlation of a donor pixel"
    )
    similarity_threshold: float = Field(
        0.4, gt=0, le=1, description="Smallest phase similarity of a donor pixel"
    )
    min_valid_fraction: float = Field(
        1.0,
        gt=0,
        le=1,
        description="Valid area fraction needed to resample A onto the B grid",
    )
    min_phasor_magnitude: float = Field(
        0.1,
        gt=0,
        le=1,
        description="Smallest mean phasor magnitude when resampling A onto the B grid",
    )
    block_size: int = Field(
        256, ge=16, le=2048, description="Processing block size, in pixels"
    )
    mask_work_mb: int = Field(
        128, ge=16, description="Memory limit for reducing one GSLC mask, in MB"
    )
    smooth_sigma_m: float = Field(
        400.0,
        gt=0,
        allow_inf_nan=False,
        description="Gaussian width in meters, used if filtering.sigma_m is None",
    )
    mask_erosion_px: int = Field(
        3,
        ge=0,
        description=(
            "Column erosion in pixels, used if filtering.donor_erosion_px is None"
        ),
    )
    filtering: IonosphereFilterOptions = Field(default_factory=IonosphereFilterOptions)
    nlooks: float = Field(
        5.0,
        gt=0,
        description=(
            "Effective looks for the correlation proxy; validate for the chosen PL"
            " settings"
        ),
    )
    resume: bool = Field(
        True, description="Reuse completed stages in the output folder"
    )
    keep_intermediates: bool = Field(
        False, description="Keep prepared rasters and unwrapper scratch files"
    )

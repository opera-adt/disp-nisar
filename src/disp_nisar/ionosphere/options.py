"""DISP-specific settings for optional main-differential estimation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

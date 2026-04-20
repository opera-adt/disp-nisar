"""Ionosphere correction for NISAR DISP.

Two workflows are provided:

GUNW-based
    Read ionosphere phase screens from NISAR GUNW HDF5 products and invert
    to a displacement timeseries.  Entry point:
    :func:`~disp_nisar.ionosphere.gunw.read_ionosphere_phase_screen`.

GSLC split-spectrum (freqA / freqB)
    Estimate ionosphere using the main-side band two-frequency method from
    co-registered freqA and freqB interferograms.  Entry points:
    :func:`~disp_nisar.ionosphere.gslc.run_ionosphere_estimation` and
    :func:`~disp_nisar.ionosphere.gslc.estimate_pair_ionosphere`.

Sub-modules
-----------
gunw        GUNW workflow
gslc        GSLC split-spectrum workflow
inversion   Least-squares network inversion (shared)
mask        Masking and gap-filling utilities (shared)
utils       Unit conversions, smoothing, and I/O helpers (shared)
"""

from .gslc import (
    estimate_pair_ionosphere,
    get_center_frequencies,
    run_ionosphere_estimation,
)
from .gunw import (
    apply_ionosphere_corrections,
    get_gunw_dates,
    read_ionosphere_from_gunw,
    read_ionosphere_phase_screen,
)
from .inversion import build_design_matrix, invert_ifg_to_timeseries
from .mask import (
    apply_similarity_mask_and_fill,
    fill_gaps,
    make_crlb_mask,
    make_similarity_mask,
    make_valid_mask,
    mask_iono_outliers,
)
from .utils import (
    C_LIGHT,
    coh_to_phase_sigma,
    disp_to_phase,
    freq_to_wavelength,
    phase_to_disp,
    read_reference_point,
    read_tif,
    resample_to_match,
    sigma_to_spatial,
    smooth_iono,
    write_tif,
)

__all__ = [
    # GUNW workflow
    "read_ionosphere_phase_screen",
    "apply_ionosphere_corrections",
    "get_gunw_dates",
    "read_ionosphere_from_gunw",
    # GSLC workflow
    "run_ionosphere_estimation",
    "estimate_pair_ionosphere",
    "get_center_frequencies",
    # Inversion
    "build_design_matrix",
    "invert_ifg_to_timeseries",
    # Masking
    "make_valid_mask",
    "fill_gaps",
    "make_similarity_mask",
    "make_crlb_mask",
    "apply_similarity_mask_and_fill",
    "mask_iono_outliers",
    # Utils
    "C_LIGHT",
    "freq_to_wavelength",
    "disp_to_phase",
    "phase_to_disp",
    "coh_to_phase_sigma",
    "sigma_to_spatial",
    "smooth_iono",
    "resample_to_match",
    "read_tif",
    "write_tif",
    "read_reference_point",
]

"""GSLC split-spectrum ionosphere estimation workflow.

Implements the main-side band (two-frequency) method to separate dispersive
(ionospheric) and non-dispersive phase from co-registered freqA and freqB
NISAR GSLC interferograms.

Reference
---------
Extracted and refined from:
  isce3/atmosphere/main_band_estimation.py
  isce3/atmosphere/ionosphere_estimation.py
  isce3/atmosphere/ionosphere_filter.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from .mask import make_valid_mask
from .utils import (
    disp_to_phase,
    freq_to_wavelength,
    phase_to_disp,
    read_reference_point,
    read_tif,
    resample_to_match,
    smooth_iono,
    write_tif,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NISAR GSLC HDF5 paths
# ---------------------------------------------------------------------------

GSLC_CENTER_FREQ_PATH = "/science/LSAR/GSLC/grids/{frequency}/centerFrequency"


# ---------------------------------------------------------------------------
# GSLC metadata
# ---------------------------------------------------------------------------


def get_center_frequencies(gslc_path: str | Path) -> tuple[float, float]:
    """Read freqA and freqB center frequencies (Hz) from a NISAR GSLC HDF5 file.

    Parameters
    ----------
    gslc_path : str or Path
        Path to a NISAR GSLC ``.h5`` file.

    Returns
    -------
    f_A, f_B : float
        Center frequencies in Hz for frequencyA and frequencyB respectively.

    """
    with h5py.File(gslc_path, "r") as f:
        f_A = float(f[GSLC_CENTER_FREQ_PATH.format(frequency="frequencyA")][()])
        f_B = float(f[GSLC_CENTER_FREQ_PATH.format(frequency="frequencyB")][()])
    return f_A, f_B


# ---------------------------------------------------------------------------
# Core two-frequency estimation
# ---------------------------------------------------------------------------


def estimate_iono_main_side(
    f_A: float,
    f_B: float,
    phi_A: np.ndarray,
    phi_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate dispersive (ionospheric) and non-dispersive phase.

    Uses the main-side band (two-frequency) method.  The system is:

    .. code-block:: text

        phi_A = phi_nondisp  +  phi_disp
        phi_B = (f_B/f_A) * phi_nondisp  +  (f_A/f_B) * phi_disp

    Solved exactly via the 2×2 matrix inverse.

    Parameters
    ----------
    f_A, f_B : float
        Center frequencies in Hz (NISAR freqA ~1.257 GHz, freqB ~1.369 GHz).
    phi_A, phi_B : np.ndarray of shape (rows, cols)
        Unwrapped phase arrays in radians.  ``phi_B`` must already be on the
        same grid as ``phi_A`` — call ``resample_to_match`` first if needed.
        NaN pixels are accepted; fill them with 0 before calling if needed.

    Returns
    -------
    dispersive : np.ndarray
        Ionospheric (dispersive) phase in radians, same shape as inputs.
    non_dispersive : np.ndarray
        Non-dispersive (surface + troposphere) phase in radians, same shape.

    """
    if phi_A.shape != phi_B.shape:
        raise ValueError(
            f"freqA {phi_A.shape} and freqB {phi_B.shape} must be on the same grid; "
            "call resample_to_match() first."
        )

    rows, cols = phi_A.shape
    M = np.array(
        [[1.0, 1.0], [f_B / f_A, f_A / f_B]],
        dtype=np.float64,
    )
    M_inv = np.linalg.inv(M)

    d = np.stack(
        [phi_A.ravel().astype(np.float64), phi_B.ravel().astype(np.float64)],
        axis=0,
    )
    out = M_inv @ d

    non_dispersive = out[0].reshape(rows, cols).astype(np.float32)
    dispersive = out[1].reshape(rows, cols).astype(np.float32)
    return dispersive, non_dispersive


def estimate_sigma_main_side(
    f_A: float,
    f_B: float,
    sig_phi_A: np.ndarray,
    sig_phi_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Theoretical phase std-dev of ionosphere and non-dispersive components.

    Propagates coherence-derived phase uncertainties through the main-side band
    linear system.

    Parameters
    ----------
    f_A, f_B : float
        Center frequencies in Hz.
    sig_phi_A, sig_phi_B : np.ndarray
        Phase standard deviations derived from coherence:
        ``sigma = sqrt(1 - coh^2) / (coh * sqrt(2 * n_looks))``.

    Returns
    -------
    sig_iono, sig_nondisp : np.ndarray
        Propagated phase standard deviations for dispersive and non-dispersive
        components respectively.

    """
    a = (f_B**2) / (f_B**2 - f_A**2)
    b = (f_A * f_B) / (f_B**2 - f_A**2)
    c = f_A / (f_B**2 - f_A**2)

    sig_iono = np.sqrt(a**2 * sig_phi_A**2 + b**2 * sig_phi_B**2)
    sig_nondisp = np.sqrt(c**2 * (f_A**2 * sig_phi_A**2 + f_B**2 * sig_phi_B**2))
    return sig_iono, sig_nondisp


# ---------------------------------------------------------------------------
# Single date-pair pipeline
# ---------------------------------------------------------------------------


def estimate_pair_ionosphere(
    disp_A_path: str | Path,
    disp_B_path: str | Path,
    f_A: float,
    f_B: float,
    ref_point: tuple[int, int] | None = None,
    smooth_sigma: float = 0.0,
    nodata_A: float = 0.0,
    nodata_B: float = 0.0,
) -> dict:
    """Estimate ionosphere for one date-pair from freqA and freqB displacements.

    FreqB is reprojected onto the freqA grid (bilinear), then re-referenced to
    the same geographic point as freqA before estimation.

    Parameters
    ----------
    disp_A_path, disp_B_path : str or Path
        Displacement GeoTIFFs in **metres** (dolphin timeseries output).
    f_A, f_B : float
        Center frequencies in Hz (from GSLC metadata via
        ``get_center_frequencies``).
    ref_point : (row, col) or None
        Reference pixel in freqA grid coordinates (from ``reference_point.txt``).
        FreqB displacement is re-referenced to this geographic location so that
        both arrays share the same zero-reference before estimation.
        Use ``read_reference_point()`` to load from file.  If None, no
        re-referencing is applied.
    smooth_sigma : float
        Gaussian smoothing sigma in pixels applied after estimation.
        0 = no smoothing.
    nodata_A, nodata_B : float
        Nodata sentinel values (dolphin default is 0.0).

    Returns
    -------
    dict
        Keys:

        ``disp_A``
            FreqA displacement (m), NaN-masked.
        ``disp_B_reproj``
            FreqB reprojected and re-referenced (m).
        ``dispersive_phase_rad``
            Ionospheric phase (rad) on freqA grid.
        ``non_dispersive_phase_rad``
            Non-dispersive phase (rad) on freqA grid.
        ``iono_disp_m``
            Ionospheric LOS displacement (m).
        ``non_disp_m``
            Non-dispersive LOS displacement (m).
        ``mask``
            Valid-pixel boolean mask.
        ``ref_offset_B_m``
            FreqB offset subtracted at the reference point (m).

    """
    wl_A = freq_to_wavelength(f_A)
    wl_B = freq_to_wavelength(f_B)

    disp_A = read_tif(disp_A_path)
    disp_B = resample_to_match(disp_B_path, disp_A_path)

    # Re-reference freqB to the freqA reference pixel so both arrays share the
    # same geographic zero-reference before estimation.
    ref_offset_B = 0.0
    if ref_point is not None:
        ref_row, ref_col = ref_point
        ref_offset_B = float(disp_B[ref_row, ref_col])
        if not np.isfinite(ref_offset_B):
            r0 = max(0, ref_row - 3)
            r1 = min(disp_B.shape[0], ref_row + 4)
            c0 = max(0, ref_col - 3)
            c1 = min(disp_B.shape[1], ref_col + 4)
            ref_offset_B = float(np.nanmean(disp_B[r0:r1, c0:c1]))
        disp_B = disp_B - ref_offset_B

    mask = make_valid_mask(disp_A, disp_B, nodata_A=nodata_A, nodata_B=nodata_B)

    phi_A = disp_to_phase(np.where(mask, disp_A, 0.0), wl_A)
    phi_B = disp_to_phase(np.where(mask, disp_B, 0.0), wl_B)

    disp_ph, non_disp_ph = estimate_iono_main_side(f_A, f_B, phi_A, phi_B)

    disp_ph[~mask] = np.nan
    non_disp_ph[~mask] = np.nan

    if smooth_sigma > 0:
        disp_ph = smooth_iono(disp_ph, sigma=smooth_sigma, mask=mask)
        non_disp_ph = smooth_iono(non_disp_ph, sigma=smooth_sigma, mask=mask)

    iono_disp_m = phase_to_disp(disp_ph, wl_A)
    non_disp_m = phase_to_disp(non_disp_ph, wl_A)

    disp_A[~mask] = np.nan
    disp_B[~mask] = np.nan

    return {
        "disp_A": disp_A,
        "disp_B_reproj": disp_B,
        "dispersive_phase_rad": disp_ph,
        "non_dispersive_phase_rad": non_disp_ph,
        "iono_disp_m": iono_disp_m,
        "non_disp_m": non_disp_m,
        "mask": mask,
        "ref_offset_B_m": ref_offset_B,
    }


# ---------------------------------------------------------------------------
# Batch pipeline over all date pairs
# ---------------------------------------------------------------------------


def run_ionosphere_estimation(
    ts_dir_A: str | Path,
    ts_dir_B: str | Path,
    out_dir: str | Path,
    f_A: float,
    f_B: float,
    ref_point_file: str | Path | None = None,
    smooth_sigma: float = 5.0,
    nodata_A: float = 0.0,
    nodata_B: float = 0.0,
    apply_correction: bool = True,
) -> list[tuple[str, Path, Path | None]]:
    """Process all matching date-pair TIFs and save ionosphere results.

    Output layout::

        {out_dir}/
            ionosphere/{pair}_iono.tif          — ionospheric displacement (m)
            iono_corrected/{pair}_corrected.tif — freqA minus ionosphere (m)
            non_dispersive/{pair}_nondisp.tif   — non-dispersive component (m)

    Parameters
    ----------
    ts_dir_A : str or Path
        Directory with freqA displacement TIFs (``2*.tif``, dolphin output).
    ts_dir_B : str or Path
        Directory with freqB displacement TIFs (same date-pair stems).
    out_dir : str or Path
        Root output directory (created if absent).
    f_A, f_B : float
        Center frequencies in Hz.
    ref_point_file : str or Path or None
        Path to freqA ``reference_point.txt`` (``row,col`` in freqA grid).
        FreqB is re-referenced to that geographic location before estimation.
        Defaults to ``{ts_dir_A}/reference_point.txt`` if the file exists.
    smooth_sigma : float
        Gaussian smoothing σ in pixels (0 = off).
    nodata_A, nodata_B : float
        Nodata sentinel values (dolphin default 0.0).
    apply_correction : bool
        If True, also write ionosphere-corrected freqA displacement.

    Returns
    -------
    list of (date_pair_str, iono_path, corrected_path)
        ``corrected_path`` is None when ``apply_correction=False``.

    """
    ts_dir_A = Path(ts_dir_A)
    ts_dir_B = Path(ts_dir_B)
    out_dir = Path(out_dir)

    if ref_point_file is None:
        ref_point_file = ts_dir_A / "reference_point.txt"
    ref_point = None
    if Path(ref_point_file).exists():
        ref_point = read_reference_point(ref_point_file)
        print(
            f"Reference point: row={ref_point[0]}, col={ref_point[1]}"
            f"  ({Path(ref_point_file).name})"
        )
    else:
        print("WARNING: no reference_point.txt found — freqB will NOT be re-referenced")

    iono_dir = out_dir / "ionosphere"
    corr_dir = out_dir / "iono_corrected"
    nondisp_dir = out_dir / "non_dispersive"
    for d in [iono_dir, corr_dir, nondisp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    files_A = sorted(f for f in ts_dir_A.glob("2*.tif") if "iono" not in f.name)
    files_B = sorted(f for f in ts_dir_B.glob("2*.tif") if "iono" not in f.name)
    map_B = {f.stem: f for f in files_B}

    if not files_A:
        raise FileNotFoundError(f"No displacement TIFs found in {ts_dir_A}")
    if not map_B:
        raise FileNotFoundError(f"No displacement TIFs found in {ts_dir_B}")

    print(f"FreqA: {len(files_A)} pairs  |  FreqB: {len(map_B)} pairs")
    print(
        f"f_A={f_A/1e9:.4f} GHz  f_B={f_B/1e9:.4f} GHz  "
        f"λ_A={freq_to_wavelength(f_A)*100:.2f} cm  "
        f"λ_B={freq_to_wavelength(f_B)*100:.2f} cm"
    )
    print(f"Smoothing σ={smooth_sigma} px\n")

    results = []

    for f_A_path in files_A:
        stem = f_A_path.stem
        if stem not in map_B:
            print(f"  [SKIP] {stem} — no freqB match")
            continue

        iono_path = iono_dir / f"{stem}_iono.tif"
        corr_path = corr_dir / f"{stem}_corrected.tif"
        nondisp_path = nondisp_dir / f"{stem}_nondisp.tif"

        print(f"  {stem} ...", end=" ", flush=True)

        result = estimate_pair_ionosphere(
            disp_A_path=f_A_path,
            disp_B_path=map_B[stem],
            f_A=f_A,
            f_B=f_B,
            ref_point=ref_point,
            smooth_sigma=smooth_sigma,
            nodata_A=nodata_A,
            nodata_B=nodata_B,
        )

        # Use f_A_path as the georef template for all outputs
        write_tif(
            iono_path, result["iono_disp_m"], like_filename=f_A_path, units="meters"
        )
        write_tif(
            nondisp_path, result["non_disp_m"], like_filename=f_A_path, units="meters"
        )

        corrected_path = None
        if apply_correction:
            corrected = result["disp_A"] - result["iono_disp_m"]
            write_tif(corr_path, corrected, like_filename=f_A_path, units="meters")
            corrected_path = corr_path

        iono = result["iono_disp_m"]
        print(
            f"iono [{np.nanmin(iono):.4f}, {np.nanmax(iono):.4f}] m  "
            f"B_offset={result['ref_offset_B_m']*100:.2f} cm  "
            f"valid={result['mask'].sum():,}"
        )
        results.append((stem, iono_path, corrected_path))

    print(f"\nDone: {len(results)}/{len(files_A)} pairs processed.")
    return results

"""Least-squares network inversion for ionosphere timeseries.

Shared by both GUNW and GSLC workflows: builds the design matrix for a
network of interferograms and inverts it to a timeseries via normal equations.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_design_matrix(
    ifg_date_pairs: list[tuple],
    unique_dates: list,
) -> np.ndarray:
    """Build the design matrix for inverting interferograms to a timeseries.

    The first date is treated as the reference (zero displacement), so there
    are ``num_dates - 1`` unknowns.

    Parameters
    ----------
    ifg_date_pairs : list[tuple]
        List of ``(reference_date, secondary_date)`` tuples for each
        interferogram.
    unique_dates : list
        Sorted list of unique dates in the network.

    Returns
    -------
    np.ndarray
        Design matrix A of shape ``(num_ifgs, num_dates - 1)``.
        Entry ``A[i, j] = +1`` if interferogram *i* has secondary at date
        ``j+1``, ``-1`` if reference at date ``j+1``, 0 otherwise.

    """
    num_ifgs = len(ifg_date_pairs)
    num_dates = len(unique_dates)

    A = np.zeros((num_ifgs, num_dates - 1), dtype=np.float32)
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    for i, (ref_date, sec_date) in enumerate(ifg_date_pairs):
        ref_idx = date_to_idx[ref_date]
        sec_idx = date_to_idx[sec_date]
        # Interferogram = secondary - reference.
        # First date (idx=0) is the implicit zero reference → skip.
        if ref_idx > 0:
            A[i, ref_idx - 1] = -1
        if sec_idx > 0:
            A[i, sec_idx - 1] = 1

    return A


def invert_ifg_to_timeseries(
    ifg_stack: np.ndarray,
    design_matrix: np.ndarray,
    AtA_inv: np.ndarray | None = None,
) -> np.ndarray | None:
    """Invert an interferogram stack to a timeseries using least squares.

    Solves ``A @ x = b`` via pre-computed normal equations
    ``x = (A^T A)^{-1} A^T b`` for all pixels simultaneously.
    NaN pixels are zeroed before the solve and restored afterwards.

    Parameters
    ----------
    ifg_stack : np.ndarray
        Stack of interferograms, shape ``(num_ifgs, rows, cols)``.
    design_matrix : np.ndarray
        Design matrix A, shape ``(num_ifgs, num_dates - 1)``.
    AtA_inv : np.ndarray, optional
        Precomputed pseudo-inverse of ``(A^T A)``, shape
        ``(num_dates-1, num_dates-1)``.  Pass this when calling per spatial
        block to avoid recomputing the decomposition each time.

    Returns
    -------
    np.ndarray or None
        Timeseries stack, shape ``(num_dates - 1, rows, cols)``.
        The reference date (first) is implicitly zero and not included.
        Returns None if the matrix inversion fails.

    """
    num_ifgs, rows, cols = ifg_stack.shape
    num_unknowns = design_matrix.shape[1]

    ifg_flat = ifg_stack.astype(np.float32).reshape(num_ifgs, -1)

    # Track pixels with no valid data in any interferogram
    all_nan_mask = ~np.isfinite(ifg_flat).any(axis=0)

    nan_mask = ~np.isfinite(ifg_flat)
    if nan_mask.any():
        ifg_flat = ifg_flat.copy()
        ifg_flat[nan_mask] = 0.0

    if AtA_inv is None:
        AtA = design_matrix.T @ design_matrix
        rank = np.linalg.matrix_rank(AtA)
        if rank < num_unknowns:
            logger.warning(
                "Design matrix is rank-deficient (rank=%d, expected=%d)."
                " Inversion may be unstable.",
                rank,
                num_unknowns,
            )
        try:
            AtA_inv = np.linalg.pinv(AtA)
        except np.linalg.LinAlgError as e:
            logger.error("Failed to invert design matrix: %s", e)
            return None

    Atb = design_matrix.T @ ifg_flat
    timeseries_flat = AtA_inv @ Atb

    timeseries_flat[:, all_nan_mask] = np.nan
    return timeseries_flat.reshape(num_unknowns, rows, cols)

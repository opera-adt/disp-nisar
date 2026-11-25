import numpy as np
import pytest

from disp_nisar.ionosphere import build_design_matrix, invert_ifg_to_timeseries
from disp_nisar.ionosphere.inversion import invert_common_phase_block


def test_build_design_matrix():
    """Test design matrix construction for interferogram inversion."""
    # 3 dates: d0 (reference), d1, d2
    # 2 ifgs: (d0, d1) and (d0, d2)
    unique_dates = ["2020-01-01", "2020-01-13", "2020-01-25"]
    ifg_date_pairs = [
        ("2020-01-01", "2020-01-13"),
        ("2020-01-01", "2020-01-25"),
    ]
    A = build_design_matrix(ifg_date_pairs, unique_dates)

    assert A.shape == (2, 2)  # (num_ifgs, num_dates - 1)
    # ifg0 = d1 - d0 => [1, 0]
    np.testing.assert_array_equal(A[0], [1, 0])
    # ifg1 = d2 - d0 => [0, 1]
    np.testing.assert_array_equal(A[1], [0, 1])


def test_build_design_matrix_non_reference():
    """Test design matrix with ifgs not involving the first date."""
    unique_dates = ["d0", "d1", "d2", "d3"]
    ifg_date_pairs = [
        ("d0", "d1"),
        ("d1", "d2"),
        ("d0", "d3"),
    ]
    A = build_design_matrix(ifg_date_pairs, unique_dates)

    assert A.shape == (3, 3)
    # ifg0: d1 - d0 => [1, 0, 0]
    np.testing.assert_array_equal(A[0], [1, 0, 0])
    # ifg1: d2 - d1 => [-1, 1, 0]
    np.testing.assert_array_equal(A[1], [-1, 1, 0])
    # ifg2: d3 - d0 => [0, 0, 1]
    np.testing.assert_array_equal(A[2], [0, 0, 1])


def test_invert_ifg_to_timeseries():
    """Test inversion of ifg stack to timeseries with known values."""
    # Create a known timeseries: 2 dates, 3x3 pixels
    ts_true = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]],
        ],
        dtype=np.float32,
    )

    # Design matrix for 2 ifgs from 3 dates: (d0,d1), (d0,d2)
    A = np.array([[1, 0], [0, 1]], dtype=np.float32)

    # ifg stack = A @ ts
    ifg_stack = np.einsum("ij,jkl->ikl", A, ts_true)

    result = invert_ifg_to_timeseries(ifg_stack, A)
    assert result.shape == ts_true.shape
    np.testing.assert_allclose(result, ts_true, atol=1e-4)


def _network():
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    design = build_design_matrix(pairs, [0, 1, 2, 3])
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(4, 3, 3))
    truth[0] = 0
    phase = np.stack([truth[j] - truth[i] for i, j in pairs]).astype(np.float32)
    return design, truth, phase


def test_invert_common_phase_block_recovers_time_series():
    design, truth, phase = _network()
    a, d = invert_common_phase_block(
        phase, 2 * phase, design, np.ones(phase.shape, bool)
    )
    np.testing.assert_allclose(a, truth[1:], atol=1e-5)
    np.testing.assert_allclose(d, 2 * truth[1:], atol=1e-5)


def test_invert_common_phase_block_drops_missing_observations():
    """A missing pair is removed, not treated as an observation of zero."""
    design, truth, phase = _network()
    valid = np.ones(phase.shape, bool)
    valid[3, 0, 0] = False
    phase[3, 0, 0] = 1e4
    a, _ = invert_common_phase_block(phase, phase, design, valid)
    np.testing.assert_allclose(a[:, 0, 0], truth[1:, 0, 0], atol=1e-5)


def test_invert_common_phase_block_split_network_is_nan():
    design, truth, phase = _network()
    valid = np.ones(phase.shape, bool)
    # Without these pairs, no observation joins epochs 0-1 to epochs 2-3
    valid[[1, 3, 4], 2, 2] = False
    a, _ = invert_common_phase_block(phase, phase, design, valid)
    assert np.isnan(a[:, 2, 2]).all()
    np.testing.assert_allclose(a[:, 0, 0], truth[1:, 0, 0], atol=1e-5)


def test_invert_common_phase_block_rejects_mismatched_shapes():
    design, _, phase = _network()
    with pytest.raises(ValueError, match="matching shapes"):
        invert_common_phase_block(phase, phase[:-1], design, np.ones(phase.shape, bool))
    with pytest.raises(ValueError, match="matching network"):
        invert_common_phase_block(
            phase[:-1], phase[:-1], design, np.ones(phase[:-1].shape, bool)
        )

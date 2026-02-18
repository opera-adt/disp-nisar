import numpy as np

from disp_nisar.ionosphere import build_design_matrix, invert_ifg_to_timeseries


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

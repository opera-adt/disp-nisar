import h5py
import numpy as np
import pytest

from disp_nisar.validate import (
    ComparisonError,
    _fmt_ratio,
    _validate_conncomp_labels,
    _validate_dataset,
    compare_groups,
)


def test_fmt_ratio():
    assert _fmt_ratio(1, 4) == "1/4 (25.000%)"
    assert _fmt_ratio(0, 10) == "0/10 (0.000%)"
    assert _fmt_ratio(3, 3) == "3/3 (100.000%)"
    assert _fmt_ratio(1, 3, digits=1) == "1/3 (33.3%)"


def test_compare_groups_matching(tmp_path):
    """Matching HDF5 groups should not raise."""
    f1 = tmp_path / "g1.h5"
    f2 = tmp_path / "g2.h5"
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    for f in (f1, f2):
        with h5py.File(f, "w") as hf:
            ds = hf.create_dataset("values", data=data)
            ds.attrs["units"] = "meters"

    with h5py.File(f1, "r") as h1, h5py.File(f2, "r") as h2:
        compare_groups(h1, h2)  # should not raise


def test_compare_groups_mismatched_keys(tmp_path):
    """Mismatched group keys should raise ComparisonError."""
    f1 = tmp_path / "g1.h5"
    f2 = tmp_path / "g2.h5"
    with h5py.File(f1, "w") as hf:
        hf.create_dataset("a", data=[1.0])
    with h5py.File(f2, "w") as hf:
        hf.create_dataset("b", data=[1.0])

    with h5py.File(f1, "r") as h1, h5py.File(f2, "r") as h2:
        with pytest.raises(ComparisonError, match="Group keys do not match"):
            compare_groups(h1, h2)


def test_validate_dataset_matching(tmp_path):
    """Matching datasets should pass validation."""
    f = tmp_path / "test.h5"
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with h5py.File(f, "w") as hf:
        ds1 = hf.create_dataset("golden", data=data)
        ds1.attrs["units"] = "m"
        ds2 = hf.create_dataset("test", data=data)
        ds2.attrs["units"] = "m"

    with h5py.File(f, "r") as hf:
        _validate_dataset(hf["test"], hf["golden"])  # should not raise


def test_validate_dataset_failing(tmp_path):
    """Datasets with large differences should raise ComparisonError."""
    f = tmp_path / "test.h5"
    golden = np.zeros(100, dtype=np.float32)
    test = np.ones(100, dtype=np.float32)
    with h5py.File(f, "w") as hf:
        ds1 = hf.create_dataset("golden", data=golden)
        ds1.attrs["units"] = "m"
        ds2 = hf.create_dataset("test", data=test)
        ds2.attrs["units"] = "m"

    with h5py.File(f, "r") as hf:
        with pytest.raises(ComparisonError, match="values do not match"):
            _validate_dataset(hf["test"], hf["golden"])


def test_validate_conncomp_labels_good_overlap(tmp_path):
    """Good overlap between conncomp labels should pass."""
    f = tmp_path / "cc.h5"
    labels = np.ones((10, 10), dtype=np.uint16)
    labels[0, 0] = 0  # one pixel different
    with h5py.File(f, "w") as hf:
        hf.create_dataset("ref", data=labels)
        hf.create_dataset("test", data=labels)

    with h5py.File(f, "r") as hf:
        _validate_conncomp_labels(hf["test"], hf["ref"])  # should not raise


def test_validate_conncomp_labels_low_overlap(tmp_path):
    """Low overlap should raise ComparisonError."""
    f = tmp_path / "cc.h5"
    ref_labels = np.ones((10, 10), dtype=np.uint16)
    test_labels = np.zeros((10, 10), dtype=np.uint16)
    test_labels[0, 0] = 1  # only 1 pixel overlaps
    with h5py.File(f, "w") as hf:
        hf.create_dataset("ref", data=ref_labels)
        hf.create_dataset("test", data=test_labels)

    with h5py.File(f, "r") as hf:
        with pytest.raises(ComparisonError, match="failed validation"):
            _validate_conncomp_labels(hf["test"], hf["ref"])

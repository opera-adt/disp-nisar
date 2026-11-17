from contextlib import chdir
from pathlib import Path

import pytest

from disp_nisar.cli.run import run_main

TEST_DATA_DIR = Path(__file__).parent / "data/delivery_data_small"


def test_run_main():
    config_file = TEST_DATA_DIR / "config_files/runconfig_historical.yaml"
    if not config_file.exists():
        pytest.skip(f"Test data does not exist at {TEST_DATA_DIR}")

    with chdir(TEST_DATA_DIR):
        run_main(config_file="config_files/runconfig_historical.yaml")


# ---------------------------------------------------------------------------
# Input prechecks
# ---------------------------------------------------------------------------
def _real(date):
    return Path(f"NISAR_{date}T000000.h5")


def _comp(ref, start="20230101"):
    return Path(f"compressed_{ref}_{start}_{ref}.h5")


def test_no_large_temporal_gaps():
    from disp_nisar.main import InputValidationError, _assert_no_large_temporal_gaps

    ok = [_real("20240101"), _real("20240113"), _comp("20231201")]
    _assert_no_large_temporal_gaps(ok)

    with pytest.raises(InputValidationError, match="gap") as exc:
        _assert_no_large_temporal_gaps([_real("20200101"), _real("20240101")])
    assert exc.value.error_code == 1000

    # Sub-year limits are reported in days, not as a "0-year" limit
    with pytest.raises(InputValidationError, match="30-day limit"):
        _assert_no_large_temporal_gaps(
            [_real("20240101"), _real("20240301")], max_gap_days=30
        )


def test_compressed_slc_conflicts():
    from disp_nisar.main import (
        InputValidationError,
        _assert_no_compressed_slc_conflicts,
    )

    fwd, hist = "DISP_NISAR_FORWARD", "DISP_NISAR_HISTORICAL"
    # No compressed SLCs: nothing to check
    _assert_no_compressed_slc_conflicts([_real("20240101")], hist)

    good = [_comp("20231201"), _real("20240101"), _real("20240113")]
    _assert_no_compressed_slc_conflicts(good, fwd)
    _assert_no_compressed_slc_conflicts(good, hist)

    same_date = [_comp("20240101"), _real("20240101"), _real("20240113")]
    for mode in (fwd, hist):
        with pytest.raises(InputValidationError) as exc:
            _assert_no_compressed_slc_conflicts(same_date, mode)
        assert exc.value.error_code == 1001

    # CCSLC between the real dates: fine for forward, rejected for historical
    middle = [_comp("20240105"), _real("20240101"), _real("20240113")]
    _assert_no_compressed_slc_conflicts(middle, fwd)
    with pytest.raises(InputValidationError) as exc:
        _assert_no_compressed_slc_conflicts(middle, hist)
    assert exc.value.error_code == 1002

    # CCSLC past the latest real date: rejected in both
    future = [_comp("20240201"), _real("20240101"), _real("20240113")]
    for mode in (fwd, hist):
        with pytest.raises(InputValidationError) as exc:
            _assert_no_compressed_slc_conflicts(future, mode)
        assert exc.value.error_code == 1002


def test_compressed_slc_conflicts_ignores_non_ccslc_paths():
    """A real GSLC under a dir named 'compressed' is not a CCSLC; no IndexError."""
    from disp_nisar.main import _assert_no_compressed_slc_conflicts

    files = [Path("/data/compressed/NISAR_20240101T000000.h5"), _real("20240113")]
    _assert_no_compressed_slc_conflicts(files, "DISP_NISAR_FORWARD")


def test_forward_mode_compressed():
    from disp_nisar.main import InputValidationError, _assert_forward_mode_compressed

    _assert_forward_mode_compressed([_comp("20231201"), _real("20240101")])
    with pytest.raises(InputValidationError) as exc:
        _assert_forward_mode_compressed([_real("20240101"), _real("20240113")])
    assert exc.value.error_code == 2000

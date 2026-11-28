import numpy as np
import pytest
from pydantic import ValidationError

from disp_nisar.ionosphere.ionosphere_filter import (
    apply_ionosphere_filter,
    filter_cache_signature,
)
from disp_nisar.ionosphere.options import IonosphereFilterOptions, IonosphereOptions

SHAPE = (120, 140)
PIXEL_M = (40.0, 40.0)
INTERIOR = (slice(30, -30), slice(30, -30))


@pytest.fixture
def plane():
    rows, cols = np.mgrid[: SHAPE[0], : SHAPE[1]]
    return 0.02 * rows - 0.01 * cols + 3.0


def _options(method="guided", **filtering):
    filtering = {"sigma_m": (80.0, 80.0), "donor_erosion_px": (0, 0), **filtering}
    return IonosphereOptions(
        filtering=IonosphereFilterOptions(method=method, **filtering)
    )


def _filter(phase, options, donor=None, footprint=None):
    return apply_ionosphere_filter(
        phase,
        donor_mask=np.isfinite(phase) if donor is None else donor,
        footprint=np.ones(SHAPE, bool) if footprint is None else footprint,
        pixel_size_m=PIXEL_M,
        options=options,
    )


@pytest.mark.parametrize("method", ["guided", "smooth_iono"])
def test_filter_preserves_a_plane(plane, method):
    out = _filter(plane, _options(method))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[INTERIOR], plane[INTERIOR], atol=1e-3)


@pytest.mark.parametrize("method", ["guided", "smooth_iono"])
def test_filter_fills_gaps_but_not_outside_the_footprint(plane, method):
    phase = plane.copy()
    phase[50:60, 60:70] = np.nan
    footprint = np.ones(SHAPE, bool)
    footprint[:, :10] = False

    out = _filter(phase, _options(method), footprint=footprint)

    np.testing.assert_allclose(out[50:60, 60:70], plane[50:60, 60:70], atol=0.5)
    assert np.isnan(out[:, :10]).all()


@pytest.mark.parametrize("method", ["guided", "smooth_iono"])
def test_filter_restore_mask_keeps_gaps(plane, method):
    phase = plane.copy()
    phase[50:60, 60:70] = np.nan
    out = _filter(phase, _options(method, restore_mask=True))
    assert np.isnan(out[50:60, 60:70]).all()
    assert np.isfinite(out[INTERIOR][0, 0])


def test_filter_erosion_removes_samples_next_to_gaps(plane):
    """Column erosion widens a gap, so a bad edge sample has no influence."""
    phase = plane.copy()
    phase[50:60, 60:70] = np.nan
    phase[50:60, 70] = 100.0
    out = _filter(phase, _options(donor_erosion_px=(0, 1), outlier_threshold=None))
    assert np.abs(out[55, 72] - plane[55, 72]) < 0.5


def test_outlier_rejection_stops_a_spike_spreading(plane):
    phase = plane.copy()
    phase[60, 70] = plane[60, 70] + 50.0

    rejected = _filter(phase, _options())
    kept = _filter(phase, _options(outlier_threshold=None))

    assert np.abs(rejected[60, 70] - plane[60, 70]) < 0.05
    assert np.abs(kept[60, 70] - plane[60, 70]) > 1.0


def test_linear_fill_follows_a_plane_across_a_gap(plane):
    """Nearest fill leaves a step inside a wide gap; linear fill does not."""
    phase = plane.copy()
    phase[40:80, 50:90] = np.nan
    options = {"guide_filter_method": "none", "restore_mask": False}

    linear = _filter(phase, _options(filling_method="linear", **options))
    nearest = _filter(phase, _options(filling_method="nearest", **options))

    gap = (slice(50, 70), slice(60, 80))
    np.testing.assert_allclose(linear[gap], plane[gap], atol=1e-3)
    assert np.abs(nearest[gap] - plane[gap]).max() > 0.05


def test_filter_uses_the_scalar_fallbacks(plane):
    options = IonosphereOptions(smooth_sigma_m=80.0, mask_erosion_px=0)
    explicit = _filter(plane, _options())
    np.testing.assert_array_equal(_filter(plane, options), explicit)


@pytest.mark.parametrize("method", ["guided", "smooth_iono"])
def test_filter_without_donors_returns_nan(plane, method):
    out = _filter(plane, _options(method), donor=np.zeros(SHAPE, bool))
    assert np.isnan(out).all()


def test_filter_rejects_wrapped_input():
    with pytest.raises(ValueError, match="not a complex interferogram"):
        _filter(np.ones(SHAPE, np.complex64), _options())


def test_filter_rejects_bad_pixel_size(plane):
    with pytest.raises(ValueError, match="positive row/column spacings"):
        apply_ionosphere_filter(
            plane,
            donor_mask=np.ones(SHAPE, bool),
            footprint=np.ones(SHAPE, bool),
            pixel_size_m=(0.0, 40.0),
            options=_options(),
        )


def test_cache_signature_tracks_settings_and_raw_file(tmp_path):
    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"one")
    first = filter_cache_signature(raw, _options())
    assert first == filter_cache_signature(raw, _options())
    assert first != filter_cache_signature(raw, _options(truncate=3.0))
    raw.write_bytes(b"longer")
    assert first != filter_cache_signature(raw, _options())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sigma_m", (0.0, 10.0), "widths must be positive"),
        ("kernel_size_px", (4, 5), "positive and odd"),
        ("guide_median_size", 4, "must be odd"),
        ("donor_erosion_px", (-1, 0), "must be nonnegative"),
        ("method", "isce3", "Input should be"),
    ],
)
def test_filter_options_reject_bad_values(field, value, message):
    with pytest.raises(ValidationError, match=message):
        IonosphereFilterOptions(**{field: value})

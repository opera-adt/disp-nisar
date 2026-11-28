"""A/(A-B) estimation with fixed date-based output paths.

Completed stages are reused only while the settings and inputs recorded in
settings.json still match; a mismatch stops the run rather than mixing results.
Filtered and exported rasters also check raw-input and filter-setting signatures.
Preparation/inversion are windowed; gap filling uses one full B-grid array.

Dolphin/SNAPHU still controls unwrap memory. Phase-linked SLCs are consumed through
existing stitched IFGs; independent A/B displacement series are not mixed here.

Connected components are not used. All donors enter the inversion, so a whole-cycle
offset between components of D reaches the correction amplified by about 12 at
NISAR frequencies. Filtering does not repair such an offset.
"""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from dolphin import io
from dolphin.utils import format_date_pair
from dolphin.workflows.config import UnwrapOptions
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from ._main_diff_io import (
    Grid,
    atomic_json,
    read_window,
)
from .gslc_mask import prepare_gslc_mask_cache
from .inversion import build_design_matrix, invert_common_phase_block
from .ionosphere_filter import (
    apply_ionosphere_filter,
    filter_cache_signature,
)
from .options import IonosphereOptions
from .utils import (
    WORKING_TIFF_OPTIONS,
    average_to_window,
    check_reduction,
    freq_to_wavelength,
    phase_to_disp,
    resample_phasor_to_match,
    source_window,
)

logger = logging.getLogger(__name__)

IONOSPHERE_METHOD = "main_diff"


@dataclass(frozen=True)
class BandInputs:
    """Stitched Dolphin outputs for one frequency band.

    Attributes
    ----------
    interferograms : Sequence[Path]
        Complex wrapped interferograms, indexed by acquisition pair.
    correlations : Sequence[Path]
        Correlation rasters for the same acquisition pairs.
    similarity : Path
        Phase-similarity raster shared by the band. All inputs for this
        band must have identical dimensions, CRS, and affine transform.

    """

    interferograms: Sequence[Path]
    correlations: Sequence[Path]
    similarity: Path


def _date_pair(path: Path) -> tuple[datetime, datetime]:
    """Extract an ordered acquisition-date pair from a raster filename.

    Parameters
    ----------
    path : Path
        Filename containing exactly two acquisition dates.

    Returns
    -------
    tuple[datetime, datetime]
        Reference and secondary dates, normalized to midnight.

    Raises
    ------
    ValueError
        If the filename does not contain two strictly increasing dates.

    """
    from opera_utils import get_dates

    dates = [
        d.replace(hour=0, minute=0, second=0, microsecond=0) for d in get_dates(path)
    ]
    if len(dates) != 2 or dates[0] >= dates[1]:
        raise ValueError(f"Expected one ordered acquisition pair: {path}")
    return dates[0], dates[1]


def _index(paths: Sequence[Path]) -> dict[tuple[datetime, datetime], Path]:
    """Index acquisition-pair rasters by their normalized dates.

    Parameters
    ----------
    paths : Sequence[Path]
        Raster filenames containing reference and secondary dates.

    Returns
    -------
    dict
        Mapping from ordered date pairs to raster paths.

    Raises
    ------
    ValueError
        If a filename has invalid dates or multiple files share a pair.

    """
    result: dict[tuple, Path] = {}
    for p in paths:
        pair = _date_pair(p)
        if pair in result:
            raise ValueError(f"Duplicate pair {pair}: {result[pair]}, {p}")
        result[pair] = Path(p)
    return result


def estimate_iono_main_diff(
    f_A: float, f_B: float, phase_a: np.ndarray, phase_d: np.ndarray
) -> np.ndarray:
    """Estimate frequency-A ionospheric phase from A and A-minus-B.

    Parameters
    ----------
    f_A, f_B : float
        Distinct positive center frequencies, in Hz.
    phase_a, phase_d : np.ndarray
        Unwrapped phases or phase time series, in radians. Arrays must
        have identical shapes, compatible spatial and temporal references,
        and the same interferometric phase convention.

    Returns
    -------
    np.ndarray
        Frequency-A ionospheric phase as float32, with the input shape
        and phase convention.

    Raises
    ------
    ValueError
        If frequencies are invalid or indistinguishable, or shapes differ.

    """
    if not (np.isfinite(f_A) and np.isfinite(f_B) and min(f_A, f_B) > 0):
        raise ValueError("Center frequencies must be finite and positive")
    if abs(f_A - f_B) <= 1e-8 * max(f_A, f_B):
        raise ValueError("Center frequencies are indistinguishable")
    if phase_a.shape != phase_d.shape:
        raise ValueError("A and D shapes differ")
    return (
        f_B / (f_A + f_B) * phase_a.astype(np.float64)
        + f_A * f_B / (f_B * f_B - f_A * f_A) * phase_d.astype(np.float64)
    ).astype(np.float32)


def _same_grid(path: Path, grid: Grid) -> None:
    """Require a raster to match a supplied grid exactly.

    Parameters
    ----------
    path : str or Path
        Raster whose spatial metadata will be checked.
    grid : Grid
        Expected dimensions, CRS, and pixel-edge affine transform.

    Raises
    ------
    ValueError
        If the raster grid differs from the expected grid.

    """
    if Grid.from_file(Path(path)) != grid:
        raise ValueError(f"Grid differs from the band's stitched IFG: {path}")


def _prepare_pair(
    inputs: dict[str, Path],
    masks: dict[str, Path],
    target: Grid,
    folder: Path,
    options: IonosphereOptions,
) -> dict[str, Path]:
    """Prepare masked A and D observations on the frequency-B grid.

    Parameters
    ----------
    inputs : dict[str, Path]
        Paths for ifg_A, ifg_B, cor_A, cor_B, sim_A, and sim_B.
    masks : dict[str, Path]
        Date-specific valid and inside masks for both frequencies and
        both pair endpoints, already reduced to their band IFG grids.
    target : Grid
        Frequency-B interferogram grid used for all prepared outputs.
    folder : Path
        Working directory for prepared rasters and partial writes.
    options : IonosphereOptions
        Quality thresholds, resampling support, block size, and resume settings.

    Returns
    -------
    dict[str, Path]
        Prepared complex observations A and D = A * conjugate(B), their
        correlation and similarity layers, and common donor/inside masks.

    Notes
    -----
    A is resampled as unit phasors onto B. Both observations use identical
    donor support. Interior non-donors receive unit placeholders with zero
    quality; pixels outside the footprint receive zero complex values.
    Partial rasters are renamed only after every writer has closed.

    """
    source = Grid.from_file(inputs["ifg_A"])
    check_reduction(source, target)

    specs = {
        "A_ifg": "complex64",
        "D_ifg": "complex64",
        "A_cor": "float32",
        "D_cor": "float32",
        "A_sim": "float32",
        "D_sim": "float32",
        "donor": "uint8",
        "inside": "uint8",
    }

    folder.mkdir(parents=True, exist_ok=True)
    outputs = {name: folder / f"{name}.tif" for name in specs}
    if options.resume and all(path.is_file() for path in outputs.values()):
        return outputs
    temporary = {
        name: path.with_suffix(".partial.tif") for name, path in outputs.items()
    }

    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in inputs.items()
        }
        mask_readers = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in masks.items()
        }
        writers = {}

        # Set spatial metadata explicitly so masks do not inherit nodata=0.
        # Write partial files and publish them only after every block succeeds.
        for name, dtype in specs.items():
            io.write_arr(
                arr=None,
                output_name=temporary[name],
                shape=(target.height, target.width),
                geotransform=target.transform.to_gdal(),
                projection=target.crs.to_wkt(),
                dtype=dtype,
                nbands=1,
                nodata=0 if name.endswith("_ifg") else None,
                options=WORKING_TIFF_OPTIONS,
            )

            writers[name] = stack.enter_context(rasterio.open(temporary[name], "r+"))

        for block in io.iter_blocks(
            arr_shape=(target.height, target.width),
            block_shape=(options.block_size, options.block_size),
        ):
            rows, cols = block
            win = Window.from_slices(rows, cols)
            sw = source_window(source, target, win)

            # Apply the same native-grid validity and quality rules to A/B.
            native = {}
            for band, window in (("A", sw), ("B", win)):
                z = read_window(readers[f"ifg_{band}"], window)
                cor = read_window(readers[f"cor_{band}"], window)
                sim = read_window(readers[f"sim_{band}"], window)

                valid = (
                    read_window(mask_readers[f"{band}_ref_valid"], window) == 1
                ) & (read_window(mask_readers[f"{band}_sec_valid"], window) == 1)

                inside_band = (
                    read_window(mask_readers[f"{band}_ref_inside"], window) == 1
                ) & (read_window(mask_readers[f"{band}_sec_inside"], window) == 1)

                good = (
                    valid
                    & inside_band
                    & np.isfinite(z)
                    & (np.abs(z) > 0)
                    & np.isfinite(cor)
                    & (cor >= options.correlation_threshold)
                    & np.isfinite(sim)
                    & (sim >= options.similarity_threshold)
                )

                native[band] = (
                    z,
                    np.where(good, np.clip(cor, 0, 1), 0),
                    np.where(good, sim, 0),
                    good,
                    inside_band,
                )

            za, qa, sa, ga, ia = native["A"]
            zb, qb, sb, gb, ib = native["B"]

            # Resample A's unit phasor and quality onto the B grid.
            ua, good_a, support = resample_phasor_to_match(
                za,
                ga,
                source,
                sw,
                target,
                win,
                options.min_valid_fraction,
                options.min_phasor_magnitude,
            )

            denominator = np.maximum(support, 1e-12)

            qa = average_to_window(qa, source, sw, target, win) / denominator
            sa = average_to_window(sa, source, sw, target, win) / denominator
            qa = np.clip(qa, 0, 1)

            # Footprint and donor support are distinct.
            inside = (average_to_window(ia, source, sw, target, win) > 0) & ib

            donor = (
                inside
                & good_a
                & gb
                & np.isfinite(ua)
                & (np.abs(ua) > 0)
                & np.isfinite(qa)
                & np.isfinite(sa)
            )

            ub = np.zeros(zb.shape, dtype=np.complex64)
            np.divide(zb, np.abs(zb), out=ub, where=gb)

            d_ifg = ua * np.conj(ub)
            d_cor = np.clip(qa * qb, 0, 1)
            d_sim = np.minimum(sa, sb)

            # Use identical donor support for both prepared observations.
            # 1+0j: interior placeholder; 0+0j: outside the footprint.
            a_ifg = np.where(
                inside,
                np.where(donor, ua, 1 + 0j),
                0 + 0j,
            )
            d_ifg = np.where(
                inside,
                np.where(donor, d_ifg, 1 + 0j),
                0 + 0j,
            )

            data = {
                "A_ifg": a_ifg,
                "D_ifg": d_ifg,
                "A_cor": np.where(donor, qa, 0),
                "D_cor": np.where(donor, d_cor, 0),
                "A_sim": np.where(donor, sa, 0),
                "D_sim": np.where(donor, d_sim, 0),
                "donor": donor,
                "inside": inside,
            }

            for name, array in data.items():
                writers[name].write(
                    array.astype(specs[name]),
                    1,
                    window=win,
                )

    # ExitStack has closed every writer before these files are published.
    for name, path in temporary.items():
        path.replace(outputs[name])
    return outputs


def _unwrap_pair(
    prepared: dict[str, Path],
    folder: Path,
    unwrap_options: UnwrapOptions,
    options: IonosphereOptions,
) -> dict[str, Path]:
    """Unwrap both prepared observations and retain original donor support.

    Parameters
    ----------
    prepared : dict[str, Path]
        Outputs from _prepare_pair, including observation quality layers,
        the footprint mask, and the original donor mask.
    folder : Path
        Pair working directory for unwrapped phase and component rasters.
    unwrap_options : dolphin.workflows.config.UnwrapOptions
        Base configuration for SNAPHU or Whirlwind.
    options : IonosphereOptions
        Effective looks, interpolation thresholds, resume behavior,
        and intermediate-file retention settings.

    Returns
    -------
    dict[str, Path]
        Unwrapped phase paths for both observations, plus the donor mask.
        Prepared inputs remain only when requested.

    Notes
    -----
    The footprint controls where unwrapping is allowed; donor quality
    controls which observations are trusted during interpolation.
    Existing unwrapped phases are reused when resume is enabled.

    """
    from dolphin.unwrap import unwrap

    bands = ("A", "D")

    unwrap_method = getattr(
        unwrap_options.unwrap_method,
        "value",
        unwrap_options.unwrap_method,
    )
    if unwrap_method not in ("snaphu", "whirlwind"):
        raise ValueError(
            f"main_diff supports snaphu or whirlwind; received {unwrap_method!r}"
        )

    # Validate all inputs before starting either unwrap.
    required = {"inside", "donor"}
    required.update(
        f"{band}_{kind}" for band in bands for kind in ("ifg", "cor", "sim")
    )
    missing = required.difference(prepared)
    if missing:
        raise ValueError(f"Missing prepared inputs for main_diff: {sorted(missing)}")

    result = dict(prepared)

    for band in bands:
        unw_path = folder / f"{band}.unw.tif"
        if options.resume and unw_path.is_file():
            result[f"{band}_unw"] = unw_path
            continue
        cfg = unwrap_options.model_copy(deep=True)

        cfg.run_unwrap = True
        cfg.run_interpolation = True
        cfg.zero_where_masked = False

        # A uses its own correlation; D uses the A*B proxy.
        cfg.preprocess_options.interpolation_cor_threshold = (
            options.correlation_threshold**2
            if band == "D"
            else options.correlation_threshold
        )
        cfg.preprocess_options.interpolation_similarity_threshold = (
            options.similarity_threshold
        )
        if unwrap_method == "whirlwind" and hasattr(cfg, "whirlwind_options"):
            # Recent Dolphin routes interpolation through Whirlwind itself.
            # Similarity has already been enforced in the prepared donor mask;
            # non-donors have zero correlation in both prepared observations.
            cfg.whirlwind_options.interpolate = True
            cfg.whirlwind_options.interp_cutoff = (
                cfg.preprocess_options.interpolation_cor_threshold
            )

        work = folder / f"_unwrap_{band}"
        work.mkdir(exist_ok=True)

        unw, _ = unwrap(
            ifg_filename=prepared[f"{band}_ifg"],
            corr_filename=prepared[f"{band}_cor"],
            unw_filename=work / f"{band}.unw.tif",
            nlooks=options.nlooks,
            mask_filename=prepared["inside"],
            similarity_filename=prepared[f"{band}_sim"],
            unwrap_options=cfg,
            unw_nodata=np.nan,
            ccl_nodata=0,
            scratchdir=work / "scratch",
            delete_scratch=not options.keep_intermediates,
        )

        Path(unw).replace(unw_path)
        result[f"{band}_unw"] = unw_path
        if not options.keep_intermediates:
            shutil.rmtree(work)

    # Preserve the original donor mask for reference selection/inversion.
    # Clean prepared files only after both unwrap operations succeed.
    if not options.keep_intermediates:
        for name, path in prepared.items():
            if name != "donor":
                Path(path).unlink()
                result.pop(name)

    return result


def _sample(path: Path, reference: tuple[int, int]) -> float:
    """Read a single raster value at a row-column reference.

    Parameters
    ----------
    path : str or Path
        Raster to sample.
    reference : tuple[int, int]
        Zero-based row and column indices.

    Returns
    -------
    float
        Sample value, or NaN for a masked value.

    """
    with rasterio.open(path) as ds:
        return float(read_window(ds, Window(reference[1], reference[0], 1, 1))[0, 0])


def select_common_reference(
    jobs: list[dict[str, Path]],
    grid: Grid,
    origin: tuple[int, int],
    output: Path,
    block_size: int,
) -> tuple[int, int]:
    """Choose a supported reference shared by every observation pair.

    Parameters
    ----------
    jobs : list[dict[str, Path]]
        Pair outputs containing donor masks and unwrapped phases for both
        observations.
    grid : Grid
        Common frequency-B grid.
    origin : tuple[int, int]
        Preferred reference row and column on the B grid.
    output : Path
        Destination for the binary common-candidate raster.
    block_size : int
        Maximum block height and width, in pixels.

    Returns
    -------
    tuple[int, int]
        Candidate row and column nearest to origin in pixel-index space.

    Raises
    ------
    ValueError
        If no donor has finite phase for both observations in every pair.

    """
    io.write_arr(
        arr=None,
        output_name=output,
        shape=(grid.height, grid.width),
        geotransform=grid.transform.to_gdal(),
        projection=grid.crs.to_wkt(),
        dtype="uint8",
        nbands=1,
        nodata=None,
        options=WORKING_TIFF_OPTIONS,
    )

    # Start with all pixels eligible, then intersect support across all pairs.
    with rasterio.open(output, "r+") as ds:
        for block in io.iter_blocks(
            arr_shape=(grid.height, grid.width),
            block_shape=(block_size, block_size),
        ):
            rows, cols = block
            win = Window.from_slices(rows, cols)
            ds.write(
                np.ones(
                    (int(win.height), int(win.width)),
                    dtype=np.uint8,
                ),
                1,
                window=win,
            )

    for job in jobs:
        with ExitStack() as stack:
            ds = stack.enter_context(rasterio.open(output, "r+"))
            readers = {
                n: stack.enter_context(rasterio.open(job[n]))
                for n in ("donor", "A_unw", "D_unw")
            }
            for block in io.iter_blocks(
                arr_shape=(grid.height, grid.width),
                block_shape=(block_size, block_size),
            ):
                rows, cols = block
                win = Window.from_slices(rows, cols)
                good = (ds.read(1, window=win) > 0) & (
                    read_window(readers["donor"], win) == 1
                )
                for band in ("A", "D"):
                    phase = read_window(readers[f"{band}_unw"], win)
                    good &= np.isfinite(phase)
                ds.write(good.astype(np.uint8), 1, window=win)
    best = None
    with rasterio.open(output) as ds:
        for block in io.iter_blocks(
            arr_shape=(grid.height, grid.width),
            block_shape=(block_size, block_size),
        ):
            rows, cols = block
            win = Window.from_slices(rows, cols)
            rr, cc = np.nonzero(ds.read(1, window=win))
            if not len(rr):
                continue
            rr, cc = rr + int(win.row_off), cc + int(win.col_off)
            distance = (rr - origin[0]) ** 2 + (cc - origin[1]) ** 2
            i = int(np.argmin(distance))
            candidate = int(distance[i]), int(rr[i]), int(cc[i])
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError(
            "No common A/D reference survives donor and finite-phase checks; "
            "completed unwraps are retained"
        )
    return best[1], best[2]


def _invert_jobs(
    jobs: list[dict[str, Path]],
    pairs: Sequence[tuple[datetime, datetime]],
    grid: Grid,
    reference: tuple[int, int],
    f_A: float,
    f_B: float,
    outputs: Sequence[Path],
    output_pairs: Sequence[tuple[datetime, datetime]],
    block_size: int,
) -> None:
    """Invert referenced pair phases and write ionosphere date-pair rasters.

    Parameters
    ----------
    jobs : list[dict[str, Path]]
        Donor and unwrapped-phase rasters in pairs order.
    pairs : sequence[tuple]
        Ordered acquisition-date pairs defining the observation network.
    grid : Grid
        Common frequency-B grid for block processing.
    reference : tuple[int, int]
        Common reference row and column on the B grid.
    f_A, f_B : float
        Center frequencies, in Hz.
    outputs : sequence[Path]
        Output paths, ordered like output_pairs.
    output_pairs : sequence[tuple]
        Requested reference/secondary epoch pairs.
    block_size : int
        Maximum block height and width, in pixels.

    Notes
    -----
    Each observation is spatially referenced to the common pixel. The
    earliest epoch is fixed at zero during inversion. Outputs contain
    frequency-A ionospheric phase in radians,
    with NaN nodata, on the B grid. No value is returned.

    """
    bands = ("A", "D")

    dates = sorted({date for pair in pairs for date in pair})
    idx = {date: i for i, date in enumerate(dates)}
    design = build_design_matrix(list(pairs), dates)

    if np.linalg.matrix_rank(design) != len(dates) - 1:
        raise ValueError("A/D network is disconnected")

    offsets = {
        band: np.array([_sample(job[f"{band}_unw"], reference) for job in jobs])
        for band in bands
    }

    if any(not np.isfinite(values).all() for values in offsets.values()):
        raise ValueError("Reference phase is not finite for every pair")

    with ExitStack() as stack:
        # These reads are inside the B grid, so Dolphin's slice reader suffices.
        # Open on demand instead of holding five handles per pair for the full run.
        readers = {
            name: [
                io.RasterReader.from_file(job[name], keep_open=False) for job in jobs
            ]
            for name in ("donor", "A_unw", "D_unw")
        }

        writers = []

        for path in outputs:
            io.write_arr(
                arr=None,
                output_name=path,
                like_filename=jobs[0]["A_unw"],
                dtype="float32",
                nbands=1,
                nodata=np.nan,
                options=WORKING_TIFF_OPTIONS,
            )

            writers.append(stack.enter_context(rasterio.open(path, "r+")))

        for dst in writers:
            dst.set_band_unit(1, "radians")
            dst.update_tags(
                phase_convention="dolphin_ifg_ref_times_conj_sec",
                ionosphere_method=IONOSPHERE_METHOD,
            )

        for block in io.iter_blocks(
            arr_shape=(grid.height, grid.width),
            block_shape=(block_size, block_size),
        ):
            rows, cols = block
            win = Window.from_slices(rows, cols)
            rows, cols = win.toslices()
            valid = (
                np.ma.stack([reader[rows, cols] for reader in readers["donor"]]).filled(
                    0
                )
                == 1
            )

            phases = {}
            for band in bands:
                stack = np.ma.stack(
                    [reader[rows, cols] for reader in readers[f"{band}_unw"]]
                )
                phases[band] = np.ma.filled(stack.astype(np.float32), np.nan)
                phases[band] -= offsets[band][:, None, None]

            phase_a, phase_d = invert_common_phase_block(
                phases["A"],
                phases["D"],
                design,
                valid,
            )

            iono = estimate_iono_main_diff(f_A, f_B, phase_a, phase_d)

            # The inversion omits the earliest epoch, whose phase is fixed at zero.
            # Restore it before differencing arbitrary requested epoch pairs.
            zero = np.where(np.isfinite(iono).all(axis=0), 0, np.nan)[None].astype(
                np.float32
            )

            series = np.concatenate((zero, iono), axis=0)

            for dst, (ref, sec) in zip(writers, output_pairs, strict=True):
                dst.write(
                    series[idx[sec]] - series[idx[ref]],
                    1,
                    window=win,
                )


def _export_correction(
    source: Path,
    match: Path,
    output: Path,
    f_A: float,
    reference_a: tuple[int, int],
    block_size: int,
) -> None:
    """Export a re-referenced ionosphere correction in meters.

    Parameters
    ----------
    source : str or Path
        Filled ionospheric phase raster in radians on the B grid.
    match : str or Path
        Nominal displacement raster defining the output grid.
    output : str or Path
        Destination correction raster, typically a partial filename.
    f_A : float
        Frequency-A center frequency, in Hz.
    reference_a : tuple[int, int]
        Nominal reference row and column on the output grid.
    block_size : int
        Maximum block height and width, in pixels.

    Raises
    ------
    ValueError
        If the nominal reference has no finite warped estimate.

    Notes
    -----
    Bilinear resampling operates on unwrapped ionospheric phase.
    Subtract the warped reference phase, then convert once using
    -wavelength / (4*pi), with the same matrix and sign as the Dolphin
    inversion. The A-B phase has no independent wavelength. The output has
    NaN nodata and meter units.
    This function writes the file and returns no value.

    """
    target = Grid.from_file(match)
    wavelength = freq_to_wavelength(f_A)
    with (
        rasterio.open(source) as src,
        WarpedVRT(
            src,
            crs=target.crs,
            transform=target.transform,
            width=target.width,
            height=target.height,
            resampling=Resampling.bilinear,
            nodata=np.nan,
            dtype="float32",
        ) as vrt,
    ):
        offset = float(
            read_window(vrt, Window(reference_a[1], reference_a[0], 1, 1))[0, 0]
        )
        if not np.isfinite(offset):
            row_a, col_a = reference_a

            # Geographic location of the A reference pixel center.
            x, y = target.transform * (col_a + 0.5, row_a + 0.5)

            # Source and target CRS must match for direct coordinate indexing.
            if target.crs != src.crs:
                from rasterio.warp import transform

                xs, ys = transform(target.crs, src.crs, [x], [y])
                x, y = xs[0], ys[0]

            row_b, col_b = src.index(x, y)
            in_bounds = 0 <= row_b < src.height and 0 <= col_b < src.width

            logger.error(
                "Unsupported reference: source=%s, match=%s, "
                "A reference=%s, source pixel=%s, source shape=%s, "
                "inside source bounds=%s, warped value=%s",
                source,
                match,
                reference_a,
                (row_b, col_b),
                src.shape,
                in_bounds,
                offset,
            )

            if in_bounds:
                # A small neighborhood is enough to identify a local gap/edge.
                r0 = max(0, row_b - 2)
                r1 = min(src.height, row_b + 3)
                c0 = max(0, col_b - 2)
                c1 = min(src.width, col_b + 3)

                patch = (
                    src.read(
                        1,
                        window=Window(c0, r0, c1 - c0, r1 - r0),
                        masked=True,
                    )
                    .astype(np.float64)
                    .filled(np.nan)
                )

                logger.error(
                    "Source nodata=%s; source reference value=%s; "
                    "source neighborhood:\n%s",
                    src.nodata,
                    patch[row_b - r0, col_b - c0],
                    np.array2string(patch, precision=5),
                )
            raw_path = Path(source).with_name(
                Path(source).name.replace("_filled.rad.tif", ".rad.tif")
            )

            with rasterio.open(raw_path) as raw_ds:
                with WarpedVRT(
                    raw_ds,
                    crs=target.crs,
                    transform=target.transform,
                    width=target.width,
                    height=target.height,
                    resampling=Resampling.bilinear,
                    nodata=np.nan,
                    dtype="float32",
                ) as raw_vrt:
                    raw_ref = float(
                        read_window(
                            raw_vrt,
                            Window(reference_a[1], reference_a[0], 1, 1),
                        )[0, 0]
                    )

            logger.error(
                "REFERENCE CHECK: raw=%s; filtered=%s; A reference=%s; raw file=%s",
                raw_ref,
                offset,
                reference_a,
                raw_path,
            )
            raise ValueError(
                "Product reference has no supported ionosphere estimate. "
                "See source/reference diagnostics above; existing unwraps are retained."
            )
        # Export on the nominal product grid, which may differ from the B grid.
        io.write_arr(
            arr=None,
            output_name=output,
            like_filename=match,
            dtype="float32",
            nbands=1,
            nodata=np.nan,
            options=WORKING_TIFF_OPTIONS,
        )
        with rasterio.open(output, "r+") as dst:
            dst.set_band_unit(1, "meters")
            dst.update_tags(
                ionosphere_method=IONOSPHERE_METHOD,
                reference_row=reference_a[0],
                reference_col=reference_a[1],
            )
            for block in io.iter_blocks(
                arr_shape=(target.height, target.width),
                block_shape=(block_size, block_size),
            ):
                rows, cols = block
                win = Window.from_slices(rows, cols)
                dst.write(
                    phase_to_disp(read_window(vrt, win) - offset, wavelength).astype(
                        np.float32
                    ),
                    1,
                    window=win,
                )


def run_main_diff_estimation(
    *,
    band_a: BandInputs,
    band_b: BandInputs,
    gslc_by_date: dict,
    output_timeseries_paths: Sequence[Path],
    reference_a: tuple[int, int],
    f_A: float,
    f_B: float,
    out_dir: Path,
    unwrap_options: UnwrapOptions,
    options: IonosphereOptions,
) -> list[Path]:
    """Run A/(A-B) ionosphere processing under an output lock.

    Parameters
    ----------
    band_a, band_b : BandInputs
        Matching stitched IFG networks on their respective band grids.
    gslc_by_date : dict
        Acquisition dates normalized to midnight, mapped to local raw
        GSLC inputs used for phase linking.
    output_timeseries_paths : Sequence[Path]
        Nominal displacement rasters defining requested pairs and grids.
    reference_a : tuple[int, int]
        Nominal reference row and column on the product grid.
    f_A, f_B : float
        Constant center frequencies, in Hz.
    out_dir : Path
        Directory for masks, pair products, inversion, and corrections.
    unwrap_options : dolphin.workflows.config.UnwrapOptions
        Base SNAPHU or Whirlwind configuration.
    options : IonosphereOptions
        Estimation, quality, memory, resume, and retention settings.

    Returns
    -------
    list[Path]
        Correction rasters in meters, ordered like the nominal products.

    Raises
    ------
    RuntimeError
        If another process already holds the output-directory lock.

    Notes
    -----
    The lock is released on exit, including exceptions. Existing nominal
    rasters and their reference metadata are not modified.

    """
    import fcntl

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "main_diff is already running in this directory"
            ) from exc
        return _run_locked(
            band_a,
            band_b,
            gslc_by_date,
            output_timeseries_paths,
            reference_a,
            f_A,
            f_B,
            out_dir,
            unwrap_options,
            options,
        )


def _settings_record(
    band_a: BandInputs,
    band_b: BandInputs,
    gslc_by_date: dict,
    reference_a: tuple[int, int],
    f_A: float,
    f_B: float,
    unwrap_options: UnwrapOptions,
    options: IonosphereOptions,
) -> dict:
    """Describe everything the masks, unwraps, and inversion depend on.

    Filter settings are left out because filtered rasters carry their own
    signature, so retuning the filter does not redo the unwrapping.
    """
    inputs = [
        *band_a.interferograms,
        *band_a.correlations,
        band_a.similarity,
        *band_b.interferograms,
        *band_b.correlations,
        band_b.similarity,
        *gslc_by_date.values(),
    ]
    stats = {str(Path(p).resolve()): Path(p).stat() for p in inputs}
    return {
        "inputs": {p: [s.st_size, s.st_mtime_ns] for p, s in sorted(stats.items())},
        "reference_a": list(reference_a),
        "frequencies": [f_A, f_B],
        "unwrap": unwrap_options.model_dump(mode="json", exclude={"n_parallel_jobs"}),
        "options": options.model_dump(
            mode="json",
            exclude={
                "block_size",
                "filtering",
                "keep_intermediates",
                "mask_erosion_px",
                "mask_work_mb",
                "resume",
                "smooth_sigma_m",
            },
        ),
    }


def _check_settings(out_dir: Path, record: dict, resume: bool) -> None:
    """Stop a resume over results from other settings; clear them if not resuming.

    Raises
    ------
    ValueError
        If resuming and the recorded settings differ or are missing.

    """
    path = out_dir / "settings.json"
    stages = [out_dir / n for n in ("masks", "pairs", "timeseries", "corrections")]
    fix = "Set ionosphere_options.resume to false to discard them and start over."
    if not resume:
        for stage in stages:
            shutil.rmtree(stage, ignore_errors=True)
    elif path.is_file():
        previous = json.loads(path.read_text())
        changed = sorted(
            k
            for k in previous.keys() | record.keys()
            if previous.get(k) != record.get(k)
        )
        if changed:
            raise ValueError(
                f"Existing results in {out_dir} used different {changed}. {fix}"
            )
    elif any(stage.exists() for stage in stages):
        raise ValueError(f"Results in {out_dir} have no settings record. {fix}")
    atomic_json(path, record)


def _filter_cache_matches(
    path: Path, raw_path: Path, options: IonosphereOptions
) -> bool:
    """Reuse a filtered/exported raster only for the same raw input and settings.

    Untagged products from earlier versions are regenerated. A failed metadata
    read is treated as a cache miss, not as a reason to reuse a partial file.
    """
    if not options.resume or not Path(path).is_file() or not Path(raw_path).is_file():
        return False
    try:
        signature = filter_cache_signature(raw_path, options)
        with rasterio.open(path) as ds:
            return ds.tags().get("ionosphere_filter_signature") == signature
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return False


def _run_locked(
    band_a: BandInputs,
    band_b: BandInputs,
    gslc_by_date: dict,
    products: Sequence[Path],
    reference_a: tuple[int, int],
    f_A: float,
    f_B: float,
    out_dir: Path,
    unwrap_options: UnwrapOptions,
    options: IonosphereOptions,
) -> list[Path]:
    """Process a matching A/B network while the caller holds the lock.

    Parameters
    ----------
    band_a, band_b : BandInputs
        Stitched wrapped phase, correlation, and similarity inputs.
    gslc_by_date : dict
        Normalized acquisition dates mapped to raw GSLC paths.
    products : Sequence[Path]
        Nominal rasters defining requested output pairs and grids.
    reference_a : tuple[int, int]
        Nominal product reference row and column.
    f_A, f_B : float
        Constant center frequencies, in Hz.
    out_dir : Path
        Resolved output-directory path.
    unwrap_options : dolphin.workflows.config.UnwrapOptions
        Base configuration for pair unwrapping.
    options : IonosphereOptions
        Method, quality thresholds, memory limits, and restart settings.

    Returns
    -------
    list[Path]
        Exported meter-valued ionosphere corrections in product order.

    Notes
    -----
    Stages are mask caching, pair preparation/unwrapping, common-reference
    selection, network inversion, filtering/filling, and product export.
    Filtering is selected by options.filtering. Its settings and the raw
    raster identity are checked before filled/correction outputs are reused.

    """
    if not products:
        return []
    final_paths = [
        out_dir / "corrections" / f"{format_date_pair(*_date_pair(Path(p)))}_iono.tif"
        for p in products
    ]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("Duplicate output date pairs")
    # Filter settings must be checked against each raw raster before reusing
    # final products. An existence-only early return would ignore config edits.
    estimate_iono_main_diff(
        f_A,
        f_B,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
    )
    maps = {
        "ifg_A": _index(band_a.interferograms),
        "cor_A": _index(band_a.correlations),
        "ifg_B": _index(band_b.interferograms),
        "cor_B": _index(band_b.correlations),
    }
    pairs = sorted(maps["ifg_A"])
    if not pairs or any(set(m) != set(pairs) for m in maps.values()):
        raise ValueError("A/B IFG and correlation networks must match exactly")
    output_pairs = [_date_pair(Path(p)) for p in products]
    dates = sorted({d for pair in pairs for d in pair})
    if any(d not in dates for pair in output_pairs for d in pair):
        raise ValueError("Output requests an epoch absent from the common network")
    if missing := set(dates) - set(gslc_by_date):
        raise ValueError(f"Missing raw GSLC masks for acquisitions: {missing}")
    grids = {b: Grid.from_file(maps[f"ifg_{b}"][pairs[0]]) for b in ("A", "B")}
    target = grids["B"]
    check_reduction(grids["A"], target)
    for b, inputs_b in (("A", band_a), ("B", band_b)):
        for p in (
            *inputs_b.interferograms,
            *inputs_b.correlations,
            inputs_b.similarity,
        ):
            _same_grid(p, grids[b])
    pg = Grid.from_file(Path(products[0]))
    if pg.crs != target.crs:
        raise ValueError("Product CRS differs from B CRS")
    for p in products:
        _same_grid(p, pg)
    r, c = reference_a
    if not (0 <= r < pg.height and 0 <= c < pg.width):
        raise ValueError("Nominal reference outside A product grid")
    x, y = pg.transform * (c + 0.5, r + 0.5)
    cb, rb = (~target.transform) * (x, y)
    origin = int(np.floor(rb)), int(np.floor(cb))
    if not (0 <= origin[0] < target.height and 0 <= origin[1] < target.width):
        raise ValueError("Nominal A reference is outside the B grid")
    _check_settings(
        out_dir,
        _settings_record(
            band_a, band_b, gslc_by_date, reference_a, f_A, f_B, unwrap_options, options
        ),
        options.resume,
    )
    raw_paths = [
        out_dir / "timeseries" / f"{format_date_pair(*pair)}_iono_B.rad.tif"
        for pair in output_pairs
    ]
    inversion_ready = options.resume and all(p.is_file() for p in raw_paths)
    need_masks = not inversion_ready or any(
        not _filter_cache_matches(final, raw, options)
        and not _filter_cache_matches(
            raw.with_name(
                raw.name.replace("_iono_B.rad.tif", "_iono_B_filled.rad.tif")
            ),
            raw,
            options,
        )
        for raw, final in zip(raw_paths, final_paths, strict=True)
    )
    cached_masks = (
        {
            (d, b): prepare_gslc_mask_cache(
                Path(gslc_by_date[d]),
                b,
                grids[b],
                out_dir / "masks",
                options.mask_reduction,
                options.block_size,
                options.mask_work_mb,
                resume=options.resume,
            )
            for d in dates
            for b in ("A", "B")
        }
        if need_masks
        else {}
    )
    jobs = []
    for i, pair in enumerate([] if inversion_ready else pairs, 1):
        name = format_date_pair(*pair)
        inputs = {n: m[pair] for n, m in maps.items()}
        inputs.update(sim_A=Path(band_a.similarity), sim_B=Path(band_b.similarity))
        masks = {
            f"{b}_{role}_{kind}": cached_masks[d, b][kind]
            for b in ("A", "B")
            for role, d in zip(("ref", "sec"), pair, strict=True)
            for kind in ("valid", "inside")
        }
        folder = out_dir / "pairs" / name
        folder.mkdir(parents=True, exist_ok=True)

        # These are the files consumed by reference selection and inversion.
        ready = {
            "A_unw": folder / "A.unw.tif",
            "D_unw": folder / "D.unw.tif",
            "donor": folder / "donor.tif",
        }
        missing_files = [str(path) for path in ready.values() if not path.is_file()]
        logger.info(
            "%s: resume=%s; missing cached files=%s",
            name,
            options.resume,
            missing_files,
        )
        if options.resume and all(p.is_file() for p in ready.values()):
            status = "reused"
        else:
            # One fixed working directory, not a new attempt directory each time.
            work = folder / "_partial"
            if work.exists() and not options.resume:
                shutil.rmtree(work)
            work.mkdir(exist_ok=True)

            generated = _unwrap_pair(
                _prepare_pair(inputs, masks, target, work, options),
                work,
                unwrap_options,
                options,
            )

            # Publish only after the pair processing completed successfully.
            ready = {}
            for key_name, path in generated.items():
                destination = folder / path.name
                path.replace(destination)
                ready[key_name] = destination

            if not options.keep_intermediates:
                shutil.rmtree(work)

            status = "processed"

        logger.info("[%d/%d] %s: %s", i, len(pairs), name, status)
        jobs.append(ready)

    # Outside the pair loop: also runs when raw inversion outputs are reused.
    transform = target.transform
    dx = float(np.hypot(transform.a, transform.d))
    dy = float(np.hypot(transform.b, transform.e))

    if not target.crs.is_projected:
        raise ValueError("Meter-based smoothing requires a projected CRS")

    _, meters_per_unit = target.crs.linear_units_factor
    dx_m = dx * meters_per_unit
    dy_m = dy * meters_per_unit

    timeseries_dir = out_dir / "timeseries"
    correction_dir = out_dir / "corrections"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    correction_dir.mkdir(parents=True, exist_ok=True)
    names = [format_date_pair(*pair) for pair in output_pairs]
    intermediate = [timeseries_dir / f"{name}_iono_B.rad.tif" for name in names]

    if options.resume and all(p.is_file() for p in intermediate):
        logger.info("Reusing existing ionosphere inversion results")
    else:
        reference = select_common_reference(
            jobs,
            target,
            origin,
            timeseries_dir / "common_candidates.tif",
            options.block_size,
        )
        atomic_json(
            timeseries_dir / "reference.json",
            {"reference_b": list(reference), "reference_a": list(reference_a)},
        )
        temporary = [p.with_name(f"{p.stem}.partial.tif") for p in intermediate]
        _invert_jobs(
            jobs,
            pairs,
            target,
            reference,
            f_A,
            f_B,
            temporary,
            output_pairs,
            options.block_size,
        )
        for source, destination in zip(temporary, intermediate, strict=True):
            source.replace(destination)

    outputs = []
    for i, (src, match, name) in enumerate(
        zip(intermediate, products, names, strict=True)
    ):
        output_path = correction_dir / f"{name}_iono.tif"
        outputs.append(output_path)
        if _filter_cache_matches(output_path, src, options):
            logger.info("%s: reusing correction", name)
            continue

        filled_path = timeseries_dir / f"{name}_iono_B_filled.rad.tif"
        if _filter_cache_matches(filled_path, src, options):
            logger.info("%s: reusing filled ionosphere", name)
        else:
            with rasterio.open(src) as ds:
                iono = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
            footprint = np.ones(iono.shape, dtype=bool)
            for date in output_pairs[i]:
                for band in ("A", "B"):
                    with rasterio.open(cached_masks[date, band]["inside"]) as mask_ds:
                        with WarpedVRT(
                            mask_ds,
                            crs=target.crs,
                            transform=target.transform,
                            width=target.width,
                            height=target.height,
                            resampling=Resampling.average,
                            src_nodata=255,
                            nodata=255,
                            dtype="float32",
                        ) as mask_vrt:
                            inside = mask_vrt.read(1, masked=True).filled(0)
                    footprint &= inside > 0
            trusted = footprint & np.isfinite(iono)
            # All filter controls come from the main algorithm parameters.
            # No calibrated inversion uncertainty is currently available, so
            # filtering uses equal precision.
            logger.info(
                "%s: applying ionosphere filter %s with settings %s",
                name,
                options.filtering.method,
                options.filtering.model_dump(mode="json"),
            )
            iono_filled = apply_ionosphere_filter(
                iono,
                donor_mask=trusted,
                footprint=footprint,
                pixel_size_m=(dy_m, dx_m),
                options=options,
            )
            logger.info(
                "%s: ionosphere valid before=%d, after=%d",
                name,
                np.count_nonzero(trusted),
                np.count_nonzero(np.isfinite(iono_filled)),
            )
            temporary_filled = filled_path.with_name(f"{filled_path.stem}.partial.tif")
            # The raw ionosphere is already on the B grid; use it as the template.
            io.write_arr(
                arr=iono_filled.astype(np.float32),
                output_name=temporary_filled,
                like_filename=src,
                dtype="float32",
                nbands=1,
                nodata=np.nan,
                units="radians",
                options=WORKING_TIFF_OPTIONS,
            )
            with rasterio.open(temporary_filled, "r+") as dst:
                dst.update_tags(
                    ionosphere_method=IONOSPHERE_METHOD,
                    phase_convention="dolphin_ifg_ref_times_conj_sec",
                    ionosphere_filter_method=options.filtering.method,
                    ionosphere_filter_signature=filter_cache_signature(src, options),
                )
            temporary_filled.replace(filled_path)

        temporary_output = output_path.with_name(f"{output_path.stem}.partial.tif")
        _export_correction(
            filled_path,
            Path(match),
            temporary_output,
            f_A,
            reference_a,
            options.block_size,
        )
        with rasterio.open(temporary_output, "r+") as dst:
            dst.update_tags(
                ionosphere_filter_method=options.filtering.method,
                ionosphere_filter_signature=filter_cache_signature(src, options),
            )
        temporary_output.replace(output_path)
    return outputs

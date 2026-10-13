"""A/(A-B) estimation with fixed date-based output paths.

Resume checks final-file existence only. Changing inputs, reference, grid or
settings requires resume=False (for the entire rerun), or removal of affected
outputs and all dependent stages. Hash/attempt caches are not auto-migrated.
Preparation/inversion are windowed; gap filling uses one full B-grid array.

Dolphin/SNAPHU still controls unwrap memory. Phase-linked SLCs are consumed through
existing stitched IFGs; independent A/B displacement series are not mixed here.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from ._main_diff_io import (
    Grid,
    atomic_json,
    check_reduction,
    read_window,
    windows,
)
from .gslc_mask import prepare_gslc_mask_cache
from .inversion import build_design_matrix, invert_common_phase_block
from .mask import apply_similarity_mask_and_fill
from .options import IonosphereOptions
from .resampling import average_to_window, resample_phasor_to_match, source_window

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandInputs:
    """Existing stitched Dolphin outputs for one frequency."""

    interferograms: Sequence[Path]
    correlations: Sequence[Path]
    similarity: Path


def _date_pair(path: Path) -> tuple:
    from opera_utils import get_dates

    dates = [
        d.replace(hour=0, minute=0, second=0, microsecond=0) for d in get_dates(path)
    ]
    if len(dates) != 2 or dates[0] >= dates[1]:
        raise ValueError(f"Expected one ordered acquisition pair: {path}")
    return tuple(dates)


def _index(paths: Sequence[Path]) -> dict:
    result: dict[tuple, Path] = {}
    for p in paths:
        pair = _date_pair(p)
        if pair in result:
            raise ValueError(f"Duplicate pair {pair}: {result[pair]}, {p}")
        result[pair] = Path(p)
    return result


def estimate_iono_main_diff(
    f_a: float, f_b: float, phase_a: np.ndarray, phase_d: np.ndarray
) -> np.ndarray:
    """Estimate ionospheric phase at frequency A from A and A-minus-B.

    Parameters
    ----------
    f_a, f_b : float
        Distinct positive center frequencies in Hz.
    phase_a, phase_d : np.ndarray
        Matched unwrapped phase or phase time series, in radians, sharing spatial
        and temporal references, support, and phase convention.

    Returns
    -------
    np.ndarray
        Ionospheric phase at frequency A, in the input phase convention.

    """
    if not (np.isfinite(f_a) and np.isfinite(f_b) and min(f_a, f_b) > 0):
        raise ValueError("Center frequencies must be finite and positive")
    if abs(f_a - f_b) <= 1e-8 * max(f_a, f_b):
        raise ValueError("Center frequencies are indistinguishable")
    if phase_a.shape != phase_d.shape:
        raise ValueError("A and D shapes differ")
    return (
        f_b / (f_a + f_b) * phase_a.astype(np.float64)
        + f_a * f_b / (f_b * f_b - f_a * f_a) * phase_d.astype(np.float64)
    ).astype(np.float32)


def _same_grid(path, grid):
    if Grid.from_file(Path(path)) != grid:
        raise ValueError(f"Grid differs from the band's stitched IFG: {path}")
def _prepare_pair(inputs, masks, target, folder, options):
    """Prepare A/B or A/D interferograms using common donor support.

    Both methods use the B grid and identical mask/quality checks.
    Rejected pixels inside the footprint receive zero-phase placeholders
    and zero quality; they are never marked as donors.
    """
    method = options.method
    if method not in ("main_side", "main_diff"):
        raise ValueError(f"Unsupported ionosphere method: {method!r}")

    second = "D" if method == "main_diff" else "B"

    source = Grid.from_file(inputs["ifg_A"])
    check_reduction(source, target)

    specs = {
        "A_ifg": "complex64",
        f"{second}_ifg": "complex64",
        "A_cor": "float32",
        f"{second}_cor": "float32",
        "A_sim": "float32",
        f"{second}_sim": "float32",
        "donor": "uint8",
        "inside": "uint8",
    }

    folder.mkdir(parents=True, exist_ok=True)
    outputs = {name: folder / f"{name}.tif" for name in specs}

    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in inputs.items()
        }
        mask_readers = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in masks.items()
        }
        writers = {
            name: stack.enter_context(
                rasterio.open(
                    outputs[name],
                    "w",
                    **target.profile(
                        dtype,
                        0 if name.endswith("_ifg") else None,
                    ),
                )
            )
            for name, dtype in specs.items()
        }

        for win in windows(target, options.block_size):
            sw = source_window(source, target, win)

            # Apply the same native-grid validity and quality rules to A/B.
            native = {}
            for band, window in (("A", sw), ("B", win)):
                z = read_window(readers[f"ifg_{band}"], window)
                cor = read_window(readers[f"cor_{band}"], window)
                sim = read_window(readers[f"sim_{band}"], window)

                valid = (
                    read_window(
                        mask_readers[f"{band}_ref_valid"], window
                    ) == 1
                ) & (
                    read_window(
                        mask_readers[f"{band}_sec_valid"], window
                    ) == 1
                )

                inside_band = (
                    read_window(
                        mask_readers[f"{band}_ref_inside"], window
                    ) == 1
                ) & (
                    read_window(
                        mask_readers[f"{band}_sec_inside"], window
                    ) == 1
                )

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

            qa = (
                average_to_window(qa, source, sw, target, win)
                / denominator
            )
            sa = (
                average_to_window(sa, source, sw, target, win)
                / denominator
            )
            qa = np.clip(qa, 0, 1)

            # Footprint and donor support are distinct.
            inside = (
                average_to_window(ia, source, sw, target, win) > 0
            ) & ib

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

            # Only the second observation and its quality differ by method.
            if method == "main_diff":
                second_ifg = ua * np.conj(ub)
                second_cor = np.clip(qa * qb, 0, 1)
                second_sim = np.minimum(sa, sb)
            else:
                second_ifg = ub
                second_cor = qb
                second_sim = sb

            # Use identical donor support for both prepared observations.
            # 1+0j: interior placeholder; 0+0j: outside the footprint.
            a_ifg = np.where(
                inside,
                np.where(donor, ua, 1 + 0j),
                0 + 0j,
            )
            second_ifg = np.where(
                inside,
                np.where(donor, second_ifg, 1 + 0j),
                0 + 0j,
            )

            data = {
                "A_ifg": a_ifg,
                f"{second}_ifg": second_ifg,
                "A_cor": np.where(donor, qa, 0),
                f"{second}_cor": np.where(donor, second_cor, 0),
                "A_sim": np.where(donor, sa, 0),
                f"{second}_sim": np.where(donor, second_sim, 0),
                "donor": donor,
                "inside": inside,
            }

            for name, array in data.items():
                writers[name].write(
                    array.astype(specs[name]),
                    1,
                    window=win,
                )

    return outputs


def _second_band(method):
    if method == "main_diff":
        return "D"
    if method == "main_side":
        return "B"
    raise ValueError(f"Unsupported ionosphere method: {method!r}")


def _estimate_iono(method, f_a, f_b, phase_a, phase_second):
    """Estimate ionospheric phase while preserving the input array shape."""
    _second_band(method)

    if phase_a.shape != phase_second.shape:
        raise ValueError("Input phase arrays must have matching shapes")

    if not (
        np.isfinite(f_a)
        and np.isfinite(f_b)
        and min(f_a, f_b) > 0
    ):
        raise ValueError("Center frequencies must be finite and positive")

    if abs(f_a - f_b) <= 1e-8 * max(f_a, f_b):
        raise ValueError("Center frequencies are indistinguishable")

    if method == "main_diff":
        return estimate_iono_main_diff(
            f_a, f_b, phase_a, phase_second
        )

    from .gslc import estimate_iono_main_side

    # The existing estimator requires 2-D arrays.
    # Its calculation is pixelwise, so flattening epochs/rows is safe.
    shape = phase_a.shape
    iono, _ = estimate_iono_main_side(
        f_a,
        f_b,
        phase_a.reshape(-1, shape[-1]),
        phase_second.reshape(-1, shape[-1]),
    )
    return iono.reshape(shape)


def _unwrap_pair(prepared, folder, unwrap_options, options):
    from dolphin.unwrap import unwrap

    ionosphere_method = options.method
    if ionosphere_method not in ("main_side", "main_diff"):
        raise ValueError(
            f"Unsupported ionosphere method: {ionosphere_method!r}"
        )

    second = "D" if ionosphere_method == "main_diff" else "B"
    bands = ("A", second)

    unwrap_method = getattr(
        unwrap_options.unwrap_method, "value", unwrap_options.unwrap_method,
    )
    if unwrap_method not in ("snaphu", "whirlwind"):
        raise ValueError(
            f"{ionosphere_method} supports snaphu or whirlwind; "
            f"received {unwrap_method!r}"
        )

    # Validate all inputs before starting either unwrap.
    required = {"inside", "donor"}
    required.update(
        f"{band}_{kind}"
        for band in bands
        for kind in ("ifg", "cor", "sim")
    )
    missing = required.difference(prepared)
    if missing:
        raise ValueError(
            f"Missing prepared inputs for {ionosphere_method}: "
            f"{sorted(missing)}"
        )


    result = dict(prepared)

    for band in bands:
        cfg = unwrap_options.model_copy(deep=True)

        if unwrap_method == "whirlwind":
            # Interpolate through Dolphin preprocessing only.
            cfg.whirlwind_options.interpolate = False

        cfg.run_unwrap = True
        cfg.run_interpolation = True
        cfg.zero_where_masked = False

        # A/B use their own correlation; D uses the A*B proxy.
        cfg.preprocess_options.interpolation_cor_threshold = (
            options.correlation_threshold**2
            if band == "D"
            else options.correlation_threshold
        )
        cfg.preprocess_options.interpolation_similarity_threshold = (
            options.similarity_threshold
        )

        unw, cc = unwrap(
            ifg_filename=prepared[f"{band}_ifg"],
            corr_filename=prepared[f"{band}_cor"],
            unw_filename=folder / f"{band}.unw.tif",
            nlooks=options.nlooks,
            mask_filename=prepared["inside"],
            similarity_filename=prepared[f"{band}_sim"],
            unwrap_options=cfg,
            unw_nodata=np.nan,
            ccl_nodata=0,
            scratchdir=folder / f"scratch_{band}",
            delete_scratch=not options.keep_intermediates,
        )

        result[f"{band}_unw"] = Path(unw)
        result[f"{band}_cc"] = Path(cc)

    # Preserve the original donor mask for reference selection/inversion.
    # Clean prepared files only after both unwrap operations succeed.
    if not options.keep_intermediates:
        for name, path in prepared.items():
            if name != "donor":
                Path(path).unlink()
                result.pop(name)

    return result

def _sample(path, reference):
    with rasterio.open(path) as ds:
        return float(read_window(ds, Window(reference[1], reference[0], 1, 1))[0, 0])


def select_common_reference(
    jobs: list[dict[str, Path]],
    grid: Grid,
    origin: tuple[int, int],
    output: Path,
    block_size: int,
    method: str,
) -> tuple[int, int]:
    """Choose a common donor after all A/D unwrapping, writing candidates on disk.

    Require original donors with finite A/D phase in every pair.
    Connected-component labels are not used. Avoid a full-frame distance transform.
    """
    second = _second_band(method)
    bands = ("A", second)
    with rasterio.open(output, "w", **grid.profile("uint8")) as ds:
        for win in windows(grid, block_size):
            ds.write(
                np.ones((int(win.height), int(win.width)), np.uint8), 1, window=win
            )
    for job in jobs:
        with ExitStack() as stack:
            ds = stack.enter_context(rasterio.open(output, "r+"))
            readers = {
                n: stack.enter_context(rasterio.open(job[n]))
                for n in ("donor", "A_unw", f"{second}_unw")
            }
            for win in windows(grid, block_size):
                good = (ds.read(1, window=win) > 0) & (
                    read_window(readers["donor"], win) == 1
                )
                for band in bands:
                    phase = read_window(readers[f"{band}_unw"], win)
                    good &= np.isfinite(phase)
                ds.write(good.astype(np.uint8), 1, window=win)
    best = None
    with rasterio.open(output) as ds:
        for win in windows(grid, block_size):
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
            f"No common A/{second} reference survives donor and "
            "finite-phase checks; completed unwraps are retained"

        )
    return best[1], best[2]


def _invert_jobs(
    jobs,
    pairs,
    grid,
    reference,
    f_a,
    f_b,
    outputs,
    output_pairs,
    block_size,
    method,
):
    second = _second_band(method)
    bands = ("A", second)

    dates = sorted({date for pair in pairs for date in pair})
    idx = {date: i for i, date in enumerate(dates)}
    design = build_design_matrix(pairs, dates)

    if np.linalg.matrix_rank(design) != len(dates) - 1:
        raise ValueError(f"A/{second} network is disconnected")

    offsets = {
        band: np.array(
            [_sample(job[f"{band}_unw"], reference) for job in jobs]
        )
        for band in bands
    }

    if any(not np.isfinite(values).all() for values in offsets.values()):
        raise ValueError("Reference phase is not finite for every pair")

    with ExitStack() as stack:
        readers = [
            {
                name: stack.enter_context(rasterio.open(job[name]))
                for name in ("donor", "A_unw", f"{second}_unw")
            }
            for job in jobs
        ]

        writers = [
            stack.enter_context(
                rasterio.open(
                    path, "w", **grid.profile("float32", np.nan)
                )
            )
            for path in outputs
        ]

        for dst in writers:
            dst.set_band_unit(1, "radians")
            dst.update_tags(
                phase_convention="dolphin_ifg_ref_times_conj_sec",
                ionosphere_method=method,
            )

        for win in windows(grid, block_size):
            valid = np.stack(
                [read_window(reader["donor"], win) == 1
                 for reader in readers]
            )

            phases = {}
            for band in bands:
                phases[band] = np.stack(
                    [read_window(reader[f"{band}_unw"], win)
                     for reader in readers]
                )
                phases[band] -= offsets[band][:, None, None]

            phase_a, phase_second = invert_common_phase_block(
                phases["A"],
                phases[second],
                design,
                valid,
            )

            iono = _estimate_iono(
                method, f_a, f_b, phase_a, phase_second
            )

            zero = np.where(
                np.isfinite(iono).all(axis=0), 0, np.nan
            )[None].astype(np.float32)

            series = np.concatenate((zero, iono), axis=0)

            for dst, (ref, sec) in zip(
                writers, output_pairs, strict=True
            ):
                dst.write(
                    series[idx[sec]] - series[idx[ref]],
                    1,
                    window=win,
                )


def _export_correction(
    source, match, output, f_a, reference_a, block_size, method
):
    # Same matrix/sign as Dolphin inversion; convert raw phase by -lambda/(4*pi)
    # exactly once. The A-B phase has no independent wavelength.
    target = Grid.from_file(match)
    factor = -299792458.0 / f_a / (4 * np.pi)
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
            in_bounds = (
                0 <= row_b < src.height
                and 0 <= col_b < src.width
            )

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

                patch = src.read(
                    1,
                    window=Window(c0, r0, c1 - c0, r1 - r0),
                    masked=True,
                ).astype(np.float64).filled(np.nan)

                logger.error(
                    "Source nodata=%s; source reference value=%s; "
                    "source neighborhood:\n%s",
                    src.nodata,
                    patch[row_b - r0, col_b - c0],
                    np.array2string(patch, precision=5),
                )

            raise ValueError(
                "Product reference has no supported ionosphere estimate. "
                "See source/reference diagnostics above; existing unwraps are retained."
            )
        with rasterio.open(output, "w", **target.profile("float32", np.nan)) as dst:
            dst.set_band_unit(1, "meters")
            dst.update_tags(
                ionosphere_method=method,
                reference_row=reference_a[0],
                reference_col=reference_a[1],
            )
            for win in windows(target, block_size):
                dst.write(
                    ((read_window(vrt, win) - offset) * factor).astype(np.float32),
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
    f_a: float,
    f_b: float,
    out_dir: Path,
    unwrap_options,
    options: IonosphereOptions,
) -> list[Path]:
    """Prepare, unwrap, reference, invert, and export A/(A-B) corrections.

    Parameters
    ----------
    band_a, band_b : BandInputs
        Full matching stitched IFG networks on their respective native grids.
    gslc_by_date : dict
        Acquisition datetime (midnight) to the actual local raw GSLC used in PL.
    output_timeseries_paths : sequence[Path]
        Nominal products whose date pairs, output order and grids must be matched.
    reference_a : tuple[int, int]
        Existing nominal A spatial reference on its product grid.
    f_a, f_b : float
        Constant center frequencies in Hz; caller checks acquisition consistency.
    out_dir : Path
        Stable mask/pair/inversion cache directory.
    unwrap_options : dolphin.workflows.config.UnwrapOptions
        SNAPHU settings, with interpolation enabled and explicit donor thresholds.
    options : IonosphereOptions
        DISP-specific donor, memory and restart options.

    Returns
    -------
    list[Path]
        Correction rasters in meters, ordered like output_timeseries_paths.
        Original nominal data and their reference metadata are not modified.

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
            f_a,
            f_b,
            out_dir,
            unwrap_options,
            options,
        )


def _run_locked(
    band_a,
    band_b,
    gslc_by_date,
    products,
    reference_a,
    f_a,
    f_b,
    out_dir,
    unwrap_options,
    options,
):
    method = options.method
    second = _second_band(method)
    if not products:
        return []
    final_paths = [
        out_dir
        / "corrections"
        / ("_".join(d.strftime("%Y%m%d") for d in _date_pair(Path(p))) + "_iono.tif")
        for p in products
    ]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("Duplicate output date pairs")
    if options.resume and all(p.is_file() for p in final_paths):
        logger.info("Reusing all existing ionosphere corrections")
        return final_paths
    _estimate_iono(
        method, f_a, f_b,
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
    for b, band in (("A", band_a), ("B", band_b)):
        for p in (*band.interferograms, *band.correlations, band.similarity):
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
    raw_paths = [
        out_dir
        / "timeseries"
        / ("_".join(d.strftime("%Y%m%d") for d in pair) + "_iono_B.rad.tif")
        for pair in output_pairs
    ]
    inversion_ready = options.resume and all(p.is_file() for p in raw_paths)
    need_masks = not inversion_ready or any(
        not final.is_file()
        and not (
            out_dir
            / "timeseries"
            / ("_".join(d.strftime("%Y%m%d") for d in pair) + "_iono_B_filled.rad.tif")
        ).is_file()
        for pair, final in zip(output_pairs, final_paths, strict=True)
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
        name = "_".join(d.strftime("%Y%m%d") for d in pair)
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
            f"{second}_unw": folder / f"{second}.unw.tif",
            "donor": folder / "donor.tif",
        }

        if options.resume and all(p.is_file() for p in ready.values()):
            status = "reused"
        else:
            # One fixed working directory, not a new attempt directory each time.
            import shutil

            work = folder / "_partial"
            if work.exists():
                shutil.rmtree(work)
            work.mkdir()

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

    fill_settings = {
        "fill_method": "gdal",
        "max_search_dist": 500,  # Gap-fill search distance in pixels.
        "smooth_sigma": (
            options.smooth_sigma_m / dy_m,
            options.smooth_sigma_m / dx_m,
        ),
        "mask_erosion_px": options.mask_erosion_px,
    }

    timeseries_dir = out_dir / "timeseries"
    correction_dir = out_dir / "corrections"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    correction_dir.mkdir(parents=True, exist_ok=True)
    names = ["_".join(d.strftime("%Y%m%d") for d in pair) for pair in output_pairs]
    intermediate = [timeseries_dir / f"{name}_iono_B.rad.tif" for name in names]

    if options.resume and all(p.is_file() for p in intermediate):
        logger.info("Reusing existing ionosphere inversion results")
    else:
        reference = select_common_reference(
<<<<<<< HEAD
            jobs, target, origin, timeseries_dir / "common_candidates.tif",
            options.block_size,  method=method,
=======
            jobs,
            target,
            origin,
            timeseries_dir / "common_candidates.tif",
            options.block_size,
>>>>>>> origin/ionosphere_main_diff_initial
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
            f_a,
            f_b,
            temporary,
            output_pairs,
            options.block_size,
<<<<<<< HEAD
            method=method,
=======
>>>>>>> origin/ionosphere_main_diff_initial
        )
        for source, destination in zip(temporary, intermediate, strict=True):
            source.replace(destination)

    outputs = []
    for i, (src, match, name) in enumerate(
        zip(intermediate, products, names, strict=True)
    ):
        output_path = correction_dir / f"{name}_iono.tif"
        outputs.append(output_path)
        if options.resume and output_path.is_file():
            logger.info("%s: reusing correction", name)
            continue

        filled_path = timeseries_dir / f"{name}_iono_B_filled.rad.tif"
        if options.resume and filled_path.is_file():
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
            # trusted = footprint & np.isfinite(iono)
            # if not trusted.any():
            #     raise ValueError(f"{name}: no valid ionosphere samples for filling")

            # # Existing helper erodes invalid stripes along columns. Check donors
            # # before calling it: a Gaussian fill with no donors is undefined.
            # from scipy.ndimage import maximum_filter1d

            # radius = fill_settings["mask_erosion_px"]
            # eroded_donors = footprint & ~maximum_filter1d(
            #     (~trusted).astype(np.uint8), size=2 * radius + 1, axis=1,
            # ).astype(bool)
            # if not eroded_donors.any():
            #     raise ValueError(f"{name}: no donors remain after mask erosion")

            # iono_filled, quality_mask = apply_similarity_mask_and_fill(
            #     iono=iono, sim_mask=trusted, existing_mask=footprint,
            #     **fill_settings,
            # )
            from .utils import smooth_iono

            trusted = footprint & np.isfinite(iono)

            if not trusted.any():
                raise ValueError(f"{name}: no valid ionosphere samples for smoothing")

            # origin is the nominal A reference mapped onto the B grid.
            rr, cc = origin

            logger.warning(
                "%s reference before smoothing: footprint=%s, trusted=%s, raw=%s",
                name,
                bool(footprint[rr, cc]),
                bool(trusted[rr, cc]),
                float(iono[rr, cc]),
            )

            iono_filled = smooth_iono(
                arr=iono,
                sigma=fill_settings["smooth_sigma"],
                mask=trusted,
                restore_mask=False,
            )

            logger.warning(
                "%s reference after smoothing, before footprint: %s",
                name,
                float(iono_filled[rr, cc]),
            )

            iono_filled[~footprint] = np.nan
            quality_mask = trusted

            logger.warning(
                "%s reference after footprint: %s",
                name,
                float(iono_filled[rr, cc]),

            )
            logger.info(
                "%s: ionosphere valid before=%d, after=%d",
                name,
                np.count_nonzero(trusted),
                np.count_nonzero(np.isfinite(iono_filled)),
            )
            temporary_filled = filled_path.with_name(f"{filled_path.stem}.partial.tif")
            with rasterio.open(
                temporary_filled,
                "w",
                **target.profile("float32", np.nan),
            ) as dst:
                dst.write(iono_filled.astype(np.float32), 1)
                dst.set_band_unit(1, "radians")
                dst.update_tags(
                    ionosphere_method=method,
                    phase_convention="dolphin_ifg_ref_times_conj_sec",
                )
            temporary_filled.replace(filled_path)

        temporary_output = output_path.with_name(f"{output_path.stem}.partial.tif")
        _export_correction(
            filled_path,
            Path(match),
            temporary_output,
            f_a,
            reference_a,
            options.block_size,
            method=method,
        )
        temporary_output.replace(output_path)
    return outputs

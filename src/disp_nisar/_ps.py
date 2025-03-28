from __future__ import annotations

import logging
import multiprocessing as mp
from enum import Enum
from pathlib import Path

import dolphin.ps
import numpy as np
from dolphin import io, masking
from dolphin._types import PathOrStr
from dolphin.utils import DummyProcessPoolExecutor
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.wrapped_phase import _get_mask

logger = logging.getLogger(__name__)


class WeightScheme(str, Enum):
    """Methods for weighted combination of old and current amplitudes."""

    LINEAR = "linear"
    EQUAL = "equal"
    EXPONENTIAL = "exponential"


def precompute_ps(cfg: DisplacementWorkflow) -> tuple[list[Path], list[Path]]:
    # TODO: Check if this is working after removing burst

    # ######################################
    # 1. Wrapped phase estimation
    # ######################################
    # grab the only key ("") and use that
    cfg.create_dir_tree()

    combined_dispersion_files: list[Path] = []
    combined_mean_files: list[Path] = []
    Executor = DummyProcessPoolExecutor
    mw = cfg.worker_settings.n_parallel_bursts
    ctx = mp.get_context("spawn")
    with Executor(
        max_workers=mw,
        mp_context=ctx,
    ) as exc:
        future = exc.submit(run_frame_ps, cfg)
        combined_dispersion_file, combined_mean_file = future.result()
        combined_dispersion_files.append(combined_dispersion_file)
        combined_mean_files.append(combined_mean_file)

    return combined_dispersion_files, combined_mean_files


def run_frame_ps(cfg: DisplacementWorkflow) -> tuple[Path, Path]:
    input_file_list = cfg.cslc_file_list
    if not input_file_list:
        msg = "No input files found"
        raise ValueError(msg)

    subdataset = cfg.input_options.subdataset
    # Mark any files beginning with "compressed" as compressed
    is_compressed = ["compressed" in str(f).lower() for f in input_file_list]

    non_compressed_slcs = [
        f for f, is_comp in zip(input_file_list, is_compressed) if not is_comp
    ]
    vrt_stack = io.VRTStack(
        non_compressed_slcs,
        subdataset=subdataset,
        outfile=cfg.work_directory / "non_compressed_slc_stack.vrt",
    )

    layover_shadow_mask = (
        cfg.layover_shadow_mask_files[0] if cfg.layover_shadow_mask_files else None
    )
    mask_filename = _get_mask(
        output_dir=cfg.work_directory,
        output_bounds=cfg.output_options.bounds,
        output_bounds_wkt=cfg.output_options.bounds_wkt,
        output_bounds_epsg=cfg.output_options.bounds_epsg,
        like_filename=vrt_stack.outfile,
        layover_shadow_mask=layover_shadow_mask,
        cslc_file_list=non_compressed_slcs,
    )
    nodata_mask = masking.load_mask_as_numpy(mask_filename) if mask_filename else None

    output_file_list = [
        cfg.ps_options._output_file,
        cfg.ps_options._amp_mean_file,
        cfg.ps_options._amp_dispersion_file,
    ]
    ps_output = cfg.ps_options._output_file
    if not all(f.exists() for f in output_file_list):
        logger.info(f"Creating persistent scatterer file {ps_output}")
        # dispersions: np.ndarray, means: np.ndarray, N: ArrayLike | Sequence
        dolphin.ps.create_ps(
            reader=vrt_stack,
            output_file=output_file_list[0],
            output_amp_mean_file=output_file_list[1],
            output_amp_dispersion_file=output_file_list[2],
            like_filename=vrt_stack.outfile,
            amp_dispersion_threshold=cfg.ps_options.amp_dispersion_threshold,
            nodata_mask=nodata_mask,
            block_shape=cfg.worker_settings.block_shape,
        )
        # Remove the actual PS mask, since we're going to redo after combining
        cfg.ps_options._output_file.unlink()

    compressed_slc_files = [
        f for f, is_comp in zip(input_file_list, is_compressed) if is_comp
    ]
    logger.info(f"Combining existing means/dispersions from {compressed_slc_files}")
    return run_combine(
        cfg.ps_options._amp_mean_file,
        cfg.ps_options._amp_dispersion_file,
        compressed_slc_files,
        num_slc=len(non_compressed_slcs),
        subdataset=subdataset,
    )


def run_combine(
    cur_mean: Path,
    cur_dispersion: Path,
    compressed_slc_files: list[PathOrStr],
    num_slc: int,
    weight_scheme: WeightScheme = WeightScheme.EXPONENTIAL,
    subdataset: str = "/science/LSAR/GSLC/grids/frequencyA/HH",
) -> tuple[Path, Path]:
    out_dispersion = cur_dispersion.parent / "combined_dispersion.tif"
    out_mean = cur_mean.parent / "combined_mean.tif"
    if out_dispersion.exists() and out_mean.exists():
        logger.info(f"{out_mean} and {out_dispersion} exist, skipping")
        return out_dispersion, out_mean

    reader_compslc = io.HDF5StackReader.from_file_list(
        file_list=compressed_slc_files,
        dset_names=subdataset,
        nodata=np.nan,
    )
    reader_compslc_dispersion = io.HDF5StackReader.from_file_list(
        file_list=compressed_slc_files,
        dset_names="/data/amplitude_dispersion",
        nodata=np.nan,
    )
    reader_mean = io.RasterReader.from_file(cur_mean, band=1)
    reader_dispersion = io.RasterReader.from_file(cur_dispersion, band=1)

    num_images = 1 + len(compressed_slc_files)
    if weight_scheme == WeightScheme.LINEAR:
        # Increase the weights from older to newer.
        N = np.linspace(0, 1, num=num_images) * num_slc
    elif weight_scheme == WeightScheme.EQUAL:
        # Increase the weights from older to newer.
        N = num_slc * np.ones((num_images,))
    elif weight_scheme == WeightScheme.EXPONENTIAL:
        alpha = 0.5
        weights = np.exp(alpha * np.arange(num_images))
        weights /= weights.max()
        N = weights.round().astype(int)
    else:
        raise ValueError(f"Unrecognized {weight_scheme = }")

    io.write_arr(arr=None, output_name=out_dispersion, like_filename=cur_dispersion)
    io.write_arr(arr=None, output_name=out_mean, like_filename=cur_mean)

    block_manager = io.StridedBlockManager(
        arr_shape=reader_compslc.shape[-2:], block_shape=(256, 256)
    )
    for out_idxs, _trim_idxs, in_idxs, _, _ in block_manager.iter_blocks():
        in_rows, in_cols = in_idxs
        out_rows, out_cols = out_idxs

        rows, cols = in_rows, in_cols
        compslc_mean = np.abs(reader_compslc[:, rows, cols].filled(0))
        if compslc_mean.ndim == 2:
            compslc_mean = compslc_mean[np.newaxis]
        compslc_dispersion = reader_compslc_dispersion[:, rows, cols].filled(0)
        if compslc_dispersion.ndim == 2:
            compslc_dispersion = compslc_dispersion[np.newaxis]

        mean = reader_mean[rows, cols][np.newaxis].filled(0)
        dispersion = reader_dispersion[rows, cols][np.newaxis].filled(0)

        dispersions = np.vstack([compslc_dispersion, dispersion])

        means = np.vstack([compslc_mean, mean])
        new_dispersion, new_mean = dolphin.ps.combine_amplitude_dispersions(
            dispersions=dispersions,
            means=means,
            N=N,
        )

        io.write_block(
            np.nan_to_num(new_dispersion),
            filename=out_dispersion,
            row_start=out_rows.start,
            col_start=out_cols.start,
        )
        io.write_block(
            np.nan_to_num(new_mean),
            filename=out_mean,
            row_start=out_rows.start,
            col_start=out_cols.start,
        )

    return (out_dispersion, out_mean)

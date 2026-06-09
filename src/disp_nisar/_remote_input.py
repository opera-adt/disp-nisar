"""Pre-stage remote NISAR GSLC URLs to local files before Dolphin runs.

GDAL's HDF5 driver cannot reliably stream from NASA Earthdata HTTPS endpoints,
so we fetch the inputs once up front and point the workflow at local copies.

Strategy: download whole granules in parallel via ``earthaccess.download``
(one bulk HTTPS stream per file is *much* faster than many byte-range reads
over a high-latency connection), then locally extract only the layers the
workflow reads (one polarization + small metadata groups) and delete the
full download. Net effect: fast network transfer + small final disk usage.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import h5netcdf
import h5py
from opera_utils import is_remote_url
from tenacity import retry, stop_after_attempt, wait_fixed

from ._streaming import S3Path

logger = logging.getLogger(__name__)


# Groups copied wholesale (small — orbit, identification, processing metadata).
_ALWAYS_KEEP_GROUPS = (
    "/science/LSAR/identification",
    "/science/LSAR/GSLC/metadata/radarGrid",
    "/science/LSAR/GSLC/metadata/orbit",
    "/science/LSAR/GSLC/metadata/sourceData/swaths",
    "/science/LSAR/GSLC/metadata/sourceData/processingInformation/parameters/frequencyA",
)

# Groups copied wholesale for gunw.
_ALWAYS_KEEP_GROUPS_GUNW = ("/science/LSAR/identification",)

# Per-frequency grid datasets kept besides the chosen polarization.
_GRID_AUX_DATASETS = (
    "projection",
    "xCoordinates",
    "yCoordinates",
    "xCoordinateSpacing",
    "yCoordinateSpacing",
    "centerFrequency",
    "listOfPolarizations",
    "mask",
)

# Per-frequency grid datasets kept besides the chosen polarization for gunw.
_GRID_AUX_DATASETS_GUNW = (
    "projection",
    "xCoordinates",
    "yCoordinates",
    "xCoordinateSpacing",
    "yCoordinateSpacing",
    "mask",
)

# Per-frequency grid datasets kept from the chosen polarization for gunw.
_GRID_AUX_DATASETS_GUNW_POL = (
    "ionospherePhaseScreen",
    "projection",
    "xCoordinates",
    "yCoordinates",
    "xCoordinateSpacing",
    "yCoordinateSpacing",
    "mask",
)


def _normalize_url(url: str) -> str:
    return re.sub(r"^(https?|s3):/(?!/)", r"\1://", url)


def _extract_subset(
    src_path: Path, dst_path: Path, frequencies: Sequence[str], polarization: str
) -> None:
    """Locally copy only the needed datasets/groups into a smaller file.

    HDF5 dimension scales use object references (DIMENSION_LIST / REFERENCE_LIST)
    that don't survive a cross-file copy. We copy the coordinate datasets first,
    then the polarization dataset, then rebuild the dimension-scale links so
    GDAL's NETCDF driver can derive a geotransform from the result.

    Multiple ``frequencies`` may be requested (e.g. ``frequencyA`` and
    ``frequencyB`` for split-spectrum ionosphere) — each frequency's grid is
    copied into the same staged file while the frequency-agnostic metadata
    groups are copied only once.
    """
    if "GSLC" in str(src_path):
        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            for grp in _ALWAYS_KEEP_GROUPS:
                if grp in src and grp not in dst:
                    src.copy(grp, dst, name=grp)

            for frequency in frequencies:
                grid = f"/science/LSAR/GSLC/grids/{frequency}"
                # Order matters: aux/coord datasets first so dimension scales exist
                # before the polarization dataset is copied.
                for name in (*_GRID_AUX_DATASETS, polarization):
                    src_path_h5 = f"{grid}/{name}"
                    if src_path_h5 in src and src_path_h5 not in dst:
                        src.copy(src_path_h5, dst, name=src_path_h5)

                _rebuild_dimension_scales(dst, grid, polarization)
    elif "GUNW" in str(src_path):
        # 1. Open the source file normally with h5py
        # 2. Open the destination using h5netcdf (which forces NetCDF compliance)
        with (
            h5py.File(src_path, "r") as src,
            h5netcdf.File(dst_path, "w") as dst,
        ):
            # 1. Copy global attributes
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            def copy_structure(full_group_path, include_datasets=None):
                if full_group_path not in src:
                    return

                src_grp = src[full_group_path]

                try:
                    dst_grp = dst.create_group(full_group_path)
                except ValueError:
                    dst_grp = dst.groups[full_group_path]

                # Set of attributes that netCDF-4 uses internally and will reject if
                # written manually
                RESERVED_ATTRS = {
                    "DIMENSION_LIST",
                    "REFERENCE_LIST",
                    "CLASS",
                    "NAME",
                    "_Netcdf4Dimid",
                    "_Netcdf4Coordinates",
                }

                # Step 1: Pre-register coordinate dimensions if they are in your
                # whitelist. This guarantees that NetCDF creates a shared dimension
                # scale for 2D rasters
                for coord_name in ["xCoordinates", "yCoordinates"]:
                    if coord_name in src_grp and isinstance(
                        src_grp[coord_name], h5py.Dataset
                    ):
                        if include_datasets is None or coord_name in include_datasets:
                            coord_ds = src_grp[coord_name]
                            if coord_name not in dst_grp.dimensions:
                                dst_grp.dimensions[coord_name] = coord_ds.shape[0]

                # Step 2: Determine which items to copy
                # If include_datasets is provided, we only iterate over those
                # specific names
                items_to_copy = (
                    src_grp.keys() if include_datasets is None else include_datasets
                )

                for name in items_to_copy:
                    if name not in src_grp:
                        continue
                    item = src_grp[name]

                    if isinstance(item, h5py.Dataset):
                        dims = []

                        # Map the raster to its correct dimensions to preserve NetCDF
                        # compatibility
                        if name in ["xCoordinates", "yCoordinates"]:
                            dims = [name]
                        elif item.ndim == 2:
                            # NISAR convention for 2D rasters: (yCoordinates,
                            # xCoordinates)
                            dim_y = (
                                "yCoordinates"
                                if "yCoordinates" in dst_grp.dimensions
                                else f"{name}_dim_0"
                            )
                            dim_x = (
                                "xCoordinates"
                                if "xCoordinates" in dst_grp.dimensions
                                else f"{name}_dim_1"
                            )

                            if dim_y not in dst_grp.dimensions:
                                dst_grp.dimensions[dim_y] = item.shape[0]
                            if dim_x not in dst_grp.dimensions:
                                dst_grp.dimensions[dim_x] = item.shape[1]
                            dims = [dim_y, dim_x]
                        else:
                            # Fallback for 1D or 3D ancillary arrays
                            for i, dim_len in enumerate(item.shape):
                                dim_name = f"{name}_dim_{i}"
                                if dim_name not in dst_grp.dimensions:
                                    dst_grp.dimensions[dim_name] = dim_len
                                dims.append(dim_name)

                        var = dst_grp.create_variable(
                            name, data=item[...], dimensions=dims
                        )

                        # Copy variable attributes, skipping system-reserved ones
                        for k, v in item.attrs.items():
                            if k not in RESERVED_ATTRS:
                                var.attrs[k] = v

            # 2. Copy standard groups
            for grp in _ALWAYS_KEEP_GROUPS_GUNW:
                copy_structure(grp)

            # 3. Copy frequency grid pathways
            for frequency in frequencies:
                grid = f"/science/LSAR/GUNW/grids/{frequency}/unwrappedInterferogram"

                # Copy the base grid datasets (like xCoordinates, yCoordinates)
                copy_structure(grid, include_datasets=_GRID_AUX_DATASETS_GUNW)

                # Copy the polarization sub-group (like HH, VV or ionospherePhaseScreen)
                copy_structure(
                    f"{grid}/{polarization}",
                    include_datasets=_GRID_AUX_DATASETS_GUNW_POL,
                )

    else:
        logger.info(f"{src_path} not recognized for subsetting")


def _rebuild_dimension_scales(dst: h5py.File, grid: str, polarization: str) -> None:
    """Reattach xCoordinates/yCoordinates as dimension scales of the SLC dataset.

    NISAR convention: SLC array shape is (yCoordinates, xCoordinates).
    """
    pol_path = f"{grid}/{polarization}"
    x_path = f"{grid}/xCoordinates"
    y_path = f"{grid}/yCoordinates"

    if pol_path not in dst or x_path not in dst or y_path not in dst:
        return

    pol = dst[pol_path]
    x = dst[x_path]
    y = dst[y_path]

    # Drop stale references inherited from the source file.
    for attr in ("DIMENSION_LIST",):
        if attr in pol.attrs:
            del pol.attrs[attr]
    for d in (x, y):
        for attr in ("REFERENCE_LIST", "DIMENSION_LIST"):
            if attr in d.attrs:
                del d.attrs[attr]
        for attr in ("CLASS", "NAME", "_Netcdf4Dimid"):
            if attr in d.attrs:
                del d.attrs[attr]

    # Re-create dimension scales and attach them in NISAR axis order.
    x.make_scale("xCoordinates")
    y.make_scale("yCoordinates")
    pol.dims[0].attach_scale(y)
    pol.dims[1].attach_scale(x)


def _strip_slc_arrays(path: Path, frequencies: Sequence[str], polarization: str) -> int:
    """Drop the heavy SLC pixel arrays from a staged GSLC, keeping metadata.

    Once Dolphin's displacement workflows have consumed the input grids, the
    only large datasets left in a staged file are the ``/grids/{freq}/{pol}``
    complex SLC arrays. ``create_products`` reads identification, orbit, and
    coordinate metadata but never the SLC pixels, so those arrays are dead
    weight for the rest of the run.

    HDF5 cannot reclaim deleted-dataset space in place, so we rewrite the file
    to a fresh copy that omits the SLC arrays (cheap — only small metadata and
    coordinate datasets remain) and atomically replace the original. Coordinate
    and projection datasets, ``/identification``, ``/metadata/orbit``, and
    ``/metadata/radarGrid`` are all preserved.

    Returns the number of bytes reclaimed (0 if no SLC array was present).
    """
    drop = {f"/science/LSAR/GSLC/grids/{fr}/{polarization}" for fr in frequencies}
    with h5py.File(path, "r") as src:
        present = [p for p in drop if p in src]
    if not present:
        return 0  # nothing to strip (e.g. compressed SLC or already trimmed)

    size_before = path.stat().st_size
    tmp = path.with_suffix(path.suffix + ".trim")
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def _visit(name: str, obj) -> None:
            full = f"/{name}"
            if full in drop:
                return  # skip the heavy SLC array
            if isinstance(obj, h5py.Group):
                grp = dst.require_group(full)
                for k, v in obj.attrs.items():
                    grp.attrs[k] = v
            elif isinstance(obj, h5py.Dataset):
                src.copy(obj, dst, name=full)

        src.visititems(_visit)

        # The coordinate datasets carry dimension-scale references to the now
        # removed SLC; drop them so the new file has no dangling references.
        for fr in frequencies:
            grid = f"/science/LSAR/GSLC/grids/{fr}"
            for coord in ("xCoordinates", "yCoordinates"):
                p = f"{grid}/{coord}"
                if p in dst:
                    for attr in ("REFERENCE_LIST", "DIMENSION_LIST"):
                        if attr in dst[p].attrs:
                            del dst[p].attrs[attr]

    tmp.replace(path)
    reclaimed = size_before - path.stat().st_size
    logger.info(
        f"Trimmed SLC arrays from {path.name} ({reclaimed / 1e6:.1f} MB reclaimed)"
    )
    return reclaimed


def trim_staged_slc_arrays(
    cslc_files: Iterable[str | Path],
    stage_dir: Path,
    frequencies: Sequence[str],
    polarization: str,
) -> int:
    """Strip SLC pixel arrays from staged GSLCs after they are no longer needed.

    Only files that live under ``stage_dir`` (i.e. were staged by this run) are
    touched — user-provided local GSLCs are never modified. Compressed SLCs are
    skipped. Returns the total bytes reclaimed.
    """
    stage_dir = Path(stage_dir).resolve()
    total = 0
    for f in cslc_files:
        path = Path(f)
        if "compressed" in path.name.lower():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(stage_dir):
            continue  # not staged by us — leave the user's original untouched
        try:
            total += _strip_slc_arrays(resolved, frequencies, polarization)
        except OSError as e:
            logger.warning(f"Could not trim {path.name}: {e}")
    if total:
        logger.info(f"Reclaimed {total / 1e6:.1f} MB from staged GSLC SLC arrays")
    return total


def _download_and_trim(
    url: str,
    scratch_dir: Path,
    raw_dir: Path,
    frequencies: Sequence[str],
    polarization: str,
) -> Path:
    """Download one granule then extract the subset; remove the full download."""
    import earthaccess

    norm = _normalize_url(url)
    fname = Path(norm).name
    final = scratch_dir / fname
    if final.exists() and final.stat().st_size > 0:
        logger.info(f"Reusing cached {final.name}")
        return final

    raw = raw_dir / fname
    if not raw.exists() or raw.stat().st_size == 0:
        logger.info(f"Downloading {fname}")
        earthaccess.download([norm], local_path=str(raw_dir), provider="ASF")

    tmp = final.with_suffix(final.suffix + ".part")
    logger.info(f"Extracting {polarization}@{','.join(frequencies)} from {fname}")
    _extract_subset(raw, tmp, frequencies, polarization)
    tmp.replace(final)

    # Free disk: drop the full granule once the trimmed copy is in place.
    try:
        raw.unlink()
    except OSError:
        pass

    logger.info(f"Staged {final.name} ({final.stat().st_size / 1e6:.1f} MB)")
    return final


@retry(stop=stop_after_attempt(5), wait=wait_fixed(15))
def parallel_s3_download(
    s3_urls: Sequence[str],
    output_dir: Path,
    raw_dir: Path,
    frequencies: Sequence[str],
    polarization: str,
    max_workers: int = 5,
) -> list[Path]:
    """Download using an authorized Boto client in parallel."""
    max_workers = min(len(s3_urls), max_workers)
    downloaded_files: list[Path] = []

    import concurrent.futures

    from ._streaming import get_authorized_s3_client

    s3_client = get_authorized_s3_client(dataset="nisar")
    # for url in s3_urls:
    #     out = _download_file(
    #         s3_client,
    #         url,
    #         output_dir,
    #         raw_dir,
    #         frequencies,
    #         polarization,
    #     )
    #     downloaded_files.append[out]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {}
        for url in s3_urls:
            future = executor.submit(
                _download_file,
                s3_client,
                url,
                output_dir,
                raw_dir,
                frequencies,
                polarization,
            )
            future_to_url[future] = url

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                file_path = future.result()
                downloaded_files.append(file_path)
            except Exception:
                logger.exception(f"{url} generated an exception")
                raise

    return downloaded_files


def _download_file(
    s3_client,
    url: str,
    output_dir: Path,
    raw_dir: Path,
    frequencies: Sequence[str],
    polarization: str,
) -> Path:
    s3_path = S3Path(url)
    logger.info(f"Downloading {s3_path} to {output_dir}")

    final = output_dir / s3_path.path.name
    if final.exists() and final.stat().st_size > 0:
        logger.info(f"Reusing cached {final.name}")
        return final

    raw = raw_dir / s3_path.path.name

    if not raw.exists() or raw.stat().st_size == 0:
        logger.info(f"Downloading {final.name}")
        s3_client.download_file(
            Bucket=s3_path.bucket, Key=s3_path.key, Filename=str(raw)
        )

    logger.info(f"Downloading {url} to {final}")

    tmp = final.with_suffix(final.suffix + ".part")
    logger.info(f"Extracting {polarization}@{','.join(frequencies)} from {raw}")
    _extract_subset(raw, tmp, frequencies, polarization)
    tmp.replace(final)

    # Free disk: drop the full granule once the trimmed copy is in place.
    try:
        raw.unlink()
    except OSError:
        pass

    logger.info(f"Staged {final.name} ({final.stat().st_size / 1e6:.1f} MB)")

    return final


def stage_remote_inputs(
    urls: Iterable[str | Path],
    scratch_dir: Path,
    frequencies: Sequence[str] = ("frequencyA",),
    polarization: str = "HH",
    n_workers: int = 6,
) -> list[Path]:
    """Download remote NISAR GSLCs (parallel) then trim each to needed layers.

    Already-local paths and previously-staged files pass through unchanged.

    Parameters
    ----------
    urls : iterable of str | Path
        Mix of local paths and remote ``https://`` / ``s3://`` URLs.
    scratch_dir : Path
        Directory to write trimmed local outputs into.
    frequencies : list of str
        NISAR frequencies to keep (``"frequencyA"`` and/or ``"frequencyB"``).
        Both are staged in a single download pass when split-spectrum
        ionosphere will run (no GUNW files provided).
    polarization : str
        Polarization to keep (``"HH"``, ``"HV"``, ``"VV"``, ``"VH"``).
    n_workers : int
        Parallel download+trim workers.

    """
    import earthaccess

    scratch_dir = Path(scratch_dir).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = scratch_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    url_list = [str(u) for u in urls]
    remote_indices = [i for i, u in enumerate(url_list) if is_remote_url(u)]
    if remote_indices:
        earthaccess.login()

    out_paths: list[Path | None] = [None] * len(url_list)
    for i, u in enumerate(url_list):
        if not is_remote_url(u):
            out_paths[i] = Path(u).resolve()

    if not remote_indices:
        return [p for p in out_paths if p is not None]

    logger.info(
        f"Staging {len(remote_indices)} remote GSLCs -> {scratch_dir}"
        f" (keeping {polarization}@{','.join(frequencies)},"
        f" {n_workers} parallel workers)"
    )

    https_urls = [u for u in url_list if u.startswith("https://")]
    s3_urls = [u for u in url_list if u.startswith("s3://")]

    if https_urls:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(
                    _download_and_trim,
                    url_list[i],
                    scratch_dir,
                    raw_dir,
                    frequencies,
                    polarization,
                ): i
                for i in remote_indices
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                out_paths[i] = fut.result()
    elif s3_urls:
        out_paths = parallel_s3_download(
            s3_urls=s3_urls,
            output_dir=scratch_dir,
            raw_dir=raw_dir,
            frequencies=frequencies,
            polarization=polarization,
            max_workers=n_workers,
        )

    # Best-effort cleanup of the empty raw dir.
    try:
        raw_dir.rmdir()
    except OSError:
        pass

    return [p for p in out_paths if p is not None]


def run_dolphin(
    config_file: str | Path,
    debug: bool = False,
    n_workers: int = 6,
) -> None:
    """Run the disp-nisar workflow, pre-staging any remote GSLC inputs."""
    from disp_nisar.main import run
    from disp_nisar.pge_runconfig import RunConfig

    pge_runconfig = RunConfig.from_yaml(str(config_file))

    scratch_gslc_dir = pge_runconfig.product_path_group.scratch_path / "gslc"
    scratch_gunw_dir = pge_runconfig.product_path_group.scratch_path / "gunw"
    gslc_files = pge_runconfig.input_file_group.gslc_file_list
    gunw_files = pge_runconfig.dynamic_ancillary_file_group.gunw_files
    if any(is_remote_url(f) for f in gslc_files):
        frequency = pge_runconfig.input_file_group.frequency
        if not isinstance(frequency, str):
            frequency = frequency.value
        polarization = pge_runconfig.input_file_group.polarization
        if not isinstance(polarization, str):
            polarization = polarization.value

        # When no GUNW files are provided, main.run() runs a second displacement
        # workflow on frequencyB for split-spectrum ionosphere (see main.py).
        # Stage both frequencies in this single download pass so the freqB run
        # has its grid without re-downloading every granule.
        frequencies = [frequency]
        if not pge_runconfig.dynamic_ancillary_file_group.gunw_files:
            if "frequencyB" not in frequencies:
                frequencies.append("frequencyB")

        local_files = stage_remote_inputs(
            gslc_files,
            scratch_dir=Path(scratch_gslc_dir),
            frequencies=frequencies,
            polarization=polarization,
            n_workers=n_workers,
        )
        pge_runconfig.input_file_group.gslc_file_list = local_files
    if len(gunw_files) > 0 and any(is_remote_url(f) for f in gunw_files):
        frequency = pge_runconfig.input_file_group.frequency
        if not isinstance(frequency, str):
            frequency = frequency.value
        polarization = pge_runconfig.input_file_group.polarization
        if not isinstance(polarization, str):
            polarization = polarization.value

        frequencies = [frequency]

        local_files_gunw = stage_remote_inputs(
            gunw_files,
            scratch_dir=Path(scratch_gunw_dir),
            frequencies=frequencies,
            polarization=polarization,
            n_workers=n_workers,
        )
        pge_runconfig.dynamic_ancillary_file_group.gunw_files = local_files_gunw

    cfg = pge_runconfig.to_workflow()
    run(cfg, pge_runconfig=pge_runconfig, debug=debug)

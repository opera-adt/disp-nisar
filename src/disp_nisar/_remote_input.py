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
from typing import Iterable

import h5py
from opera_utils import is_remote_url

logger = logging.getLogger(__name__)


# Groups copied wholesale (small — orbit, identification, processing metadata).
_ALWAYS_KEEP_GROUPS = (
    "/science/LSAR/identification",
    "/science/LSAR/GSLC/metadata/radarGrid",
    "/science/LSAR/GSLC/metadata/orbit",
)

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


def _normalize_url(url: str) -> str:
    return re.sub(r"^(https?|s3):/(?!/)", r"\1://", url)


def _extract_subset(
    src_path: Path, dst_path: Path, frequency: str, polarization: str
) -> None:
    """Locally copy only the needed datasets/groups into a smaller file.

    HDF5 dimension scales use object references (DIMENSION_LIST / REFERENCE_LIST)
    that don't survive a cross-file copy. We copy the coordinate datasets first,
    then the polarization dataset, then rebuild the dimension-scale links so
    GDAL's NETCDF driver can derive a geotransform from the result.
    """
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        for grp in _ALWAYS_KEEP_GROUPS:
            if grp in src and grp not in dst:
                src.copy(grp, dst, name=grp)

        grid = f"/science/LSAR/GSLC/grids/{frequency}"
        # Order matters: aux/coord datasets first so dimension scales exist
        # before the polarization dataset is copied.
        for name in (*_GRID_AUX_DATASETS, polarization):
            src_path_h5 = f"{grid}/{name}"
            if src_path_h5 in src and src_path_h5 not in dst:
                src.copy(src_path_h5, dst, name=src_path_h5)

        _rebuild_dimension_scales(dst, grid, polarization)


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


def _download_and_trim(
    url: str,
    scratch_dir: Path,
    raw_dir: Path,
    frequency: str,
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
    logger.info(f"Extracting {polarization}@{frequency} from {fname}")
    _extract_subset(raw, tmp, frequency, polarization)
    tmp.replace(final)

    # Free disk: drop the full granule once the trimmed copy is in place.
    try:
        raw.unlink()
    except OSError:
        pass

    logger.info(f"Staged {final.name} ({final.stat().st_size / 1e6:.1f} MB)")
    return final


def stage_remote_gslcs(
    urls: Iterable[str | Path],
    scratch_dir: Path,
    frequency: str = "frequencyA",
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
    frequency : str
        NISAR frequency to keep (``"frequencyA"`` or ``"frequencyB"``).
    polarization : str
        Polarization to keep (``"HH"``, ``"HV"``, ``"VV"``, ``"VH"``).
    n_workers : int
        Parallel download+trim workers.

    """
    import earthaccess

    scratch_dir = Path(scratch_dir)
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
            out_paths[i] = Path(u)

    if not remote_indices:
        return [p for p in out_paths if p is not None]

    logger.info(
        f"Staging {len(remote_indices)} remote GSLCs -> {scratch_dir}"
        f" (keeping {polarization}@{frequency}, {n_workers} parallel workers)"
    )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {
            pool.submit(
                _download_and_trim,
                url_list[i],
                scratch_dir,
                raw_dir,
                frequency,
                polarization,
            ): i
            for i in remote_indices
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            out_paths[i] = fut.result()

    # Best-effort cleanup of the empty raw dir.
    try:
        raw_dir.rmdir()
    except OSError:
        pass

    return [p for p in out_paths if p is not None]


def run_dolphin_with_earthaccess(
    config_file: str | Path,
    debug: bool = False,
    n_workers: int = 6,
) -> None:
    """Run the disp-nisar workflow, pre-staging any remote GSLC inputs."""
    from disp_nisar.main import run
    from disp_nisar.pge_runconfig import RunConfig

    pge_runconfig = RunConfig.from_yaml(str(config_file))

    scratch_dir = pge_runconfig.product_path_group.scratch_path / "stage_inputs"
    gslc_files = pge_runconfig.input_file_group.gslc_file_list
    if any(is_remote_url(f) for f in gslc_files):
        frequency = pge_runconfig.input_file_group.frequency
        if not isinstance(frequency, str):
            frequency = frequency.value
        polarization = pge_runconfig.input_file_group.polarization
        if not isinstance(polarization, str):
            polarization = polarization.value

        local_files = stage_remote_gslcs(
            gslc_files,
            scratch_dir=Path(scratch_dir),
            frequency=frequency,
            polarization=polarization,
            n_workers=n_workers,
        )
        pge_runconfig.input_file_group.gslc_file_list = local_files

    cfg = pge_runconfig.to_workflow()
    run(cfg, pge_runconfig=pge_runconfig, debug=debug)

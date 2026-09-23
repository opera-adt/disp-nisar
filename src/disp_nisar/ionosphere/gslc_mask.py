"""Reduce GSLC masks directly to looked IFG grids using bounded HDF5 reads."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from ._main_diff_io import (
    ALGORITHM_VERSION,
    Grid,
    check_reduction,
    completed,
    finish,
    key,
    stamp,
    windows,
)


def _intervals(origin, step, dst_origin, dst_step, n):
    edges = (dst_origin + np.arange(n + 1) * dst_step - origin) / step
    rounded = np.rint(edges)
    edges = np.where(np.abs(edges - rounded) < 1e-6, rounded, edges)
    return (
        np.floor(np.minimum(edges[:-1], edges[1:])).astype(int),
        np.ceil(np.maximum(edges[:-1], edges[1:])).astype(int),
    )


def _sums(a, r0, r1, c0, c1):
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
    np.cumsum(a, axis=0, dtype=np.int64, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
    return (
        ii[r1[:, None], c1]
        - ii[r0[:, None], c1]
        - ii[r1[:, None], c0]
        + ii[r0[:, None], c0]
    )


def reduce_gslc_mask(dataset, source: Grid, target: Grid, mode: str, tile_size: int):
    """Yield window, valid donor, and footprint arrays without a native-sized read.

    Codes: 1 valid, 0 invalid inside, 255 outside. Unknown codes are errors.
    all_valid includes every native pixel with positive-area overlap.
    """
    check_reduction(source, target)
    if mode not in ("all_valid", "nearest"):
        raise ValueError(f"Unknown reduction: {mode}")
    st, dt = source.transform, target.transform
    y0, y1 = _intervals(st.f, st.e, dt.f, dt.e, target.height)
    x0, x1 = _intervals(st.c, st.a, dt.c, dt.a, target.width)
    for win in windows(target, tile_size):
        r, c, h, w = map(int, (win.row_off, win.col_off, win.height, win.width))
        a0, a1 = y0[r : r + h], y1[r : r + h]
        b0, b1 = x0[c : c + w], x1[c : c + w]
        if mode == "nearest":
            a0 = np.floor(
                (dt.f + (np.arange(r, r + h) + 0.5) * dt.e - st.f) / st.e
            ).astype(int)
            b0 = np.floor(
                (dt.c + (np.arange(c, c + w) + 0.5) * dt.a - st.c) / st.a
            ).astype(int)
            a1, b1 = a0 + 1, b0 + 1
        ly, hy = np.clip(a0, 0, source.height), np.clip(a1, 0, source.height)
        lx, hx = np.clip(b0, 0, source.width), np.clip(b1, 0, source.width)
        ry, ey, cx, ex = int(ly.min()), int(hy.max()), int(lx.min()), int(hx.max())
        good: np.ndarray = np.zeros((h, w), bool)
        inside = good.copy()
        if ey > ry and ex > cx:
            a = np.asarray(dataset[ry:ey, cx:ex])
            if not np.isin(a, (0, 1, 255)).all():
                raise ValueError("Unexpected GSLC mask codes; inspect product encoding")
            args = ly - ry, hy - ry, lx - cx, hx - cx
            area = (a1 - a0)[:, None] * (b1 - b0)[None, :]
            good = (_sums(a == 1, *args) == area) & (area > 0)
            inside = _sums(a != 255, *args) > 0
        yield win, good, inside


def prepare_gslc_mask_cache(
    source: Path,
    band: str,
    grid: Grid,
    cache: Path,
    mode: str = "all_valid",
    tile_size: int = 256,
    work_mb: int = 128,
) -> dict[str, Path]:
    """Cache each acquisition/frequency mask once per target grid and policy."""
    if band not in ("A", "B"):
        raise ValueError("band must be A or B")
    sig = {
        "version": ALGORITHM_VERSION,
        "source": stamp(source),
        "band": band,
        "grid": grid.record(),
        "mode": mode,
        "valid_values": [1],
        "outside_values": [255],
    }
    folder = cache / key(sig)
    if outputs := completed(folder, sig):
        return outputs
    folder.mkdir(parents=True, exist_ok=True)
    attempt = Path(mkdtemp(prefix="attempt_", dir=folder))
    outputs = {n: attempt / f"{n}.tif" for n in ("valid", "inside")}
    with h5py.File(source, "r", rdcc_nbytes=16 * 1024**2) as h:
        g = h[f"/science/LSAR/GSLC/grids/frequency{band}"]
        x, y = np.asarray(g["xCoordinates"]), np.asarray(g["yCoordinates"])
        if min(len(x), len(y)) < 2 or g["mask"].shape != (len(y), len(x)):
            raise ValueError("Invalid GSLC coordinate/mask dimensions")
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dy)):
            raise ValueError("Nonuniform GSLC coordinates")
        proj = g["projection"]
        wkt = proj.attrs.get("spatial_ref", proj.attrs.get("crs_wkt"))
        if isinstance(wkt, bytes):
            wkt = wkt.decode()
        crs = CRS.from_wkt(wkt) if wkt is not None else CRS.from_epsg(int(proj[()]))
        native = Grid(
            len(y), len(x), crs, Affine(dx, 0, x[0] - dx / 2, 0, dy, y[0] - dy / 2)
        )
        check_reduction(native, grid)
        ratio = abs(grid.transform.a / dx * grid.transform.e / dy)
        tile = max(1, min(tile_size, int(np.sqrt(work_mb * 1024**2 / (64 * ratio)))))
        with (
            rasterio.open(outputs["valid"], "w", **grid.profile("uint8")) as vd,
            rasterio.open(outputs["inside"], "w", **grid.profile("uint8")) as fd,
        ):
            for win, valid, inside in reduce_gslc_mask(
                g["mask"], native, grid, mode, tile
            ):
                vd.write(valid.astype("uint8"), 1, window=win)
                fd.write(inside.astype("uint8"), 1, window=win)
    finish(folder, sig, outputs)
    return outputs

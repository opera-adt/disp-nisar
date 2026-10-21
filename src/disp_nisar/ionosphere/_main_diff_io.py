"""Bounded raster I/O and restart records for main-differential estimation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from dolphin import io
from rasterio.windows import Window

ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class Grid:
    """Pixel-edge transform and shape of a north-up raster."""

    height: int
    width: int
    crs: Any
    transform: Affine

    from dolphin import io

    @classmethod
    def from_file(cls, path: Path) -> Grid:
        """Read raster grid metadata using Dolphin.

        Parameters
        ----------
        path : Path
            Input raster readable by Dolphin's RasterReader.

        Returns
        -------
        Grid
            Raster dimensions, coordinate reference system, and
            pixel-edge affine transform.

        """
        reader = io.RasterReader.from_file(path, keep_open=False)
        height, width = reader.shape

        return cls(
            height=height,
            width=width,
            crs=reader.crs,
            transform=reader.transform,
        )

    def record(self) -> dict:
        """Return JSON-compatible spatial metadata."""
        return {
            "height": self.height,
            "width": self.width,
            "crs": self.crs.to_wkt(),
            "transform": list(self.transform)[:6],
        }

    def profile(self, dtype: str, nodata=None) -> dict:
        """Build a tiled working GeoTIFF profile."""
        return {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "crs": self.crs,
            "transform": self.transform,
            "count": 1,
            "dtype": dtype,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "lzw",
            "BIGTIFF": "IF_SAFER",
        }

    def create_raster(self, path: Path, dtype: str, nodata=None):
        """Create a working raster with Dolphin, then open it for block writes."""
        io.write_arr(
            arr=None,
            output_name=path,
            shape=(self.height, self.width),
            dtype=dtype,
            geotransform=self.transform.to_gdal(),
            projection=self.crs,
            nodata=nodata,
            options=[
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "COMPRESS=LZW",
                "BIGTIFF=IF_SAFER",
            ],
        )
        return rasterio.open(path, "r+")


def read_window(ds, window: Window) -> np.ndarray:
    """Read declared nodata as NaN, preserving complex data and uint32 labels."""
    outside = (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > ds.width
        or window.row_off + window.height > ds.height
    )
    a = ds.read(1, window=window, boundless=outside, masked=True)
    dtype = np.complex64 if np.iscomplexobj(a) else np.float32
    if np.issubdtype(a.dtype, np.integer) and a.dtype.itemsize >= 4:
        dtype = np.float64
    return a.astype(dtype).filled(np.nan)


def stamp(path: Path) -> dict:
    """Identify a local input, including a VRT's backing files."""
    path = Path(path).resolve()
    s = path.stat()
    result = {"path": str(path), "size": s.st_size, "mtime_ns": s.st_mtime_ns}
    if path.suffix.lower() == ".vrt":
        with rasterio.open(path) as ds:
            result["dependencies"] = [
                stamp(Path(p)) for p in sorted(ds.files) if Path(p).resolve() != path
            ]
    return result


def key(record: dict) -> str:
    """Hash stable inputs and algorithm versions, never Jupyter cell identity."""
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:24]


def atomic_json(path: Path, record: dict) -> None:
    """Publish a complete JSON record atomically."""
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def completed(folder: Path, signature: dict) -> dict[str, Path] | None:
    """Reuse only finished stages with unchanged, readable output rasters."""
    try:
        record = json.loads((folder / "complete.json").read_text())
        if record["signature"] != signature:
            return None
        outputs = {n: Path(p) for n, p in record["outputs"].items()}
        for n, p in outputs.items():
            if stamp(p) != record["stamps"][n]:
                return None
            with rasterio.open(p) as ds:
                ds.read(1, window=Window(0, 0, 1, 1))
        return outputs
    except (OSError, ValueError, KeyError, TypeError):
        return None


def finish(folder: Path, signature: dict, outputs: dict[str, Path]) -> None:
    """Publish completion after all stage outputs finish."""
    atomic_json(
        folder / "complete.json",
        {
            "signature": signature,
            "outputs": {n: str(p.resolve()) for n, p in outputs.items()},
            "stamps": {n: stamp(p) for n, p in outputs.items()},
        },
    )

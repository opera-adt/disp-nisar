"""Bounded raster I/O helpers for main-differential estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from dolphin import io
from rasterio.io import DatasetReader
from rasterio.windows import Window


@dataclass(frozen=True)
class Grid:
    """Pixel-edge transform and shape of a north-up raster."""

    height: int
    width: int
    crs: Any
    transform: Affine

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


def read_window(ds: DatasetReader, window: Window) -> np.ndarray:
    """Read declared nodata as NaN, preserving complex data and uint32 labels."""
    outside = (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > ds.width
        or window.row_off + window.height > ds.height
    )
    a = ds.read(1, window=window, boundless=outside, masked=True)
    dtype: type = np.complex64 if np.iscomplexobj(a) else np.float32
    if np.issubdtype(a.dtype, np.integer) and a.dtype.itemsize >= 4:
        dtype = np.float64
    return a.astype(dtype).filled(np.nan)


def atomic_json(path: Path, record: dict) -> None:
    """Publish a complete JSON record atomically."""
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)

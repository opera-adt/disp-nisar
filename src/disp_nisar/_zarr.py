"""NISAR GSLC → Zarr converter and ZarrStack reader.

Zarr store layout
-----------------
<output>.zarr/
  freqA/
    HH    : (n_dates, ny, nx_a)  complex64
    mask  : (n_dates, ny, nx_a)  uint8
    .attrs: dates, xCoordinates, yCoordinates, xCoordinateSpacing,
            yCoordinateSpacing, centerFrequency, projection
  freqB/              [optional]
    HH    : (n_dates, ny, nx_b)  complex64
    mask  : (n_dates, ny, nx_b)  uint8
  .zattrs : dates, file_list

Chunk shape is (n_dates, spatial_chunks, spatial_chunks) — all dates per
spatial tile, which is optimal for phase-linking (reads one chunk per block).

Usage
-----
    from disp_nisar._zarr import gslc_to_zarr, ZarrStack

    # one-time preprocessing (~16 GB/file × n_dates disk write)
    gslc_to_zarr(files, 'stack.zarr', freqs=('A',))

    # drop-in for VRTStack anywhere in the dolphin workflow
    stack = ZarrStack('stack.zarr', freq='A')
    block = stack[:, 0:512, 0:512]          # numpy array, shape (n_dates, 512, 512)
    da_arr = stack.as_dask()                # dask array for parallel ops
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._streaming import open_h5_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(fname: str | Path) -> str:
    """Extract YYYYMMDD from a NISAR GSLC filename."""
    m = re.search(r"_(\d{8})T\d{6}_", Path(fname).name)
    if m:
        return m.group(1)
    # fallback: first 8-digit sequence
    m = re.search(r"(\d{8})", Path(fname).name)
    return m.group(1) if m else "unknown"


def _make_compressor(name: str = "lz4"):
    """Return a zarr v3 BloscCodec for the requested algorithm."""
    from zarr.codecs import BloscCodec  # zarr ≥ 3.x

    cname_map = {"lz4": "lz4", "zstd": "zstd", "lz4hc": "lz4hc", "none": None}
    cname = cname_map.get(name, "lz4")
    if cname is None:
        return []
    return [BloscCodec(cname=cname, clevel=5, shuffle="shuffle")]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def gslc_to_zarr(
    file_list: Sequence[str | Path],
    output_path: str | Path,
    freqs: Sequence[str] = ("A",),
    pol: str = "HH",
    spatial_chunks: int = 512,
    row_batch: int = 1024,
    compressor: str = "lz4",
    overwrite: bool = False,
    progress: bool = True,
) -> Any:
    """Convert NISAR GSLC HDF5 files to a single Zarr store.

    Parameters
    ----------
    file_list:
        Chronologically sorted list of NISAR GSLC .h5 files.
    output_path:
        Output Zarr store path, e.g. ``'stack.zarr'``.
    freqs:
        Frequencies to extract: ``('A',)`` or ``('A', 'B')``.
    pol:
        Polarization, default ``'HH'``.
    spatial_chunks:
        Spatial chunk size in pixels (both x and y). The time dimension is
        always chunked as ``n_dates`` so every phase-linking block is one chunk.
    row_batch:
        Number of HDF5 rows to read at once.  Controls peak memory usage:
        ``n_dates × row_batch × nx × 8 bytes``.
        Default 1024 rows → ~3.4 GB peak for freqA with 6 dates.
        Use 512 for tighter memory budgets.
    compressor:
        Blosc compressor: ``'lz4'`` (fast), ``'zstd'`` (best ratio), ``'none'``.
    overwrite:
        Recreate the store if it already exists.
    progress:
        Show a tqdm progress bar.

    Returns
    -------
    zarr.Group
        Opened root group of the written store (read-only).

    Notes
    -----
    For 6 dates of freqA, the uncompressed stack is ~222 GB.  Blosc-lz4
    typically achieves ~2–3× on complex SAR data, so expect ~80–110 GB on disk.
    Full write time depends on I/O bandwidth; ~30–90 min on a fast NFS/SSD.

    """
    import zarr

    try:
        from tqdm.auto import tqdm
    except ImportError:

        def tqdm(it, **_kw):
            return it

    file_list = [Path(f) for f in file_list]
    n_dates = len(file_list)
    dates = [_parse_date(f) for f in file_list]
    output_path = Path(output_path)

    if output_path.exists():
        if not overwrite:
            print(f"{output_path} already exists — returning existing store.")
            print("Pass overwrite=True to recreate.")
            return zarr.open_group(str(output_path), mode="r")
        import shutil

        shutil.rmtree(output_path)

    codecs = _make_compressor(compressor)
    root = zarr.open_group(str(output_path), mode="w")
    root.attrs["dates"] = dates
    root.attrs["n_dates"] = n_dates
    root.attrs["file_list"] = [str(f) for f in file_list]

    # Keep all HDF5 files open for the duration of the write
    handles = [open_h5_file(f, "r") for f in file_list]

    try:
        for freq_letter in freqs:
            freq_key = f"frequency{freq_letter.upper()}"
            grp_path = f"science/LSAR/GSLC/grids/{freq_key}"

            ds0 = handles[0][f"{grp_path}/{pol}"]
            ny, nx = ds0.shape
            dtype = ds0.dtype

            # Read coordinate metadata from first file
            meta: dict[str, Any] = {}
            for mkey in [
                "xCoordinates",
                "yCoordinates",
                "xCoordinateSpacing",
                "yCoordinateSpacing",
                "centerFrequency",
                "projection",
            ]:
                node = handles[0][grp_path].get(mkey)
                if node is None:
                    continue
                val = node[()]
                meta[mkey] = val.tolist() if hasattr(val, "tolist") else float(val)

            freq_grp = root.require_group(f"freq{freq_letter.upper()}")
            freq_grp.attrs.update(meta)
            freq_grp.attrs["dates"] = dates
            freq_grp.attrs["pol"] = pol
            freq_grp.attrs["shape_yx"] = [ny, nx]

            chunks = (n_dates, spatial_chunks, spatial_chunks)

            slc_arr = freq_grp.create_array(
                pol,
                shape=(n_dates, ny, nx),
                chunks=chunks,
                dtype=dtype,
                compressors=codecs,
            )
            mask_arr = freq_grp.create_array(
                "mask",
                shape=(n_dates, ny, nx),
                chunks=chunks,
                dtype="uint8",
                compressors=codecs,
            )

            raw_gb = n_dates * ny * nx * np.dtype(dtype).itemsize / 1e9
            print(
                f"freq{freq_letter.upper()}/{pol}: ({n_dates}, {ny}, {nx}) "
                f"{dtype}  raw={raw_gb:.1f} GB  chunks={chunks}"
            )

            row_starts = range(0, ny, row_batch)
            for row_start in tqdm(
                row_starts,
                desc=f"freq{freq_letter.upper()}→zarr",
                unit="batch",
                disable=not progress,
            ):
                row_end = min(row_start + row_batch, ny)
                nrows = row_end - row_start

                slc_buf = np.empty((n_dates, nrows, nx), dtype=dtype)
                mask_buf = np.empty((n_dates, nrows, nx), dtype="uint8")

                for i, h in enumerate(handles):
                    slc_buf[i] = h[f"{grp_path}/{pol}"][row_start:row_end, :]
                    mask_buf[i] = h[f"{grp_path}/mask"][row_start:row_end, :]

                # Zero out invalid pixels (mask==0) so downstream JAX/cuSolver
                # never receives NaN. Dolphin skips blocks that are all-zero,
                # so this converts nodata edge regions to skippable zeros.
                slc_buf[mask_buf == 0] = 0

                slc_arr[:, row_start:row_end, :] = slc_buf
                mask_arr[:, row_start:row_end, :] = mask_buf

    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass

    zarr.consolidate_metadata(str(output_path))
    print(f"\nDone → {output_path}")
    return zarr.open_group(str(output_path), mode="r")


# ---------------------------------------------------------------------------
# Reader — drop-in replacement for dolphin's VRTStack
# ---------------------------------------------------------------------------


class ZarrStack:
    """Read a Zarr SLC stack produced by :func:`gslc_to_zarr`.

    Implements the ``DatasetReader`` protocol expected by dolphin
    (``shape``, ``dtype``, ``ndim``, ``__getitem__``), so it can be passed
    anywhere a ``VRTStack`` is used.

    Parameters
    ----------
    zarr_path:
        Path to the ``.zarr`` store written by :func:`gslc_to_zarr`.
    freq:
        Frequency band: ``'A'`` or ``'B'``.
    pol:
        Polarization: ``'HH'`` or ``'HV'``.

    Examples
    --------
    >>> stack = ZarrStack('stack.zarr')
    >>> stack.shape                          # (n_dates, ny, nx)
    >>> block = stack[:, 256:768, 256:768]   # numpy array
    >>> da_arr = stack.as_dask()             # dask array, same shape
    >>> da_arr.rechunk({0: -1, 1: 256, 2: 256})  # rechunk for experiments

    """

    def __init__(
        self,
        zarr_path: str | Path,
        freq: str = "A",
        pol: str = "HH",
    ) -> None:
        import zarr

        self._path = Path(zarr_path)
        self._freq = freq.upper()
        self._pol = pol

        root = zarr.open_group(str(zarr_path), mode="r")
        freq_grp = root[f"freq{self._freq}"]
        self._arr = freq_grp[pol]  # zarr.Array (n_dates, ny, nx)
        self._meta = dict(freq_grp.attrs)
        self._root_meta = dict(root.attrs)

    # -- DatasetReader protocol ------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._arr.dtype)

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._arr[key])

    # -- VRTStack compatibility ------------------------------------------------

    @property
    def outfile(self) -> Path:
        """Mimic VRTStack.outfile — returns the zarr store path."""
        return self._path

    @property
    def file_list(self) -> list[Path]:
        return [Path(f) for f in self._root_meta.get("file_list", [])]

    @property
    def subdataset(self) -> str:
        """HDF5 subdataset path — for compatibility with dolphin's sequential.py."""
        return f"/science/LSAR/GSLC/grids/frequency{self._freq}/{self._pol}"

    @property
    def gdal_path(self) -> str:
        """GDAL-readable path to the last source HDF5 file.

        Use this wherever dolphin expects a ``like_filename`` that GDAL can
        open to read geotransform / projection / size metadata::

            dolphin.ps.create_ps(..., like_filename=zarr_stack.gdal_path)
        """
        src = self.file_list
        if not src:
            raise RuntimeError(
                "No source files in zarr metadata. "
                "Pass like_filename=<path_to_any_gslc_h5> explicitly."
            )
        return f'NETCDF:"{src[-1]}":{self.subdataset}'

    def __fspath__(self) -> str:
        """Make fspath(zarr_stack) return the GDAL-readable HDF5 path.

        This is what dolphin's io._core._get_gdal_ds calls via
        ``gdal.Open(fspath(like_filename))``.
        """
        return self.gdal_path

    # -- Extra helpers ---------------------------------------------------------

    @property
    def dates(self) -> list[str]:
        return self._meta.get("dates") or self._root_meta.get("dates") or []

    @property
    def n_dates(self) -> int:
        return self.shape[0]

    @property
    def mask(self) -> "ZarrMaskView":
        """Access the mask sub-array (same shape as SLC stack)."""
        return ZarrMaskView(self._path, self._freq)

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_coords, y_coords) 1-D arrays."""
        x = np.asarray(self._meta["xCoordinates"])
        y = np.asarray(self._meta["yCoordinates"])
        return x, y

    def as_dask(self, chunks: tuple | None = None) -> Any:
        """Return a dask array backed by this Zarr store.

        Parameters
        ----------
        chunks:
            Override chunk shape. Default uses the stored chunk shape
            ``(n_dates, spatial_chunks, spatial_chunks)``.

        """
        import dask.array as da

        da_arr = da.from_zarr(
            str(self._path), component=f"freq{self._freq}/{self._pol}"
        )
        if chunks is not None:
            da_arr = da_arr.rechunk(chunks)
        return da_arr

    def __repr__(self) -> str:
        ny, nx = self.shape[1], self.shape[2]
        return (
            f"ZarrStack(freq{self._freq}/{self._pol}  "
            f"shape=({self.n_dates}, {ny}, {nx})  "
            f"dtype={self.dtype}  dates={self.dates[0]}…{self.dates[-1]})"
        )


class ZarrMaskView:
    """Thin wrapper for the mask array inside a ZarrStack store."""

    def __init__(self, zarr_path: Path, freq: str) -> None:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        self._arr = root[f"freq{freq}"]["mask"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._arr[key])

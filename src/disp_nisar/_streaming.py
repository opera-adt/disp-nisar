"""Utilities for streaming CSLC/GSLC files from remote sources using earthaccess."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import h5py
import numpy as np

logger = logging.getLogger(__name__)

try:
    import earthaccess
    import xarray as xr

    HAS_EARTHACCESS = True
except ImportError:
    HAS_EARTHACCESS = False
    logger.debug("earthaccess not available, remote streaming disabled")


def is_remote_url(file_path: str | Path) -> bool:
    """Check if a file path is a remote URL (https:// or s3://).

    Parameters
    ----------
    file_path : str | Path
        File path to check.

    Returns
    -------
    bool
        True if the path is a remote URL, False otherwise.

    """
    path_str = str(file_path)
    parsed = urlparse(path_str)
    return parsed.scheme in ("https", "http", "s3")


def authenticate_earthdata() -> Any:
    """Authenticate with NASA Earthdata Login.

    Returns
    -------
    Any
        Earthaccess authentication object.

    Raises
    ------
    ImportError
        If earthaccess is not installed.

    """
    if not HAS_EARTHACCESS:
        msg = (
            "earthaccess is required for streaming remote files. "
            "Install with: pip install earthaccess"
        )
        raise ImportError(msg)

    logger.info("Authenticating with NASA Earthdata Login")
    auth = earthaccess.login()
    return auth


def open_remote_file(file_url: str) -> Any:
    """Open a remote file using earthaccess.

    Parameters
    ----------
    file_url : str
        URL of the remote file (https:// or s3://).

    Returns
    -------
    Any
        File-like object that can be used with h5py.

    Raises
    ------
    ImportError
        If earthaccess is not installed.

    """
    if not HAS_EARTHACCESS:
        msg = (
            "earthaccess is required for streaming remote files. "
            "Install with: pip install earthaccess"
        )
        raise ImportError(msg)

    logger.info(f"Opening remote file: {file_url}")

    # Authenticate if not already done
    authenticate_earthdata()

    # For s3:// URLs, we can directly open them with earthaccess
    # For https:// URLs, we need to search for the granule first
    if file_url.startswith("s3://"):
        # Open S3 file directly
        file_obj = earthaccess.open([file_url])[0]
    else:
        # For https URLs, try to open directly
        # earthaccess.open can handle both granule objects and URLs
        file_obj = earthaccess.open([file_url])[0]

    return file_obj


def open_h5_file(file_path: str | Path, mode: str = "r") -> h5py.File:
    """Open an HDF5 file, supporting both local and remote paths.

    Parameters
    ----------
    file_path : str | Path
        Path to the HDF5 file (local path, https://, or s3://).
    mode : str, optional
        File open mode, by default "r".

    Returns
    -------
    h5py.File
        Opened HDF5 file object.

    """
    if is_remote_url(file_path):
        file_obj = open_remote_file(str(file_path))
        return h5py.File(file_obj, mode)
    else:
        return h5py.File(file_path, mode)


def open_xarray_dataset(
    file_paths: list[str | Path],
    group: str = "/science/LSAR/GSLC/grids/frequencyA",
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Open remote GSLC files as an xarray Dataset using earthaccess.

    This function streams data without downloading files locally.

    Parameters
    ----------
    file_paths : list[str | Path]
        List of file paths or URLs to open.
    group : str, optional
        HDF5 group path to open, by default "/science/LSAR/GSLC/grids/frequencyA".
    chunks : dict[str, int] | None, optional
        Chunk sizes for dask arrays, by default {"row": 4000, "col": 4000}.

    Returns
    -------
    xr.Dataset
        Xarray dataset with the data.

    Raises
    ------
    ImportError
        If earthaccess is not installed.

    Examples
    --------
    >>> file_paths = ["s3://opera-adt/path/to/file1.h5", "s3://opera-adt/path/to/file2.h5"]
    >>> ds = open_xarray_dataset(file_paths)  # doctest: +SKIP
    >>> hh_cube = ds['HH']  # doctest: +SKIP

    """
    if not HAS_EARTHACCESS:
        msg = (
            "earthaccess and xarray are required for streaming remote files. "
            "Install with: pip install earthaccess xarray h5netcdf"
        )
        raise ImportError(msg)

    if chunks is None:
        chunks = {"row": 4000, "col": 4000}

    # Separate remote and local files
    remote_files = [f for f in file_paths if is_remote_url(f)]
    local_files = [f for f in file_paths if not is_remote_url(f)]

    # Authenticate if we have remote files
    if remote_files:
        authenticate_earthdata()
        logger.info(f"Opening {len(remote_files)} remote files with earthaccess")
        file_objects = earthaccess.open(remote_files)
    else:
        file_objects = []

    # Combine with local files
    all_files = file_objects + [str(f) for f in local_files]

    # Open as xarray dataset
    ds = xr.open_mfdataset(
        all_files,
        group=group,
        engine="h5netcdf",
        phony_dims="access",
        concat_dim="time",
        combine="nested",
        chunks=chunks,
    )

    return ds


class XarrayStackReader:
    """Reader for HDF5 stacks using xarray/dask for remote file support.

    This class provides a similar interface to dolphin's VRTStack and HDF5StackReader
    but uses xarray with dask for chunked parallel processing, especially useful for
    remote files accessed via earthaccess.

    Parameters
    ----------
    file_list : list[str | Path]
        List of HDF5 file paths (local or remote URLs).
    subdataset : str
        HDF5 dataset path to read (e.g., "/science/LSAR/GSLC/grids/frequencyA/HH").
    chunks : dict[str, int] | None, optional
        Chunk sizes for dask arrays, by default {"time": 1, "row": 2048, "col": 2048}.
        Can also be a tuple (row, col) which will be converted to dict.
    nodata : float | None, optional
        Nodata value to use for masked arrays, by default np.nan.
    overlap : dict[int, int] | None, optional
        Overlap/halo pixels for each dimension (e.g., {0: 0, 1: 200, 2: 0} for
        200 pixels overlap in row direction). Used for operations that need
        neighboring pixels like phase linking.
    n_workers : int | None, optional
        Number of dask workers for parallel processing. If None, uses dask's default.
        Should typically be set from worker_settings.n_parallel_bursts in runconfig.

    Examples
    --------
    >>> reader = XarrayStackReader(
    ...     file_list=["s3://bucket/file1.h5", "s3://bucket/file2.h5"],
    ...     subdataset="/science/LSAR/GSLC/grids/frequencyA/HH",
    ...     chunks={"time": 1, "row": 2048, "col": 2048},
    ...     overlap={0: 0, 1: 200, 2: 0}  # 200 pixel overlap in azimuth
    ... )  # doctest: +SKIP
    >>> data = reader[:, 100:200, 100:200]  # Read a block  # doctest: +SKIP

    """

    def __init__(
        self,
        file_list: list[str | Path],
        subdataset: str,
        chunks: dict[str, int] | tuple[int, int] | None = None,
        nodata: float | None = None,
        overlap: dict[int, int] | None = None,
        n_workers: int | None = None,
    ):
        """Initialize the xarray stack reader."""
        self.file_list = [Path(f) if not is_remote_url(f) else f for f in file_list]
        self.subdataset = subdataset
        self.nodata = nodata if nodata is not None else np.nan
        self.overlap = overlap
        self.n_workers = n_workers

        # Handle different chunk formats
        if chunks is None:
            chunks = {"time": 1, "row": 2048, "col": 2048}
        elif isinstance(chunks, tuple):
            # Convert (row, col) tuple to dict
            chunks = {"time": 1, "row": chunks[0], "col": chunks[1]}

        self.chunks = chunks

        # Configure dask if n_workers specified
        if self.n_workers is not None:
            self._configure_dask(self.n_workers)

        # Check if any files are remote
        self.has_remote = any(is_remote_url(f) for f in self.file_list)

        # Get dataset name from subdataset path
        self.dset_name = subdataset.split("/")[-1]

        # Determine group path (everything except the last component)
        parts = subdataset.split("/")
        self.group = "/".join(parts[:-1]) if len(parts) > 1 else None

        # Open the dataset
        self._open_dataset(chunks)

    def _configure_dask(self, n_workers: int) -> None:
        """Configure dask for parallel processing.

        This also configures JAX/NumPy threading to avoid oversubscription.

        Parameters
        ----------
        n_workers : int
            Number of dask workers to use.

        """
        import multiprocessing
        import os

        # Calculate threads per worker to avoid oversubscription
        # Leave 1-2 cores for system if possible
        total_cores = multiprocessing.cpu_count()
        threads_per_worker = max(1, total_cores // n_workers)

        # Configure JAX and NumPy threading BEFORE creating workers
        # This affects all spawned processes
        logger.info(
            f"Configuring parallelism: {n_workers} workers × {threads_per_worker} "
            f"threads = {n_workers * threads_per_worker}/{total_cores} cores"
        )

        # JAX configuration
        os.environ['XLA_FLAGS'] = (
            f'--xla_cpu_multi_thread_eigen=true '
            f'intra_op_parallelism_threads={threads_per_worker} '
            f'inter_op_parallelism_threads=1'
        )

        # NumPy/BLAS threading (used by JAX backend)
        os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
        os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
        os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
        os.environ['NUMEXPR_NUM_THREADS'] = str(threads_per_worker)

        # Prevent JAX from pre-allocating all GPU memory
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

        logger.info(
            f"Set thread limits: OMP={threads_per_worker}, "
            f"JAX intra_op={threads_per_worker}"
        )

        try:
            from dask.distributed import Client, LocalCluster

            # Check if a dask client already exists
            try:
                from dask.distributed import get_client
                client = get_client()
                logger.info(
                    f"Using existing dask client with {len(client.cluster.workers)} workers"
                )
            except ValueError:
                # No client exists, create one
                logger.info(f"Creating dask LocalCluster with {n_workers} workers")
                cluster = LocalCluster(
                    n_workers=n_workers,
                    threads_per_worker=threads_per_worker,
                    processes=True,  # CRITICAL: Isolates JAX in each worker
                    silence_logs=logging.ERROR,
                )
                client = Client(cluster)
                logger.info(f"Dask client created: {client}")
                logger.info(f"Dashboard: {client.dashboard_link}")
        except ImportError:
            # dask.distributed not available, fall back to threaded scheduler
            logger.warning(
                "dask.distributed not available, using threaded scheduler. "
                "Install with: pip install 'dask[distributed]'"
            )
            import dask
            dask.config.set(scheduler='threads', num_workers=n_workers)

    def _open_dataset(self, chunks: dict[str, int]) -> None:
        """Open the dataset using xarray."""
        if not HAS_EARTHACCESS and self.has_remote:
            msg = "earthaccess is required for remote files"
            raise ImportError(msg)

        # Separate remote and local files
        remote_files = [f for f in self.file_list if is_remote_url(f)]
        local_files = [str(f) for f in self.file_list if not is_remote_url(f)]

        # Open remote files with earthaccess
        if remote_files:
            authenticate_earthdata()
            file_objects = earthaccess.open(remote_files)
        else:
            file_objects = []

        # Combine with local files
        all_files = file_objects + local_files

        # Open with xarray
        try:
            self.ds = xr.open_mfdataset(
                all_files,
                group=self.group,
                engine="h5netcdf",
                concat_dim="time",
                combine="nested",
                chunks=chunks,
                mask_and_scale=False,
            )

            # Get the data array
            if self.dset_name in self.ds:
                self.data = self.ds[self.dset_name]
            else:
                # Try to find the dataset in the group
                available = list(self.ds.data_vars.keys())
                if len(available) == 1:
                    self.data = self.ds[available[0]]
                else:
                    raise ValueError(
                        f"Dataset {self.dset_name} not found. Available: {available}"
                    )
        except Exception as e:
            logger.error(f"Failed to open dataset with xarray: {e}")
            raise

    def __getitem__(self, key):
        """Get data using numpy-style indexing."""
        result = self.data[key]

        # Load the data if it's a dask array
        if hasattr(result, 'compute'):
            result = result.compute()

        # Convert to numpy array and handle nodata
        arr = np.asarray(result)

        # Return as masked array if nodata is set
        if self.nodata is not None and not np.isnan(self.nodata):
            mask = arr == self.nodata
            return np.ma.masked_array(arr, mask=mask)

        return arr

    def map_overlap(
        self,
        func,
        depth: dict[int, int] | None = None,
        boundary: str = "reflect",
        trim: bool = True,
        **kwargs,
    ):
        """Apply a function to overlapping blocks with halos.

        This enables processing with neighboring pixels for operations like
        phase linking, filtering, etc.

        Parameters
        ----------
        func : callable
            Function to apply to each block. Should accept a numpy array
            with shape (time, row + 2*overlap, col + 2*overlap).
        depth : dict[int, int] | None, optional
            Overlap depth for each dimension. If None, uses self.overlap.
            Format: {0: time_overlap, 1: row_overlap, 2: col_overlap}
        boundary : str, optional
            How to handle boundaries. Options: "reflect", "periodic", "nearest", "none".
            Default is "reflect".
        trim : bool, optional
            Whether to trim overlaps after processing. Default is True.
        **kwargs : dict
            Additional arguments to pass to the function.

        Returns
        -------
        xr.DataArray
            Processed data array with same coordinates and dimensions.

        Examples
        --------
        >>> def process_block(block):
        ...     # Your processing with neighboring pixels
        ...     return block * 1.0  # doctest: +SKIP
        >>> result = reader.map_overlap(process_block, depth={0: 0, 1: 200, 2: 0})  # doctest: +SKIP

        """
        import dask.array as da

        if depth is None:
            if self.overlap is None:
                raise ValueError("No overlap specified. Provide depth or set overlap in __init__")
            depth = self.overlap

        # Get the dask array from xarray
        dask_data = self.data.data

        # Apply map_overlap
        processed = da.map_overlap(
            func,
            dask_data,
            depth=depth,
            boundary=boundary,
            dtype=dask_data.dtype,
            trim=trim,
            **kwargs,
        )

        # Return as xarray DataArray with same coords/dims
        import xarray as xr
        return xr.DataArray(
            processed,
            coords=self.data.coords,
            dims=self.data.dims,
            attrs=self.data.attrs,
        )

    @property
    def shape(self) -> tuple:
        """Shape of the data array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def dtype(self):
        """Data type."""
        return self.data.dtype

    def close(self) -> None:
        """Close the dataset."""
        if hasattr(self, 'ds'):
            self.ds.close()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()


class StreamingFileManager:
    """Context manager for handling both local and remote file access.

    This class provides a unified interface for working with files that may be
    local or remote (https:// or s3://).

    Parameters
    ----------
    file_list : list[str | Path]
        List of file paths (local or remote URLs).
    authenticate : bool, optional
        Whether to authenticate with earthaccess on initialization, by default True.

    Examples
    --------
    >>> with StreamingFileManager(file_list) as manager:  # doctest: +SKIP
    ...     for file_path in file_list:  # doctest: +SKIP
    ...         with manager.open_h5(file_path) as hf:  # doctest: +SKIP
    ...             data = hf['dataset'][:]  # doctest: +SKIP

    """

    def __init__(
        self, file_list: list[str | Path], authenticate: bool = True
    ) -> None:
        """Initialize the streaming file manager."""
        self.file_list = file_list
        self.remote_files = [f for f in file_list if is_remote_url(f)]
        self.local_files = [f for f in file_list if not is_remote_url(f)]
        self._authenticated = False

        if authenticate and self.remote_files:
            self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with earthaccess if remote files are present."""
        if not self._authenticated and self.remote_files and HAS_EARTHACCESS:
            authenticate_earthdata()
            self._authenticated = True

    def __enter__(self) -> "StreamingFileManager":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        # Clean up if needed
        pass

    def open_h5(self, file_path: str | Path, mode: str = "r") -> h5py.File:
        """Open an HDF5 file (local or remote).

        Parameters
        ----------
        file_path : str | Path
            Path to the file.
        mode : str, optional
            File open mode, by default "r".

        Returns
        -------
        h5py.File
            Opened HDF5 file.

        """
        return open_h5_file(file_path, mode)

    def get_file_objects(self, file_paths: list[str | Path] | None = None) -> list:
        """Get file-like objects for a list of files.

        For remote files, this returns earthaccess file objects.
        For local files, this returns the file paths as strings.

        Parameters
        ----------
        file_paths : list[str | Path] | None, optional
            File paths to open. If None, uses self.file_list.

        Returns
        -------
        list
            List of file objects (earthaccess objects for remote, paths for local).

        """
        if file_paths is None:
            file_paths = self.file_list

        remote_files = [f for f in file_paths if is_remote_url(f)]
        local_files = [f for f in file_paths if not is_remote_url(f)]

        result = []

        # Open remote files with earthaccess
        if remote_files and HAS_EARTHACCESS:
            self._authenticate()
            result.extend(earthaccess.open(remote_files))

        # Add local file paths
        result.extend([str(f) for f in local_files])

        return result

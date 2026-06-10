"""Utilities for streaming CSLC/GSLC files from remote sources using earthaccess."""

from __future__ import annotations

import copy
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Self, Union
from urllib.parse import ParseResult, urlparse

import h5py
import numpy as np
from opera_utils import is_remote_url
from opera_utils.credentials import AWSCredentials

logger = logging.getLogger(__name__)

try:
    import earthaccess
    import xarray as xr

    HAS_EARTHACCESS = True
except ImportError:
    HAS_EARTHACCESS = False
    logger.debug("earthaccess not available, remote streaming disabled")


class ASFCredentialEndpoints(Enum):
    """Enumeration of ASF temporary credentials endpoints."""

    OPERA = "https://cumulus.asf.alaska.edu/s3credentials"
    OPERA_UAT = "https://cumulus-test.asf.alaska.edu/s3credentials"
    SENTINEL1 = "https://sentinel1.asf.alaska.edu/s3credentials"
    NISAR = "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"


class GeneralPath(Protocol):
    """A protocol to handle paths that can be either local or S3 paths."""

    def parent(self): ...

    def suffix(self): ...

    def read_text(self): ...

    def __truediv__(self, other): ...

    def __str__(self) -> str: ...

    def __fspath__(self) -> str:
        return str(self)


class S3Path(GeneralPath):
    """A convenience class to handle paths on S3.

    This class relies on `pathlib.Path` for operations using `urllib` to parse the url.

    If passing a url with a trailing slash, the slash will be preserved
    when converting back to string.

    Note that pure path manipulation functions do *not* require `boto3`,
    but functions which interact with S3 (e.g. `exists()`, `.read_text()`) do.

    Attributes
    ----------
    bucket : str
        Name of bucket in the url
    path : pathlib.Path
        The URL path after s3://<bucket>/
    key : str
        Alias of `path` converted to a string

    Examples
    --------
    >>> from orca.paths import S3Path
    >>> s3_path = S3Path("s3://bucket/path/to/file.txt")
    >>> str(s3_path)
    's3://bucket/path/to/file.txt'
    >>> s3_path.parent
    S3Path("s3://bucket/path/to/")
    >>> str(s3_path.parent)
    's3://bucket/path/to/'

    """

    def __init__(self, s3_url: Union[str, "S3Path"]):
        """Create an S3Path.

        Parameters
        ----------
        s3_url : str or S3Path
            The S3 url to parse.

        """
        # Names come from the urllib.parse.ParseResult
        if isinstance(s3_url, S3Path):
            self._scheme: str = s3_url._scheme
            self._netloc: str = s3_url._netloc
            self.bucket: str = s3_url.bucket
            self.path: Path = s3_url.path
            self._trailing_slash: str = s3_url._trailing_slash
        else:
            parsed: ParseResult = urlparse(s3_url)
            self._scheme = parsed.scheme
            self._netloc = self.bucket = parsed.netloc
            # self._parsed = parsed
            self.path = Path(parsed.path)
            self._trailing_slash = "/" if s3_url.endswith("/") else ""

        if self._scheme != "s3":
            raise ValueError(f"{s3_url} is not an S3 url")

    @classmethod
    def from_bucket_key(cls, bucket: str, key: str):
        """Create a `S3Path` from the bucket name and key/prefix.

        Matches API of some Boto3 functions which use this format.

        Parameters
        ----------
        bucket : str
            Name of S3 bucket.
        key : str
            S3 url of path after the bucket.

        """
        return cls(f"s3://{bucket}/{key}")

    def get_path(self) -> str:
        """Get the full S3 URI as a string."""
        # For S3 paths, we need to add the double slash and netloc back to the front
        return f"{self._scheme}://{self._netloc}{self.path.as_posix()}{self._trailing_slash}"

    @property
    def key(self) -> str:
        """Name of key/prefix within the bucket with leading slash removed."""
        return f"{str(self.path.as_posix()).lstrip('/')}{self._trailing_slash}"

    @property
    def parent(self):
        """The S3Path to the parent directory."""
        parent_path = self.path.parent
        # Since the constructor only accepts s3:// URIs,
        # the else case will never be triggered. So we could simplify it to:
        return S3Path(f"{self._scheme}://{self._netloc}{parent_path.as_posix()}/")
        # # Since this is a parent, it will will always end in a slash
        # if self._scheme == "s3":
        #     # For S3 paths, we need to add the scheme and netloc back to the front
        #     return S3Path(f"{self._scheme}://{self._netloc}{parent_path.as_posix()}/")
        # else:
        #     # For local paths, we can just convert the path to a string
        #     return S3Path(str(parent_path) + "/")

    @property
    def suffix(self):
        """The file extension (including the dot) or '' if there is no extension."""
        return self.path.suffix

    def _get_client(self):
        import boto3

        return boto3.client("s3")

    def exists(self) -> bool:
        """Whether this path exists on S3."""
        client = self._get_client()
        resp = client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.key,
            MaxKeys=1,
        )
        return resp.get("KeyCount") == 1

    def read_text(self) -> str:
        """Download/read the S3 file as text."""
        return self._download_as_bytes().decode()

    def read_bytes(self) -> bytes:
        """Download/read the S3 file as bytes."""
        return self._download_as_bytes()

    def write_bytes(self, data: bytes) -> None:
        """Write bytes to a file on S3."""
        client = self._get_client()
        client.put_object(Bucket=self.bucket, Key=self.key, Body=data)

    def _download_as_bytes(self) -> bytes:
        """Download file to a `BytesIO` buffer to read as bytes."""
        from io import BytesIO

        client = self._get_client()

        bio = BytesIO()
        client.download_fileobj(self.bucket, self.key, bio)
        bio.seek(0)
        out = bio.read()
        bio.close()
        return out

    def write_text(self, data: str) -> None:
        """Write a string to a file on S3."""
        return self.write_bytes(data.encode("utf-8"))

    def __truediv__(self, other):
        new = copy.deepcopy(self)
        new.path = self.path / other
        new._trailing_slash = "/" if str(other).endswith("/") else ""
        return new

    def __repr__(self):
        return f'S3Path("{self.get_path()}")'

    def __str__(self):
        return self.get_path()

    def glob(self, pattern):
        """Perform a glob-style search for S3 objects.

        Parameters
        ----------
        pattern : str
            The glob pattern to match against S3 object keys.

        Returns
        -------
        list
            A list of S3 objects matching the given pattern.

        """
        full_pattern = str(self) + pattern
        logger.debug(f"Searching {full_pattern}")
        return list_bucket(full_bucket_glob=full_pattern)


def list_bucket(
    bucket: str | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    full_bucket_glob: Optional[str] = None,
    aws_profile: str | None = None,
    num_workers: int = 10,
) -> list[str]:
    """Use `s5cmd` to quickly list items in a bucket.

    Parameters
    ----------
    bucket : str
        Name of the bucket.
    prefix : str, optional
        Prefix to filter by, by default "".
    suffix : str, optional
        Suffix to filter by, by default "".
    full_bucket_glob : str, optional
        Alternate to prefix/suffix. Full glob to filter by, by default None.
    aws_profile : str, optional
        AWS profile to use, by default None.
    num_workers : int, default = 10
        Number of workers to use for parallel downloads.

    Returns
    -------
    list[str]
        list of items in the bucket.

    """
    import json
    import subprocess

    cmd = ["s5cmd", "--json", "--numworkers", str(num_workers)]
    if aws_profile:
        cmd += ["--profile", aws_profile]

    if full_bucket_glob:
        bucket_str = full_bucket_glob
    else:
        bucket_str = f"s3://{bucket}"
        if prefix:
            bucket_str += f"/{prefix}"
        if suffix:
            bucket_str += f"*{suffix}"
    cmd += ["ls", bucket_str.strip()]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=90)
    except subprocess.CalledProcessError as e:
        try:
            # no matching files found. structured error is returned:
            s5cmd_error = json.loads(e.stderr)
            if "no object found" in s5cmd_error["error"]:
                return []
        except json.JSONDecodeError:
            pass
        if "ExpiredToken" in e.stderr:
            raise CredentialsError(
                "Error downloading files: AWS credentials have expired."
            ) from e
        # Unknown error:
        raise RuntimeError(f"Error listing bucket {bucket_str}: {e.stderr}") from e

    out: list[str] = []
    for line in p.stdout.splitlines():
        item = json.loads(line)
        if item["type"] == "directory":
            continue
        out.append(item["key"])
    return out


class CredentialsError(Exception):
    """Raised when AWS credentials have expired."""


def get_earthaccess_s3_creds(
    dataset: str | ASFCredentialEndpoints = ASFCredentialEndpoints.NISAR,
) -> AWSCredentials:
    """Get S3 credentials for the specified dataset.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset to get credentials for.
        Options are "OPERA", "SENTINEL1", or "NISAR".

    Returns
    -------
    AWSCredentials
        Object containing S3 credentials

    Raises
    ------
    ValueError
        If an unknown dataset is specified.

    Notes
    -----
    Uses the `earthaccess` library to login, which requires one of the following
    auth strategies:
        - "all": (default) try all methods until one works
        - "interactive": enter username and password.
        - "netrc": retrieve username and password from ~/.netrc.
        - "environment": retrieve username and password from
            `$EARTHDATA_USERNAME` and `$EARTHDATA_PASSWORD`.

    """
    if isinstance(dataset, str):
        endpoint: ASFCredentialEndpoints = getattr(
            ASFCredentialEndpoints, dataset.upper()
        )
    else:
        endpoint = dataset
    return AWSCredentials.from_asf(endpoint=endpoint)


def get_authorized_s3_client(
    dataset: str | ASFCredentialEndpoints = ASFCredentialEndpoints.NISAR,
    aws_credentials: AWSCredentials | None = None,
):
    """Get an authorized S3 client for the specified dataset.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset to get credentials for. Default is "opera".
    aws_credentials : AWSCredentials, optional
        Pre-configured s3 credentials.
        If not provided, fetches using earthaccess

    Returns
    -------
    boto3.S3Client
        An authorized S3 client.

    """
    import boto3

    # from botocore.config import Config

    if aws_credentials is None:
        aws_credentials = get_earthaccess_s3_creds(dataset=dataset)

    # return client
    return boto3.client(
        "s3",
        aws_access_key_id=aws_credentials.access_key_id,
        aws_secret_access_key=aws_credentials.secret_access_key,
        aws_session_token=aws_credentials.session_token,
        region_name="us-west-2",
    )


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


def open_h5_file(file_path: str | Path, mode: str = "r"):
    """Open an HDF5 file, supporting both local and remote paths.

    For remote files, uses h5netcdf with earthaccess for better compatibility.
    For local files, uses h5py (faster).

    Parameters
    ----------
    file_path : str | Path
        Path to the HDF5 file (local path, https://, or s3://).
    mode : str, optional
        File open mode, by default "r".

    Returns
    -------
    h5py.File or h5netcdf.legacyapi.File
        Opened HDF5 file object. Both have compatible interfaces.

    Examples
    --------
    >>> with open_h5_file("/local/file.h5") as f:  # doctest: +SKIP
    ...     data = f['/dataset'][:]

    >>> with open_h5_file("s3://bucket/file.h5") as f:  # doctest: +SKIP
    ...     data = f['/dataset'][:]

    """
    if is_remote_url(file_path):
        # Use h5netcdf + earthaccess for remote files
        if not HAS_EARTHACCESS:
            msg = (
                "earthaccess and h5netcdf are required for remote files. "
                "Install with: pip install earthaccess h5netcdf"
            )
            raise ImportError(msg)

        import h5netcdf

        # Authenticate with earthaccess
        authenticate_earthdata()

        # Get fsspec file object from earthaccess
        file_objs = earthaccess.open([str(file_path)])
        if not file_objs:
            raise FileNotFoundError(f"Could not open remote file: {file_path}")

        file_obj = file_objs[0]

        # Open with h5netcdf (compatible with h5py interface)
        return h5netcdf.File(file_obj, mode, invalid_netcdf=True)
    else:
        # Use h5py for local files (faster)
        return h5py.File(file_path, mode)


def open_xarray_group(
    file_path: str | Path,
    group: str,
    phony_dims: str = "access",
) -> "xr.Dataset":
    """Open a specific group from an HDF5 file using xarray.

    This is useful for reading metadata and coordinates from remote files.

    Parameters
    ----------
    file_path : str | Path
        Path to the HDF5 file (local or remote).
    group : str
        HDF5 group path (e.g., "/science/LSAR/GSLC/grids/frequencyA").
    phony_dims : str, optional
        How to handle dimensions, by default "access".

    Returns
    -------
    xr.Dataset
        Xarray dataset for the group.

    Examples
    --------
    >>> # Read metadata from remote file
    >>> ds = open_xarray_group(
    ...     "s3://bucket/file.h5",
    ...     "/science/LSAR/GSLC/grids/frequencyA"
    ... )  # doctest: +SKIP
    >>> print(ds.coords)  # Shows row, col, x, y coordinates  # doctest: +SKIP

    """
    if not HAS_EARTHACCESS:
        msg = (
            "earthaccess and xarray required. Install with: pip install earthaccess"
            " xarray h5netcdf"
        )
        raise ImportError(msg)

    import xarray as xr

    if is_remote_url(file_path):
        # Authenticate and open remote file
        authenticate_earthdata()
        file_objs = earthaccess.open([str(file_path)])
        if not file_objs:
            raise FileNotFoundError(f"Could not open: {file_path}")
        file_obj = file_objs[0]
    else:
        file_obj = str(file_path)

    # Open with xarray
    return xr.open_dataset(
        file_obj,
        group=group,
        engine="h5netcdf",
        phony_dims=phony_dims,
    )


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
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={threads_per_worker} "
            "inter_op_parallelism_threads=1"
        )

        # NumPy/BLAS threading (used by JAX backend)
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)

        # Prevent JAX from pre-allocating all GPU memory
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

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
                    f"Using existing dask client with {len(client.cluster.workers)}"
                    " workers"
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

            dask.config.set(scheduler="threads", num_workers=n_workers)

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
        if hasattr(result, "compute"):
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
        >>> result = reader.map_overlap(
        ...     process_block, depth={0: 0, 1: 200, 2: 0}
        ... )  # doctest: +SKIP

        """
        import dask.array as da

        if depth is None:
            if self.overlap is None:
                raise ValueError(
                    "No overlap specified. Provide depth or set overlap in __init__"
                )
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
        if hasattr(self, "ds"):
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

    def __init__(self, file_list: list[str | Path], authenticate: bool = True) -> None:
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

    def __enter__(self) -> Self:
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

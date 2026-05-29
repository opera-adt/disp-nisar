"""Utilities for GDAL to access remote files via virtual file system."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def to_gdal_path(file_path: str | Path) -> str:
    """Convert a file path to GDAL-compatible format, handling remote URLs.

    GDAL can access remote files using virtual file systems:
    - /vsicurl/ for HTTP(S) URLs
    - /vsis3/ for S3 URLs (requires AWS credentials)

    Parameters
    ----------
    file_path : str | Path
        File path (local or remote URL).

    Returns
    -------
    str
        GDAL-compatible path string.

    Examples
    --------
    >>> to_gdal_path("/local/path/file.h5")
    '/local/path/file.h5'
    >>> to_gdal_path("https://example.com/file.h5")
    '/vsicurl/https://example.com/file.h5'
    >>> to_gdal_path("s3://bucket/path/file.h5")
    '/vsis3/bucket/path/file.h5'

    """
    path_str = str(file_path)
    parsed = urlparse(path_str)

    if parsed.scheme == "":
        # Local file
        return path_str

    elif parsed.scheme in ("http", "https"):
        # HTTP(S) URL - use /vsicurl/
        return f"/vsicurl/{path_str}"

    elif parsed.scheme == "s3":
        # S3 URL - use /vsis3/
        # Convert s3://bucket/path to /vsis3/bucket/path
        s3_path = f"{parsed.netloc}{parsed.path}"
        return f"/vsis3/{s3_path}"

    else:
        # Unknown scheme, return as-is
        logger.warning(f"Unknown URL scheme '{parsed.scheme}' for {path_str}")
        return path_str


def to_gdal_netcdf_path(file_path: str | Path, dataset: str) -> str:
    """Convert file path and dataset to GDAL NETCDF format.

    Parameters
    ----------
    file_path : str | Path
        HDF5/NetCDF file path (local or remote).
    dataset : str
        Dataset path within the file.

    Returns
    -------
    str
        GDAL NETCDF path: "NETCDF:file:dataset"

    Examples
    --------
    >>> to_gdal_netcdf_path("/local/file.h5", "/data/HH")
    'NETCDF:/local/file.h5:/data/HH'
    >>> to_gdal_netcdf_path("https://example.com/file.h5", "/data/HH")
    'NETCDF:/vsicurl/https://example.com/file.h5:/data/HH'
    >>> to_gdal_netcdf_path("s3://bucket/file.h5", "/data/HH")
    'NETCDF:/vsis3/bucket/file.h5:/data/HH'

    """
    gdal_path = to_gdal_path(file_path)
    return f"NETCDF:{gdal_path}:{dataset}"


def configure_gdal_for_remote(
    max_retry: int = 3,
    timeout: int = 60,
    chunk_size: int = 4 * 1024 * 1024,
) -> None:
    """Configure GDAL for optimal remote file access.

    Parameters
    ----------
    max_retry : int, optional
        Maximum number of retries for failed requests, by default 3.
    timeout : int, optional
        Timeout in seconds for HTTP requests, by default 60.
    chunk_size : int, optional
        Chunk size for HTTP range requests in bytes, by default 4MB.

    """
    from osgeo import gdal

    # Enable GDAL exceptions
    gdal.UseExceptions()

    # Configure HTTP/HTTPS access
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", str(max_retry))
    gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "1")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".h5,.hdf5,.nc,.tif")
    gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", str(timeout))

    # Enable caching
    gdal.SetConfigOption("VSI_CACHE", "YES")
    gdal.SetConfigOption("VSI_CACHE_SIZE", str(chunk_size))

    # For S3 access (requires AWS credentials in environment)
    # Set these if you have AWS credentials:
    # gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")  # For public buckets
    # Or configure credentials via AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    logger.info("GDAL configured for remote file access")
    logger.debug(f"  Max retry: {max_retry}")
    logger.debug(f"  Timeout: {timeout}s")
    logger.debug(f"  Chunk size: {chunk_size / 1024 / 1024:.1f} MB")

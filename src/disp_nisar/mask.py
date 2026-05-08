"""Public API for water mask creation — intended for notebook use.

Typical usage::

    from disp_nisar.mask import create_water_mask, load_mask

    # From a GSLC file (downloads from S3, writes binary TIF)
    create_water_mask("NISAR_L2_GSLC_....h5", output="water_mask.tif")

    # From an explicit lon/lat bounding box (WSEN)
    create_water_mask(bbox=(-120.5, 34.0, -118.0, 36.0), output="water_mask.tif")

    # Read the result back as a numpy array
    mask = load_mask("water_mask.tif")   # True = land, False = water
"""

import logging
from pathlib import Path

import numpy as np
from dolphin.io import load_gdal

from disp_nisar._masking import convert_distance_to_binary
from disp_nisar._water import (
    check_dateline,
    create_mask_from_distance,
    download_map,
    polygon_from_bounding_box,
    set_aws_env_from_saml,
    warp_mask_to_gslc_grid,
)
from disp_nisar._water import (
    create_water_mask as _create_water_mask,
)

__all__ = [
    "create_water_mask",
    "download_water_distance",
    "load_mask",
    "warp_mask_to_gslc_grid",
    "convert_distance_to_binary",
    "create_mask_from_distance",
]

logger = logging.getLogger(__name__)


def create_water_mask(
    gslc_file: str | Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    output: str | Path = "water_mask.tif",
    margin: int = 5,
    land_buffer: int = 1,
    ocean_buffer: int = 1,
    aws_profile: str = "saml-pub",
    aws_region: str = "us-west-2",
    verbose: bool = True,
) -> Path:
    """Download and create a binary water mask.

    Parameters
    ----------
    gslc_file : str or Path, optional
        NISAR GSLC HDF5 file; spatial extent is read from frequencyA grid.
    bbox : tuple[float, float, float, float], optional
        Bounding box as (West, South, East, North) in decimal degrees.
    output : str or Path, optional
        Output binary TIF path. Default: ``"water_mask.tif"``.
    margin : int, optional
        Margin in km to add around the extent. Default: 5.
    land_buffer : int, optional
        km buffer that shrinks inland-water masking (keeps more land pixels).
        Default: 1.
    ocean_buffer : int, optional
        km buffer that shrinks ocean masking (keeps more coastal pixels).
        Default: 1.
    aws_profile : str, optional
        AWS profile for S3 authentication. Default: ``"saml-pub"``.
    aws_region : str, optional
        AWS region. Default: ``"us-west-2"``.
    verbose : bool, optional
        Enable INFO logging to stdout. Default: True.

    Returns
    -------
    Path
        Path to the created binary mask TIF.

    Raises
    ------
    ValueError
        If neither gslc_file nor bbox is provided.

    Examples
    --------
    >>> from disp_nisar.mask import create_water_mask
    >>> mask_path = create_water_mask("NISAR_L2_GSLC_....h5")
    >>> mask_path = create_water_mask(bbox=(-120.5, 34.0, -118.0, 36.0))

    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if gslc_file is None and bbox is None:
        raise ValueError("Provide gslc_file or bbox")

    output = Path(output)
    _create_water_mask(
        gslc_file=Path(gslc_file) if gslc_file is not None else None,
        bbox=bbox,
        output=output,
        margin=margin,
        land_buffer=land_buffer,
        ocean_buffer=ocean_buffer,
        aws_profile=aws_profile,
        aws_region=aws_region,
    )
    return output


def download_water_distance(
    gslc_file: str | Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    output: str | Path = "water_distance.vrt",
    margin: int = 5,
    aws_profile: str = "saml-pub",
    aws_region: str = "us-west-2",
    verbose: bool = True,
) -> Path:
    """Download the raw water distance VRT for use as PGE runconfig ``mask_file``.

    The PGE (``main.py``) expects the raw uint8 distance raster as
    ``dynamic_ancillary_file_group.mask_file`` and handles the binary
    conversion internally (with land/ocean buffers).  Pass the path returned
    by this function directly to the runconfig — do NOT pass the binary TIF
    produced by :func:`create_water_mask`.

    Encoding: ``0`` = land, ``1–99`` = ocean (km to shore),
    ``100–200`` = inland water (km to land), ``255`` = nodata.

    Parameters
    ----------
    gslc_file : str or Path, optional
        NISAR GSLC HDF5 file; spatial extent read from frequencyA grid.
    bbox : tuple[float, float, float, float], optional
        Bounding box as (West, South, East, North) in decimal degrees.
    output : str or Path, optional
        Output VRT path. Default: ``"water_distance.vrt"``.
    margin : int, optional
        Margin in km to add around the extent. Default: 5.
    aws_profile : str, optional
        AWS profile for S3 authentication. Default: ``"saml-pub"``.
    aws_region : str, optional
        AWS region. Default: ``"us-west-2"``.
    verbose : bool, optional
        Enable INFO logging to stdout. Default: True.

    Returns
    -------
    Path
        Path to the downloaded VRT — set this as ``mask_file`` in the runconfig.

    Examples
    --------
    >>> from disp_nisar.mask import download_water_distance
    >>> vrt = download_water_distance(gslc_file="NISAR_L2_GSLC_....h5")
    >>> # In runconfig:
    >>> pge_runconfig.dynamic_ancillary_file_group.mask_file = vrt

    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if gslc_file is None and bbox is None:
        raise ValueError("Provide gslc_file or bbox")

    set_aws_env_from_saml(profile_name=aws_profile, region=aws_region)

    if gslc_file is not None:
        from opera_utils.nisar._info import get_nisar_bbox

        bbox_obj = get_nisar_bbox(Path(gslc_file))
        bbox = (bbox_obj.left, bbox_obj.bottom, bbox_obj.right, bbox_obj.top)
        logger.info(f"GSLC extent (WSEN): {bbox}")

    assert bbox is not None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    polys = check_dateline(polygon_from_bounding_box(bbox, margin))
    download_map(polys, output)
    logger.info(f"Water distance VRT ready for PGE runconfig: {output}")
    return output


def load_mask(mask_file: str | Path) -> np.ndarray:
    """Load a binary water mask as a boolean numpy array.

    Parameters
    ----------
    mask_file : str or Path
        Path to a binary water mask TIF (1 = land, 0 = water).

    Returns
    -------
    np.ndarray
        Boolean array where ``True`` = land (valid), ``False`` = water (masked).

    Examples
    --------
    >>> from disp_nisar.mask import load_mask
    >>> mask = load_mask("water_mask.tif")
    >>> print(f"Land fraction: {mask.mean():.1%}")

    """
    data = load_gdal(mask_file)
    # nodata (255) and 0 are both treated as masked (water/invalid)
    return (data == 1).astype(bool)

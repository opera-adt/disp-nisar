import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import rioxarray  # noqa: F401  (registers the `.rio` accessor)
import xarray as xr
from dolphin._types import Filename, PathOrStr
from dolphin.io import format_nc_filename, load_gdal, write_arr

# from opera_utils import group_by_burst
from scipy import ndimage

from ._common import GSLC_MASK_DSET

logger = logging.getLogger(__name__)

# Value used in a GSLC `mask` dataset for pixels outside the radar extent (fill)
GSLC_MASK_FILL = 255


def create_mask_from_distance(
    water_distance_file: PathOrStr,
    output_file: PathOrStr,
    land_buffer: int = 0,
    ocean_buffer: int = 0,
) -> None:
    """Create a binary mask from the NISAR water distance mask with buffer zones.

    This function reads a water distance file, converts it to a binary mask
    with consideration for buffer zones, and writes the result to a new file.

    Parameters
    ----------
    water_distance_file : PathOrStr
        Path to the input water distance file.
    output_file : PathOrStr
        Path to save the output binary mask file.
    land_buffer : int, optional
        Buffer distance (in km) for land pixels. Only pixels this far or farther
        from water will be considered land. Default is 0.
    ocean_buffer : int, optional
        Buffer distance (in km) for ocean pixels. Only pixels this far or farther
        from land will be considered water. Default is 0.

    Notes
    -----
    Format of `water_distance_file` is UInt8, where:
    - 0 means "land"
    - 1 - 99 are ocean water pixels. The value is the distance (in km) to the shore.
      Value is rounded up to the nearest integer.
    - 100 - 200 are inland water pixels. Value is the distance to land.

    Output is a mask where 0 represents water pixels ("bad" pixels to ignore during
    processing/unwrapping), and 1 are land pixels to use.

    The buffer arguments make the masking more conservative. For example, a land_buffer
    of 2 means only pixels 2 km or farther from water will be masked as land. This helps
    account for potential changes in water levels.

    """
    # Load the water distance data
    water_distance_data = load_gdal(water_distance_file, masked=True)

    binary_mask = convert_distance_to_binary(
        water_distance_data, land_buffer, ocean_buffer
    )

    write_arr(
        arr=binary_mask.astype(np.uint8),
        output_name=output_file,
        like_filename=water_distance_file,
        dtype="uint8",
        nodata=255,
    )


def convert_distance_to_binary(
    water_distance_data: np.ma.MaskedArray, land_buffer: int = 0, ocean_buffer: int = 0
) -> np.ndarray:
    """Convert water distance data to a binary mask considering buffer zones.

    Parameters
    ----------
    water_distance_data : np.ma.MaskedArray
        Input water distance data as a masked array.
    land_buffer : int, optional
        Buffer distance (in km) for land pixels. Only pixels this far or farther
        from water will be considered land. Default is 0.
    ocean_buffer : int, optional
        Buffer distance (in km) for ocean pixels. Only pixels this far or farther
        from land will be considered water. Default is 0.

    Returns
    -------
    np.ndarray
        Binary mask where True represents land pixels and False represents water pixels.

    Notes
    -----
    The function applies the following logic:
    - Starts with all pixels as land (True).
    - Masks inland water pixels as water (False) if they are farther from land
        than the land_buffer.
    - Masks ocean pixels as water (False) if they are farther from shore than
        `ocean_buffer`.

    """
    # Create the binary mask with buffer considerations. Start all on (assume all land)
    binary_mask = np.ma.MaskedArray(
        np.ones_like(water_distance_data, dtype=bool), mask=water_distance_data.mask
    )

    # Mask inland water pixels (considering land buffer): anything 101 or higher is land
    inland_water_mask = water_distance_data > land_buffer + 100
    binary_mask[inland_water_mask] = False
    # For ocean, only look at values 1-100, then consider buffer
    ocean_water_mask = (water_distance_data <= 100) & (
        water_distance_data > ocean_buffer
    )
    binary_mask[ocean_water_mask] = False
    # Erode away small single-pixels
    closed_mask = ndimage.binary_closing(
        binary_mask.filled(0), structure=np.ones((3, 3)), border_value=1
    )
    return closed_mask


# TODO: No layover/shadow mask in nisar static layers
# modify or remove this function
def create_layover_shadow_masks(
    cslc_static_files: Sequence[Filename],
    output_dir: Filename,
) -> list[Path]:
    """Create binary masks from the layover shadow CSLC static files.

    In the outputs, 0 indicates a bad masked pixel, 1 is a good pixel.

    Parameters
    ----------
    cslc_static_files : Sequence[Filename]
        List of CSLC static layer files to process
    output_dir : Filename
        Directory where output masks will be saved

    Returns
    -------
    list[Path]
        List of paths to the created binary layover shadow mask files

    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_files = []

    if len(cslc_static_files) > 1:
        logger.warning(f"Found multiple static files: {cslc_static_files}")
    f = cslc_static_files[0]
    input_name = format_nc_filename(f, ds_name="/data/layover_shadow_mask")
    out_file = output_path / "layover_shadow.tif"
    if out_file.exists():
        output_files.append(out_file)

    logger.info(f"Extracting layover shadow mask from {f} to {out_file}")
    layover_data = load_gdal(input_name)
    # we'll ignore the nodata region to be conservative
    layover_data[layover_data == 127] = 0
    not_layover_pixels = layover_data == 0
    write_arr(
        arr=not_layover_pixels,
        output_name=out_file,
        like_filename=input_name,
        nodata=127,
    )

    output_files.append(out_file)

    return output_files


def read_gslc_validity_mask(
    gslc_file: PathOrStr,
    frequency: str = "frequencyA",
    strides: tuple[int, int] = (1, 1),
) -> xr.DataArray:
    """Read the validity mask of a GSLC (or a compressed SLC) on its native grid.

    Compressed SLCs store a `mask` dataset in the same place and encoding as a GSLC,
    so the same reader handles both.

    In the `mask` dataset, valid pixels have a subswath number >= 1, invalid pixels
    are 0, and 255 is fill (outside the radar extent).

    Parameters
    ----------
    gslc_file : PathOrStr
        Path to the GSLC (or compressed SLC) HDF5 file. Can be remote.
    frequency : str
        Frequency layer to read, e.g. "frequencyA".
    strides : tuple[int, int]
        (row, column) strides applied while reading.

    Returns
    -------
    xr.DataArray
        uint8 array with 1 where the pixel is valid and 0 elsewhere, with CRS and
        coordinates attached.

    """
    from ._streaming import open_h5_file

    sy, sx = strides
    with open_h5_file(gslc_file, "r") as f:
        grp = f[f"science/LSAR/GSLC/grids/{frequency}"]
        raw = grp[GSLC_MASK_DSET][::sy, ::sx]
        x = grp["xCoordinates"][::sx]
        y = grp["yCoordinates"][::sy]
        epsg = int(grp["projection"][()])

    valid = (raw >= 1) & (raw != GSLC_MASK_FILL)
    return xr.DataArray(
        valid.astype(np.uint8),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs={"long_name": "valid sample mask"},
    ).rio.write_crs(f"EPSG:{epsg}")


def _mask_values_on_grid(mask: xr.DataArray, ref: xr.DataArray) -> np.ndarray:
    """Get the values of `mask` on the grid of `ref`.

    Pixels of `ref` that `mask` does not cover are returned as valid (1), so an input
    with a smaller footprint (e.g. cropped to the output bounds) does not mask the
    rest of the frame.
    """
    if (
        mask.shape == ref.shape
        and mask.rio.crs == ref.rio.crs
        and np.allclose(mask.x, ref.x, atol=1e-3)
        and np.allclose(mask.y, ref.y, atol=1e-3)
    ):
        return mask.values

    from rasterio.enums import Resampling

    matched = mask.rio.reproject_match(
        ref, resampling=Resampling.nearest, nodata=GSLC_MASK_FILL
    )
    return np.where(matched.values == GSLC_MASK_FILL, 1, matched.values).astype(
        np.uint8
    )


def combine_gslc_masks(
    gslc_files: Sequence[PathOrStr],
    frequency: str = "frequencyA",
    strides: tuple[int, int] = (1, 1),
) -> xr.DataArray:
    """Combine the validity masks of all inputs into one.

    The masks differ from one acquisition to the next, so a pixel is kept only if it
    is valid in every input. The result is on the grid of the first readable input;
    inputs on another grid are resampled to it (nearest neighbor).

    Inputs whose mask cannot be read (e.g. no `mask` dataset) are skipped with a
    warning.

    Parameters
    ----------
    gslc_files : Sequence[PathOrStr]
        GSLCs and/or compressed SLCs. Put a real GSLC first, since its grid is used
        as the reference.
    frequency : str
        Frequency layer to read, e.g. "frequencyA".
    strides : tuple[int, int]
        (row, column) strides applied while reading.

    Returns
    -------
    xr.DataArray
        uint8 array, 1 where the pixel is valid in all inputs.

    Raises
    ------
    ValueError
        If none of the masks could be read.

    """
    combined: xr.DataArray | None = None
    num_used = 0
    for f in gslc_files:
        try:
            mask = read_gslc_validity_mask(f, frequency=frequency, strides=strides)
        except Exception as e:
            logger.warning(f"Could not read the mask of {f}: {e}; skipping it")
            continue
        if combined is None:
            combined = mask
        else:
            values = _mask_values_on_grid(mask, combined)
            combined = combined.copy(data=np.bitwise_and(combined.values, values))
        num_used += 1

    if combined is None:
        msg = f"No readable GSLC masks in {len(gslc_files)} input files"
        raise ValueError(msg)
    logger.info(f"Combined the GSLC masks of {num_used}/{len(gslc_files)} inputs")
    return combined


def mask_on_grid(
    mask: xr.DataArray, y: np.ndarray, x: np.ndarray, crs_wkt: str
) -> np.ndarray:
    """Get `mask` as a uint8 array on the grid defined by the `y`, `x` centers."""
    template = xr.DataArray(
        np.zeros((len(y), len(x)), dtype=np.uint8),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    ).rio.write_crs(crs_wkt)
    return _mask_values_on_grid(mask, template)

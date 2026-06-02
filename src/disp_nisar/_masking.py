import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from dolphin._types import Filename, PathOrStr
from dolphin.io import format_nc_filename, load_gdal, write_arr

# from opera_utils import group_by_burst
from scipy import ndimage

logger = logging.getLogger(__name__)


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
    """Convert water distance data to a binary land/water mask.

    Auto-detects the input convention. Two formats are supported:

    A. **OPERA distance** (UInt8): ``0`` = land, ``1–99`` = ocean (km to shore),
       ``100–200`` = inland water (km to land).
    B. **Distance-from-water** (UInt8): ``0`` = water, ``1+`` = land (km to
       water, often capped at 100). This is the convention used by some NISAR
       ancillary water masks.

    Detection heuristic: if more than ~15% of valid pixels are ≥100 (the
    "capped" land tail in convention B), input is treated as convention B;
    otherwise OPERA convention is assumed.

    Parameters
    ----------
    water_distance_data : np.ma.MaskedArray
        Input water distance data as a masked array.
    land_buffer : int, optional
        Buffer (in km) for land. Pixels closer than ``land_buffer`` to water
        are treated as water. Default 0.
    ocean_buffer : int, optional
        Buffer (in km) for ocean (OPERA convention only). Pixels closer than
        ``ocean_buffer`` km to shore are treated as land. Default 0.

    Returns
    -------
    np.ndarray
        Binary mask where ``True`` (1) = land/valid and ``False`` (0) = water.

    """
    valid_values = water_distance_data.compressed()
    if valid_values.size == 0:
        return np.zeros(water_distance_data.shape, dtype=bool)

    pct_high = float((valid_values >= 100).mean())
    convention_b = pct_high > 0.15

    if convention_b:
        # 0 = water; positive = distance-from-water (land).
        # Treat pixels closer than land_buffer to water as water.
        is_land = water_distance_data > max(land_buffer, 0)
        binary_mask = np.ma.MaskedArray(
            np.asarray(is_land, dtype=bool),
            mask=water_distance_data.mask,
        )
    else:
        # OPERA convention: 0 = land, 1-99 = ocean, 100-200 = inland water.
        binary_mask = np.ma.MaskedArray(
            np.ones_like(water_distance_data, dtype=bool),
            mask=water_distance_data.mask,
        )
        # Inland water (101–200, considering land buffer)
        binary_mask[water_distance_data > land_buffer + 100] = False
        # Ocean water (1–100, considering ocean buffer)
        binary_mask[
            (water_distance_data <= 100) & (water_distance_data > ocean_buffer)
        ] = False

    # Close small holes inside land regions.
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

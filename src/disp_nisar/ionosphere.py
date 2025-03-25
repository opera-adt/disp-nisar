# Read Ionosphere correction layers from GUNW products

# It has to read them from interferograms, invert to timeseries
# if required for each date (or pair)

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_ionosphere_phase_screen(
    gunw_files: list[Path] | None, output_timeseries_files: list[Path] | None
) -> list[Path] | None:
    """Read ionosphere correction layers from GUNW products.

    Read ionosphere correction layers from interferograms and
    invert them to a time series for each date or pair.

    Parameters
    ----------
    gunw_files : List[Path]
        Provide a list of paths to GUNW products containing interferograms.
    output_timeseries_files : List[Path]
        Specify a list of paths to output timeseries for which ionosphere
        corrections are needed.

    Returns
    -------
    None
        This function currently serves as an interface/placeholder.

    """
    if gunw_files is not None and output_timeseries_files is not None:
        logger.info("Reading ionosphere phase screen from GUNW products...")
        logger.info(f"Number of input files: {len(gunw_files)}")
        logger.info(f"Number of output files: {len(output_timeseries_files)}")

        # Placeholder for future implementation
        # TODO: check for the design matrix of the gunw files network
        # to be full rank
        output_paths = None  # output_timeseries_files
        return output_paths
    else:
        return None

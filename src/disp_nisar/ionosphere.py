# Read Ionosphere correction layers from GUNW products

# It has to read them from interferograms, invert to timeseries
# if required for each date (or pair)

from __future__ import annotations

import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def read_ionosphere_phase_screen(
    gunw_files: list[Path], 
    output_timeseries_files: list[Path]
) -> None:
    """
    Placeholder function to read ionosphere correction layers from GUNW products.
    
    This function is intended to:
    - Read ionosphere correction layers from interferograms.
    - Invert to time series for each date or pair.
    
    Parameters:
    gunw_files : List[Path]
        List of paths to GUNW products containing interferograms.
    output_timeseries_files : List[Path]
        List of paths to output timeseries for which ionosphere corrections are needed.
    
    Returns:
    None
        This function currently serves as an interface/placeholder.
    """
    logger.info("Reading ionosphere phase screen from GUNW products...")
    logger.info(f"Number of input files: {len(gunw_files)}")
    logger.info(f"Number of output files: {len(output_timeseries_files)}")
    
    # Placeholder for future implementation
    pass

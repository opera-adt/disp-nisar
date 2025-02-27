from enum import Enum

__all__ = [
    "ProcessingMode",
]


class ProcessingMode(str, Enum):
    """DisplacementWorkflow processing modes for SDS operation."""

    FORWARD = "forward"
    """New data: only output one incremental result."""

    HISTORICAL = "historical"
    """Past stack of data: output multiple results."""

class ImagingFrequency(str, Enum):
    """The frequency of the imaging radar"""

    A = "frequencyA"
    B = "frequencyB"

class Polarization(str, Enum):
    """Polarization of the images"""
    HH = "HH"
    VV = "VV"
    HV = 'HV'
    VH = 'VH'

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
NISAR_GSLC_GRIDS_GROUP = "/science/LSAR/GSLC/grids"
NISAR_GSLC_PARAMETERS_GROUP = (
    "/science/LSAR/GSLC/metadata/sourceData/processingInformation/parameters"
)
DEFAULT_FREQUENCY = "frequencyA"
DEFAULT_POLARIZATION = "HH"


def gslc_dataset_name(
    frequency: str = DEFAULT_FREQUENCY, polarization: str = DEFAULT_POLARIZATION
) -> str:
    """HDF5 path of the SLC dataset of a GSLC (or compressed SLC).

    Parameters
    ----------
    frequency : str
        Frequency group, e.g. "frequencyA" or "frequencyB". Enums are accepted.
    polarization : str
        Polarization, e.g. "HH", "VV", "HV", "VH". Enums are accepted.

    """
    frequency = getattr(frequency, "value", frequency)
    polarization = getattr(polarization, "value", polarization)
    return f"{NISAR_GSLC_GRIDS_GROUP}/{frequency}/{polarization}"


# SLC dataset for the default frequency/polarization (a default; the workflow builds
# the real one from `input_file_group.frequency` and `.polarization`)
NISAR_DATASET_NAME = gslc_dataset_name()
NISAR_IDENTIFICATION_GROUP = "/science/LSAR/identification"
AMPLITUDE_DISPERSION_DSET = "amplitude_dispersion"
GSLC_MASK_DSET = "mask"
# Per-pixel flags for input data exceptions (e.g. QFSP); 0 marks an invalid pixel
GSLC_EXCEPTION_MASK_DSET = "inputDataExceptionMask"

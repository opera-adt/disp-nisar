"""CLI and function for generating a disp-nisar runconfig YAML."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

import click
from dolphin.stack import CompressedSlcPlan

from disp_nisar.pge_runconfig import (
    AlgorithmParameters,
    DynamicAncillaryFileGroup,
    InputFileGroup,
    PrimaryExecutable,
    ProductPathGroup,
    RunConfig,
    StaticAncillaryFileGroup,
)


def _parse_value(raw: str) -> Any:
    """Parse a CLI value string to a Python scalar.

    Tries JSON first (handles ``true``/``false``, numbers, lists, null),
    falls back to plain string.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _apply_dot_overrides(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge dot-notation key overrides into a nested dict in-place.

    Parameters
    ----------
    base : dict
        Nested dict (e.g. from ``model.model_dump()``).
    overrides : dict
        Flat mapping of ``"a.b.c"`` keys to already-parsed values.

    Returns
    -------
    dict
        The same ``base`` dict with overrides applied.

    """
    for dotkey, value in overrides.items():
        keys = dotkey.split(".")
        node = base
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise click.BadParameter(
                    f"'{dotkey}': '{k}' is not a valid nested key.",
                    param_hint="--set",
                )
            node = node[k]
        leaf = keys[-1]
        if leaf not in node:
            raise click.BadParameter(
                f"'{dotkey}': '{leaf}' is not a recognised parameter.",
                param_hint="--set",
            )
        node[leaf] = value
    return base


def generate_algorithm_parameters(
    output_dir: Path,
    mode: str = "forward",
    algorithm_parameters_file: Optional[Path] = None,
    # --- phase linking ---
    compressed_slc_plan: str = "last_per_ministack",
    write_crlb: bool = True,
    amp_dispersion_threshold: Optional[float] = None,
    half_window: Optional[tuple[int, int]] = None,
    # --- unwrapping ---
    run_interpolation: bool = True,
    n_parallel_jobs: int = 4,
    snaphu_ntiles: tuple[int, int] = (5, 5),
    snaphu_tile_overlap: tuple[int, int] = (500, 500),
    snaphu_single_tile_reoptimize: bool = True,
    interpolation_similarity_threshold: Optional[float] = None,
    interpolation_max_radius: Optional[int] = None,
    # --- output ---
    output_strides: Optional[tuple[int, int]] = None,
    epsg: Optional[int] = None,
    num_parallel_products: int = 3,
    forward_mode_network_size: int = 3,
    # --- arbitrary overrides ---
    set_params: Optional[dict[str, Any]] = None,
) -> Path:
    """Write an algorithm parameters YAML, optionally overriding any defaults.

    Parameters
    ----------
    output_dir : Path
        Root output directory.  The config subdirectory is created automatically.
    mode : str
        Processing mode: ``"forward"`` or ``"historical"``.  Used only to name
        the output file when ``algorithm_parameters_file`` is not given.
    algorithm_parameters_file : Path or None
        Explicit output path.  Defaults to
        ``{output_dir}/config/algorithm_parameters_{mode}_{date}.yaml``.
    compressed_slc_plan : str
        Compressed SLC plan for phase linking.  One of
        ``"always_first"``, ``"first_per_ministack"``, ``"last_per_ministack"``.
    write_crlb : bool
        Write Cramer-Rao Lower Bound rasters.
    amp_dispersion_threshold : float or None
        Amplitude dispersion threshold for PS detection.
        If None, uses dolphin default (``0.25``).
    half_window : (y, x) or None
        Phase-linking half-window size in pixels.
        If None, uses dolphin default (``y=7, x=14``).
    run_interpolation : bool
        Run interpolation step on wrapped interferogram before unwrapping.
    n_parallel_jobs : int
        Number of interferograms to unwrap in parallel.
    snaphu_ntiles : (rows, cols)
        Number of tiles for SNAPHU tiling.
    snaphu_tile_overlap : (rows, cols)
        Tile overlap in pixels for SNAPHU tiling.
    snaphu_single_tile_reoptimize : bool
        After multi-tile unwrapping, run a single-tile re-optimisation pass.
    interpolation_similarity_threshold : float or None
        Similarity threshold for the interpolation pre-processing step.
        Pixels below this value are interpolated before unwrapping.
        If None, uses dolphin default (``0.3``).
    interpolation_max_radius : int or None
        Maximum radius in pixels to search for valid neighbours during
        interpolation.  If None, uses dolphin default (``51``).
    output_strides : (y, x) or None
        Decimation strides for output products.  If None, uses dolphin default
        (``y=1, x=1``, i.e. no decimation).
    epsg : int or None
        Output projection EPSG code.  If None, uses the most common input CRS.
    num_parallel_products : int
        Number of output products to create in parallel.
    forward_mode_network_size : int
        Interferogram network size for forward mode (3 or 4).
    set_params : dict or None
        Arbitrary dot-notation overrides applied last, e.g.
        ``{"phase_linking.ministack_size": 10,
        "unwrap_options.unwrap_method": "spurt"}``.
        Values must already be parsed (use ``_parse_value`` for CLI strings).

    Returns
    -------
    Path
        Path to the written algorithm parameters YAML.

    """
    output_dir = Path(output_dir)
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    mode = mode.lower()
    if mode not in ("forward", "historical"):
        raise ValueError(f"mode must be 'forward' or 'historical', got '{mode}'")

    if algorithm_parameters_file is None:
        today = date.today().strftime("%Y%m%d")
        algorithm_parameters_file = (
            config_dir / f"algorithm_parameters_{mode}_{today}.yaml"
        )

    params = AlgorithmParameters()

    # Phase linking
    params.phase_linking.compressed_slc_plan = CompressedSlcPlan(compressed_slc_plan)
    params.phase_linking.write_crlb = write_crlb
    if amp_dispersion_threshold is not None:
        params.ps_options.amp_dispersion_threshold = amp_dispersion_threshold
    if half_window is not None:
        params.phase_linking.half_window.y = half_window[0]
        params.phase_linking.half_window.x = half_window[1]

    # Unwrapping
    params.unwrap_options.run_interpolation = run_interpolation
    params.unwrap_options.n_parallel_jobs = n_parallel_jobs
    params.unwrap_options.snaphu_options.ntiles = list(snaphu_ntiles)
    params.unwrap_options.snaphu_options.tile_overlap = list(snaphu_tile_overlap)
    params.unwrap_options.snaphu_options.single_tile_reoptimize = (
        snaphu_single_tile_reoptimize
    )
    if interpolation_similarity_threshold is not None:
        params.unwrap_options.preprocess_options.interpolation_similarity_threshold = (
            interpolation_similarity_threshold
        )
    if interpolation_max_radius is not None:
        params.unwrap_options.preprocess_options.max_radius = interpolation_max_radius

    # Output
    if output_strides is not None:
        params.output_options.strides.y = output_strides[0]
        params.output_options.strides.x = output_strides[1]
    if epsg is not None:
        params.output_options.epsg = epsg
    params.num_parallel_products = num_parallel_products
    params.forward_mode_network_size = forward_mode_network_size

    # Arbitrary dot-notation overrides (applied last)
    if set_params:
        base = params.model_dump()
        _apply_dot_overrides(base, set_params)
        params = AlgorithmParameters.model_validate(base)

    params.to_yaml(algorithm_parameters_file)
    return Path(algorithm_parameters_file)


@click.command(name="make-algorithm-parameters")
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Root output directory (config/ subdirectory created automatically).",
)
@click.option(
    "--mode",
    "-m",
    default="forward",
    type=click.Choice(["forward", "historical"], case_sensitive=False),
    show_default=True,
    help="Processing mode (used in output filename).",
)
@click.option(
    "--outfile",
    type=click.Path(),
    default=None,
    help=(
        "Output path (default:"
        " {output_dir}/config/algorithm_parameters_{mode}_{date}.yaml)."
    ),
)
# Phase linking
@click.option(
    "--compressed-slc-plan",
    default="last_per_ministack",
    show_default=True,
    type=click.Choice(
        ["always_first", "first_per_ministack", "last_per_ministack"],
        case_sensitive=False,
    ),
    help="Compressed SLC plan for phase linking.",
)
@click.option(
    "--write-crlb/--no-write-crlb",
    default=True,
    show_default=True,
    help="Write Cramer-Rao Lower Bound rasters.",
)
@click.option(
    "--amp-dispersion-threshold",
    default=None,
    type=float,
    help="PS amplitude dispersion threshold (dolphin default: 0.25).",
)
@click.option(
    "--half-window",
    nargs=2,
    type=int,
    default=None,
    metavar="Y X",
    help="Phase-linking half-window size in pixels (dolphin default: 7 14).",
)
# Unwrapping
@click.option(
    "--run-interpolation/--no-run-interpolation",
    default=True,
    show_default=True,
    help="Run interpolation step on wrapped interferogram before unwrapping.",
)
@click.option(
    "--n-parallel-jobs",
    default=4,
    show_default=True,
    type=int,
    help="Number of interferograms to unwrap in parallel.",
)
@click.option(
    "--snaphu-ntiles",
    nargs=2,
    type=int,
    default=(5, 5),
    show_default=True,
    metavar="ROWS COLS",
    help="SNAPHU tiling: number of tiles (rows cols).",
)
@click.option(
    "--snaphu-tile-overlap",
    nargs=2,
    type=int,
    default=(500, 500),
    show_default=True,
    metavar="ROWS COLS",
    help="SNAPHU tiling: tile overlap in pixels (rows cols).",
)
@click.option(
    "--snaphu-single-tile-reoptimize/--no-snaphu-single-tile-reoptimize",
    default=True,
    show_default=True,
    help="Run single-tile re-optimisation pass after multi-tile unwrapping.",
)
@click.option(
    "--interpolation-similarity-threshold",
    default=None,
    type=float,
    help="Similarity threshold for pre-unwrap interpolation (dolphin default: 0.3).",
)
@click.option(
    "--interpolation-max-radius",
    default=None,
    type=int,
    help=(
        "Max search radius in pixels for pre-unwrap interpolation (dolphin default:"
        " 51)."
    ),
)
# Output
@click.option(
    "--output-strides",
    nargs=2,
    type=int,
    default=None,
    metavar="Y X",
    help="Output decimation strides (dolphin default: 1 1, i.e. no decimation).",
)
@click.option(
    "--epsg",
    default=None,
    type=int,
    help="Output projection EPSG code (default: most common input CRS).",
)
@click.option(
    "--num-parallel-products",
    default=3,
    show_default=True,
    type=int,
    help="Number of output products to create in parallel.",
)
@click.option(
    "--forward-mode-network-size",
    default=3,
    show_default=True,
    type=click.Choice(["3", "4"]),
    help="Interferogram network size for forward mode.",
)
# Generic override + schema
@click.option(
    "--set",
    "set_params",
    metavar="KEY=VALUE",
    multiple=True,
    help=(
        "Override any parameter with dot-notation (applied last). "
        "Values parsed as JSON. Repeat for multiple. "
        "E.g. --set phase_linking.ministack_size=10"
    ),
)
@click.option(
    "--print-defaults",
    is_flag=True,
    default=False,
    help="Print full YAML schema with all keys and defaults, then exit.",
)
def make_algorithm_parameters_cli(
    output_dir,
    mode,
    outfile,
    compressed_slc_plan,
    write_crlb,
    amp_dispersion_threshold,
    half_window,
    run_interpolation,
    n_parallel_jobs,
    snaphu_ntiles,
    snaphu_tile_overlap,
    snaphu_single_tile_reoptimize,
    interpolation_similarity_threshold,
    interpolation_max_radius,
    output_strides,
    epsg,
    num_parallel_products,
    forward_mode_network_size,
    set_params,
    print_defaults,
):
    r"""Write an algorithm parameters YAML, optionally overriding any parameter.

    Use ``--print-defaults`` to see every available key with its type and
    default value.  Use ``--set KEY=VALUE`` with dot-notation to reach any
    nested parameter not covered by the named options above.

    \b
    Show all available parameters:
      disp-nisar make-algorithm-parameters -o ./work --print-defaults

    \b
    Example — historical run with custom tiling:
      disp-nisar make-algorithm-parameters -o ./work --mode historical \\
          --snaphu-ntiles 4 4 --snaphu-tile-overlap 200 200 \\
          --n-parallel-jobs 14

    \b
    Arbitrary overrides with --set:
      --set phase_linking.ministack_size=10
      --set unwrap_options.unwrap_method=spurt
      --set timeseries_options.correlation_threshold=0.3
    """
    if print_defaults:
        click.echo("# All available algorithm parameters (defaults shown):\n")
        AlgorithmParameters.print_yaml_schema()
        return

    parsed_set: dict[str, Any] = {}
    for item in set_params:
        if "=" not in item:
            raise click.BadParameter(
                f"'{item}' is not in KEY=VALUE format.", param_hint="--set"
            )
        key, _, raw = item.partition("=")
        parsed_set[key.strip()] = _parse_value(raw.strip())

    alg_params_path = generate_algorithm_parameters(
        output_dir=Path(output_dir),
        mode=mode,
        algorithm_parameters_file=Path(outfile) if outfile else None,
        compressed_slc_plan=compressed_slc_plan,
        write_crlb=write_crlb,
        amp_dispersion_threshold=amp_dispersion_threshold,
        half_window=tuple(half_window) if half_window else None,
        run_interpolation=run_interpolation,
        n_parallel_jobs=n_parallel_jobs,
        snaphu_ntiles=tuple(snaphu_ntiles),
        snaphu_tile_overlap=tuple(snaphu_tile_overlap),
        snaphu_single_tile_reoptimize=snaphu_single_tile_reoptimize,
        interpolation_similarity_threshold=interpolation_similarity_threshold,
        interpolation_max_radius=interpolation_max_radius,
        output_strides=tuple(output_strides) if output_strides else None,
        epsg=epsg,
        num_parallel_products=num_parallel_products,
        forward_mode_network_size=int(forward_mode_network_size),
        set_params=parsed_set or None,
    )
    click.echo(f"Algorithm parameters: {alg_params_path}")
    click.echo("\nReference in runconfig with:")
    click.echo(f"  --algorithm-parameters-file {alg_params_path}")


def generate_runconfig(
    gslc_files: list[Path],
    frame_id: int,
    output_dir: Path,
    mode: str = "forward",
    frequency: str = "frequencyA",
    polarization: str = "HH",
    gunw_files: Optional[list[Path]] = None,
    mask_file: Optional[Path] = None,
    dem_file: Optional[Path] = None,
    troposphere_files: Optional[list[Path]] = None,
    scratch_dir: Optional[Path] = None,
    algorithm_parameters_file: Optional[Path] = None,
    frame_to_bounds_json: Optional[Path] = None,
    reference_date_json: Optional[Path] = None,
    algorithm_parameters_overrides_json: Optional[Path] = None,
    save_compressed_slc: bool = False,
    threads_per_worker: int = 2,
    n_parallel_bursts: int = 1,
    block_shape: tuple[int, int] = (512, 512),
    product_version: str = "0.4",
    runconfig_file: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Generate a disp-nisar runconfig YAML and algorithm parameters YAML.

    Parameters
    ----------
    gslc_files : list[Path]
        Paths to input NISAR GSLC HDF5 files.
    frame_id : int
        Frame ID of the bursts contained in the GSLC files.
    output_dir : Path
        Root output directory for products, scratch, and config files.
    mode : str
        Processing mode: ``"forward"`` or ``"historical"``.
    frequency : str
        GSLC frequency band, e.g. ``"frequencyA"`` or ``"frequencyB"``.
    polarization : str
        Polarization, e.g. ``"HH"``, ``"VV"``, ``"HV"``, ``"VH"``.
    gunw_files : list[Path] or None
        Paths to NISAR GUNW HDF5 files for ionosphere correction.
        If None, ionosphere correction is skipped.
    mask_file : Path or None
        Water/bad-data mask (uint8, 0=invalid, 1=valid).
    dem_file : Path or None
        DEM file covering the full frame.  If None, DEM corrections skipped.
    troposphere_files : list[Path] or None
        Troposphere correction files (1 per date).  If None, skipped.
    scratch_dir : Path or None
        Scratch directory.  Defaults to ``{output_dir}/scratch``.
    algorithm_parameters_file : Path or None
        Path where the algorithm parameters YAML will be written.
        Defaults to ``{output_dir}/config/algorithm_parameters_{mode}.yaml``.
    frame_to_bounds_json : Path or None
        JSON mapping frame_id → bounds.  If None, downloaded automatically.
    reference_date_json : Path or None
        JSON with reference date changes per frame.
    algorithm_parameters_overrides_json : Path or None
        JSON with frame-specific algorithm parameter overrides.
    save_compressed_slc : bool
        Whether to save compressed SLCs in the output.
    threads_per_worker : int
        Number of OMP threads per worker process.
    n_parallel_bursts : int
        Number of bursts/chunks to process in parallel.
    block_shape : (rows, cols)
        Block size for PS detection and wrapped-phase estimation
        (``worker_settings.block_shape``).  Default ``(512, 512)``.
        Use larger values e.g. ``(2048, 2048)`` to reduce I/O overhead on
        systems with sufficient RAM.
    product_version : str
        Product version string in ``<major>.<minor>`` format.
    runconfig_file : Path or None
        Output path for the runconfig YAML.
        Defaults to ``{output_dir}/config/runconfig_{mode}.yaml``.

    Returns
    -------
    runconfig_path : Path
        Path to the written runconfig YAML.
    algorithm_parameters_path : Path
        Path to the written algorithm parameters YAML.

    """
    output_dir = Path(output_dir)
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    mode = mode.lower()
    if mode not in ("forward", "historical"):
        raise ValueError(f"mode must be 'forward' or 'historical', got '{mode}'")

    today = date.today().strftime("%Y%m%d")
    if scratch_dir is None:
        scratch_dir = output_dir / "scratch"
    if algorithm_parameters_file is None:
        algorithm_parameters_file = (
            config_dir / f"algorithm_parameters_{mode}_{today}.yaml"
        )
    if runconfig_file is None:
        runconfig_file = config_dir / f"runconfig_{mode}_{today}.yaml"

    product_type = f"DISP_NISAR_{mode.upper()}"

    # Write algorithm parameters with defaults
    AlgorithmParameters().to_yaml(algorithm_parameters_file)

    runconfig = RunConfig(
        input_file_group=InputFileGroup(
            gslc_file_list=gslc_files,
            frame_id=frame_id,
            frequency=frequency,
            polarization=polarization,
        ),
        dynamic_ancillary_file_group=DynamicAncillaryFileGroup(
            algorithm_parameters_file=algorithm_parameters_file,
            mask_file=mask_file,
            dem_file=dem_file,
            gunw_files=gunw_files or [],
            troposphere_files=troposphere_files,
        ),
        static_ancillary_file_group=StaticAncillaryFileGroup(
            frame_to_bounds_json=frame_to_bounds_json,
            reference_date_database_json=reference_date_json,
            algorithm_parameters_overrides_json=algorithm_parameters_overrides_json,
        ),
        primary_executable=PrimaryExecutable(product_type=product_type),
        product_path_group=ProductPathGroup(
            product_path=output_dir / "output",
            scratch_path=scratch_dir,
            sas_output_path=output_dir / "output",
            product_version=product_version,
            save_compressed_slc=save_compressed_slc,
        ),
        log_file=output_dir / "output" / f"disp_nisar_{mode}.log",
    )
    runconfig.worker_settings.threads_per_worker = threads_per_worker
    runconfig.worker_settings.n_parallel_bursts = n_parallel_bursts
    runconfig.worker_settings.block_shape = list(block_shape)

    runconfig.to_yaml(runconfig_file)

    return Path(runconfig_file), Path(algorithm_parameters_file)


@click.command(name="make-runconfig")
@click.argument("gslc_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--frame-id", "-f", required=True, type=int, help="Frame ID of the GSLC bursts."
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Root output directory for products, scratch, and configs.",
)
@click.option(
    "--mode",
    "-m",
    default="forward",
    type=click.Choice(["forward", "historical"], case_sensitive=False),
    show_default=True,
    help="Processing mode.",
)
@click.option(
    "--frequency", default="frequencyA", show_default=True, help="GSLC frequency band."
)
@click.option("--polarization", default="HH", show_default=True, help="Polarization.")
@click.option(
    "--gunw-files",
    multiple=True,
    type=click.Path(exists=True),
    help="GUNW files for ionosphere correction (repeat for multiple).",
)
@click.option(
    "--mask-file",
    type=click.Path(exists=True),
    default=None,
    help="Water/bad-data mask file (uint8).",
)
@click.option(
    "--dem-file",
    type=click.Path(exists=True),
    default=None,
    help="DEM file covering the full frame.",
)
@click.option(
    "--scratch-dir",
    type=click.Path(),
    default=None,
    help="Scratch directory (default: {output_dir}/scratch).",
)
@click.option(
    "--algorithm-parameters-file",
    type=click.Path(),
    default=None,
    help="Path for the algorithm parameters YAML output.",
)
@click.option(
    "--frame-to-bounds-json",
    type=click.Path(exists=True),
    default=None,
    help="JSON mapping frame_id → bounds. Downloaded automatically if omitted.",
)
@click.option(
    "--reference-date-json",
    type=click.Path(exists=True),
    default=None,
    help="JSON with reference date changes per frame.",
)
@click.option(
    "--overrides-json",
    type=click.Path(exists=True),
    default=None,
    help="JSON with frame-specific algorithm parameter overrides.",
)
@click.option(
    "--save-compressed-slc",
    is_flag=True,
    default=False,
    help="Save compressed SLCs in the output.",
)
@click.option(
    "--threads-per-worker",
    default=2,
    show_default=True,
    type=int,
    help="OMP threads per worker.",
)
@click.option(
    "--n-parallel-bursts",
    default=1,
    show_default=True,
    type=int,
    help="Number of bursts/chunks to process in parallel.",
)
@click.option(
    "--block-shape",
    nargs=2,
    type=int,
    default=(512, 512),
    show_default=True,
    metavar="ROWS COLS",
    help="Block size for PS detection and wrapped-phase estimation.",
)
@click.option(
    "--product-version",
    default="0.4",
    show_default=True,
    help="Product version string.",
)
@click.option(
    "--outfile",
    type=click.Path(),
    default=None,
    help=(
        "Output runconfig YAML path (default:"
        " {output_dir}/config/runconfig_{mode}.yaml)."
    ),
)
def make_runconfig_cli(
    gslc_files,
    frame_id,
    output_dir,
    mode,
    frequency,
    polarization,
    gunw_files,
    mask_file,
    dem_file,
    scratch_dir,
    algorithm_parameters_file,
    frame_to_bounds_json,
    reference_date_json,
    overrides_json,
    save_compressed_slc,
    threads_per_worker,
    n_parallel_bursts,
    block_shape,
    product_version,
    outfile,
):
    r"""Generate a disp-nisar runconfig YAML from GSLC files.

    GSLC_FILES: one or more paths to NISAR GSLC HDF5 files.

    Example:
    \b
    disp-nisar make-runconfig \\
        input_slcs/*.h5 \\
        --frame-id 26410 \\
        --output-dir ./work \\
        --mode forward \\
        --gunw-files gunw_files/*.h5 \\
        --mask-file water_mask.vrt

    """
    runconfig_path, alg_params_path = generate_runconfig(
        gslc_files=[Path(f) for f in gslc_files],
        frame_id=frame_id,
        output_dir=Path(output_dir),
        mode=mode,
        frequency=frequency,
        polarization=polarization,
        gunw_files=[Path(f) for f in gunw_files] if gunw_files else None,
        mask_file=Path(mask_file) if mask_file else None,
        dem_file=Path(dem_file) if dem_file else None,
        scratch_dir=Path(scratch_dir) if scratch_dir else None,
        algorithm_parameters_file=(
            Path(algorithm_parameters_file) if algorithm_parameters_file else None
        ),
        frame_to_bounds_json=(
            Path(frame_to_bounds_json) if frame_to_bounds_json else None
        ),
        reference_date_json=Path(reference_date_json) if reference_date_json else None,
        algorithm_parameters_overrides_json=(
            Path(overrides_json) if overrides_json else None
        ),
        save_compressed_slc=save_compressed_slc,
        threads_per_worker=threads_per_worker,
        n_parallel_bursts=n_parallel_bursts,
        block_shape=tuple(block_shape),
        product_version=product_version,
        runconfig_file=Path(outfile) if outfile else None,
    )
    click.echo(f"Runconfig:             {runconfig_path}")
    click.echo(f"Algorithm parameters:  {alg_params_path}")
    click.echo("\nRun with:")
    click.echo(f"  disp-nisar run {runconfig_path}")

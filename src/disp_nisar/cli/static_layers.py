"""CLI for static layers generation."""

import logging
from pathlib import Path

import click
from dolphin._log import setup_logging

from disp_nisar.main_static_layers import run_static_layers
from disp_nisar.pge_runconfig import StaticLayersRunConfig

logger = logging.getLogger(__name__)


@click.command(name="static-layers")
@click.option(
    "--runconfig",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to static layers runconfig YAML file",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def static_layers_cli(runconfig: Path, debug: bool) -> None:
    """Generate static geometry layers for a NISAR frame.

    This command creates reusable geometry products including LOS vectors,
    incidence angles, layover/shadow masks, and warped DEM.
    """
    setup_logging(logger_name="disp_nisar", debug=debug)

    logger.info(f"Loading runconfig from {runconfig}")
    rc = StaticLayersRunConfig.from_yaml(runconfig)

    outputs = run_static_layers(rc)

    click.echo("\nStatic layers generation complete!")
    click.echo(f"Output directory: {rc.product_path_group.output_directory}")
    click.echo("\nOutputs:")
    click.echo(f"  - Incidence angle: {outputs.incidence_angle_path.name}")
    click.echo(f"  - LOS East: {outputs.los_east_path.name}")
    click.echo(f"  - LOS North: {outputs.los_north_path.name}")
    click.echo(f"  - Layover/Shadow mask: {outputs.layover_shadow_mask_path.name}")
    click.echo(f"  - DEM (UTM): {outputs.dem_warped_path.name}")
    if outputs.los_combined_path:
        click.echo(f"  - LOS combined (3-band): {outputs.los_combined_path.name}")

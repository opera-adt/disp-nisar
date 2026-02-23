import click

__all__ = ["run_cli", "run_main"]


def run_main(config_file: str, debug: bool = False) -> None:
    """Run the displacement workflow for CONFIG_FILE."""
    # Disable GPU before any JAX import to suppress CUDA discovery errors
    # on nodes without a GPU. Must happen before `import dolphin`.
    _disable_gpu_early(config_file)

    # rest of imports here so --help doesn't take forever
    from disp_nisar.main import run
    from disp_nisar.pge_runconfig import RunConfig

    pge_runconfig = RunConfig.from_yaml(config_file)
    cfg = pge_runconfig.to_workflow()
    run(cfg, pge_runconfig=pge_runconfig, debug=debug)


def _disable_gpu_early(config_file: str) -> None:
    """Set CUDA_VISIBLE_DEVICES before JAX is imported if gpu_enabled is false."""
    import os

    import yaml  # type: ignore[import-untyped]

    with open(config_file) as f:
        raw = yaml.safe_load(f)
    gpu_enabled = (raw.get("worker_settings") or {}).get("gpu_enabled", False)
    if not gpu_enabled:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


@click.command("run")
@click.argument("config_file", type=click.Path(exists=True))
@click.pass_context
def run_cli(
    ctx: click.Context,
    config_file: str,
) -> None:
    """Run the displacement workflow for CONFIG_FILE."""
    run_main(config_file=config_file, debug=ctx.obj["debug"])

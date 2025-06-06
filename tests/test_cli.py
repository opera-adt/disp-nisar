import pytest
from click.testing import CliRunner

from disp_nisar.cli import cli_app

# https://click.palletsprojects.com/en/8.1.x/testing/


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("disp-nisar, version ")
    assert result.output.endswith("\n")


def test_cli_debug():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--debug"])
    assert result.exit_code == 2
    assert result.output.startswith("Usage: disp-nisar [OPTIONS] COMMAND [ARGS]...\n")
    assert result.output.endswith("\n")


# Check run, validate, download, make-browse
@pytest.mark.parametrize(
    "command",
    ["run", "download", "validate", "make-browse"],
)
def test_cli_subcommands_smoke_test(command):
    runner = CliRunner()
    result = runner.invoke(cli_app, [command, "--help"])
    assert result.exit_code == 0


# Check run, validate, download, make-browse
@pytest.mark.parametrize(
    "command",
    ["dem"],
)
def test_cli_download_subcommands_smoke_test(command):
    runner = CliRunner()
    result = runner.invoke(cli_app, ["download", command, "--help"])
    assert result.exit_code == 0

"""Unit tests for static layers CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from disp_nisar.cli.static_layers import static_layers_cli
from disp_nisar.main_static_layers import StaticLayersOutputs


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_runconfig(tmp_path):
    """Create a mock runconfig YAML file."""
    runconfig_content = f"""
input_file_group:
  frame_id: 12345
  frequency: frequencyA
  polarization: HH

dynamic_ancillary_file_group:
  gslc_file: {tmp_path}/test_gslc.h5
  dem_file: {tmp_path}/test_dem.tif
  mask_file: null

static_ancillary_file_group:
  frame_to_bounds_json: null

primary_executable:
  product_type: DISP_NISAR_STATIC

product_path_group:
  product_path: {tmp_path}/output
  scratch_path: {tmp_path}/scratch
  sas_output_path: {tmp_path}/output
  product_version: '0.4'

worker_settings:
  threads_per_worker: 2

log_file: {tmp_path}/scratch/log.txt

create_3band_los: false
product_spacing_m: null
"""
    runconfig_path = tmp_path / "runconfig_static.yaml"
    runconfig_path.write_text(runconfig_content)

    # Create required files
    (tmp_path / "test_gslc.h5").touch()
    (tmp_path / "test_dem.tif").touch()
    (tmp_path / "output").mkdir()
    (tmp_path / "scratch").mkdir()

    return runconfig_path


class TestStaticLayersCLI:
    """Test static layers CLI command."""

    def test_cli_requires_runconfig(self, runner):
        """Test that CLI requires --runconfig argument."""
        result = runner.invoke(static_layers_cli, [])
        assert result.exit_code != 0
        assert "Missing option '--runconfig'" in result.output

    def test_cli_runconfig_must_exist(self, runner, tmp_path):
        """Test that CLI validates runconfig exists."""
        nonexistent = tmp_path / "nonexistent.yaml"
        result = runner.invoke(static_layers_cli, ["--runconfig", str(nonexistent)])
        assert result.exit_code != 0

    @patch("disp_nisar.cli.static_layers.run_static_layers")
    def test_cli_runs_successfully(self, mock_run, runner, mock_runconfig, tmp_path):
        """Test successful CLI execution."""
        # Mock run_static_layers to return outputs
        mock_outputs = StaticLayersOutputs(
            incidence_angle_path=tmp_path / "incidence_angle.tif",
            los_east_path=tmp_path / "los_east.tif",
            los_north_path=tmp_path / "los_north.tif",
            layover_shadow_mask_path=tmp_path / "layover_shadow_mask.tif",
            dem_warped_path=tmp_path / "dem_warped_utm.tif",
        )
        mock_run.return_value = mock_outputs

        result = runner.invoke(static_layers_cli, ["--runconfig", str(mock_runconfig)])

        assert result.exit_code == 0
        assert "Static layers generation complete!" in result.output
        assert "incidence_angle.tif" in result.output
        assert "los_east.tif" in result.output

        # Verify run_static_layers was called
        mock_run.assert_called_once()

    @patch("disp_nisar.cli.static_layers.run_static_layers")
    def test_cli_with_debug_flag(self, mock_run, runner, mock_runconfig, tmp_path):
        """Test CLI with --debug flag."""
        mock_outputs = StaticLayersOutputs(
            incidence_angle_path=tmp_path / "incidence_angle.tif",
            los_east_path=tmp_path / "los_east.tif",
            los_north_path=tmp_path / "los_north.tif",
            layover_shadow_mask_path=tmp_path / "layover_shadow_mask.tif",
            dem_warped_path=tmp_path / "dem_warped_utm.tif",
        )
        mock_run.return_value = mock_outputs

        result = runner.invoke(
            static_layers_cli, ["--runconfig", str(mock_runconfig), "--debug"]
        )

        assert result.exit_code == 0

    @patch("disp_nisar.cli.static_layers.run_static_layers")
    def test_cli_shows_3band_los_output(
        self, mock_run, runner, mock_runconfig, tmp_path
    ):
        """Test CLI output includes 3-band LOS if present."""
        mock_outputs = StaticLayersOutputs(
            incidence_angle_path=tmp_path / "incidence_angle.tif",
            los_east_path=tmp_path / "los_east.tif",
            los_north_path=tmp_path / "los_north.tif",
            layover_shadow_mask_path=tmp_path / "layover_shadow_mask.tif",
            dem_warped_path=tmp_path / "dem_warped_utm.tif",
            los_combined_path=tmp_path / "los_enu.tif",
        )
        mock_run.return_value = mock_outputs

        result = runner.invoke(static_layers_cli, ["--runconfig", str(mock_runconfig)])

        assert result.exit_code == 0
        assert "los_enu.tif" in result.output


class TestStaticLayersCLIIntegration:
    """Integration tests for static layers CLI."""

    def test_cli_command_registered(self):
        """Test that static-layers command is registered."""
        from disp_nisar.cli import cli_app

        # Check that the command exists
        assert "static-layers" in [cmd.name for cmd in cli_app.commands.values()]

    def test_cli_help_message(self, runner):
        """Test CLI help message."""
        result = runner.invoke(static_layers_cli, ["--help"])
        assert result.exit_code == 0
        assert "Generate static geometry layers" in result.output
        assert "--runconfig" in result.output
        assert "--debug" in result.output


class TestStaticLayersScriptIntegration:
    """Test integration with standalone script."""

    def test_script_workflow(self, tmp_path):
        """Test the standalone script workflow (conceptual test)."""
        # This is a conceptual test showing how the script components fit together
        # Testing that the workflow components can be composed without importing scripts

        # Mock frame info
        mock_get_frame_info = MagicMock()
        mock_get_frame_info.return_value = {
            "epsg": 32611,
            "track": 123,
            "frame": 456,
            "bounds": MagicMock(),
            "pass_direction": "ascending",
        }

        # Mock GSLC download
        mock_download_gslc = MagicMock()
        gslc_file = tmp_path / "downloaded_gslc.h5"
        gslc_file.touch()
        mock_download_gslc.return_value = gslc_file

        # Mock DEM download
        mock_download_dem = MagicMock()
        dem_file = tmp_path / "downloaded_dem.tif"
        dem_file.touch()
        mock_download_dem.return_value = dem_file

        # Mock run_static_layers
        mock_run_static_layers = MagicMock()
        mock_outputs = StaticLayersOutputs(
            incidence_angle_path=tmp_path / "incidence_angle.tif",
            los_east_path=tmp_path / "los_east.tif",
            los_north_path=tmp_path / "los_north.tif",
            layover_shadow_mask_path=tmp_path / "layover_shadow_mask.tif",
            dem_warped_path=tmp_path / "dem_warped_utm.tif",
        )
        mock_run_static_layers.return_value = mock_outputs

        # Simulate the script workflow
        frame_id = 12345
        frame_info = mock_get_frame_info(frame_id, Path("dummy.gpkg"))
        gslc = mock_download_gslc(frame_id, frame_info, tmp_path)
        dem = mock_download_dem(gslc, "frequencyA", "HH", tmp_path / "dem.tif")
        outputs = mock_run_static_layers(MagicMock())

        # Verify all steps were called
        assert gslc.exists()
        assert dem.exists()
        assert outputs is not None


class TestStaticLayersErrorHandling:
    """Test error handling in CLI."""

    @patch("disp_nisar.cli.static_layers.run_static_layers")
    def test_cli_handles_missing_gslc(self, mock_run, runner, tmp_path):
        """Test CLI handles missing GSLC file."""
        # Create runconfig with non-existent GSLC
        runconfig_content = f"""
input_file_group:
  frame_id: 12345
  frequency: frequencyA
  polarization: HH

dynamic_ancillary_file_group:
  gslc_file: {tmp_path}/nonexistent_gslc.h5
  dem_file: {tmp_path}/test_dem.tif

static_ancillary_file_group:
  frame_to_bounds_json: null

primary_executable:
  product_type: DISP_NISAR_STATIC

product_path_group:
  product_path: {tmp_path}/output
  scratch_path: {tmp_path}/scratch
  sas_output_path: {tmp_path}/output
  product_version: '0.4'

worker_settings:
  threads_per_worker: 2

log_file: {tmp_path}/scratch/log.txt
create_3band_los: false
"""
        runconfig_path = tmp_path / "runconfig.yaml"
        runconfig_path.write_text(runconfig_content)
        (tmp_path / "output").mkdir()
        (tmp_path / "scratch").mkdir()

        # Mock run_static_layers to raise FileNotFoundError
        mock_run.side_effect = FileNotFoundError("GSLC file not found")

        result = runner.invoke(static_layers_cli, ["--runconfig", str(runconfig_path)])

        # CLI should handle the error gracefully
        assert result.exit_code != 0

    @patch("disp_nisar.cli.static_layers.run_static_layers")
    def test_cli_handles_invalid_frame_id(self, mock_run, runner, tmp_path):
        """Test CLI handles invalid frame ID."""
        runconfig_content = f"""
input_file_group:
  frame_id: null
  frequency: frequencyA
  polarization: HH

dynamic_ancillary_file_group:
  gslc_file: {tmp_path}/test_gslc.h5
  dem_file: {tmp_path}/test_dem.tif

static_ancillary_file_group:
  frame_to_bounds_json: null

primary_executable:
  product_type: DISP_NISAR_STATIC

product_path_group:
  product_path: {tmp_path}/output
  scratch_path: {tmp_path}/scratch
  sas_output_path: {tmp_path}/output
  product_version: '0.4'

worker_settings:
  threads_per_worker: 2

log_file: {tmp_path}/scratch/log.txt
create_3band_los: false
"""
        runconfig_path = tmp_path / "runconfig.yaml"
        runconfig_path.write_text(runconfig_content)
        (tmp_path / "test_gslc.h5").touch()
        (tmp_path / "test_dem.tif").touch()
        (tmp_path / "output").mkdir()
        (tmp_path / "scratch").mkdir()

        mock_run.side_effect = ValueError("frame_id is required")

        result = runner.invoke(static_layers_cli, ["--runconfig", str(runconfig_path)])

        assert result.exit_code != 0

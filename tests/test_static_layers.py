"""Unit tests for static layers workflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dolphin import Bbox

from disp_nisar.main_static_layers import (
    StaticLayersOutputs,
    _create_template_raster,
    _make_3band_los,
    add_product_metadata,
    run_static_layers,
    warp_dem_to_utm,
)
from disp_nisar.pge_runconfig import (
    InputFileGroup,
    PrimaryExecutable,
    ProductPathGroup,
    StaticAncillaryFileGroup,
    StaticLayersDynamicAncillaryFileGroup,
    StaticLayersRunConfig,
    WorkerSettings,
)


@pytest.fixture
def static_layers_input_file_group():
    """Create InputFileGroup for static layers."""
    return InputFileGroup(
        frame_id=12345,
        frequency="frequencyA",
        polarization="HH",
    )


@pytest.fixture
def static_layers_dynamic_ancillary_file_group(tmp_path):
    """Create DynamicAncillaryFileGroup for static layers."""
    gslc_file = tmp_path / "test_gslc.h5"
    dem_file = tmp_path / "test_dem.tif"
    # Create dummy files
    gslc_file.touch()
    dem_file.touch()
    return StaticLayersDynamicAncillaryFileGroup(
        gslc_file=gslc_file,
        dem_file=dem_file,
        mask_file=None,
    )


@pytest.fixture
def static_layers_runconfig(
    tmp_path,
    static_layers_input_file_group,
    static_layers_dynamic_ancillary_file_group,
):
    """Create StaticLayersRunConfig for testing."""
    output_dir = tmp_path / "output"
    scratch_dir = tmp_path / "scratch"
    output_dir.mkdir()
    scratch_dir.mkdir()

    return StaticLayersRunConfig(
        input_file_group=static_layers_input_file_group,
        dynamic_ancillary_file_group=static_layers_dynamic_ancillary_file_group,
        static_ancillary_file_group=StaticAncillaryFileGroup(),
        primary_executable=PrimaryExecutable(product_type="DISP_NISAR_STATIC"),
        product_path_group=ProductPathGroup(
            product_path=output_dir,
            scratch_path=scratch_dir,
            sas_output_path=output_dir,
            product_version="0.4",
        ),
        worker_settings=WorkerSettings(threads_per_worker=2),
        log_file=scratch_dir / "log.txt",
        create_3band_los=False,
        product_spacing_m=None,
    )


class TestStaticLayersRunConfig:
    """Test StaticLayersRunConfig configuration class."""

    def test_config_creation(self, static_layers_runconfig):
        """Test that StaticLayersRunConfig can be created."""
        assert static_layers_runconfig.input_file_group.frame_id == 12345
        assert static_layers_runconfig.input_file_group.frequency == "frequencyA"
        assert static_layers_runconfig.create_3band_los is False

    def test_config_to_yaml(self, static_layers_runconfig, tmp_path):
        """Test that config can be serialized to YAML."""
        yaml_path = tmp_path / "config.yaml"
        static_layers_runconfig.to_yaml(yaml_path)
        assert yaml_path.exists()

        # Test loading back
        loaded_config = StaticLayersRunConfig.from_yaml(yaml_path)
        assert loaded_config.input_file_group.frame_id == 12345

    def test_config_with_3band_los(self, static_layers_runconfig):
        """Test config with 3-band LOS enabled."""
        static_layers_runconfig.create_3band_los = True
        assert static_layers_runconfig.create_3band_los is True

    def test_config_with_product_spacing(self, static_layers_runconfig):
        """Test config with product spacing."""
        static_layers_runconfig.product_spacing_m = 90
        assert static_layers_runconfig.product_spacing_m == 90


class TestStaticLayersDynamicAncillaryFileGroup:
    """Test StaticLayersDynamicAncillaryFileGroup configuration class."""

    def test_optional_files(self):
        """Test that gslc_file and dem_file can be None."""
        config = StaticLayersDynamicAncillaryFileGroup(
            gslc_file=None,
            dem_file=None,
            mask_file=None,
        )
        assert config.gslc_file is None
        assert config.dem_file is None
        assert config.mask_file is None

    def test_with_files(self, tmp_path):
        """Test with actual file paths."""
        gslc_file = tmp_path / "test.h5"
        dem_file = tmp_path / "dem.tif"
        mask_file = tmp_path / "mask.tif"

        config = StaticLayersDynamicAncillaryFileGroup(
            gslc_file=gslc_file,
            dem_file=dem_file,
            mask_file=mask_file,
        )
        assert config.gslc_file == gslc_file
        assert config.dem_file == dem_file
        assert config.mask_file == mask_file


class TestCreateTemplateRaster:
    """Test _create_template_raster function."""

    def test_create_template_basic(self, tmp_path):
        """Test basic template raster creation."""
        output_path = tmp_path / "template.tif"
        bounds = Bbox(left=500000, bottom=4000000, right=600000, top=4100000)
        epsg = 32611
        spacing = 30.0

        result = _create_template_raster(output_path, bounds, epsg, spacing)

        assert result.exists()
        assert result == output_path

        # Check raster properties
        from osgeo import gdal

        ds = gdal.Open(str(output_path))
        assert ds is not None
        assert ds.RasterXSize == int((bounds.right - bounds.left) / spacing)
        assert ds.RasterYSize == int((bounds.top - bounds.bottom) / spacing)
        ds = None

    def test_template_geotransform(self, tmp_path):
        """Test that template has correct geotransform."""
        output_path = tmp_path / "template.tif"
        bounds = Bbox(left=500000, bottom=4000000, right=600000, top=4100000)
        epsg = 32611
        spacing = 30.0

        _create_template_raster(output_path, bounds, epsg, spacing)

        from osgeo import gdal

        ds = gdal.Open(str(output_path))
        gt = ds.GetGeoTransform()
        assert gt[0] == bounds.left  # Origin X
        assert gt[3] == bounds.top  # Origin Y
        assert gt[1] == spacing  # Pixel width
        assert gt[5] == -spacing  # Pixel height (negative)
        ds = None


class TestWarpDemToUtm:
    """Test warp_dem_to_utm function."""

    @pytest.fixture
    def mock_dem_file(self, tmp_path):
        """Create a mock DEM file."""
        from osgeo import gdal, osr

        dem_path = tmp_path / "input_dem.tif"
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(dem_path), 100, 100, 1, gdal.GDT_Float32)

        # Set WGS84 projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())

        # Set geotransform (1 degree coverage)
        ds.SetGeoTransform([-118.0, 0.01, 0, 34.0, 0, -0.01])

        # Fill with dummy elevation data
        band = ds.GetRasterBand(1)
        data = np.random.rand(100, 100).astype(np.float32) * 1000
        band.WriteArray(data)
        band.FlushCache()
        ds = None

        return dem_path

    def test_warp_dem_basic(self, tmp_path, mock_dem_file):
        """Test basic DEM warping."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        bounds = Bbox(left=400000, bottom=3760000, right=500000, top=3860000)
        epsg = 32611
        spacing = 30.0

        result = warp_dem_to_utm(
            dem_file=mock_dem_file,
            epsg=epsg,
            bounds=bounds,
            output_dir=output_dir,
            spacing=spacing,
        )

        assert result.exists()
        assert result.name == "dem_warped_utm.tif"

        # Check output properties
        from osgeo import gdal

        ds = gdal.Open(str(result))
        assert ds is not None
        assert ds.RasterXSize > 0
        assert ds.RasterYSize > 0
        ds = None


class TestMake3BandLos:
    """Test _make_3band_los function."""

    @pytest.fixture
    def mock_los_files(self, tmp_path):
        """Create mock LOS east and north files."""
        import rasterio as rio

        los_east = tmp_path / "los_east.tif"
        los_north = tmp_path / "los_north.tif"

        # Create test data - ensure east^2 + north^2 < 1
        shape = (100, 100)
        # Generate values between -0.5 and 0.5 to ensure magnitude is < 1
        los_east_data = np.random.uniform(-0.5, 0.5, shape).astype(np.float32)
        los_north_data = np.random.uniform(-0.5, 0.5, shape).astype(np.float32)

        # Normalize where needed to ensure east^2 + north^2 < 1
        mag_sq = los_east_data**2 + los_north_data**2
        scale = np.where(mag_sq >= 1.0, 0.99 / np.sqrt(mag_sq), 1.0)
        los_east_data *= scale
        los_north_data *= scale

        profile = {
            "driver": "GTiff",
            "height": shape[0],
            "width": shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:32611",
            "transform": rio.transform.from_bounds(
                500000, 4000000, 503000, 4003000, shape[1], shape[0]
            ),
        }

        with rio.open(los_east, "w", **profile) as dst:
            dst.write(los_east_data, 1)

        with rio.open(los_north, "w", **profile) as dst:
            dst.write(los_north_data, 1)

        return los_east, los_north

    def test_3band_los_creation(self, tmp_path, mock_los_files):
        """Test 3-band LOS file creation."""
        los_east, los_north = mock_los_files
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = _make_3band_los(
            los_east_path=los_east,
            los_north_path=los_north,
            output_dir=output_dir,
        )

        assert result.exists()
        assert result.name == "los_enu.tif"

        # Check that it's a 3-band file
        import rasterio as rio

        with rio.open(result) as src:
            assert src.count == 3
            assert src.descriptions[0] == "LOS East"
            assert src.descriptions[1] == "LOS North"
            assert src.descriptions[2] == "LOS Up"

            # Check that up component is computed correctly
            east = src.read(1)
            north = src.read(2)
            up = src.read(3)

            # Verify magnitude is approximately 1
            mag = np.sqrt(east**2 + north**2 + up**2)
            np.testing.assert_allclose(mag, 1.0, rtol=0.01)

    def test_3band_los_compression(self, tmp_path, mock_los_files):
        """Test that 3-band LOS uses compression."""
        los_east, los_north = mock_los_files
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = _make_3band_los(
            los_east_path=los_east,
            los_north_path=los_north,
            output_dir=output_dir,
        )

        import rasterio as rio

        with rio.open(result) as src:
            assert src.compression is not None
            assert src.profile["compress"] == "deflate"


class TestAddProductMetadata:
    """Test add_product_metadata function."""

    @pytest.fixture
    def mock_output_files(self, tmp_path):
        """Create mock output files."""
        from osgeo import gdal

        files = []
        for name in [
            "incidence_angle.tif",
            "los_east.tif",
            "los_north.tif",
            "layover_shadow_mask.tif",
            "dem_warped_utm.tif",
        ]:
            path = tmp_path / name
            driver = gdal.GetDriverByName("GTiff")
            driver.Create(str(path), 10, 10, 1, gdal.GDT_Float32)
            files.append(path)

        return StaticLayersOutputs(
            incidence_angle_path=files[0],
            los_east_path=files[1],
            los_north_path=files[2],
            layover_shadow_mask_path=files[3],
            dem_warped_path=files[4],
        )

    def test_add_metadata(
        self, tmp_path, mock_output_files, static_layers_runconfig  # noqa: ARG002
    ):
        """Test adding metadata to output files."""
        from datetime import datetime, timezone

        processing_datetime = datetime.now(tz=timezone.utc)

        add_product_metadata(
            static_layers_paths=mock_output_files,
            pge_runconfig=static_layers_runconfig,
            frame_id=12345,
            orbit_direction="ascending",
            frequency="frequencyA",
            processing_datetime=processing_datetime,
        )

        # Check that metadata was added
        from osgeo import gdal

        ds = gdal.Open(str(mock_output_files.incidence_angle_path))
        metadata = ds.GetMetadata()
        assert "platform" in metadata
        assert metadata["platform"] == "NISAR"
        assert metadata["frame_id"] == "12345"
        assert metadata["orbit_direction"] == "ascending"
        ds = None


class TestRunStaticLayers:
    """Test run_static_layers main function."""

    @pytest.mark.skip(reason="Complex integration test - requires full mocking")
    @patch("disp_nisar.main_static_layers.prepare_geometry_layers")
    @patch("disp_nisar.main_static_layers.warp_dem_to_utm")
    @patch("disp_nisar.main_static_layers.add_product_metadata")
    @patch("disp_nisar.main_static_layers.create_outputs")
    @patch("disp_nisar.main_static_layers._create_template_raster")
    @patch("disp_nisar.main_static_layers.h5py.File")
    @patch("disp_nisar._utils.get_nisar_frame_bbox")
    def test_run_static_layers_basic(
        self,
        mock_get_bbox,
        mock_h5py,
        mock_template,  # noqa: ARG002
        mock_create_outputs,
        mock_add_metadata,
        mock_warp_dem,
        mock_prepare_geometry,
        static_layers_runconfig,
        tmp_path,  # noqa: ARG002
    ):
        """Test basic run_static_layers execution."""
        # Setup mocks
        scratch_dir = static_layers_runconfig.product_path_group.scratch_path

        mock_get_bbox.return_value = (
            32611,
            Bbox(left=500000, bottom=4000000, right=600000, top=4100000),
        )

        # Mock h5py.File to return orbit direction
        mock_hf = MagicMock()
        mock_hf.__enter__.return_value = mock_hf
        mock_hf.__getitem__.return_value = MagicMock(__call__=lambda: b"ascending")
        mock_h5py.return_value = mock_hf

        # Mock prepare_geometry_layers to return file paths
        mock_prepare_geometry.return_value = {
            "incidence_angle": scratch_dir / "incidence_angle.tif",
            "los_east": scratch_dir / "los_east.tif",
            "los_north": scratch_dir / "los_north.tif",
            "layover_shadow_mask": scratch_dir / "layover_shadow_mask.tif",
        }

        # Create dummy output files
        for name in [
            "incidence_angle.tif",
            "los_east.tif",
            "los_north.tif",
            "layover_shadow_mask.tif",
        ]:
            (scratch_dir / name).touch()

        mock_warp_dem.return_value = scratch_dir / "dem_warped_utm.tif"
        (scratch_dir / "dem_warped_utm.tif").touch()

        # Run the workflow
        outputs = run_static_layers(static_layers_runconfig)

        # Verify outputs
        assert isinstance(outputs, StaticLayersOutputs)
        assert outputs.incidence_angle_path.exists()
        assert outputs.los_east_path.exists()
        assert outputs.los_north_path.exists()
        assert outputs.layover_shadow_mask_path.exists()
        assert outputs.dem_warped_path.exists()

        # Verify function calls
        mock_prepare_geometry.assert_called_once()
        mock_warp_dem.assert_called_once()
        mock_add_metadata.assert_called_once()
        mock_create_outputs.assert_called_once()

    def test_run_static_layers_validates_frame_id(self, static_layers_runconfig):
        """Test that run_static_layers validates frame_id."""
        static_layers_runconfig.input_file_group.frame_id = None

        with pytest.raises(ValueError, match="frame_id is required"):
            run_static_layers(static_layers_runconfig)

    @pytest.mark.skip(reason="Complex integration test - requires full mocking")
    @patch("disp_nisar.main_static_layers.h5py.File")
    def test_run_static_layers_handles_empty_paths(
        self, mock_h5py, static_layers_runconfig  # noqa: ARG002
    ):
        """Test that empty string paths are converted to None."""
        # Set files to empty strings
        static_layers_runconfig.dynamic_ancillary_file_group.gslc_file = Path("")
        static_layers_runconfig.dynamic_ancillary_file_group.dem_file = Path("")

        # This should trigger auto-download logic or raise appropriate error
        with pytest.raises((ValueError, FileNotFoundError)):
            run_static_layers(static_layers_runconfig)

    @pytest.mark.skip(reason="Complex integration test - requires full mocking")
    @patch("disp_nisar.main_static_layers.prepare_geometry_layers")
    @patch("disp_nisar.main_static_layers.warp_dem_to_utm")
    @patch("disp_nisar.main_static_layers._make_3band_los")
    @patch("disp_nisar.main_static_layers.add_product_metadata")
    @patch("disp_nisar.main_static_layers.create_outputs")
    @patch("disp_nisar.main_static_layers._create_template_raster")
    @patch("disp_nisar.main_static_layers.h5py.File")
    @patch("disp_nisar.main_static_layers.get_nisar_frame_bbox")
    def test_run_static_layers_with_3band_los(
        self,
        mock_get_bbox,
        mock_h5py,
        mock_template,  # noqa: ARG002
        mock_create_outputs,  # noqa: ARG002
        mock_add_metadata,  # noqa: ARG002
        mock_3band_los,
        mock_warp_dem,
        mock_prepare_geometry,
        static_layers_runconfig,
        tmp_path,  # noqa: ARG002
    ):
        """Test run_static_layers with 3-band LOS enabled."""
        # Enable 3-band LOS
        static_layers_runconfig.create_3band_los = True
        scratch_dir = static_layers_runconfig.product_path_group.scratch_path

        # Setup mocks
        mock_get_bbox.return_value = (
            32611,
            Bbox(left=500000, bottom=4000000, right=600000, top=4100000),
        )

        mock_hf = MagicMock()
        mock_hf.__enter__.return_value = mock_hf
        mock_hf.__getitem__.return_value = MagicMock(__call__=lambda: b"ascending")
        mock_h5py.return_value = mock_hf

        mock_prepare_geometry.return_value = {
            "incidence_angle": scratch_dir / "incidence_angle.tif",
            "los_east": scratch_dir / "los_east.tif",
            "los_north": scratch_dir / "los_north.tif",
            "layover_shadow_mask": scratch_dir / "layover_shadow_mask.tif",
        }

        # Create dummy files
        for name in [
            "incidence_angle.tif",
            "los_east.tif",
            "los_north.tif",
            "layover_shadow_mask.tif",
        ]:
            (scratch_dir / name).touch()

        mock_warp_dem.return_value = scratch_dir / "dem_warped_utm.tif"
        (scratch_dir / "dem_warped_utm.tif").touch()

        mock_3band_los.return_value = scratch_dir / "los_enu.tif"
        (scratch_dir / "los_enu.tif").touch()

        # Run the workflow
        outputs = run_static_layers(static_layers_runconfig)

        # Verify 3-band LOS was created
        assert outputs.los_combined_path is not None
        assert outputs.los_combined_path.exists()
        mock_3band_los.assert_called_once()


class TestStaticLayersOutputs:
    """Test StaticLayersOutputs NamedTuple."""

    def test_outputs_creation(self, tmp_path):
        """Test creating StaticLayersOutputs."""
        outputs = StaticLayersOutputs(
            incidence_angle_path=tmp_path / "incidence.tif",
            los_east_path=tmp_path / "los_east.tif",
            los_north_path=tmp_path / "los_north.tif",
            layover_shadow_mask_path=tmp_path / "mask.tif",
            dem_warped_path=tmp_path / "dem.tif",
        )

        assert outputs.incidence_angle_path.name == "incidence.tif"
        assert outputs.los_combined_path is None

    def test_outputs_with_combined_los(self, tmp_path):
        """Test StaticLayersOutputs with combined LOS."""
        outputs = StaticLayersOutputs(
            incidence_angle_path=tmp_path / "incidence.tif",
            los_east_path=tmp_path / "los_east.tif",
            los_north_path=tmp_path / "los_north.tif",
            layover_shadow_mask_path=tmp_path / "mask.tif",
            dem_warped_path=tmp_path / "dem.tif",
            los_combined_path=tmp_path / "los_enu.tif",
        )

        assert outputs.los_combined_path.name == "los_enu.tif"


class TestStaticLayersSkipExisting:
    """Test that static layers workflow skips existing files."""

    @patch("disp_nisar.main_static_layers.prepare_geometry_layers")
    @patch("disp_nisar.main_static_layers.warp_dem_to_utm")
    @patch("disp_nisar.main_static_layers.add_product_metadata")
    @patch("disp_nisar.main_static_layers.create_outputs")
    @patch("disp_nisar.main_static_layers._create_template_raster")
    @patch("disp_nisar.main_static_layers.h5py.File")
    @patch("disp_nisar._utils.get_nisar_frame_bbox")
    def test_skip_existing_geometry(
        self,
        mock_get_bbox,
        mock_h5py,
        mock_template,  # noqa: ARG002
        mock_create_outputs,  # noqa: ARG002
        mock_add_metadata,  # noqa: ARG002
        mock_warp_dem,
        mock_prepare_geometry,
        static_layers_runconfig,
        tmp_path,  # noqa: ARG002
    ):
        """Test that existing geometry files are skipped."""
        scratch_dir = static_layers_runconfig.product_path_group.scratch_path

        # Create existing geometry files
        for name in [
            "incidence_angle.tif",
            "los_east.tif",
            "los_north.tif",
            "layover_shadow_mask.tif",
        ]:
            (scratch_dir / name).touch()

        mock_get_bbox.return_value = (
            32611,
            Bbox(left=500000, bottom=4000000, right=600000, top=4100000),
        )

        mock_hf = MagicMock()
        mock_hf.__enter__.return_value = mock_hf
        mock_hf.__getitem__.return_value = MagicMock(__call__=lambda: b"ascending")
        mock_h5py.return_value = mock_hf

        mock_warp_dem.return_value = scratch_dir / "dem_warped_utm.tif"
        (scratch_dir / "dem_warped_utm.tif").touch()

        outputs = run_static_layers(static_layers_runconfig)

        # Verify prepare_geometry_layers was NOT called
        mock_prepare_geometry.assert_not_called()

        # Verify outputs structure is correct
        # (Files may have been moved/deleted by create_outputs mock)
        assert outputs.incidence_angle_path.name == "incidence_angle.tif"
        assert outputs.los_east_path.name == "los_east.tif"
        assert outputs.los_north_path.name == "los_north.tif"

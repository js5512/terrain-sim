import io
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terrain_mapper.dem import parse_ascii_dem, parse_dem, parse_geotiff
from terrain_mapper.elevation import ElevationGrid, load_dem


def test_parse_ascii_dem_from_string_and_file(tmp_path: Path) -> None:
    ascii_content = """NCOLS 4
NROWS 3
XLLCORNER 500000.0
YLLCORNER 4000000.0
CELLSIZE 25.0
NODATA_VALUE -9999
10.0 20.0 30.0 40.0
15.0 25.0 -9999 45.0
20.0 30.0 40.0 50.0
"""
    # Test from string
    grid = parse_ascii_dem(ascii_content)
    assert grid.values_m.shape == (3, 4)
    assert grid.cell_size_x_m == 25.0
    assert grid.cell_size_y_m == 25.0
    assert grid.bounds == (500000.0, 4000000.0, 500100.0, 4000075.0)
    assert np.isnan(grid.values_m[1, 2])
    assert grid.values_m[0, 0] == 10.0

    # Test from file path
    dem_file = tmp_path / "terrain.asc"
    dem_file.write_text(ascii_content)
    grid_from_file = parse_dem(dem_file)
    assert grid_from_file.values_m.shape == (3, 4)
    assert grid_from_file.bounds == grid.bounds

    # Test from raw bytes
    grid_from_bytes = parse_dem(ascii_content.encode("utf-8"))
    assert grid_from_bytes.values_m.shape == (3, 4)


def test_parse_ascii_dem_center_and_missing_nodata() -> None:
    ascii_content = """ncols 2
nrows 2
xllcenter 100.0
yllcenter 200.0
cellsize 10.0
1 2
3 4
"""
    grid = parse_ascii_dem(ascii_content)
    assert grid.bounds == (95.0, 195.0, 115.0, 215.0)
    assert grid.values_m[0, 0] == 1.0
    assert grid.values_m[1, 1] == 4.0


def test_parse_ascii_dem_invalid_header() -> None:
    invalid_content = "some random text with no header"
    with pytest.raises(ValueError, match="missing required 'ncols' or 'nrows'"):
        parse_ascii_dem(invalid_content)


def test_parse_geotiff_with_projected_crs(tmp_path: Path) -> None:
    elevations = np.array([
        [100.0, 120.0, 140.0],
        [110.0, 130.0, 150.0],
        [105.0, -9999.0, 160.0],
    ], dtype=np.float32)

    transform = from_origin(600000.0, 5000000.0, 10.0, 10.0)
    tif_path = tmp_path / "sample.tif"

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevations, 1)

    grid = parse_geotiff(tif_path)
    assert grid.values_m.shape == (3, 3)
    assert grid.cell_size_x_m == 10.0
    assert grid.cell_size_y_m == 10.0
    assert "32632" in grid.crs
    assert grid.bounds[0] == 600000.0
    assert grid.bounds[3] == 5000000.0
    assert np.isnan(grid.values_m[2, 1])

    # Test via universal parse_dem and load_dem
    grid2 = parse_dem(tif_path)
    assert grid2.bounds == grid.bounds
    grid3 = load_dem(tif_path)
    assert grid3.bounds == grid.bounds


def test_parse_geotiff_from_in_memory_bytes() -> None:
    elevations = np.ones((4, 4), dtype=np.float32) * 50.0
    transform = from_origin(200000.0, 3000000.0, 5.0, 5.0)

    stream = io.BytesIO()
    with rasterio.open(
        stream,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(elevations, 1)

    raw_bytes = stream.getvalue()
    grid = parse_dem(raw_bytes)
    assert grid.values_m.shape == (4, 4)
    assert grid.cell_size_x_m == 5.0
    assert grid.cell_size_y_m == 5.0
    assert grid.bounds[0] == 200000.0
    assert grid.bounds[3] == 3000000.0


def test_parse_geotiff_geographic_crs(tmp_path: Path) -> None:
    # WGS84 geographic coordinates in degrees
    elevations = np.ones((5, 5), dtype=np.float32) * 100.0
    transform = from_origin(10.0, 50.0, 0.001, 0.001)
    tif_path = tmp_path / "geographic.tif"

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elevations, 1)

    grid = parse_geotiff(tif_path)
    # 0.001 degree should convert to roughly ~70-111 meters
    assert 50.0 < grid.cell_size_x_m < 120.0
    assert 100.0 < grid.cell_size_y_m < 120.0
    assert grid.crs == "EPSG:4326"


def test_elevation_grid_coordinate_transformations() -> None:
    elevations = np.zeros((10, 20), dtype=np.float32)
    # Top-left at (500000, 4000000), resolution 10m x 10m
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    bounds = (500000.0, 3999900.0, 500200.0, 4000000.0)

    grid = ElevationGrid(
        values_m=elevations,
        cell_size_x_m=10.0,
        cell_size_y_m=10.0,
        crs="EPSG:32632",
        bounds=bounds,
        transform=transform,
    )

    # Test grid_to_world
    x0, y0 = grid.grid_to_world(0, 0)
    assert np.isclose(x0, 500000.0)
    assert np.isclose(y0, 4000000.0)

    x_last, y_last = grid.grid_to_world(9, 19)
    assert np.isclose(x_last, 500190.0)
    assert np.isclose(y_last, 3999910.0)

    # Test world_to_grid
    r0, c0 = grid.world_to_grid(500000.0, 4000000.0)
    assert (r0, c0) == (0, 0)

    r_last, c_last = grid.world_to_grid(500190.0, 3999910.0)
    assert (r_last, c_last) == (9, 19)

    # Test properties
    assert grid.x_coords.shape == (20,)
    assert grid.y_coords.shape == (10,)
    assert grid.x_m.shape == (10, 20)
    assert grid.y_m.shape == (10, 20)
    assert np.isclose(grid.x_m[0, 0], 500000.0)
    assert np.isclose(grid.y_m[0, 0], 4000000.0)

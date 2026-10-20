import numpy as np

from matplotlib.animation import PillowWriter

from terrain_mapper.simulation import (
    DEFAULT_GRID_SIZE,
    HEIGHTMAP_RELIEF_M,
    TERRAIN_EXTENT_M,
    TerrainConfig,
    animate_scenario,
    build_synthetic_scenario,
)


def test_synthetic_scenario_produces_traversable_route() -> None:
    scenario = build_synthetic_scenario(size=40)

    assert scenario.grid.values_m.shape == (40, 40)
    assert np.isclose(scenario.x_m.min(), 0.0)
    assert np.isclose(scenario.x_m.max(), TERRAIN_EXTENT_M)
    assert np.isclose(scenario.grid.cell_size_x_m, TERRAIN_EXTENT_M / 39)
    assert np.isfinite(scenario.grid.values_m).all()
    assert scenario.path[0] == scenario.start
    assert scenario.path[-1] == scenario.goal
    assert len(scenario.path) > 2
    assert scenario.route_cost > 0


def test_default_synthetic_scenario_uses_expanded_map() -> None:
    scenario = build_synthetic_scenario()

    assert scenario.grid.values_m.shape == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
    assert np.isclose(scenario.x_m[-1, -1], TERRAIN_EXTENT_M)
    assert np.isclose(scenario.grid.cell_size_x_m, TERRAIN_EXTENT_M / (DEFAULT_GRID_SIZE - 1))


def test_heightmap_uses_meter_scale() -> None:
    heightmap = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    scenario = build_synthetic_scenario(size=32, heightmap=heightmap)

    assert np.isclose(float(scenario.grid.values_m.max()), HEIGHTMAP_RELIEF_M)


def test_custom_terrain_config_changes_geometry() -> None:
    standard = build_synthetic_scenario(size=32)
    custom = build_synthetic_scenario(
        size=32,
        config=TerrainConfig(
            slope_steepness=2.5,
            height_variation=3.0,
            obstacle_density=0.10,
            roughness=1.5,
        ),
    )

    assert custom.grid.values_m.shape == (32, 32)
    assert np.isfinite(custom.grid.values_m).all()
    assert np.max(custom.grid.values_m) >= np.max(standard.grid.values_m)


def test_extreme_custom_terrain_reports_unreachable_route(tmp_path) -> None:
    scenario = build_synthetic_scenario(
        size=40,
        config=TerrainConfig(
            slope_steepness=3.0,
            height_variation=4.0,
            obstacle_density=1.0,
            roughness=3.0,
            max_slope_degrees=20.0,
            route_cost_weight=10.0,
        ),
    )

    assert scenario.path == []
    assert np.isinf(scenario.route_cost)
    animation = animate_scenario(scenario)
    output_path = tmp_path / "no-route.gif"
    animation.save(output_path, writer=PillowWriter(fps=1))
    assert output_path.exists()


def test_interactive_editor_applies_brush_edits_and_obstacles() -> None:
    import matplotlib.pyplot as plt

    from terrain_mapper.interactive import InteractiveTerrainEditor

    scenario = build_synthetic_scenario(size=32)
    editor = InteractiveTerrainEditor(scenario, TerrainConfig())
    row, col = editor.path[len(editor.path) // 2]
    original = float(editor.elevation[row, col])
    editor.radius_widget.set_val(2)
    editor.strength_widget.set_val(1.2)
    editor.mode = "Raise"
    editor._paint_cell(row, col)

    assert editor.elevation[row, col] > original
    assert editor.path

    row, col = editor.path[len(editor.path) // 2]
    editor.mode = "Obstacle"
    editor._paint_cell(row, col)
    assert editor.obstacles[row, col]
    assert (row, col) not in editor.path
    editor.animation.event_source.stop()
    plt.close(editor.figure)


def test_build_dem_scenario_from_geotiff(tmp_path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from terrain_mapper.simulation import build_dem_scenario

    elevations = np.linspace(10, 50, 400, dtype=np.float32).reshape((20, 20))
    tif_path = tmp_path / "test_terrain.tif"
    transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(elevations, 1)

    scenario = build_dem_scenario(tif_path)
    assert scenario.grid.bounds[0] == 500000.0
    assert scenario.grid.bounds[3] == 4000000.0
    assert scenario.path
    assert scenario.route_cost > 0
    assert np.isclose(scenario.x_m[0, 0], 500000.0)
    assert np.isclose(scenario.y_m[0, 0], 4000000.0)


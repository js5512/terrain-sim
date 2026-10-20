import numpy as np

from terrain_mapper.config import PlannerConfig
from terrain_mapper.costmap import (
    build_costmap,
    calculate_elevation_contours,
    calculate_slope_steepness,
)
from terrain_mapper.elevation import ElevationGrid


def test_costmap_blocks_steep_and_invalid_cells() -> None:
    elevations = np.zeros((5, 5), dtype=np.float32)
    elevations[:, 3:] = np.array([50, 100], dtype=np.float32)
    elevations[0, 0] = np.nan
    grid = ElevationGrid(elevations, 1.0, 1.0, "EPSG:32610")

    costs = build_costmap(grid, PlannerConfig(max_slope_degrees=30.0))

    assert np.isinf(costs[0, 0])
    assert np.isinf(costs[2, 4])
    assert costs[2, 1] == 1.0


def test_calculate_slope_steepness_accuracy() -> None:
    # 45-degree planar ramp along X: z increases by 1m per 1m run
    x_indices = np.arange(10, dtype=np.float32)
    ramp = np.tile(x_indices, (10, 1))  # dz/dx = 1.0, dz/dy = 0.0 -> slope = 45 degrees
    grid = ElevationGrid(ramp, 1.0, 1.0, "EPSG:32632")

    slopes = calculate_slope_steepness(grid)
    # Interior cells should have exactly 45.0 degrees slope
    interior_slopes = slopes[2:-2, 2:-2]
    assert np.allclose(interior_slopes, 45.0, atol=1e-1)


def test_nan_boundary_cells_avoid_artificial_cliffs() -> None:
    elevations = np.ones((5, 5), dtype=np.float32) * 100.0
    elevations[0, 0] = np.nan  # only corner is NaN
    grid = ElevationGrid(elevations, 1.0, 1.0, "EPSG:32632")

    slopes = calculate_slope_steepness(grid)
    assert np.isnan(slopes[0, 0])
    # Cells adjacent to NaN should evaluate to 0 degrees, not infinite cliffs
    assert np.isclose(slopes[0, 1], 0.0)
    assert np.isclose(slopes[1, 0], 0.0)
    assert np.isclose(slopes[1, 1], 0.0)


def test_calculate_elevation_contours() -> None:
    # Create a cone / hill with concentric contour rings
    y, x = np.mgrid[0:21, 0:21]
    dist_from_center = np.hypot(x - 10, y - 10).astype(np.float32)
    elevation = 100.0 - 4.0 * dist_from_center  # center is 100m, edges ~40m
    bounds = (500000.0, 4000000.0, 500200.0, 4000200.0)
    grid = ElevationGrid(elevation, 10.0, 10.0, "EPSG:32632", bounds=bounds)

    contours = calculate_elevation_contours(grid, interval_m=10.0)
    assert len(contours) > 0

    levels = [c.level for c in contours]
    assert all(40.0 <= lvl <= 100.0 for lvl in levels)

    # Check real-world coordinates of contour paths
    for c in contours:
        for path in c.paths:
            assert path.ndim == 2 and path.shape[1] == 2
            # Ensure coordinates lie within the real-world bounds
            assert np.all(path[:, 0] >= bounds[0] - 1e-2)
            assert np.all(path[:, 0] <= bounds[2] + 1e-2)
            assert np.all(path[:, 1] >= bounds[1] - 1e-2)
            assert np.all(path[:, 1] <= bounds[3] + 1e-2)

    # Check to_dict serialization
    c_dict = contours[0].to_dict()
    assert "level" in c_dict
    assert "paths" in c_dict
    assert isinstance(c_dict["paths"], list)
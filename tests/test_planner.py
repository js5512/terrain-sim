import numpy as np
import pytest

from terrain_mapper.elevation import ElevationGrid
from terrain_mapper.planner import (
    astar,
    find_route,
    find_route_through_waypoints,
    find_route_world,
    grid_path_to_world,
)


def test_route_stitches_optimal_legs_through_waypoints_in_order() -> None:
    costmap = np.ones((7, 7), dtype=np.float32)
    start = (3, 0)
    waypoints = [(0, 3), (3, 6)]
    goal = (6, 3)

    path, total_cost = find_route_through_waypoints(costmap, start, waypoints, goal)

    assert path[0] == start
    assert path[-1] == goal
    previous_index = 0
    for waypoint in waypoints:
        waypoint_index = path.index(waypoint, previous_index + 1)
        previous_index = waypoint_index
    assert all(first != second for first, second in zip(path, path[1:]))

    targets = [*waypoints, goal]
    starts = [start, *waypoints]
    expected_cost = sum(
        find_route(costmap, leg_start, leg_goal)[1]
        for leg_start, leg_goal in zip(starts, targets)
    )
    assert np.isclose(total_cost, expected_cost)


def test_astar_navigates_around_obstacles() -> None:
    # 5x5 grid with a vertical wall of infinite cost in the middle
    costmap = np.ones((5, 5), dtype=np.float32)
    costmap[1:4, 2] = np.inf  # wall at col 2, rows 1..3

    start = (2, 0)
    goal = (2, 4)

    path, cost = astar(costmap, start, goal)
    assert path[0] == start
    assert path[-1] == goal
    # Wall cells must not be in path
    for r in range(1, 4):
        assert (r, 2) not in path
    assert cost > 0


def test_astar_unreachable_raises_value_error() -> None:
    costmap = np.ones((5, 5), dtype=np.float32)
    costmap[:, 2] = np.inf  # full impenetrable wall
    with pytest.raises(ValueError, match="No traversable route found"):
        astar(costmap, (2, 0), (2, 4))


def test_find_route_world_uses_real_world_coordinates() -> None:
    elevations = np.zeros((10, 10), dtype=np.float32)
    grid = ElevationGrid(
        values_m=elevations,
        cell_size_x_m=20.0,
        cell_size_y_m=20.0,
        crs="EPSG:32632",
        bounds=(500000.0, 4000000.0, 500180.0, 4000180.0),
    )
    costmap = np.ones((10, 10), dtype=np.float32)

    start_xy = (500000.0, 4000000.0)
    goal_xy = (500180.0, 4000180.0)

    world_path, grid_path, cost = find_route_world(costmap, grid, start_xy, goal_xy)

    assert grid_path[0] == (0, 0)
    assert grid_path[-1] == (9, 9)
    assert np.isclose(world_path[0][0], start_xy[0])
    assert np.isclose(world_path[0][1], start_xy[1])
    assert np.isclose(world_path[-1][0], goal_xy[0])
    assert np.isclose(world_path[-1][1], goal_xy[1])

    # Convert path back to world manually and verify match
    mapped_world = grid_path_to_world(grid_path, grid)
    assert len(mapped_world) == len(world_path)
    for p1, p2 in zip(mapped_world, world_path):
        assert np.isclose(p1[0], p2[0])
        assert np.isclose(p1[1], p2[1])
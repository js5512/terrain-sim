from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from terrain_mapper.elevation import ElevationGrid


def astar(
    costmap: NDArray[np.float32],
    start: tuple[int, int],
    goal: tuple[int, int],
    cell_size_x: float = 1.0,
    cell_size_y: float = 1.0,
    min_cost: float = 1.0,
) -> tuple[list[tuple[int, int]], float]:
    """Find the optimal 8-connected cell path through a cost surface using A* search."""
    rows, cols = costmap.shape
    for name, (row, column) in (("start", start), ("goal", goal)):
        if not (0 <= row < rows and 0 <= column < cols):
            raise ValueError(f"{name} cell is outside the costmap")
        if not np.isfinite(costmap[row, column]):
            raise ValueError(f"{name} cell is not traversable")

    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])

    if (sr, sc) == (gr, gc):
        return [(sr, sc)], 0.0

    dx = float(cell_size_x)
    dy = float(cell_size_y)
    diag = math.hypot(dx, dy)

    # 8 neighbors and their metric distances
    neighbors = (
        (-1, 0, dy),
        (1, 0, dy),
        (0, -1, dx),
        (0, 1, dx),
        (-1, -1, diag),
        (-1, 1, diag),
        (1, -1, diag),
        (1, 1, diag),
    )

    def heuristic(r: int, c: int) -> float:
        return math.hypot((r - gr) * dy, (c - gc) * dx) * min_cost

    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[sr, sc] = 0.0

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    visited = np.zeros((rows, cols), dtype=bool)

    # Priority queue stores (f_score, g_score, row, col)
    open_heap = [(heuristic(sr, sc), 0.0, sr, sc)]

    while open_heap:
        _, g, r, c = heapq.heappop(open_heap)
        if visited[r, c]:
            continue
        visited[r, c] = True

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = [(r, c)]
            curr = (r, c)
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return path, float(g)

        c_curr = float(costmap[r, c])

        for dr, dc, dist in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if visited[nr, nc]:
                    continue
                c_next = float(costmap[nr, nc])
                if not math.isfinite(c_next):
                    continue

                if dr != 0 and dc != 0:
                    if not math.isfinite(float(costmap[r + dr, c])) or not math.isfinite(float(costmap[r, c + dc])):
                        continue

                step_cost = dist * 0.5 * (c_curr + c_next)
                tentative_g = g + step_cost
                if tentative_g < g_score[nr, nc]:
                    g_score[nr, nc] = tentative_g
                    came_from[(nr, nc)] = (r, c)
                    f_next = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_heap, (f_next, tentative_g, nr, nc))

    raise ValueError("No traversable route found between start and goal")


def find_route(
    costmap: NDArray[np.float32],
    start: tuple[int, int],
    goal: tuple[int, int],
    cell_size_x: float = 1.0,
    cell_size_y: float = 1.0,
) -> tuple[list[tuple[int, int]], float]:
    """Find the least-cost 8-connected cell path through a cost surface using A* search."""
    return astar(
        costmap,
        start,
        goal,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
    )


def find_route_through_waypoints(
    costmap: NDArray[np.float32],
    start: tuple[int, int],
    waypoints: Sequence[tuple[int, int]],
    goal: tuple[int, int],
    cell_size_x: float = 1.0,
    cell_size_y: float = 1.0,
) -> tuple[list[tuple[int, int]], float]:
    """Find and stitch optimal legs through waypoints in the supplied order using A* search."""
    route = [start]
    total_cost = 0.0
    current = start
    for target in (*waypoints, goal):
        leg, leg_cost = find_route(
            costmap,
            current,
            target,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
        )
        route.extend(leg[1:])
        total_cost += leg_cost
        current = target
    return route, total_cost


def grid_path_to_world(
    path: Sequence[tuple[int, int]], grid: ElevationGrid
) -> list[tuple[float, float]]:
    """Convert a sequence of (row, col) grid cells to real-world (x, y) coordinates."""
    return [grid.grid_to_world(r, c) for r, c in path]


def find_route_world(
    costmap: NDArray[np.float32],
    grid: ElevationGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    waypoints_xy: Sequence[tuple[float, float]] = (),
) -> tuple[list[tuple[float, float]], list[tuple[int, int]], float]:
    """Find the optimal route given real-world (x, y) coordinates."""
    start_cell = grid.world_to_grid(start_xy[0], start_xy[1])
    goal_cell = grid.world_to_grid(goal_xy[0], goal_xy[1])
    waypoint_cells = [grid.world_to_grid(x, y) for x, y in waypoints_xy]

    grid_path, cost = find_route_through_waypoints(
        costmap,
        start_cell,
        waypoint_cells,
        goal_cell,
        cell_size_x=grid.cell_size_x_m,
        cell_size_y=grid.cell_size_y_m,
    )
    world_path = grid_path_to_world(grid_path, grid)
    return world_path, grid_path, cost
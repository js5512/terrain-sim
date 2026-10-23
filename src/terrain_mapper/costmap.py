from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.measure import find_contours

from terrain_mapper.config import PlannerConfig
from terrain_mapper.elevation import ElevationGrid


@dataclass(frozen=True)
class ElevationContour:
    """An elevation contour line at a specified elevation height level."""

    level: float
    paths: list[NDArray[np.float32]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "paths": [path.tolist() for path in self.paths],
        }


def calculate_slope_steepness(grid: ElevationGrid) -> NDArray[np.float32]:
    """Calculate slope steepness in degrees for each cell of the elevation grid.

    Uses Horn's weighted 3x3 window method where neighbors are valid, with robust
    adaptive fallbacks for boundary and nodata cells to prevent artificial edge cliffs.
    """
    elevations = grid.values_m.astype(np.float64, copy=False)
    valid = np.isfinite(elevations)
    rows, cols = elevations.shape

    dx = float(grid.cell_size_x_m)
    dy = float(grid.cell_size_y_m)

    # Initialize gradient arrays with zero
    grad_x = np.zeros((rows, cols), dtype=np.float64)
    grad_y = np.zeros((rows, cols), dtype=np.float64)

    # Pad elevations and valid mask with NaN
    pad_z = np.full((rows + 2, cols + 2), np.nan, dtype=np.float64)
    pad_z[1:-1, 1:-1] = elevations

    # 3x3 neighborhood slices
    z00 = pad_z[0:-2, 0:-2]
    z01 = pad_z[0:-2, 1:-1]
    z02 = pad_z[0:-2, 2:]
    z10 = pad_z[1:-1, 0:-2]
    z11 = pad_z[1:-1, 1:-1]
    z12 = pad_z[1:-1, 2:]
    z20 = pad_z[2:, 0:-2]
    z21 = pad_z[2:, 1:-1]
    z22 = pad_z[2:, 2:]

    # Check which cells have full 3x3 valid neighborhood
    full_valid = (
        np.isfinite(z00)
        & np.isfinite(z01)
        & np.isfinite(z02)
        & np.isfinite(z10)
        & np.isfinite(z11)
        & np.isfinite(z12)
        & np.isfinite(z20)
        & np.isfinite(z21)
        & np.isfinite(z22)
    )

    # For full 3x3 neighborhoods, use Horn's algorithm
    if np.any(full_valid):
        grad_x[full_valid] = (
            (z02[full_valid] + 2.0 * z12[full_valid] + z22[full_valid])
            - (z00[full_valid] + 2.0 * z10[full_valid] + z20[full_valid])
        ) / (8.0 * dx)
        grad_y[full_valid] = (
            (z20[full_valid] + 2.0 * z21[full_valid] + z22[full_valid])
            - (z00[full_valid] + 2.0 * z01[full_valid] + z02[full_valid])
        ) / (8.0 * dy)

    # For cells that are valid but have some missing neighbors (edges / nodata boundaries)
    partial_valid = valid & ~full_valid
    if np.any(partial_valid):
        # Adaptive gradient along X:
        has_right = np.isfinite(z12) & partial_valid
        has_left = np.isfinite(z10) & partial_valid
        both_x = has_right & has_left
        only_right = has_right & ~has_left
        only_left = has_left & ~has_right

        grad_x[both_x] = (z12[both_x] - z10[both_x]) / (2.0 * dx)
        grad_x[only_right] = (z12[only_right] - z11[only_right]) / dx
        grad_x[only_left] = (z11[only_left] - z10[only_left]) / dx

        # Adaptive gradient along Y:
        has_down = np.isfinite(z21) & partial_valid
        has_up = np.isfinite(z01) & partial_valid
        both_y = has_down & has_up
        only_down = has_down & ~has_up
        only_up = has_up & ~has_down

        grad_y[both_y] = (z21[both_y] - z01[both_y]) / (2.0 * dy)
        grad_y[only_down] = (z21[only_down] - z11[only_down]) / dy
        grad_y[only_up] = (z11[only_up] - z01[only_up]) / dy

    # Calculate slope in degrees
    slope_radians = np.arctan(np.hypot(grad_x, grad_y))
    slope_degrees = np.degrees(slope_radians)
    slope_degrees[~valid] = np.nan
    return slope_degrees.astype(np.float32)


def calculate_elevation_contours(
    grid: ElevationGrid,
    interval_m: float | None = None,
    num_levels: int = 10,
    levels: Sequence[float] | None = None,
) -> list[ElevationContour]:
    """Calculate elevation contour lines from the elevation data in real-world coordinates.

    Returns a list of ElevationContour objects, each containing the elevation level
    and one or more continuous paths formatted as (N, 2) arrays of (x, y) coordinates.
    """
    elevations = grid.values_m
    valid = np.isfinite(elevations)
    if not valid.any():
        return []

    z_min = float(np.nanmin(elevations))
    z_max = float(np.nanmax(elevations))

    if levels is not None:
        target_levels = [float(lvl) for lvl in levels if z_min <= lvl <= z_max]
    elif interval_m is not None and interval_m > 0:
        start_lvl = math.ceil(z_min / interval_m) * interval_m
        target_levels = []
        curr = start_lvl
        while curr <= z_max:
            target_levels.append(float(curr))
            curr += interval_m
    else:
        if z_max > z_min:
            raw_levels = np.linspace(z_min, z_max, num_levels + 2)[1:-1]
            target_levels = [float(lvl) for lvl in raw_levels]
        else:
            target_levels = [z_min]

    # Transform parameters for grid_to_world mapping
    a, b, c, d, e, f = grid.transform  # type: ignore[misc]

    results: list[ElevationContour] = []
    # Fill NaN for contour finding if necessary
    safe_z = np.where(valid, elevations, np.nanmin(elevations))

    for level in target_levels:
        raw_contours = find_contours(safe_z, level=level)
        world_paths: list[NDArray[np.float32]] = []
        for contour in raw_contours:
            if len(contour) < 2:
                continue
            pts_r = contour[:, 0]
            pts_c = contour[:, 1]
            pts_x = c + pts_c * a + pts_r * b
            pts_y = f + pts_c * d + pts_r * e
            world_paths.append(np.column_stack([pts_x, pts_y]).astype(np.float32))

        if world_paths:
            results.append(ElevationContour(level=float(level), paths=world_paths))

    return results


def build_costmap(
    grid: ElevationGrid, config: PlannerConfig
) -> NDArray[np.float32]:
    """Return per-cell travel costs; NaN and over-limit slopes are blocked."""
    slope_degrees = calculate_slope_steepness(grid)
    valid = np.isfinite(slope_degrees)

    costs = np.full(grid.values_m.shape, np.inf, dtype=np.float32)
    # Travel cost based on tangent of slope angle
    tangent_slope = np.tan(np.radians(slope_degrees[valid]))
    costs[valid] = config.minimum_cost + config.slope_cost_weight * tangent_slope

    # Blocked cells: invalid elevations or slope exceeding maximum threshold
    blocked = ~valid | (slope_degrees > config.max_slope_degrees)

    # Boundary validation: bordering empty space or outside valid dimensions
    invalid = ~np.isfinite(grid.values_m)
    rows, cols = grid.values_m.shape
    pad_invalid = np.full((rows + 2, cols + 2), True, dtype=bool)
    pad_invalid[1:-1, 1:-1] = invalid
    
    borders_empty = (
        pad_invalid[0:-2, 0:-2] | pad_invalid[0:-2, 1:-1] | pad_invalid[0:-2, 2:] |
        pad_invalid[1:-1, 0:-2] | pad_invalid[1:-1, 1:-1] | pad_invalid[1:-1, 2:] |
        pad_invalid[2:, 0:-2] | pad_invalid[2:, 1:-1] | pad_invalid[2:, 2:]
    )
    
    blocked |= borders_empty

    costs[blocked] = np.inf
    return costs.astype(np.float32)
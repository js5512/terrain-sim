from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

from terrain_mapper.elevation import ElevationGrid
from terrain_mapper.navigation.geo import GeoReference, normalize_angle
from terrain_mapper.navigation.types import Waypoint


def ramer_douglas_peucker(
    points: Sequence[Tuple[float, float]], epsilon: float
) -> List[Tuple[float, float]]:
    """Simplify a 2D polyline using the Ramer-Douglas-Peucker (RDP) algorithm."""
    if len(points) <= 2:
        return list(points)

    start = np.array(points[0], dtype=np.float64)
    end = np.array(points[-1], dtype=np.float64)
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    dmax = 0.0
    index = 0

    for i in range(1, len(points) - 1):
        pt = np.array(points[i], dtype=np.float64)
        if line_len < 1e-9:
            dist = float(np.linalg.norm(pt - start))
        else:
            # Perpendicular distance from pt to line segment (start -> end)
            dist = float(np.abs(line_vec[0] * (start[1] - pt[1]) - line_vec[1] * (start[0] - pt[0])) / line_len)

        if dist > dmax:
            index = i
            dmax = dist

    if dmax > epsilon:
        # Recursive split
        rec1 = ramer_douglas_peucker(points[: index + 1], epsilon)
        rec2 = ramer_douglas_peucker(points[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [points[0], points[-1]]


def resample_polyline(
    points: Sequence[Tuple[float, float]],
    max_spacing: float = 10.0,
    min_spacing: float = 1.0,
) -> List[Tuple[float, float]]:
    """Resample a 2D polyline to enforce maximum and minimum distance between points."""
    if len(points) <= 1:
        return list(points)

    result: List[Tuple[float, float]] = [points[0]]

    for i in range(len(points) - 1):
        p1 = np.array(result[-1], dtype=np.float64)
        p2 = np.array(points[i + 1], dtype=np.float64)
        dist = float(np.linalg.norm(p2 - p1))

        if dist < min_spacing and i < len(points) - 2:
            # Skip points that are too close, unless it's the final goal point
            continue

        if dist > max_spacing:
            # Subdivide into intermediate points
            num_segments = int(math.ceil(dist / max_spacing))
            for k in range(1, num_segments):
                t = k / num_segments
                interp = p1 + t * (p2 - p1)
                result.append((float(interp[0]), float(interp[1])))

        result.append((float(p2[0]), float(p2[1])))

    return result


class WaypointGenerator:
    """Generates an optimal sequence of intermediate navigation waypoints

    from a destination coordinate, supporting local coordinates, GPS,
    elevation mapping, and path optimization.
    """

    def __init__(
        self,
        grid: Optional[ElevationGrid] = None,
        geo_ref: Optional[GeoReference] = None,
        default_speed: float = 2.0,
        min_speed: float = 0.5,
        simplification_tolerance: float = 2.0,  # RDP epsilon in meters
        max_spacing: float = 15.0,  # Max distance between waypoints in meters
        min_spacing: float = 1.5,  # Min distance between waypoints in meters
        default_acceptance_radius: float = 1.5,
    ) -> None:
        self.grid = grid
        self.geo_ref = geo_ref
        self.default_speed = float(default_speed)
        self.min_speed = float(min_speed)
        self.simplification_tolerance = float(simplification_tolerance)
        self.max_spacing = float(max_spacing)
        self.min_spacing = float(min_spacing)
        self.default_acceptance_radius = float(default_acceptance_radius)

    def sample_elevation(self, x: float, y: float) -> float:
        """Sample elevation at local coordinate (x, y) if grid is available."""
        if self.grid is None:
            return 0.0
        try:
            r, c = self.grid.world_to_grid(x, y)
            val = float(self.grid.values_m[r, c])
            return val if math.isfinite(val) else 0.0
        except Exception:
            return 0.0

    def generate_from_path(
        self,
        path_xy: Sequence[Tuple[float, float]],
        target_speed: Optional[float] = None,
    ) -> List[Waypoint]:
        """Convert a dense 2D path sequence (e.g. from A* planner) into a sparse,

        smooth sequence of intermediate waypoints with speed and heading profiling.
        """
        if not path_xy:
            return []
        if len(path_xy) == 1:
            x, y = path_xy[0]
            z = self.sample_elevation(x, y)
            lat, lon, alt = None, None, None
            if self.geo_ref:
                lat, lon, alt = self.geo_ref.enu_to_gps(x, y, z)
            return [
                Waypoint(
                    x=x,
                    y=y,
                    z=z,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                    target_speed=0.0,
                    heading=0.0,
                    acceptance_radius=self.default_acceptance_radius,
                )
            ]

        # 1. Simplify path geometry using Ramer-Douglas-Peucker
        simplified = ramer_douglas_peucker(path_xy, self.simplification_tolerance)

        # 2. Resample to maintain spacing bounds
        resampled = resample_polyline(
            simplified, max_spacing=self.max_spacing, min_spacing=self.min_spacing
        )

        base_speed = target_speed if target_speed is not None else self.default_speed
        waypoints: List[Waypoint] = []
        n = len(resampled)

        for i, (x, y) in enumerate(resampled):
            # Calculate heading to next point or from previous point
            if i < n - 1:
                next_x, next_y = resampled[i + 1]
                heading = math.atan2(next_y - y, next_x - x)
            else:
                prev_x, prev_y = resampled[i - 1]
                heading = math.atan2(y - prev_y, x - prev_x)

            # Curvature-based speed profiling: slow down for sharp turns
            speed = base_speed
            if 0 < i < n - 1:
                prev_x, prev_y = resampled[i - 1]
                next_x, next_y = resampled[i + 1]
                h_in = math.atan2(y - prev_y, x - prev_x)
                h_out = math.atan2(next_y - y, next_x - x)
                turn_angle = abs(normalize_angle(h_out - h_in))
                # Reduce speed proportionally to turn sharpness
                speed_factor = max(0.2, math.cos(turn_angle / 2.0))
                speed = max(self.min_speed, base_speed * speed_factor)

            # Final goal approaches zero velocity
            if i == n - 1:
                speed = 0.0

            z = self.sample_elevation(x, y)

            lat, lon, alt = None, None, None
            if self.geo_ref:
                lat, lon, alt = self.geo_ref.enu_to_gps(x, y, z)

            waypoints.append(
                Waypoint(
                    x=x,
                    y=y,
                    z=z,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                    target_speed=speed,
                    heading=heading,
                    acceptance_radius=self.default_acceptance_radius,
                )
            )

        return waypoints

    def generate_waypoints(
        self,
        start_xy: Tuple[float, float],
        destination_xy: Tuple[float, float],
        costmap: Optional[NDArray[np.float32]] = None,
        target_speed: Optional[float] = None,
    ) -> List[Waypoint]:
        """Generate a sequence of waypoints from start to destination local coordinates.

        If a costmap and ElevationGrid are present, leverages A* trajectory planning.
        Otherwise, generates a direct path with intermediate waypoints.
        """
        if costmap is not None and self.grid is not None:
            from terrain_mapper.planner import find_route_world

            world_path, _, _ = find_route_world(
                costmap=costmap,
                grid=self.grid,
                start_xy=start_xy,
                goal_xy=destination_xy,
            )
            return self.generate_from_path(world_path, target_speed=target_speed)

        # Free space direct line generation with uniform intermediate spacing
        dx = destination_xy[0] - start_xy[0]
        dy = destination_xy[1] - start_xy[1]
        dist = math.hypot(dx, dy)

        if dist < 1e-3:
            return self.generate_from_path([start_xy], target_speed=target_speed)

        num_segments = max(1, int(math.ceil(dist / self.max_spacing)))
        path = [
            (
                start_xy[0] + (i / num_segments) * dx,
                start_xy[1] + (i / num_segments) * dy,
            )
            for i in range(num_segments + 1)
        ]
        return self.generate_from_path(path, target_speed=target_speed)

    def generate_from_gps(
        self,
        start_gps: Tuple[float, float],
        destination_gps: Tuple[float, float],
        costmap: Optional[NDArray[np.float32]] = None,
        target_speed: Optional[float] = None,
    ) -> List[Waypoint]:
        """Generate waypoints from GPS coordinates (lat, lon)."""
        if self.geo_ref is None:
            # Automatically establish origin at start_gps
            self.geo_ref = GeoReference(origin_lat=start_gps[0], origin_lon=start_gps[1])

        start_x, start_y, _ = self.geo_ref.gps_to_enu(start_gps[0], start_gps[1])
        dest_x, dest_y, _ = self.geo_ref.gps_to_enu(destination_gps[0], destination_gps[1])

        return self.generate_waypoints(
            start_xy=(start_x, start_y),
            destination_xy=(dest_x, dest_y),
            costmap=costmap,
            target_speed=target_speed,
        )

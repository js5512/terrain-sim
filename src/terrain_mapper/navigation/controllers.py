from __future__ import annotations

import abc
import math
from typing import Optional, Sequence, Tuple
import numpy as np

from terrain_mapper.navigation.geo import normalize_angle
from terrain_mapper.navigation.types import ControlCommand, VehicleState, Waypoint


class BaseController(abc.ABC):
    """Abstract base class for path-tracking / steering controllers."""

    @abc.abstractmethod
    def compute_control(
        self,
        state: VehicleState,
        waypoints: Sequence[Waypoint],
        active_index: int = 0,
    ) -> ControlCommand:
        """Compute steering and velocity commands to track waypoints."""
        pass


class PurePursuitController(BaseController):
    """Adaptive Pure Pursuit path-tracking controller.

    Computes steering based on lookahead distance geometric arc.
    Suitable for Ackermann steering and differential-drive platforms.
    """

    def __init__(
        self,
        lookahead_base: float = 2.5,
        lookahead_min: float = 1.2,
        lookahead_max: float = 15.0,
        k_lookahead: float = 0.8,  # Lookahead distance velocity gain (s)
        max_steering_angle: float = math.radians(35.0),  # radians (~35 deg)
        wheelbase: float = 2.0,  # meters
    ) -> None:
        self.lookahead_base = float(lookahead_base)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_max = float(lookahead_max)
        self.k_lookahead = float(k_lookahead)
        self.max_steering_angle = float(max_steering_angle)
        self.wheelbase = float(wheelbase)

    def calculate_lookahead_distance(self, speed: float) -> float:
        """Calculate dynamic lookahead distance scaled by current speed."""
        ld = self.lookahead_base + self.k_lookahead * abs(speed)
        return float(np.clip(ld, self.lookahead_min, self.lookahead_max))

    def find_lookahead_point(
        self,
        state: VehicleState,
        waypoints: Sequence[Waypoint],
        active_index: int,
        lookahead_dist: float,
    ) -> Tuple[Tuple[float, float], int]:
        """Find the target point on the path at lookahead_dist ahead of the vehicle."""
        if not waypoints:
            return (state.x, state.y), 0

        if active_index >= len(waypoints) - 1:
            last = waypoints[-1]
            return (last.x, last.y), len(waypoints) - 1

        vx, vy = state.x, state.y
        best_point = (waypoints[active_index].x, waypoints[active_index].y)
        best_index = active_index

        # Search forward along line segments between waypoints
        for i in range(active_index, len(waypoints) - 1):
            p1 = np.array([waypoints[i].x, waypoints[i].y], dtype=np.float64)
            p2 = np.array([waypoints[i + 1].x, waypoints[i + 1].y], dtype=np.float64)
            seg = p2 - p1
            seg_len = float(np.linalg.norm(seg))

            if seg_len < 1e-6:
                continue

            # Vector from p1 to vehicle
            v_p1 = np.array([vx, vy], dtype=np.float64) - p1

            # Intersection between circle centered at (vx, vy) with radius lookahead_dist
            # and line segment p1 -> p2
            # Solve || p1 + t * seg - v ||^2 = lookahead_dist^2
            a = float(np.dot(seg, seg))
            b = 2.0 * float(np.dot(-v_p1, seg))
            c = float(np.dot(v_p1, v_p1)) - lookahead_dist**2

            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)

                # Pick the forward intersection on the segment (t in [0, 1])
                t_cand = max(t for t in (t1, t2) if 0.0 <= t <= 1.0) if any(0.0 <= t <= 1.0 for t in (t1, t2)) else None
                if t_cand is not None:
                    interp = p1 + t_cand * seg
                    return (float(interp[0]), float(interp[1])), i + 1

            # Check distance to p2
            d2 = math.hypot(p2[0] - vx, p2[1] - vy)
            if d2 >= lookahead_dist:
                return (float(p2[0]), float(p2[1])), i + 1
            else:
                best_point = (float(p2[0]), float(p2[1]))
                best_index = i + 1

        return best_point, best_index

    def compute_control(
        self,
        state: VehicleState,
        waypoints: Sequence[Waypoint],
        active_index: int = 0,
    ) -> ControlCommand:
        """Calculate Ackermann steering angle and differential drive commands."""
        if not waypoints:
            return ControlCommand(
                steering_angle=0.0,
                target_velocity=0.0,
                angular_velocity=0.0,
                curvature=0.0,
                cross_track_error=0.0,
                heading_error=0.0,
                lookahead_point=(state.x, state.y),
                target_waypoint_index=0,
            )

        lookahead_dist = self.calculate_lookahead_distance(state.v)
        (tx, ty), target_idx = self.find_lookahead_point(
            state, waypoints, active_index, lookahead_dist
        )

        # Vector to target in world frame
        dx = tx - state.x
        dy = ty - state.y
        dist_to_target = max(math.hypot(dx, dy), 1e-4)

        # Alpha is the heading difference to target in vehicle body frame
        target_heading = math.atan2(dy, dx)
        alpha = normalize_angle(target_heading - state.yaw)

        # Curvature kappa = 2 * sin(alpha) / lookahead_distance
        effective_ld = max(dist_to_target, self.lookahead_min)
        curvature = (2.0 * math.sin(alpha)) / effective_ld

        # Bicycle model steering angle delta = atan(2 * L * sin(alpha) / Ld)
        wheelbase = state.wheelbase or self.wheelbase
        raw_steering = math.atan2(2.0 * wheelbase * math.sin(alpha), effective_ld)
        steering_angle = float(np.clip(raw_steering, -self.max_steering_angle, self.max_steering_angle))

        # Target velocity from active waypoint
        target_wp = waypoints[min(active_index, len(waypoints) - 1)]
        target_velocity = target_wp.target_speed

        # Differential drive / unicycle angular velocity omega = v * kappa
        angular_velocity = float(target_velocity * curvature)

        # Cross-track error estimate: perpendicular distance to segment
        cte = dist_to_target * math.sin(alpha)

        return ControlCommand(
            steering_angle=steering_angle,
            target_velocity=target_velocity,
            angular_velocity=angular_velocity,
            curvature=curvature,
            cross_track_error=float(cte),
            heading_error=float(alpha),
            lookahead_point=(tx, ty),
            target_waypoint_index=target_idx,
        )


class StanleyController(BaseController):
    """Stanley cross-track and heading path-tracking controller.

    Uses front-axle kinematics to minimize lateral cross-track error
    and heading alignment error.
    """

    def __init__(
        self,
        k_gain: float = 1.0,  # Cross-track error gain
        k_soft: float = 1.0,  # Low-speed softening constant (m/s)
        max_steering_angle: float = math.radians(35.0),
        wheelbase: float = 2.0,
    ) -> None:
        self.k_gain = float(k_gain)
        self.k_soft = float(k_soft)
        self.max_steering_angle = float(max_steering_angle)
        self.wheelbase = float(wheelbase)

    def compute_control(
        self,
        state: VehicleState,
        waypoints: Sequence[Waypoint],
        active_index: int = 0,
    ) -> ControlCommand:
        """Calculate front-wheel steering command using the Stanley algorithm."""
        if not waypoints:
            return ControlCommand(
                steering_angle=0.0,
                target_velocity=0.0,
                angular_velocity=0.0,
                curvature=0.0,
                cross_track_error=0.0,
                heading_error=0.0,
                lookahead_point=(state.x, state.y),
                target_waypoint_index=0,
            )

        wheelbase = state.wheelbase or self.wheelbase
        # Front axle position
        fx = state.x + wheelbase * math.cos(state.yaw)
        fy = state.y + wheelbase * math.sin(state.yaw)

        # Find closest path segment to front axle
        idx = max(0, min(active_index, len(waypoints) - 1))
        if idx >= len(waypoints) - 1 and len(waypoints) > 1:
            idx = len(waypoints) - 2

        p1 = waypoints[idx]
        p2 = waypoints[min(idx + 1, len(waypoints) - 1)]

        dx = p2.x - p1.x
        dy = p2.y - p1.y
        path_heading = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else p1.heading or state.yaw

        # Heading error = path_heading - vehicle_yaw
        theta_e = normalize_angle(path_heading - state.yaw)

        # Signed cross-track error from front axle to line segment (p1 -> p2)
        # Normal vector pointing to the left of the path
        nx = -math.sin(path_heading)
        ny = math.cos(path_heading)
        d_fx = fx - p1.x
        d_fy = fy - p1.y
        cross_track_error = d_fx * nx + d_fy * ny

        # Stanley steering law:
        # Cross-track error vector from path to front axle.
        # If vehicle is to the left of path (cross_track_error > 0), steering must steer right (< 0).
        speed = max(0.0, state.v)
        crosstrack_steering = math.atan2(-self.k_gain * cross_track_error, speed + self.k_soft)
        raw_steering = theta_e + crosstrack_steering
        steering_angle = float(np.clip(raw_steering, -self.max_steering_angle, self.max_steering_angle))

        target_wp = waypoints[min(active_index, len(waypoints) - 1)]
        target_velocity = target_wp.target_speed

        curvature = math.tan(steering_angle) / wheelbase if wheelbase > 1e-3 else 0.0
        angular_velocity = float(target_velocity * curvature)

        return ControlCommand(
            steering_angle=steering_angle,
            target_velocity=target_velocity,
            angular_velocity=angular_velocity,
            curvature=curvature,
            cross_track_error=float(cross_track_error),
            heading_error=float(theta_e),
            lookahead_point=(p2.x, p2.y),
            target_waypoint_index=min(idx + 1, len(waypoints) - 1),
        )

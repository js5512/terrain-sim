from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Tuple

from terrain_mapper.navigation.geo import normalize_angle


class NavigationState(str, Enum):
    """Lifecycle states of the autonomous navigation executive."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    NAVIGATING = "NAVIGATING"
    WAYPOINT_REACHED = "WAYPOINT_REACHED"
    GOAL_REACHED = "GOAL_REACHED"
    OBSTACLE_DETECTED = "OBSTACLE_DETECTED"
    GPS_LOST = "GPS_LOST"
    ERROR = "ERROR"


class HeadingSource(str, Enum):
    """Source used to estimate vehicle orientation."""
    MANUAL = "MANUAL"
    GPS = "GPS"
    IMU = "IMU"
    LOCKED = "LOCKED"


@dataclass(frozen=True)
class Waypoint:
    """Represents an active navigation waypoint in local and/or geodetic coordinates."""
    x: float
    y: float
    z: float = 0.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    target_speed: float = 1.0  # m/s
    heading: Optional[float] = None  # Expected tangent heading in radians
    acceptance_radius: float = 1.5  # meters

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class HeadingEstimator:
    """Dynamic heading/yaw estimator using consecutive GPS positions and IMU orientation stream fallback.

    When the platform moves (speed and displacement exceed noise thresholds), heading is
    derived from consecutive Cartesian GPS displacements with phase-unwrapped low-pass filtering.
    When stationary or moving below threshold, it falls back to an active IMU orientation stream,
    or locks the last known heading to avoid erratic sensor jitter spins.
    """

    def __init__(
        self,
        min_moving_speed: float = 0.3,  # m/s
        min_displacement: float = 0.15,  # meters
        smoothing_alpha: float = 0.35,  # angular EMA filter factor (0 < alpha <= 1.0)
        initial_yaw: float = 0.0,
    ) -> None:
        self.min_moving_speed = float(min_moving_speed)
        self.min_displacement = float(min_displacement)
        self.smoothing_alpha = float(min(1.0, max(0.01, smoothing_alpha)))
        self.yaw: float = normalize_angle(float(initial_yaw))
        self.heading_source: HeadingSource = HeadingSource.MANUAL

        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._has_acquired_gps_heading: bool = False

    def reset(self, initial_yaw: float = 0.0) -> None:
        """Reset internal heading state."""
        self.yaw = normalize_angle(float(initial_yaw))
        self.heading_source = HeadingSource.MANUAL
        self._last_x = None
        self._last_y = None
        self._last_timestamp = None
        self._has_acquired_gps_heading = False

    def estimate(
        self,
        x: float,
        y: float,
        speed: Optional[float] = None,
        timestamp: Optional[float] = None,
        imu_yaw: Optional[float] = None,
    ) -> Tuple[float, HeadingSource]:
        """Estimate vehicle heading given current position, speed, and optional IMU stream."""
        # Initial point anchor
        if self._last_x is None or self._last_y is None:
            self._last_x = float(x)
            self._last_y = float(y)
            self._last_timestamp = timestamp
            if imu_yaw is not None:
                self.yaw = normalize_angle(float(imu_yaw))
                self.heading_source = HeadingSource.IMU
            else:
                self.heading_source = HeadingSource.MANUAL
            return (self.yaw, self.heading_source)

        dx = x - self._last_x
        dy = y - self._last_y
        displacement = math.hypot(dx, dy)

        dt: Optional[float] = None
        if timestamp is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp

        if speed is not None:
            effective_speed = speed
        elif dt is not None and dt > 1e-4:
            effective_speed = displacement / dt
        else:
            effective_speed = 0.0

        # Determine motion condition
        is_moving = displacement >= self.min_displacement and (
            effective_speed >= self.min_moving_speed or speed is None
        )

        if is_moving:
            # Dynamic GPS heading calculation from Cartesian displacement vector
            gps_heading = math.atan2(dy, dx)
            if not self._has_acquired_gps_heading and self.heading_source == HeadingSource.MANUAL:
                # First motion acquisition establishes initial heading immediately
                self.yaw = gps_heading
                self._has_acquired_gps_heading = True
            else:
                # Angular low-pass filter accounting for [-pi, pi] wrap-around
                angular_err = normalize_angle(gps_heading - self.yaw)
                self.yaw = normalize_angle(self.yaw + self.smoothing_alpha * angular_err)

            self.heading_source = HeadingSource.GPS
        else:
            # Stationary or sub-threshold displacement
            if imu_yaw is not None:
                # Fallback to IMU stream
                self.yaw = normalize_angle(float(imu_yaw))
                self.heading_source = HeadingSource.IMU
            else:
                # Preserve locked heading to prevent jitter spin
                self.heading_source = HeadingSource.LOCKED

        self._last_x = float(x)
        self._last_y = float(y)
        self._last_timestamp = timestamp

        return (self.yaw, self.heading_source)


@dataclass
class VehicleState:
    """Kinematic state of the mobile platform / autonomous ground vehicle."""
    x: float
    y: float
    yaw: float  # Heading in radians (counter-clockwise from East / +X axis)
    v: float = 0.0  # Longitudinal velocity in m/s
    yaw_rate: float = 0.0  # Angular velocity in rad/s
    z: float = 0.0
    wheelbase: float = 2.0  # Distance between front and rear axles (m)
    imu_yaw: Optional[float] = None  # Optional IMU stream orientation in radians
    timestamp: float = 0.0
    heading_source: HeadingSource = HeadingSource.MANUAL
    _heading_estimator: Optional[HeadingEstimator] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._heading_estimator is None:
            self._heading_estimator = HeadingEstimator(initial_yaw=self.yaw)

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def front_axle(self) -> Tuple[float, float]:
        """Front axle center position in world coordinates."""
        fx = self.x + self.wheelbase * math.cos(self.yaw)
        fy = self.y + self.wheelbase * math.sin(self.yaw)
        return (fx, fy)

    def update_from_gps(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        speed: Optional[float] = None,
        timestamp: Optional[float] = None,
        imu_yaw: Optional[float] = None,
    ) -> VehicleState:
        """Dynamically update position, velocity, and heading from consecutive GPS positions or IMU stream."""
        if self._heading_estimator is None:
            self._heading_estimator = HeadingEstimator(initial_yaw=self.yaw)

        # Update heading estimator
        new_yaw, source = self._heading_estimator.estimate(
            x=x,
            y=y,
            speed=speed,
            timestamp=timestamp,
            imu_yaw=imu_yaw,
        )

        dt = (timestamp - self.timestamp) if (timestamp is not None and self.timestamp > 0.0) else None
        if dt is not None and dt > 1e-4:
            self.yaw_rate = normalize_angle(new_yaw - self.yaw) / dt

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = new_yaw
        if speed is not None:
            self.v = float(speed)
        self.imu_yaw = imu_yaw
        if timestamp is not None:
            self.timestamp = float(timestamp)
        self.heading_source = source

        return self


@dataclass(frozen=True)
class ControlCommand:
    """Actuator control command emitted by the navigation controller."""
    steering_angle: float  # Front wheel steering angle (rad), positive = left
    target_velocity: float  # Longitudinal target speed (m/s)
    angular_velocity: float  # Curvature * velocity (rad/s) for differential drive
    curvature: float  # Path curvature kappa = 1 / R (1/m)
    cross_track_error: float  # Lateral distance to path (m)
    heading_error: float  # Orientation error relative to path (rad)
    lookahead_point: Tuple[float, float]
    target_waypoint_index: int


@dataclass(frozen=True)
class NavigationResult:
    """Status emitted at each step of the real-time navigation loop."""
    state: NavigationState
    command: Optional[ControlCommand]
    active_waypoint_index: int
    total_waypoints: int
    distance_to_active: float
    distance_to_goal: float
    progress_fraction: float

from __future__ import annotations

import math
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from terrain_mapper.navigation.controllers import (
    BaseController,
    PurePursuitController,
)
from terrain_mapper.navigation.geo import GeoReference, normalize_angle
from terrain_mapper.navigation.gps_source import (
    FilteredPosition,
    GPSFix,
    GPSQuality,
    GPSSource,
    parse_nmea_sentence,
)
from terrain_mapper.navigation.types import (
    ControlCommand,
    HeadingEstimator,
    HeadingSource,
    NavigationResult,
    NavigationState,
    VehicleState,
    Waypoint,
)
from terrain_mapper.navigation.waypoint_generator import WaypointGenerator


class Navigator:
    """Autonomous navigation executive, mission state machine, and closed-loop GPS tracker.

    Monitors vehicle position, checks waypoint proximity and boundary crossing,
    dynamically switches waypoints, and executes path-tracking control (Pure Pursuit / Stanley).
    Fully supports live GPS telemetry streams, coordinate transformations, jitter filtering,
    dynamic heading estimation, and safety watchdogs for signal loss and coordinate drift.
    """

    def __init__(
        self,
        controller: Optional[BaseController] = None,
        goal_tolerance: float = 1.0,  # Goal acceptance radius in meters
        waypoint_tolerance: float = 1.5,  # Intermediate waypoint radius in meters
        geo_ref: Optional[GeoReference] = None,
        gps_source: Optional[GPSSource] = None,
        waypoint_generator: Optional[WaypointGenerator] = None,
        vehicle_state: Optional[VehicleState] = None,
        max_gps_stale_time: float = 2.5,
    ) -> None:
        self.controller: BaseController = controller or PurePursuitController()
        self.goal_tolerance = float(goal_tolerance)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.max_gps_stale_time = float(max_gps_stale_time)

        self.geo_ref: Optional[GeoReference] = geo_ref
        self.gps_source: GPSSource = gps_source or GPSSource(
            geo_ref=self.geo_ref, signal_timeout=self.max_gps_stale_time
        )
        if self.geo_ref is not None:
            self.gps_source.set_geo_reference(self.geo_ref)

        self.waypoint_generator: WaypointGenerator = (
            waypoint_generator or WaypointGenerator(geo_ref=self.geo_ref)
        )
        self.vehicle_state: VehicleState = vehicle_state or VehicleState(
            x=0.0, y=0.0, yaw=0.0
        )

        self._state: NavigationState = NavigationState.IDLE
        self._waypoints: List[Waypoint] = []
        self._active_index: int = 0
        self._total_distance: float = 0.0

        # Pending GPS destination coordinates for cold-start anchoring
        self._pending_gps_destination: Optional[Tuple[float, float, float]] = None
        self._pending_costmap: Optional[NDArray[np.float32]] = None
        self._pending_target_speed: Optional[float] = None

    @property
    def state(self) -> NavigationState:
        return self._state

    @property
    def waypoints(self) -> List[Waypoint]:
        return list(self._waypoints)

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def is_active(self) -> bool:
        return self._state in {
            NavigationState.NAVIGATING,
            NavigationState.WAYPOINT_REACHED,
        }

    def set_waypoints(self, waypoints: Sequence[Waypoint]) -> None:
        """Load a new sequence of waypoints and reset navigation state."""
        self._waypoints = list(waypoints)
        self._active_index = 0
        if not self._waypoints:
            self._state = NavigationState.IDLE
            self._total_distance = 0.0
            return

        self._state = NavigationState.NAVIGATING
        # Compute total route length for progress calculations
        dist = 0.0
        for i in range(len(self._waypoints) - 1):
            p1 = self._waypoints[i]
            p2 = self._waypoints[i + 1]
            dist += math.hypot(p2.x - p1.x, p2.y - p1.y)
        self._total_distance = max(dist, 1.0)

    def set_gps_destination(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0,
        current_gps: Optional[Tuple[float, float]] = None,
        costmap: Optional[NDArray[np.float32]] = None,
        target_speed: Optional[float] = None,
    ) -> List[Waypoint]:
        """Accept a global GPS destination (lat, lon), dynamically generate the waypoint array,

        and prime the navigation executive.

        If current_gps is provided or geo_ref datum is already initialized, waypoints are
        computed immediately. Otherwise, destination is registered and computed automatically
        upon the first valid live GPS fix.
        """
        # Determine origin datum
        origin_known = False
        start_coord: Optional[Tuple[float, float]] = None

        if current_gps is not None:
            start_coord = (current_gps[0], current_gps[1])
            if self.geo_ref is None:
                self.geo_ref = GeoReference(
                    origin_lat=current_gps[0],
                    origin_lon=current_gps[1],
                    origin_alt=alt,
                )
                self.gps_source.set_geo_reference(self.geo_ref)
                self.waypoint_generator.geo_ref = self.geo_ref
            origin_known = True
        elif self.geo_ref is not None:
            # Derive starting GPS from current vehicle state
            curr_lat, curr_lon, _ = self.geo_ref.enu_to_gps(
                self.vehicle_state.x, self.vehicle_state.y, self.vehicle_state.z
            )
            start_coord = (curr_lat, curr_lon)
            origin_known = True

        if origin_known and start_coord is not None:
            self.waypoint_generator.geo_ref = self.geo_ref
            waypoints = self.waypoint_generator.generate_from_gps(
                start_gps=start_coord,
                destination_gps=(lat, lon),
                costmap=costmap,
                target_speed=target_speed,
            )
            self.set_waypoints(waypoints)
            self._pending_gps_destination = None
            self._pending_costmap = None
            self._pending_target_speed = None
            return waypoints
        else:
            # Register pending destination to be planned when first valid GPS fix arrives
            self._pending_gps_destination = (float(lat), float(lon), float(alt))
            self._pending_costmap = costmap
            self._pending_target_speed = target_speed
            self._state = NavigationState.PLANNING
            return []

    def reset(self) -> None:
        """Reset state machine, GPS source, and waypoints to IDLE."""
        self._state = NavigationState.IDLE
        self._waypoints = []
        self._active_index = 0
        self._total_distance = 0.0
        self._pending_gps_destination = None
        self._pending_costmap = None
        self._pending_target_speed = None
        self.gps_source.reset()

    def _has_passed_waypoint(
        self,
        vehicle_x: float,
        vehicle_y: float,
        wp_curr: Waypoint,
        wp_next: Waypoint,
    ) -> bool:
        """Check if vehicle has crossed the normal plane perpendicular to the path at wp_curr.

        Prevents vehicle circling back if an intermediate waypoint is slightly clipped.
        """
        seg_x = wp_next.x - wp_curr.x
        seg_y = wp_next.y - wp_curr.y
        seg_len_sq = seg_x**2 + seg_y**2
        if seg_len_sq < 1e-4:
            return False

        # Vector from wp_curr to vehicle
        vec_x = vehicle_x - wp_curr.x
        vec_y = vehicle_y - wp_curr.y

        # Dot product with forward segment vector
        projection = vec_x * seg_x + vec_y * seg_y
        return projection >= 0.0 and math.hypot(vec_x, vec_y) < self.waypoint_tolerance * 2.5

    def step(self, vehicle: VehicleState) -> NavigationResult:
        """Execute one cycle of the navigation executive using an explicitly provided VehicleState."""
        if not self._waypoints or self._state in {
            NavigationState.IDLE,
            NavigationState.PLANNING,
            NavigationState.GPS_LOST,
            NavigationState.ERROR,
        }:
            return NavigationResult(
                state=self._state,
                command=None,
                active_waypoint_index=0,
                total_waypoints=len(self._waypoints),
                distance_to_active=0.0,
                distance_to_goal=0.0,
                progress_fraction=0.0,
            )

        if self._state == NavigationState.GOAL_REACHED:
            # Maintain zero velocity command at goal
            last_wp = self._waypoints[-1]
            d_goal = math.hypot(last_wp.x - vehicle.x, last_wp.y - vehicle.y)
            return NavigationResult(
                state=NavigationState.GOAL_REACHED,
                command=ControlCommand(
                    steering_angle=0.0,
                    target_velocity=0.0,
                    angular_velocity=0.0,
                    curvature=0.0,
                    cross_track_error=0.0,
                    heading_error=0.0,
                    lookahead_point=(last_wp.x, last_wp.y),
                    target_waypoint_index=len(self._waypoints) - 1,
                ),
                active_waypoint_index=len(self._waypoints) - 1,
                total_waypoints=len(self._waypoints),
                distance_to_active=d_goal,
                distance_to_goal=d_goal,
                progress_fraction=1.0,
            )

        current_wp = self._waypoints[self._active_index]
        dist_to_active = math.hypot(current_wp.x - vehicle.x, current_wp.y - vehicle.y)

        goal_wp = self._waypoints[-1]
        dist_to_goal = math.hypot(goal_wp.x - vehicle.x, goal_wp.y - vehicle.y)

        # Check for terminal goal arrival
        if dist_to_goal <= max(goal_wp.acceptance_radius, self.goal_tolerance):
            self._state = NavigationState.GOAL_REACHED
            self._active_index = len(self._waypoints) - 1
            return NavigationResult(
                state=NavigationState.GOAL_REACHED,
                command=ControlCommand(
                    steering_angle=0.0,
                    target_velocity=0.0,
                    angular_velocity=0.0,
                    curvature=0.0,
                    cross_track_error=0.0,
                    heading_error=0.0,
                    lookahead_point=(goal_wp.x, goal_wp.y),
                    target_waypoint_index=self._active_index,
                ),
                active_waypoint_index=self._active_index,
                total_waypoints=len(self._waypoints),
                distance_to_active=dist_to_goal,
                distance_to_goal=dist_to_goal,
                progress_fraction=1.0,
            )

        # Dynamic Waypoint Switching Logic
        is_last_waypoint = self._active_index >= len(self._waypoints) - 1

        if not is_last_waypoint:
            # Check intermediate waypoint arrival (proximity OR pass-by)
            next_wp = self._waypoints[self._active_index + 1]
            within_radius = dist_to_active <= max(current_wp.acceptance_radius, self.waypoint_tolerance)
            passed_boundary = self._has_passed_waypoint(vehicle.x, vehicle.y, current_wp, next_wp)

            if within_radius or passed_boundary:
                self._active_index += 1
                self._state = NavigationState.WAYPOINT_REACHED
                # Re-evaluate distance to new active waypoint
                current_wp = self._waypoints[self._active_index]
                dist_to_active = math.hypot(current_wp.x - vehicle.x, current_wp.y - vehicle.y)
            else:
                self._state = NavigationState.NAVIGATING
        else:
            self._state = NavigationState.NAVIGATING

        # Calculate tracking command
        cmd = self.controller.compute_control(
            state=vehicle,
            waypoints=self._waypoints,
            active_index=self._active_index,
        )

        # Smooth deceleration when tracking the final approach segment
        if is_last_waypoint and cmd.target_velocity <= 0.0:
            approach_speed = max(0.5, min(2.0, dist_to_goal * 0.8))
            cmd = ControlCommand(
                steering_angle=cmd.steering_angle,
                target_velocity=approach_speed,
                angular_velocity=approach_speed * cmd.curvature,
                curvature=cmd.curvature,
                cross_track_error=cmd.cross_track_error,
                heading_error=cmd.heading_error,
                lookahead_point=cmd.lookahead_point,
                target_waypoint_index=cmd.target_waypoint_index,
            )

        progress = float(self._active_index / max(1, len(self._waypoints) - 1))

        return NavigationResult(
            state=self._state,
            command=cmd,
            active_waypoint_index=self._active_index,
            total_waypoints=len(self._waypoints),
            distance_to_active=dist_to_active,
            distance_to_goal=dist_to_goal,
            progress_fraction=progress,
        )

    def step_gps(
        self,
        gps_data: Union[str, GPSFix, Tuple[float, float], Tuple[float, float, float]],
        imu_yaw: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> NavigationResult:
        """Execute one closed-loop cycle using live GPS telemetry and optional IMU orientation.

        Ingests raw NMEA sentences, GPSFix objects, or coordinates, runs them through the
        transformation and filtering pipeline, updates the vehicle pose and dynamic heading estimator,
        and computes the path-tracking control command.

        Safely handles GPS signal loss (timeout or degraded fixes) by transitioning to GPS_LOST
        and issuing a fail-safe zero-velocity command. Also mitigates multipath coordinate jumps.
        """
        now = timestamp if timestamp is not None else time.time()

        # Ingestion through GPS pipeline
        filtered_pos: Optional[FilteredPosition] = None
        if isinstance(gps_data, str):
            filtered_pos = self.gps_source.ingest_nmea(gps_data, current_time=now)
        elif isinstance(gps_data, GPSFix):
            filtered_pos = self.gps_source.ingest_fix(gps_data, current_time=now)
        elif isinstance(gps_data, (tuple, list)):
            lat = float(gps_data[0])
            lon = float(gps_data[1])
            alt = float(gps_data[2]) if len(gps_data) > 2 else 0.0
            filtered_pos = self.gps_source.ingest_coordinates(
                lat=lat, lon=lon, alt=alt, timestamp=now
            )

        # Synchronize geo_ref with gps_source
        if self.geo_ref is None and self.gps_source.geo_ref is not None:
            self.geo_ref = self.gps_source.geo_ref
            self.waypoint_generator.geo_ref = self.geo_ref

        # Check for pending destination planning on initial fix
        if (
            self._pending_gps_destination is not None
            and filtered_pos is not None
            and self.geo_ref is not None
        ):
            dest_lat, dest_lon, _ = self._pending_gps_destination
            start_coord = (filtered_pos.lat, filtered_pos.lon)
            self.waypoint_generator.geo_ref = self.geo_ref
            waypoints = self.waypoint_generator.generate_from_gps(
                start_gps=start_coord,
                destination_gps=(dest_lat, dest_lon),
                costmap=self._pending_costmap,
                target_speed=self._pending_target_speed,
            )
            self.set_waypoints(waypoints)
            self._pending_gps_destination = None
            self._pending_costmap = None
            self._pending_target_speed = None

        # GPS Signal Loss / Timeout Handling
        if filtered_pos is None or self.gps_source.has_signal_loss(current_time=now):
            self._state = NavigationState.GPS_LOST
            d_act = 0.0
            d_goal = 0.0
            if self._waypoints:
                cur = self._waypoints[min(self._active_index, len(self._waypoints) - 1)]
                d_act = math.hypot(cur.x - self.vehicle_state.x, cur.y - self.vehicle_state.y)
                goal = self._waypoints[-1]
                d_goal = math.hypot(goal.x - self.vehicle_state.x, goal.y - self.vehicle_state.y)

            # Fail-safe command: immediate braking stop
            fail_safe_cmd = ControlCommand(
                steering_angle=0.0,
                target_velocity=0.0,
                angular_velocity=0.0,
                curvature=0.0,
                cross_track_error=0.0,
                heading_error=0.0,
                lookahead_point=(self.vehicle_state.x, self.vehicle_state.y),
                target_waypoint_index=self._active_index,
            )
            return NavigationResult(
                state=NavigationState.GPS_LOST,
                command=fail_safe_cmd,
                active_waypoint_index=self._active_index,
                total_waypoints=len(self._waypoints),
                distance_to_active=d_act,
                distance_to_goal=d_goal,
                progress_fraction=float(self._active_index / max(1, len(self._waypoints) - 1)) if self._waypoints else 0.0,
            )

        # Signal recovery
        if self._state == NavigationState.GPS_LOST and self._waypoints:
            self._state = NavigationState.NAVIGATING

        # Update vehicle kinematic state from filtered position and dynamic heading estimator
        self.vehicle_state.update_from_gps(
            x=filtered_pos.x,
            y=filtered_pos.y,
            z=filtered_pos.z,
            speed=filtered_pos.speed_m_s,
            timestamp=filtered_pos.timestamp,
            imu_yaw=imu_yaw,
        )

        # Execute tracking control step
        result = self.step(self.vehicle_state)

        # Mitigate coordinate drift / multipath spike: if outlier was flagged, clamp speed command
        if filtered_pos.is_outlier and result.command is not None:
            safe_speed = min(result.command.target_velocity, 0.5)
            adjusted_cmd = ControlCommand(
                steering_angle=result.command.steering_angle,
                target_velocity=safe_speed,
                angular_velocity=safe_speed * result.command.curvature,
                curvature=result.command.curvature,
                cross_track_error=result.command.cross_track_error,
                heading_error=result.command.heading_error,
                lookahead_point=result.command.lookahead_point,
                target_waypoint_index=result.command.target_waypoint_index,
            )
            result = NavigationResult(
                state=result.state,
                command=adjusted_cmd,
                active_waypoint_index=result.active_waypoint_index,
                total_waypoints=result.total_waypoints,
                distance_to_active=result.distance_to_active,
                distance_to_goal=result.distance_to_goal,
                progress_fraction=result.progress_fraction,
            )

        return result

    def simulate(
        self,
        initial_state: VehicleState,
        dt: float = 0.05,
        max_steps: int = 3000,
    ) -> List[Tuple[VehicleState, NavigationResult]]:
        """Run a closed-loop kinematic simulation to completion or timeout."""
        history: List[Tuple[VehicleState, NavigationResult]] = []
        state = VehicleState(
            x=initial_state.x,
            y=initial_state.y,
            yaw=initial_state.yaw,
            v=initial_state.v,
            yaw_rate=initial_state.yaw_rate,
            z=initial_state.z,
            wheelbase=initial_state.wheelbase,
        )

        for _ in range(max_steps):
            result = self.step(state)
            history.append((
                VehicleState(
                    x=state.x,
                    y=state.y,
                    yaw=state.yaw,
                    v=state.v,
                    yaw_rate=state.yaw_rate,
                    z=state.z,
                    wheelbase=state.wheelbase,
                ),
                result,
            ))

            if result.state == NavigationState.GOAL_REACHED:
                break

            cmd = result.command
            if cmd is None:
                break

            # Kinematic bicycle model forward integration
            state.v = cmd.target_velocity
            beta = math.atan(0.5 * math.tan(cmd.steering_angle))
            state.x += state.v * math.cos(state.yaw + beta) * dt
            state.y += state.v * math.sin(state.yaw + beta) * dt
            state.yaw = normalize_angle(
                state.yaw + (state.v / state.wheelbase) * math.tan(cmd.steering_angle) * math.cos(beta) * dt
            )

        return history

from __future__ import annotations

import math
import numpy as np
import pytest

from terrain_mapper.costmap import build_costmap
from terrain_mapper.config import PlannerConfig
from terrain_mapper.elevation import ElevationGrid
from terrain_mapper.navigation import (
    ControlCommand,
    FilteredPosition,
    GPSFilter,
    GPSFix,
    GPSPipeline,
    GPSQuality,
    GPSSource,
    GeoReference,
    HeadingEstimator,
    HeadingSource,
    NavigationResult,
    NavigationState,
    Navigator,
    PurePursuitController,
    StanleyController,
    VehicleState,
    Waypoint,
    WaypointGenerator,
    compute_nmea_checksum,
    haversine_distance,
    initial_bearing,
    normalize_angle,
    parse_nmea_coord,
    parse_nmea_sentence,
    ramer_douglas_peucker,
    resample_polyline,
)


def test_angle_normalization():
    assert math.isclose(normalize_angle(0.0), 0.0)
    assert math.isclose(normalize_angle(math.pi), -math.pi, abs_tol=1e-6)
    assert math.isclose(normalize_angle(3 * math.pi), -math.pi, abs_tol=1e-6)
    assert math.isclose(normalize_angle(-3 * math.pi), -math.pi, abs_tol=1e-6)
    assert math.isclose(normalize_angle(math.pi / 2), math.pi / 2, abs_tol=1e-6)


def test_haversine_and_bearing():
    # San Francisco to San Jose (~67 km)
    sf_lat, sf_lon = 37.7749, -122.4194
    sj_lat, sj_lon = 37.3382, -121.8863

    dist = haversine_distance(sf_lat, sf_lon, sj_lat, sj_lon)
    assert 65000 < dist < 72000

    bearing = initial_bearing(sf_lat, sf_lon, sj_lat, sj_lon)
    # Bearing should be roughly South-East (between 90 deg and 180 deg)
    assert 1.5 < bearing < 2.8


def test_geo_reference_round_trip():
    origin_lat, origin_lon = 37.7749, -122.4194
    geo = GeoReference(origin_lat=origin_lat, origin_lon=origin_lon, origin_alt=15.0)

    # Convert offset of 100m East, 200m North, 5m Up
    lat, lon, alt = geo.enu_to_gps(100.0, 200.0, 20.0)
    x, y, z = geo.gps_to_enu(lat, lon, alt)

    assert math.isclose(x, 100.0, abs_tol=1e-3)
    assert math.isclose(y, 200.0, abs_tol=1e-3)
    assert math.isclose(z, 20.0, abs_tol=1e-3)


def test_ramer_douglas_peucker():
    # Straight line with intermediate collinear points
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    simplified = ramer_douglas_peucker(points, epsilon=0.1)
    assert len(simplified) == 2
    assert simplified[0] == (0.0, 0.0)
    assert simplified[-1] == (4.0, 4.0)

    # Line with a corner deviation
    points_corner = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    simplified_corner = ramer_douglas_peucker(points_corner, epsilon=0.1)
    assert len(simplified_corner) == 3


def test_resample_polyline():
    points = [(0.0, 0.0), (100.0, 0.0)]
    resampled = resample_polyline(points, max_spacing=25.0, min_spacing=2.0)
    assert len(resampled) == 5
    assert resampled[0] == (0.0, 0.0)
    assert resampled[-1] == (100.0, 0.0)


def test_waypoint_generator_direct():
    generator = WaypointGenerator(default_speed=3.0, max_spacing=10.0)
    waypoints = generator.generate_waypoints(start_xy=(0.0, 0.0), destination_xy=(30.0, 40.0))

    assert len(waypoints) >= 5
    assert math.isclose(waypoints[0].x, 0.0)
    assert math.isclose(waypoints[0].y, 0.0)
    assert math.isclose(waypoints[-1].x, 30.0)
    assert math.isclose(waypoints[-1].y, 40.0)
    # Final waypoint speed should ramp to 0
    assert waypoints[-1].target_speed == 0.0
    # Tangent heading along 30, 40 should be atan2(40, 30)
    expected_heading = math.atan2(40.0, 30.0)
    assert math.isclose(waypoints[0].heading, expected_heading, abs_tol=1e-3)


def test_waypoint_generator_from_gps():
    generator = WaypointGenerator(default_speed=2.0)
    start_gps = (37.7749, -122.4194)
    dest_gps = (37.7759, -122.4184)

    waypoints = generator.generate_from_gps(start_gps=start_gps, destination_gps=dest_gps)
    assert len(waypoints) >= 2
    assert waypoints[0].lat is not None
    assert waypoints[-1].lat is not None
    assert math.isclose(waypoints[0].lat, start_gps[0], abs_tol=1e-5)
    assert math.isclose(waypoints[-1].lat, dest_gps[0], abs_tol=1e-5)


def test_pure_pursuit_controller_straight():
    controller = PurePursuitController(lookahead_base=3.0, wheelbase=2.0)
    waypoints = [
        Waypoint(x=0.0, y=0.0, target_speed=2.0),
        Waypoint(x=10.0, y=0.0, target_speed=2.0),
        Waypoint(x=20.0, y=0.0, target_speed=2.0),
    ]

    # Vehicle is on track heading east (+X)
    state = VehicleState(x=2.0, y=0.0, yaw=0.0, v=2.0, wheelbase=2.0)
    cmd = controller.compute_control(state, waypoints, active_index=1)

    # Steering angle should be close to 0 on a straight line
    assert abs(cmd.steering_angle) < 0.05
    assert math.isclose(cmd.target_velocity, 2.0)


def test_pure_pursuit_controller_steers_towards_path():
    controller = PurePursuitController(lookahead_base=4.0, wheelbase=2.0)
    waypoints = [
        Waypoint(x=0.0, y=0.0, target_speed=2.0),
        Waypoint(x=20.0, y=0.0, target_speed=2.0),
    ]

    # Vehicle is displaced to the right (y = -2.0) heading along X-axis
    state = VehicleState(x=0.0, y=-2.0, yaw=0.0, v=2.0, wheelbase=2.0)
    cmd = controller.compute_control(state, waypoints, active_index=0)

    # Vehicle should steer LEFT (positive steering angle) to regain track
    assert cmd.steering_angle > 0.1


def test_stanley_controller_tracking():
    controller = StanleyController(k_gain=1.0, k_soft=1.0, wheelbase=2.0)
    waypoints = [
        Waypoint(x=0.0, y=0.0, target_speed=2.0),
        Waypoint(x=20.0, y=0.0, target_speed=2.0),
    ]

    # Vehicle is displaced to the left of the path (y = +1.0) heading East
    state = VehicleState(x=0.0, y=1.0, yaw=0.0, v=2.0, wheelbase=2.0)
    cmd = controller.compute_control(state, waypoints, active_index=0)

    # With front axle at y=1.0 (left of path), cross track error is positive,
    # steering should turn RIGHT (negative steering angle) to converge
    assert cmd.cross_track_error > 0.5
    assert cmd.steering_angle < -0.1


def test_navigator_state_machine_switching():
    waypoints = [
        Waypoint(x=0.0, y=0.0, target_speed=2.0, acceptance_radius=1.0),
        Waypoint(x=10.0, y=0.0, target_speed=2.0, acceptance_radius=1.0),
        Waypoint(x=20.0, y=0.0, target_speed=0.0, acceptance_radius=1.0),
    ]

    navigator = Navigator(waypoint_tolerance=1.5, goal_tolerance=1.0)
    navigator.set_waypoints(waypoints)
    assert navigator.state == NavigationState.NAVIGATING
    assert navigator.active_index == 0

    # Vehicle approaches waypoint 0
    vehicle = VehicleState(x=0.2, y=0.1, yaw=0.0, v=2.0)
    result = navigator.step(vehicle)
    # Should switch to waypoint 1
    assert result.state == NavigationState.WAYPOINT_REACHED
    assert navigator.active_index == 1

    # Vehicle drives towards waypoint 1
    vehicle = VehicleState(x=5.0, y=0.0, yaw=0.0, v=2.0)
    result = navigator.step(vehicle)
    assert result.state == NavigationState.NAVIGATING
    assert navigator.active_index == 1

    # Vehicle reaches waypoint 1
    vehicle = VehicleState(x=10.0, y=0.0, yaw=0.0, v=2.0)
    result = navigator.step(vehicle)
    assert result.state == NavigationState.WAYPOINT_REACHED
    assert navigator.active_index == 2

    # Vehicle arrives at final goal waypoint
    vehicle = VehicleState(x=19.8, y=0.0, yaw=0.0, v=0.5)
    result = navigator.step(vehicle)
    assert result.state == NavigationState.GOAL_REACHED
    assert result.command.target_velocity == 0.0


def test_navigator_simulation_convergence():
    waypoints = [
        Waypoint(x=0.0, y=0.0, target_speed=2.0),
        Waypoint(x=10.0, y=10.0, target_speed=2.0),
        Waypoint(x=20.0, y=10.0, target_speed=2.0),
        Waypoint(x=30.0, y=0.0, target_speed=0.0),
    ]

    controller = PurePursuitController(lookahead_base=3.0, wheelbase=1.5)
    navigator = Navigator(controller=controller, waypoint_tolerance=2.0, goal_tolerance=1.5)
    navigator.set_waypoints(waypoints)

    initial_state = VehicleState(x=0.0, y=0.0, yaw=math.pi / 4, v=2.0, wheelbase=1.5)
    history = navigator.simulate(initial_state, dt=0.05, max_steps=1000)

    final_state, final_result = history[-1]
    assert final_result.state == NavigationState.GOAL_REACHED
    assert math.hypot(final_state.x - 30.0, final_state.y - 0.0) < 2.0


def test_end_to_end_with_grid_and_costmap():
    # Synthetic flat elevation grid with a central obstacle
    values = np.zeros((30, 30), dtype=np.float32)
    grid = ElevationGrid(
        values_m=values,
        cell_size_x_m=2.0,
        cell_size_y_m=2.0,
        crs="EPSG:32610",
        bounds=(0.0, 0.0, 60.0, 60.0),
    )
    costmap = build_costmap(grid, PlannerConfig())

    # Generate waypoints using A* through costmap
    gen = WaypointGenerator(grid=grid, default_speed=2.5, simplification_tolerance=1.5)
    waypoints = gen.generate_waypoints(start_xy=(4.0, 4.0), destination_xy=(50.0, 50.0), costmap=costmap)

    assert len(waypoints) >= 3
    assert math.isclose(waypoints[0].x, 4.0, abs_tol=3.0)
    assert math.isclose(waypoints[-1].x, 50.0, abs_tol=3.0)

    # Initialize Navigator and step through
    nav = Navigator()
    nav.set_waypoints(waypoints)
    state = VehicleState(x=4.0, y=4.0, yaw=math.pi / 4, v=2.0)
    res = nav.step(state)
    assert res.command is not None
    assert res.command.target_velocity > 0.0


# ---------------------------------------------------------------------------
# GPS Pipeline & NMEA Ingestion Tests
# ---------------------------------------------------------------------------

def test_nmea_gga_parsing():
    # Example GGA sentence with valid checksum:
    # 48 deg 07.038 min N = 48.1173 deg N
    # 11 deg 31.000 min E = 11.516667 deg E
    raw = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    fix = parse_nmea_sentence(raw)

    assert fix is not None
    assert fix.is_valid is True
    assert fix.quality == GPSQuality.SINGLE_SPS
    assert math.isclose(fix.lat, 48.1173, abs_tol=1e-4)
    assert math.isclose(fix.lon, 11.516667, abs_tol=1e-4)
    assert fix.num_sats == 8
    assert math.isclose(fix.hdop, 0.9)
    assert math.isclose(fix.alt, 545.4)


def test_nmea_rmc_parsing():
    # RMC sentence with speed 22.4 knots (~11.52 m/s) and course 84.4 degrees (~1.473 rad)
    raw = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
    fix = parse_nmea_sentence(raw)

    assert fix is not None
    assert fix.is_valid is True
    assert math.isclose(fix.lat, 48.1173, abs_tol=1e-4)
    assert fix.speed_m_s is not None
    assert math.isclose(fix.speed_m_s, 22.4 * 0.514444, abs_tol=0.1)
    assert fix.course_rad is not None
    assert math.isclose(fix.course_rad, math.radians(84.4), abs_tol=1e-3)


def test_nmea_malformed_and_checksum():
    # Invalid checksum
    corrupt = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*FF"
    assert parse_nmea_sentence(corrupt) is None

    # Garbage string
    assert parse_nmea_sentence("NOT_AN_NMEA_STRING") is None
    assert parse_nmea_sentence("") is None

    # Status 'V' (Void) in RMC
    void_rmc = "$GPRMC,123519,V,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*7D"
    void_fix = parse_nmea_sentence(void_rmc)
    assert void_fix is not None
    assert void_fix.is_valid is False


def test_gps_filter_jitter_mitigation():
    # Feed stationary points with Gaussian jitter around (10.0, 20.0)
    filter_ = GPSFilter(window_size=5, alpha=0.3, max_speed_threshold=15.0)

    # Stationary ground truth
    gx, gy = 10.0, 20.0
    jitter_offsets = [0.0, 0.4, -0.3, 0.5, -0.4, 0.2, -0.2]

    filtered_positions = []
    for i, offset in enumerate(jitter_offsets):
        x = gx + offset
        y = gy - offset
        fx, fy, fz, is_outlier, speed, jitter = filter_.update(x, y, 0.0, timestamp=i * 0.1)
        filtered_positions.append((fx, fy))

    # The filtered trajectory should stay very close to the true center (less than 0.2m error)
    final_fx, final_fy = filtered_positions[-1]
    assert math.hypot(final_fx - gx, final_fy - gy) < 0.25


def test_gps_filter_outlier_rejection():
    filter_ = GPSFilter(window_size=5, alpha=0.5, max_speed_threshold=10.0, max_jump_m=10.0)

    # Normal updates at 1 Hz, speed 1 m/s along X
    filter_.update(0.0, 0.0, 0.0, timestamp=0.0)
    filter_.update(1.0, 0.0, 0.0, timestamp=1.0)
    fx, fy, _, is_outlier, _, _ = filter_.update(2.0, 0.0, 0.0, timestamp=2.0)
    assert not is_outlier

    # Sudden multipath spike: jumps 100 meters away in 1 second
    fx_spike, fy_spike, _, is_outlier, _, _ = filter_.update(102.0, 50.0, 0.0, timestamp=3.0)
    assert is_outlier is True
    # Filter should reject the spike and retain previous filtered position
    assert math.isclose(fx_spike, fx, abs_tol=1e-4)


def test_gps_source_pipeline_ingestion():
    geo = GeoReference(origin_lat=37.7749, origin_lon=-122.4194)
    source = GPSSource(geo_ref=geo, filter_alpha=0.4)

    # Ingest coordinates 100m East, 50m North
    lat, lon, _ = geo.enu_to_gps(100.0, 50.0)
    pos = source.ingest_coordinates(lat=lat, lon=lon, alt=10.0, timestamp=1.0)

    assert pos is not None
    assert math.isclose(pos.x, 100.0, abs_tol=0.5)
    assert math.isclose(pos.y, 50.0, abs_tol=0.5)
    assert source.is_healthy(current_time=1.5)
    assert not source.has_signal_loss(current_time=1.5)


def test_gps_source_signal_loss():
    source = GPSSource(signal_timeout=2.0)
    source.ingest_coordinates(lat=37.7749, lon=-122.4194, timestamp=10.0)

    assert source.is_healthy(current_time=11.0) is True
    # After 2.5s (exceeding 2.0s timeout), signal loss should trigger
    assert source.is_healthy(current_time=12.5) is False
    assert source.has_signal_loss(current_time=12.5) is True


# ---------------------------------------------------------------------------
# Heading Estimator & VehicleState Tests
# ---------------------------------------------------------------------------

def test_heading_estimator_dynamic_moving():
    estimator = HeadingEstimator(min_moving_speed=0.3, min_displacement=0.2, smoothing_alpha=0.5)

    # First point: anchor
    yaw, source = estimator.estimate(x=0.0, y=0.0, timestamp=0.0)
    assert source == HeadingSource.MANUAL

    # Move North (+Y direction) at 1 m/s: heading should point towards +pi/2 (90 deg)
    for t in [1.0, 2.0, 3.0]:
        yaw, source = estimator.estimate(x=0.0, y=t * 1.0, speed=1.0, timestamp=t)
        assert source == HeadingSource.GPS

    assert math.isclose(yaw, math.pi / 2, abs_tol=0.1)


def test_heading_estimator_stationary_imu_fallback():
    estimator = HeadingEstimator(min_moving_speed=0.5, min_displacement=0.3)

    # Anchor at (0, 0)
    estimator.estimate(x=0.0, y=0.0, timestamp=0.0)

    # Platform is stationary (jittering less than 0.05m)
    # IMU stream provides orientation pointing South (-pi/2)
    yaw, source = estimator.estimate(
        x=0.02, y=0.01, speed=0.0, timestamp=1.0, imu_yaw=-math.pi / 2
    )

    assert source == HeadingSource.IMU
    assert math.isclose(yaw, -math.pi / 2, abs_tol=1e-3)


def test_heading_estimator_stationary_lock_without_imu():
    estimator = HeadingEstimator(min_moving_speed=0.5, min_displacement=0.3, initial_yaw=1.0)
    estimator.estimate(x=0.0, y=0.0, timestamp=0.0)

    # Tiny GPS jitter without IMU stream
    yaw, source = estimator.estimate(x=0.03, y=0.02, speed=0.05, timestamp=1.0, imu_yaw=None)
    assert source == HeadingSource.LOCKED
    assert math.isclose(yaw, 1.0, abs_tol=1e-3)


def test_vehicle_state_dynamic_gps_update():
    vehicle = VehicleState(x=0.0, y=0.0, yaw=0.0)

    # Move diagonally (+X, +Y) at 2 m/s -> angle should approach pi/4
    for t in range(1, 5):
        vehicle.update_from_gps(
            x=t * 1.5,
            y=t * 1.5,
            speed=2.0,
            timestamp=float(t),
        )

    assert vehicle.heading_source == HeadingSource.GPS
    assert math.isclose(vehicle.yaw, math.pi / 4, abs_tol=0.15)


# ---------------------------------------------------------------------------
# Navigator GPS Closed-Loop Execution Loop Tests
# ---------------------------------------------------------------------------

def test_navigator_set_gps_destination_and_step():
    origin_lat, origin_lon = 37.7749, -122.4194
    geo = GeoReference(origin_lat=origin_lat, origin_lon=origin_lon)

    # Target point: 50m East, 0m North
    target_lat, target_lon, _ = geo.enu_to_gps(50.0, 0.0)

    nav = Navigator(
        controller=PurePursuitController(lookahead_base=3.0),
        geo_ref=geo,
        waypoint_tolerance=2.0,
        goal_tolerance=1.5,
    )
    waypoints = nav.set_gps_destination(
        lat=target_lat,
        lon=target_lon,
        current_gps=(origin_lat, origin_lon),
        target_speed=2.0,
    )

    assert len(waypoints) >= 2
    assert nav.state == NavigationState.NAVIGATING

    # Live telemetry feed moving towards target
    res = nav.step_gps((origin_lat, origin_lon), imu_yaw=0.0, timestamp=1.0)
    assert res.state in {NavigationState.NAVIGATING, NavigationState.WAYPOINT_REACHED}
    assert res.command is not None
    assert res.command.target_velocity > 0.0

    # Drive vehicle along trajectory towards target
    for t in range(2, 35):
        x = min(50.0, t * 2.0)
        curr_lat, curr_lon, _ = geo.enu_to_gps(x, 0.0)
        res_near = nav.step_gps((curr_lat, curr_lon), imu_yaw=0.0, timestamp=float(t))

    assert res_near.state == NavigationState.GOAL_REACHED
    assert res_near.command.target_velocity == 0.0


def test_navigator_gps_signal_loss_failsafe():
    origin_lat, origin_lon = 37.7749, -122.4194
    geo = GeoReference(origin_lat=origin_lat, origin_lon=origin_lon)
    dest_lat, dest_lon, _ = geo.enu_to_gps(100.0, 0.0)

    nav = Navigator(geo_ref=geo, max_gps_stale_time=2.0)
    nav.set_gps_destination(dest_lat, dest_lon, current_gps=(origin_lat, origin_lon))

    # Valid step
    res1 = nav.step_gps((origin_lat, origin_lon), timestamp=10.0)
    assert res1.state in {NavigationState.NAVIGATING, NavigationState.WAYPOINT_REACHED}

    # Signal loss: step after timeout with invalid data or time jump
    res_loss = nav.step_gps("INVALID_NMEA_STRING", timestamp=15.0)
    assert res_loss.state == NavigationState.GPS_LOST
    assert res_loss.command is not None
    # Must command safe zero velocity stop
    assert res_loss.command.target_velocity == 0.0
    assert res_loss.command.steering_angle == 0.0

    # Signal recovery: valid fix arrives
    res_recovered = nav.step_gps((origin_lat, origin_lon), timestamp=16.0)
    assert res_recovered.state == NavigationState.NAVIGATING
    assert res_recovered.command.target_velocity > 0.0


def test_navigator_stanley_controller_live_gps():
    origin_lat, origin_lon = 37.7749, -122.4194
    geo = GeoReference(origin_lat=origin_lat, origin_lon=origin_lon)
    dest_lat, dest_lon, _ = geo.enu_to_gps(0.0, 100.0)  # Path along North (+Y)

    stanley = StanleyController(k_gain=1.0, k_soft=1.0, wheelbase=2.0)
    nav = Navigator(controller=stanley, geo_ref=geo)
    nav.set_gps_destination(dest_lat, dest_lon, current_gps=(origin_lat, origin_lon))

    # Vehicle is displaced to the right of path (x = 2.0, y = 10.0)
    curr_lat, curr_lon, _ = geo.enu_to_gps(2.0, 10.0)
    # Heading North (pi/2)
    res = nav.step_gps((curr_lat, curr_lon), imu_yaw=math.pi / 2, timestamp=5.0)

    assert res.state == NavigationState.NAVIGATING
    assert res.command is not None
    # Since vehicle is right of path (positive X), steering should turn LEFT (positive steering angle)
    assert res.command.steering_angle > 0.0


from __future__ import annotations

from terrain_mapper.navigation.controllers import (
    BaseController,
    PurePursuitController,
    StanleyController,
)
from terrain_mapper.navigation.geo import (
    GeoReference,
    haversine_distance,
    initial_bearing,
    normalize_angle,
)
from terrain_mapper.navigation.gps_source import (
    FilteredPosition,
    GPSFilter,
    GPSFix,
    GPSPipeline,
    GPSQuality,
    GPSSource,
    compute_nmea_checksum,
    parse_nmea_coord,
    parse_nmea_sentence,
    parse_nmea_time,
    verify_nmea_checksum,
)
from terrain_mapper.navigation.navigator import Navigator
from terrain_mapper.navigation.types import (
    ControlCommand,
    HeadingEstimator,
    HeadingSource,
    NavigationResult,
    NavigationState,
    VehicleState,
    Waypoint,
)
from terrain_mapper.navigation.waypoint_generator import (
    WaypointGenerator,
    ramer_douglas_peucker,
    resample_polyline,
)

__all__ = [
    "BaseController",
    "ControlCommand",
    "FilteredPosition",
    "GPSFilter",
    "GPSFix",
    "GPSPipeline",
    "GPSQuality",
    "GPSSource",
    "GeoReference",
    "HeadingEstimator",
    "HeadingSource",
    "NavigationResult",
    "NavigationState",
    "Navigator",
    "PurePursuitController",
    "StanleyController",
    "VehicleState",
    "Waypoint",
    "WaypointGenerator",
    "compute_nmea_checksum",
    "haversine_distance",
    "initial_bearing",
    "normalize_angle",
    "parse_nmea_coord",
    "parse_nmea_sentence",
    "parse_nmea_time",
    "ramer_douglas_peucker",
    "resample_polyline",
    "verify_nmea_checksum",
]

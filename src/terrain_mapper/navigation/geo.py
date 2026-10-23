from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

# WGS-84 Ellipsoid constants
WGS84_A = 6378137.0  # semi-major axis in meters
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # first eccentricity squared


def normalize_angle(angle: float) -> float:
    """Normalize an angle in radians to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle distance between two GPS coordinates in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return WGS84_A * c


def initial_bearing(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate initial compass bearing from (lat1, lon1) to (lat2, lon2) in radians."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.atan2(y, x)


class GeoReference:
    """Provides bi-directional conversion between WGS-84 (lat, lon, alt)
    and local East-North-Up (ENU) Cartesian coordinates (x, y, z) in meters.
    """

    def __init__(
        self,
        origin_lat: float,
        origin_lon: float,
        origin_alt: float = 0.0,
    ) -> None:
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.origin_alt = float(origin_alt)

        # Radians of origin
        self._lat_rad = math.radians(self.origin_lat)
        self._lon_rad = math.radians(self.origin_lon)

        # Meridional and prime vertical radius of curvature at origin
        sin_lat = math.sin(self._lat_rad)
        cos_lat = math.cos(self._lat_rad)
        w = math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        self._rn = WGS84_A / w
        self._rm = WGS84_A * (1.0 - WGS84_E2) / (w**3)

        # Meters per degree at the origin
        self.meters_per_deg_lat = math.radians(1.0) * (self._rm + self.origin_alt)
        self.meters_per_deg_lon = math.radians(1.0) * (self._rn + self.origin_alt) * cos_lat

    def gps_to_enu(
        self, lat: float, lon: float, alt: float = 0.0
    ) -> Tuple[float, float, float]:
        """Convert GPS (lat, lon, alt) to local East-North-Up (x, y, z) in meters."""
        d_lat = lat - self.origin_lat
        d_lon = lon - self.origin_lon
        x = d_lon * self.meters_per_deg_lon
        y = d_lat * self.meters_per_deg_lat
        z = alt - self.origin_alt
        return float(x), float(y), float(z)

    def enu_to_gps(
        self, x: float, y: float, z: float = 0.0
    ) -> Tuple[float, float, float]:
        """Convert local East-North-Up (x, y, z) in meters to GPS (lat, lon, alt)."""
        d_lon = x / self.meters_per_deg_lon if abs(self.meters_per_deg_lon) > 1e-9 else 0.0
        d_lat = y / self.meters_per_deg_lat if abs(self.meters_per_deg_lat) > 1e-9 else 0.0
        lat = self.origin_lat + d_lat
        lon = self.origin_lon + d_lon
        alt = self.origin_alt + z
        return float(lat), float(lon), float(alt)

    def distance_enu(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """Euclidean distance in local ENU frame between two 2D points (x, y)."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# Re-export GPS ingestion and filtering classes for convenience
from terrain_mapper.navigation.gps_source import (
    FilteredPosition,
    GPSFilter,
    GPSFix,
    GPSPipeline,
    GPSQuality,
    GPSSource,
    parse_nmea_sentence,
)

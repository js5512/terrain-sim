from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import re
import time
from typing import Deque, List, Optional, Sequence, Tuple, Union

import numpy as np

from terrain_mapper.navigation.geo import GeoReference, normalize_angle


class GPSQuality(str, Enum):
    """NMEA GPS fix quality indicator."""
    INVALID = "INVALID"  # 0: Fix not available or invalid
    SINGLE_SPS = "SINGLE_SPS"  # 1: Standard GPS SPS fix
    DGPS = "DGPS"  # 2: Differential GPS SPS fix
    PPS = "PPS"  # 3: Precise Positioning Service
    RTK_FIX = "RTK_FIX"  # 4: Real-Time Kinematic fixed integer
    RTK_FLOAT = "RTK_FLOAT"  # 5: Real-Time Kinematic float solution
    DEAD_RECKONING = "DEAD_RECKONING"  # 6: Estimated / dead reckoning fix
    MANUAL = "MANUAL"  # 7: Manual input mode
    SIMULATION = "SIMULATION"  # 8: Simulation mode


@dataclass(frozen=True)
class GPSFix:
    """Raw or parsed GPS fix report."""
    lat: float
    lon: float
    alt: float = 0.0
    timestamp: float = 0.0
    quality: GPSQuality = GPSQuality.SINGLE_SPS
    num_sats: int = 0
    hdop: float = 1.0
    speed_m_s: Optional[float] = None  # Speed over ground in m/s
    course_rad: Optional[float] = None  # Track angle / course over ground in radians
    is_valid: bool = True
    raw_sentence: Optional[str] = None


@dataclass(frozen=True)
class FilteredPosition:
    """Filtered ENU position and associated GPS telemetry."""
    x: float
    y: float
    z: float
    lat: float
    lon: float
    alt: float
    speed_m_s: float
    timestamp: float
    hdop: float
    quality: GPSQuality
    is_outlier: bool = False
    jitter_estimate: float = 0.0  # Estimated standard deviation of position (m)

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def compute_nmea_checksum(sentence: str) -> str:
    """Calculate the 2-character hexadecimal XOR checksum of an NMEA sentence."""
    body = sentence.strip()
    if body.startswith("$") or body.startswith("!"):
        body = body[1:]
    if "*" in body:
        body = body.split("*", 1)[0]
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"{cs:02X}"


def verify_nmea_checksum(sentence: str) -> bool:
    """Verify if the NMEA sentence checksum matches."""
    sentence = sentence.strip()
    if "*" not in sentence:
        return True  # If checksum is omitted, allow lenient ingestion
    parts = sentence.split("*")
    if len(parts) != 2 or len(parts[1]) < 2:
        return False
    expected = parts[1][:2].upper()
    calculated = compute_nmea_checksum(parts[0])
    return expected == calculated


def parse_nmea_coord(value: str, direction: str) -> Optional[float]:
    """Parse NMEA coordinate string (ddmm.mmmm or dddmm.mmmm) to decimal degrees."""
    if not value or not direction:
        return None
    try:
        val = float(value)
        direction = direction.strip().upper()
        # Latitude has 2 degree digits (ddmm.mmmm), Longitude has 3 (dddmm.mmmm)
        if direction in {"N", "S"}:
            deg = int(val / 100)
            minutes = val - (deg * 100)
            deg_dec = deg + (minutes / 60.0)
            if direction == "S":
                deg_dec = -deg_dec
            return deg_dec
        elif direction in {"E", "W"}:
            deg = int(val / 100)
            minutes = val - (deg * 100)
            deg_dec = deg + (minutes / 60.0)
            if direction == "W":
                deg_dec = -deg_dec
            return deg_dec
        return None
    except (ValueError, TypeError):
        return None


def parse_nmea_time(time_str: str) -> float:
    """Parse NMEA UTC time string (hhmmss or hhmmss.ss) into seconds of day."""
    if not time_str:
        return time.time()
    try:
        val = float(time_str)
        hh = int(val // 10000)
        mm = int((val % 10000) // 100)
        ss = val % 100
        return hh * 3600.0 + mm * 60.0 + ss
    except (ValueError, TypeError):
        return time.time()


def parse_nmea_sentence(sentence: str) -> Optional[GPSFix]:
    """Parse an NMEA 0183 sentence ($GPGGA, $GNGGA, $GPRMC, $GNRMC, etc.) into a GPSFix."""
    if not sentence or not isinstance(sentence, str):
        return None

    cleaned = sentence.strip()
    if not cleaned.startswith("$"):
        return None

    if not verify_nmea_checksum(cleaned):
        return None

    body = cleaned[1:].split("*")[0]
    tokens = body.split(",")
    if not tokens:
        return None

    talker_type = tokens[0].upper()

    # GGA parsing (Global Positioning System Fix Data)
    if talker_type.endswith("GGA"):
        if len(tokens) < 10:
            return None
        time_sec = parse_nmea_time(tokens[1])
        lat = parse_nmea_coord(tokens[2], tokens[3])
        lon = parse_nmea_coord(tokens[4], tokens[5])
        if lat is None or lon is None:
            return None

        # Quality indicator
        raw_quality = tokens[6].strip()
        quality_map = {
            "0": GPSQuality.INVALID,
            "1": GPSQuality.SINGLE_SPS,
            "2": GPSQuality.DGPS,
            "3": GPSQuality.PPS,
            "4": GPSQuality.RTK_FIX,
            "5": GPSQuality.RTK_FLOAT,
            "6": GPSQuality.DEAD_RECKONING,
            "7": GPSQuality.MANUAL,
            "8": GPSQuality.SIMULATION,
        }
        quality = quality_map.get(raw_quality, GPSQuality.INVALID)
        is_valid = quality != GPSQuality.INVALID

        try:
            num_sats = int(tokens[7]) if tokens[7] else 0
        except ValueError:
            num_sats = 0

        try:
            hdop = float(tokens[8]) if tokens[8] else 1.0
        except ValueError:
            hdop = 1.0

        try:
            alt = float(tokens[9]) if tokens[9] else 0.0
        except ValueError:
            alt = 0.0

        return GPSFix(
            lat=lat,
            lon=lon,
            alt=alt,
            timestamp=time_sec,
            quality=quality,
            num_sats=num_sats,
            hdop=hdop,
            speed_m_s=None,
            course_rad=None,
            is_valid=is_valid,
            raw_sentence=cleaned,
        )

    # RMC parsing (Recommended Minimum Specific GPS/TRANSIT Data)
    elif talker_type.endswith("RMC"):
        if len(tokens) < 9:
            return None
        time_sec = parse_nmea_time(tokens[1])
        status = tokens[2].strip().upper()
        is_valid = status == "A"

        lat = parse_nmea_coord(tokens[3], tokens[4])
        lon = parse_nmea_coord(tokens[5], tokens[6])
        if lat is None or lon is None:
            return None

        # Speed over ground in knots -> m/s
        speed_m_s = None
        if tokens[7]:
            try:
                speed_m_s = float(tokens[7]) * 0.5144444444444445
            except ValueError:
                pass

        # Course over ground in degrees -> radians
        course_rad = None
        if tokens[8]:
            try:
                course_rad = math.radians(float(tokens[8]))
            except ValueError:
                pass

        quality = GPSQuality.SINGLE_SPS if is_valid else GPSQuality.INVALID

        return GPSFix(
            lat=lat,
            lon=lon,
            alt=0.0,
            timestamp=time_sec,
            quality=quality,
            num_sats=0,
            hdop=1.0,
            speed_m_s=speed_m_s,
            course_rad=course_rad,
            is_valid=is_valid,
            raw_sentence=cleaned,
        )

    return None


class GPSFilter:
    """Lightweight moving-average and low-pass filter for ENU coordinates.

    Mitigates high-frequency GPS position jitter and rejects multipath outlier jumps.
    """

    def __init__(
        self,
        window_size: int = 5,
        alpha: float = 0.35,  # Low-pass EMA weighting (0 < alpha <= 1.0)
        max_speed_threshold: float = 25.0,  # Max plausible velocity in m/s
        max_jump_m: float = 12.0,  # Max position jump in meters between steps
        max_consecutive_outliers: int = 4,  # Auto-realign if outlier persists
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.max_speed_threshold = float(max_speed_threshold)
        self.max_jump_m = float(max_jump_m)
        self.max_consecutive_outliers = max(1, int(max_consecutive_outliers))

        self._history_x: Deque[float] = deque(maxlen=self.window_size)
        self._history_y: Deque[float] = deque(maxlen=self.window_size)
        self._history_z: Deque[float] = deque(maxlen=self.window_size)

        self._filtered_x: Optional[float] = None
        self._filtered_y: Optional[float] = None
        self._filtered_z: Optional[float] = None

        self._last_raw_x: Optional[float] = None
        self._last_raw_y: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._consecutive_outliers: int = 0
        self._last_speed: float = 0.0

    def reset(self) -> None:
        """Reset internal filter state."""
        self._history_x.clear()
        self._history_y.clear()
        self._history_z.clear()
        self._filtered_x = None
        self._filtered_y = None
        self._filtered_z = None
        self._last_raw_x = None
        self._last_raw_y = None
        self._last_timestamp = None
        self._consecutive_outliers = 0
        self._last_speed = 0.0

    @property
    def current_position(self) -> Optional[Tuple[float, float, float]]:
        if self._filtered_x is None:
            return None
        return (self._filtered_x, self._filtered_y, self._filtered_z or 0.0)

    @property
    def current_speed(self) -> float:
        return self._last_speed

    def update(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        timestamp: Optional[float] = None,
        reported_speed: Optional[float] = None,
    ) -> Tuple[float, float, float, bool, float, float]:
        """Update filter with a new local Cartesian ENU position.

        Returns:
            (filtered_x, filtered_y, filtered_z, is_outlier, estimated_speed, jitter_std)
        """
        curr_time = timestamp if timestamp is not None else time.time()
        is_outlier = False

        # First reading: initialize filter
        if self._filtered_x is None:
            self._filtered_x = float(x)
            self._filtered_y = float(y)
            self._filtered_z = float(z)
            self._last_raw_x = float(x)
            self._last_raw_y = float(y)
            self._last_timestamp = curr_time
            self._last_speed = reported_speed or 0.0
            self._history_x.append(float(x))
            self._history_y.append(float(y))
            self._history_z.append(float(z))
            return (self._filtered_x, self._filtered_y, self._filtered_z, False, self._last_speed, 0.0)

        # Calculate displacement and delta time
        dx = x - self._filtered_x
        dy = y - self._filtered_y
        dist = math.hypot(dx, dy)
        dt = curr_time - (self._last_timestamp or curr_time)

        # Check for outlier jump
        if dt > 1e-4:
            apparent_speed = dist / dt
            if apparent_speed > self.max_speed_threshold and dist > 3.0:
                is_outlier = True
        elif dist > self.max_jump_m:
            is_outlier = True

        if is_outlier:
            self._consecutive_outliers += 1
            if self._consecutive_outliers < self.max_consecutive_outliers:
                # Reject outlier: maintain last filtered position
                jitter = self._compute_jitter()
                return (
                    self._filtered_x,
                    self._filtered_y,
                    self._filtered_z or 0.0,
                    True,
                    self._last_speed,
                    jitter,
                )
            else:
                # Too many consecutive outliers: snap to new reading to recover from relocations
                self._consecutive_outliers = 0
                is_outlier = False
        else:
            self._consecutive_outliers = 0

        # Moving average / jitter history update
        self._history_x.append(float(x))
        self._history_y.append(float(y))
        self._history_z.append(float(z))

        # Exponential low-pass filter (EMA)
        self._filtered_x = self.alpha * float(x) + (1.0 - self.alpha) * self._filtered_x
        self._filtered_y = self.alpha * float(y) + (1.0 - self.alpha) * self._filtered_y
        self._filtered_z = self.alpha * float(z) + (1.0 - self.alpha) * (self._filtered_z or 0.0)

        # Compute speed
        if reported_speed is not None and reported_speed >= 0.0:
            self._last_speed = reported_speed
        elif dt > 1e-4:
            step_dist = math.hypot(dx * self.alpha, dy * self.alpha)
            self._last_speed = float(step_dist / dt)

        self._last_raw_x = float(x)
        self._last_raw_y = float(y)
        self._last_timestamp = curr_time

        jitter = self._compute_jitter()
        return (
            self._filtered_x,
            self._filtered_y,
            self._filtered_z,
            is_outlier,
            self._last_speed,
            jitter,
        )

    def _compute_jitter(self) -> float:
        """Estimate standard deviation of recent Cartesian positions in meters."""
        if len(self._history_x) < 2:
            return 0.0
        var_x = float(np.var(self._history_x))
        var_y = float(np.var(self._history_y))
        return float(math.sqrt(var_x + var_y))


class GPSSource:
    """Complete GPS ingestion, transformation, and filtering pipeline.

    Ingests NMEA strings or GPSFix objects, transforms geodetic coordinates to local
    East-North-Up (ENU) coordinates via GeoReference, filters sensor jitter,
    and handles GPS signal loss / degradation watchdog checks.
    """

    def __init__(
        self,
        geo_ref: Optional[GeoReference] = None,
        filter_alpha: float = 0.35,
        window_size: int = 5,
        max_speed_threshold: float = 25.0,
        max_jump_m: float = 12.0,
        signal_timeout: float = 2.5,  # Max seconds without valid fix before signal loss
        max_hdop: float = 8.0,  # Max acceptable HDOP
        min_satellites: int = 3,  # Min acceptable satellites (if reported)
    ) -> None:
        self.geo_ref = geo_ref
        self.signal_timeout = float(signal_timeout)
        self.max_hdop = float(max_hdop)
        self.min_satellites = int(min_satellites)

        self.filter = GPSFilter(
            window_size=window_size,
            alpha=filter_alpha,
            max_speed_threshold=max_speed_threshold,
            max_jump_m=max_jump_m,
        )

        self._last_valid_fix: Optional[GPSFix] = None
        self._last_valid_time: Optional[float] = None
        self._latest_filtered_pos: Optional[FilteredPosition] = None
        self._total_fixes_ingested: int = 0
        self._rejected_fixes: int = 0

    @property
    def latest_position(self) -> Optional[FilteredPosition]:
        return self._latest_filtered_pos

    @property
    def last_valid_fix(self) -> Optional[GPSFix]:
        return self._last_valid_fix

    @property
    def total_fixes_ingested(self) -> int:
        return self._total_fixes_ingested

    def is_healthy(self, current_time: Optional[float] = None) -> bool:
        """Check if GPS signal is healthy, valid, and recent."""
        if self._last_valid_fix is None or self._last_valid_time is None:
            return False
        now = current_time if current_time is not None else time.time()
        if (now - self._last_valid_time) > self.signal_timeout:
            return False
        return True

    def has_signal_loss(self, current_time: Optional[float] = None) -> bool:
        """Return True if GPS signal has timed out or is lost."""
        return not self.is_healthy(current_time)

    def reset(self) -> None:
        """Reset pipeline states."""
        self.filter.reset()
        self._last_valid_fix = None
        self._last_valid_time = None
        self._latest_filtered_pos = None
        self._total_fixes_ingested = 0
        self._rejected_fixes = 0

    def set_geo_reference(self, geo_ref: GeoReference) -> None:
        """Explicitly set or update the coordinate transformation datum."""
        self.geo_ref = geo_ref

    def ingest_nmea(
        self,
        sentence: str,
        current_time: Optional[float] = None,
    ) -> Optional[FilteredPosition]:
        """Parse an NMEA sentence and process the resulting fix through the pipeline."""
        self._total_fixes_ingested += 1
        fix = parse_nmea_sentence(sentence)
        if fix is None:
            self._rejected_fixes += 1
            return None
        return self.ingest_fix(fix, current_time=current_time)

    def ingest_coordinates(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0,
        timestamp: Optional[float] = None,
        reported_speed: Optional[float] = None,
        course_rad: Optional[float] = None,
        hdop: float = 1.0,
        num_sats: int = 8,
        quality: GPSQuality = GPSQuality.SINGLE_SPS,
    ) -> Optional[FilteredPosition]:
        """Directly ingest lat, lon coordinates."""
        fix = GPSFix(
            lat=lat,
            lon=lon,
            alt=alt,
            timestamp=timestamp if timestamp is not None else time.time(),
            quality=quality,
            num_sats=num_sats,
            hdop=hdop,
            speed_m_s=reported_speed,
            course_rad=course_rad,
            is_valid=True,
        )
        return self.ingest_fix(fix, current_time=timestamp)

    def ingest_fix(
        self,
        fix: GPSFix,
        current_time: Optional[float] = None,
    ) -> Optional[FilteredPosition]:
        """Ingest a GPSFix object, convert to ENU, and apply filtering."""
        now = current_time if current_time is not None else (fix.timestamp or time.time())

        # Quality and validity checks
        if not fix.is_valid or fix.quality == GPSQuality.INVALID:
            self._rejected_fixes += 1
            return None

        if fix.hdop > self.max_hdop:
            self._rejected_fixes += 1
            return None

        if fix.num_sats > 0 and fix.num_sats < self.min_satellites:
            self._rejected_fixes += 1
            return None

        # Auto-anchor datum on first valid fix if not already set
        if self.geo_ref is None:
            self.geo_ref = GeoReference(
                origin_lat=fix.lat,
                origin_lon=fix.lon,
                origin_alt=fix.alt,
            )

        # Coordinate transformation WGS-84 -> ENU
        raw_x, raw_y, raw_z = self.geo_ref.gps_to_enu(fix.lat, fix.lon, fix.alt)

        # Filter smoothing and jitter mitigation
        filt_x, filt_y, filt_z, is_outlier, speed, jitter = self.filter.update(
            x=raw_x,
            y=raw_y,
            z=raw_z,
            timestamp=now,
            reported_speed=fix.speed_m_s,
        )

        # Convert filtered ENU back to geodetic for filtered GPS coordinates
        filt_lat, filt_lon, filt_alt = self.geo_ref.enu_to_gps(filt_x, filt_y, filt_z)

        self._last_valid_fix = fix
        self._last_valid_time = now

        filtered_pos = FilteredPosition(
            x=filt_x,
            y=filt_y,
            z=filt_z,
            lat=filt_lat,
            lon=filt_lon,
            alt=filt_alt,
            speed_m_s=speed,
            timestamp=now,
            hdop=fix.hdop,
            quality=fix.quality,
            is_outlier=is_outlier,
            jitter_estimate=jitter,
        )
        self._latest_filtered_pos = filtered_pos
        return filtered_pos


# Alias for backward-compatible or alternative naming
GPSPipeline = GPSSource

"""Module to parse standard DEM and GeoTIFF files into ElevationGrid objects."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from terrain_mapper.elevation import ElevationGrid


def parse_ascii_dem(source: str | Path | bytes | io.BytesIO) -> ElevationGrid:
    """Parse an ASCII DEM / ESRI ASCII Grid (.asc, .dem).

    Expected header keywords (case-insensitive):
      ncols <int>
      nrows <int>
      xllcorner / xllcenter <float>
      yllcorner / yllcenter <float>
      cellsize <float> (or dx / dy)
      nodata_value <float> (optional, defaults to -9999)
    Followed by nrows lines of ncols elevation values.
    """
    if isinstance(source, (str, Path)) and Path(source).exists():
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif isinstance(source, bytes):
        text = source.decode("utf-8", errors="replace")
    elif isinstance(source, io.BytesIO):
        text = source.getvalue().decode("utf-8", errors="replace")
    elif isinstance(source, str):
        text = source
    else:
        raise ValueError(f"Unsupported ASCII DEM source type: {type(source)}")

    lines = text.strip().splitlines()
    header: dict[str, float] = {}
    data_start_line = 0

    header_keys = {
        "ncols",
        "nrows",
        "xllcorner",
        "yllcorner",
        "xllcenter",
        "yllcenter",
        "cellsize",
        "dx",
        "dy",
        "nodata_value",
    }

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0].lower()
        if key in header_keys and len(parts) >= 2:
            try:
                header[key] = float(parts[1])
                data_start_line = idx + 1
            except ValueError:
                break
        else:
            break

    if "ncols" not in header or "nrows" not in header:
        raise ValueError("ASCII DEM missing required 'ncols' or 'nrows' in header")

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])

    cell_size_x = header.get("cellsize", header.get("dx", 1.0))
    cell_size_y = header.get("cellsize", header.get("dy", cell_size_x))

    if cell_size_x <= 0 or cell_size_y <= 0:
        raise ValueError("ASCII DEM cell sizes must be positive")

    nodata_val = header.get("nodata_value", -9999.0)

    # Determine bounds
    if "xllcorner" in header:
        min_x = header["xllcorner"]
    elif "xllcenter" in header:
        min_x = header["xllcenter"] - 0.5 * cell_size_x
    else:
        min_x = 0.0

    if "yllcorner" in header:
        min_y = header["yllcorner"]
    elif "yllcenter" in header:
        min_y = header["yllcenter"] - 0.5 * cell_size_y
    else:
        min_y = 0.0

    max_x = min_x + ncols * cell_size_x
    max_y = min_y + nrows * cell_size_y

    # Parse numeric data
    data_text = " ".join(lines[data_start_line:])
    values_1d = np.fromstring(data_text, dtype=np.float32, sep=" ")

    if values_1d.size != nrows * ncols:
        # Try line-by-line reading in case formatting varies
        tokens: list[float] = []
        for line in lines[data_start_line:]:
            for tok in line.strip().split():
                try:
                    tokens.append(float(tok))
                except ValueError:
                    continue
        values_1d = np.asarray(tokens, dtype=np.float32)

    if values_1d.size != nrows * ncols:
        raise ValueError(
            f"ASCII DEM data size mismatch: expected {nrows}x{ncols} = {nrows * ncols} values, got {values_1d.size}"
        )

    values_2d = values_1d.reshape((nrows, ncols))

    # Replace nodata values with NaN
    if nodata_val is not None:
        is_nodata = np.isclose(values_2d, nodata_val, atol=1e-3) | (values_2d <= -9990.0)
        values_2d[is_nodata] = np.nan

    # In ASCII DEM, row 0 is top (North, max_y)
    transform = (cell_size_x, 0.0, min_x, 0.0, -cell_size_y, max_y)

    return ElevationGrid(
        values_m=values_2d,
        cell_size_x_m=float(cell_size_x),
        cell_size_y_m=float(cell_size_y),
        crs="local-meters",
        bounds=(float(min_x), float(min_y), float(max_x), float(max_y)),
        transform=transform,
        nodata=float(nodata_val),
    )


def parse_geotiff(source: str | Path | bytes | io.BytesIO) -> ElevationGrid:
    """Parse a GeoTIFF (.tif, .tiff) elevation raster using Rasterio."""
    import rasterio
    from rasterio.io import MemoryFile

    def _extract_from_dataset(dataset: Any) -> ElevationGrid:
        masked = dataset.read(1, masked=True).astype(np.float32)
        values = np.asarray(masked.filled(np.nan), dtype=np.float32)

        transform = dataset.transform
        bounds = dataset.bounds
        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)

        crs_obj = dataset.crs
        crs_str = crs_obj.to_string() if crs_obj is not None else "local-meters"

        # Check if coordinates are geographic (degrees) and convert cell sizes to meters
        if crs_obj is not None and getattr(crs_obj, "is_geographic", False):
            center_lat = (bounds.bottom + bounds.top) / 2.0
            lat_rad = math.radians(center_lat)
            meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad)
            meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
            cell_size_x_m = cell_size_x * meters_per_deg_lon
            cell_size_y_m = cell_size_y * meters_per_deg_lat
        else:
            cell_size_x_m = cell_size_x
            cell_size_y_m = cell_size_y

        nodata_val = dataset.nodata
        transform_tuple = (
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        )

        return ElevationGrid(
            values_m=values,
            cell_size_x_m=float(cell_size_x_m),
            cell_size_y_m=float(cell_size_y_m),
            crs=crs_str,
            bounds=(
                float(bounds.left),
                float(bounds.bottom),
                float(bounds.right),
                float(bounds.top),
            ),
            transform=transform_tuple,
            nodata=float(nodata_val) if nodata_val is not None else None,
        )

    if isinstance(source, (bytes, io.BytesIO)):
        raw_bytes = source if isinstance(source, bytes) else source.getvalue()
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as dataset:
                return _extract_from_dataset(dataset)
    else:
        with rasterio.open(source) as dataset:
            return _extract_from_dataset(dataset)


def parse_dem(source: str | Path | bytes | io.BytesIO) -> ElevationGrid:
    """Universal parser for standard DEM formats (GeoTIFF, ASCII DEM, USGS DEM)."""
    # Check if raw bytes or BytesIO
    if isinstance(source, (bytes, io.BytesIO)):
        raw_bytes = source if isinstance(source, bytes) else source.getvalue()
        # Check TIFF magic bytes
        if raw_bytes.startswith(b"II*\x00") or raw_bytes.startswith(b"MM\x00*"):
            return parse_geotiff(raw_bytes)
        # Check if ASCII DEM text
        try:
            head = raw_bytes[:200].decode("utf-8", errors="ignore").lower()
            if "ncols" in head or "nrows" in head:
                return parse_ascii_dem(raw_bytes)
        except Exception:
            pass
        # Fallback to rasterio MemoryFile
        try:
            return parse_geotiff(raw_bytes)
        except Exception:
            return parse_ascii_dem(raw_bytes)

    # Check path / string
    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        return parse_geotiff(path)
    elif suffix in {".asc"}:
        return parse_ascii_dem(path)
    elif suffix in {".dem"}:
        # .dem can be ASCII grid or binary USGS DEM
        try:
            with open(path, "rb") as f:
                head = f.read(200).decode("utf-8", errors="ignore").lower()
            if "ncols" in head or "nrows" in head:
                return parse_ascii_dem(path)
        except Exception:
            pass
        # Try GeoTIFF / GDAL rasterio first, then ASCII
        try:
            return parse_geotiff(path)
        except Exception:
            return parse_ascii_dem(path)
    else:
        # Default try geotiff then ascii
        try:
            return parse_geotiff(path)
        except Exception:
            return parse_ascii_dem(path)


# Alias DEMData to ElevationGrid for intuitive typing
DEMData = ElevationGrid

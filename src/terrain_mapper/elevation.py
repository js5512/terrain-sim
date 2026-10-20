from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ElevationGrid:
    """Elevation samples and pixel spacing, with invalid cells represented by NaN."""

    values_m: NDArray[np.float32]
    cell_size_x_m: float
    cell_size_y_m: float
    crs: str
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    transform: tuple[float, float, float, float, float, float] | None = None
    nodata: float | None = None

    def __post_init__(self) -> None:
        if self.values_m.ndim != 2:
            raise ValueError("elevation grid must be a 2D array")
        if self.cell_size_x_m <= 0 or self.cell_size_y_m <= 0:
            raise ValueError("cell sizes must be positive")
        if not self.crs:
            raise ValueError("a projected CRS is required")

        rows, cols = self.values_m.shape
        if self.bounds == (0.0, 0.0, 0.0, 0.0):
            max_x = float(self.cell_size_x_m * max(cols - 1, 1))
            max_y = float(self.cell_size_y_m * max(rows - 1, 1))
            object.__setattr__(self, "bounds", (0.0, 0.0, max_x, max_y))

        if self.transform is None:
            min_x, min_y, _, _ = self.bounds
            object.__setattr__(
                self,
                "transform",
                (float(self.cell_size_x_m), 0.0, float(min_x), 0.0, float(self.cell_size_y_m), float(min_y)),
            )

    @property
    def min_x(self) -> float:
        return self.bounds[0]

    @property
    def min_y(self) -> float:
        return self.bounds[1]

    @property
    def max_x(self) -> float:
        return self.bounds[2]

    @property
    def max_y(self) -> float:
        return self.bounds[3]

    @property
    def x_coords(self) -> NDArray[np.float32]:
        """1D array of real-world X coordinates for columns."""
        a, b, c, d, e, f = self.transform  # type: ignore[misc]
        cols = self.values_m.shape[1]
        col_indices = np.arange(cols, dtype=np.float32)
        return (c + col_indices * a).astype(np.float32)

    @property
    def y_coords(self) -> NDArray[np.float32]:
        """1D array of real-world Y coordinates for rows."""
        a, b, c, d, e, f = self.transform  # type: ignore[misc]
        rows = self.values_m.shape[0]
        row_indices = np.arange(rows, dtype=np.float32)
        return (f + row_indices * e).astype(np.float32)

    @property
    def x_m(self) -> NDArray[np.float32]:
        """2D meshgrid of real-world X coordinates matching array shape."""
        a, b, c, d, e, f = self.transform  # type: ignore[misc]
        rows, cols = self.values_m.shape
        col_idx = np.arange(cols, dtype=np.float32)
        row_idx = np.arange(rows, dtype=np.float32)
        cc, rr = np.meshgrid(col_idx, row_idx)
        return (c + cc * a + rr * b).astype(np.float32)

    @property
    def y_m(self) -> NDArray[np.float32]:
        """2D meshgrid of real-world Y coordinates matching array shape."""
        a, b, c, d, e, f = self.transform  # type: ignore[misc]
        rows, cols = self.values_m.shape
        col_idx = np.arange(cols, dtype=np.float32)
        row_idx = np.arange(rows, dtype=np.float32)
        cc, rr = np.meshgrid(col_idx, row_idx)
        return (f + cc * d + rr * e).astype(np.float32)

    def grid_to_world(self, row: int | float, col: int | float) -> tuple[float, float]:
        """Convert grid cell (row, col) to real-world (x, y) coordinates."""
        a, b, c, d, e, f = self.transform  # type: ignore[misc]
        x = float(c + col * a + row * b)
        y = float(f + col * d + row * e)
        return x, y

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert real-world (x, y) coordinates to nearest (row, col) cell indices."""
        a, b, c, d, e, f = self.transform  # type: ignore[misc]
        rows, cols = self.values_m.shape
        det = a * e - b * d
        if abs(det) < 1e-9:
            # Fallback to bounds ratio
            dx = (self.max_x - self.min_x) if self.max_x != self.min_x else 1.0
            dy = (self.max_y - self.min_y) if self.max_y != self.min_y else 1.0
            col = int(np.clip(round((x - self.min_x) * (cols - 1) / dx), 0, cols - 1))
            row = int(np.clip(round((y - self.min_y) * (rows - 1) / dy), 0, rows - 1))
            return row, col

        col_f = ((x - c) * e - (y - f) * b) / det
        row_f = (a * (y - f) - d * (x - c)) / det
        col = int(np.clip(round(col_f), 0, cols - 1))
        row = int(np.clip(round(row_f), 0, rows - 1))
        return row, col


def load_dem(path: str | Path) -> ElevationGrid:
    """Load a projected DEM or GeoTIFF file using the DEM parser."""
    from terrain_mapper.dem import parse_dem

    return parse_dem(path)
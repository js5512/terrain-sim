"""Animated 3D terrain-routing demo; run with ``python -m terrain_mapper.simulation``."""

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib


def _has_desktop_display() -> bool:
    return sys.platform in {"win32", "darwin"} or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


if (
    _has_desktop_display()
    and "MPLBACKEND" not in os.environ
    and importlib.util.find_spec("PyQt6") is not None
):
    matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from terrain_mapper.config import PlannerConfig
from terrain_mapper.costmap import build_costmap
from terrain_mapper.elevation import ElevationGrid
from terrain_mapper.planner import find_route


TERRAIN_EXTENT_M = 1000.0
DEFAULT_GRID_SIZE = 160
HEIGHTMAP_RELIEF_M = 60.0


@dataclass(frozen=True)
class TerrainConfig:
    """Parameters for synthetic terrain generation and routing behavior."""

    slope_steepness: float = 1.0
    height_variation: float = 1.0
    obstacle_density: float = 0.08
    roughness: float = 1.0
    seed: int = 7
    max_slope_degrees: float = 45.0
    route_cost_weight: float = 7.0

    def __post_init__(self) -> None:
        if self.slope_steepness <= 0:
            raise ValueError("slope_steepness must be positive")
        if self.height_variation <= 0:
            raise ValueError("height_variation must be positive")
        if not 0 <= self.obstacle_density <= 1.0:
            raise ValueError("obstacle_density must be between 0 and 1")
        if self.roughness <= 0:
            raise ValueError("roughness must be positive")
        if self.max_slope_degrees <= 0 or self.max_slope_degrees >= 90:
            raise ValueError("max_slope_degrees must be between 0 and 90")
        if self.route_cost_weight <= 0:
            raise ValueError("route_cost_weight must be positive")


@dataclass(frozen=True)
class TerrainScenario:
    grid: ElevationGrid
    x_m: NDArray[np.float32]
    y_m: NDArray[np.float32]
    obstacles: NDArray[np.bool_]
    start: tuple[int, int]
    goal: tuple[int, int]
    path: list[tuple[int, int]]
    route_cost: float


def _resize_to_shape(array: NDArray[np.float32], target_shape: tuple[int, int]) -> NDArray[np.float32]:
    rows, cols = array.shape
    target_rows, target_cols = target_shape
    row_indices = np.linspace(0, rows - 1, target_rows)
    col_indices = np.linspace(0, cols - 1, target_cols)
    row_indices = np.clip(np.rint(row_indices).astype(int), 0, rows - 1)
    col_indices = np.clip(np.rint(col_indices).astype(int), 0, cols - 1)
    return array[row_indices[:, None], col_indices[None, :]].astype(np.float32)


def load_heightmap(
    path: str | Path | NDArray[np.float32], size: int | None = None
) -> NDArray[np.float32]:
    """Load and normalize a grayscale heightmap from disk or an array."""
    if isinstance(path, (str, Path)):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"heightmap not found: {path}")

        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            from PIL import Image

            image = Image.open(path).convert("L")
            if size is not None and image.size != (size, size):
                image = image.resize((size, size), resample=Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
    else:
        array = np.asarray(path)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("heightmap must be a 2D grayscale array or image")
    if size is not None and array.shape != (size, size):
        array = _resize_to_shape(array, (size, size))

    if not np.isfinite(array).all():
        raise ValueError("heightmap contains NaN or Inf values")
    array = array - np.min(array)
    maximum = float(np.max(array))
    if maximum > 0:
        array = array / maximum
    return array.astype(np.float32)


def _nearest_traversable_cell(costmap: NDArray[np.float32], target: tuple[int, int]) -> tuple[int, int]:
    traversable = np.isfinite(costmap)
    target_row, target_col = target
    if traversable[target_row, target_col]:
        return target

    candidates = np.argwhere(traversable)
    if candidates.size == 0:
        raise ValueError("terrain has no traversable cells")
    distances = np.linalg.norm(candidates - np.array([target_row, target_col], dtype=float), axis=1)
    best_index = int(np.argmin(distances))
    return (int(candidates[best_index, 0]), int(candidates[best_index, 1]))


def build_synthetic_scenario(
    size: int = DEFAULT_GRID_SIZE,
    config: TerrainConfig | None = None,
    heightmap: str | Path | NDArray[np.float32] | None = None,
) -> TerrainScenario:
    """Create a hilly local-meter grid and route between opposite sides."""
    if size < 20:
        raise ValueError("size must be at least 20 cells")
    config = config or TerrainConfig()

    coordinates = np.linspace(0.0, TERRAIN_EXTENT_M, size, dtype=np.float32)
    x_m, y_m = np.meshgrid(coordinates, coordinates)

    if heightmap is not None:
        elevation = load_heightmap(heightmap, size=size)
        elevation = elevation * HEIGHTMAP_RELIEF_M * config.height_variation
        obstacles = np.zeros((size, size), dtype=bool)
    else:
        rng = np.random.default_rng(config.seed)
        surface_noise = rng.standard_normal((size, size)).astype(np.float32)
        obstacles = rng.random((size, size)) < config.obstacle_density
        base_slope = config.slope_steepness * (0.06 * x_m + 0.045 * y_m)
        mound_a = 32.0 * config.height_variation * np.exp(-((x_m - 270.0) ** 2 + (y_m - 650.0) ** 2) / 25000.0)
        mound_b = 28.0 * config.height_variation * np.exp(-((x_m - 730.0) ** 2 + (y_m - 330.0) ** 2) / 28000.0)
        ridge = 22.0 * config.height_variation * np.exp(-((x_m - 500.0) ** 2 + (y_m - 830.0) ** 2) / 48000.0)
        roughness = config.roughness * 0.12 * surface_noise
        elevation = (
            base_slope
            + mound_a
            + mound_b
            - ridge
            + roughness
            + 1.5 * np.sin(x_m / 100.0) * np.cos(y_m / 130.0) * config.height_variation
        ).astype(np.float32)

    cell_size = float(coordinates[1] - coordinates[0])
    grid = ElevationGrid(
        values_m=elevation,
        cell_size_x_m=cell_size,
        cell_size_y_m=cell_size,
        crs="synthetic-local-meters",
        bounds=(0.0, 0.0, TERRAIN_EXTENT_M, TERRAIN_EXTENT_M),
    )
    costs = build_costmap(
        grid,
        PlannerConfig(
            max_slope_degrees=config.max_slope_degrees,
            slope_cost_weight=config.route_cost_weight,
        ),
    )
    costs[obstacles] = np.inf

    start = (size // 2, size // 12)
    goal = (size // 2, size - 1 - size // 12)
    try:
        start = _nearest_traversable_cell(costs, start)
        goal = _nearest_traversable_cell(costs, goal)
        path, route_cost = find_route(
            costs,
            start,
            goal,
            cell_size_x=grid.cell_size_x_m,
            cell_size_y=grid.cell_size_y_m,
        )
    except ValueError:
        path, route_cost = [], float("inf")

    return TerrainScenario(grid, x_m, y_m, obstacles, start, goal, path, route_cost)


def build_dem_scenario(
    dem_source: str | Path | ElevationGrid,
    config: TerrainConfig | None = None,
    size: int | None = None,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
) -> TerrainScenario:
    """Create a TerrainScenario from real-world DEM or GeoTIFF elevation data."""
    from terrain_mapper.dem import parse_dem

    if isinstance(dem_source, ElevationGrid):
        grid = dem_source
    else:
        grid = parse_dem(dem_source)

    if size is not None and grid.values_m.shape != (size, size):
        resized_values = _resize_to_shape(grid.values_m, (size, size))
        cell_size_x = (grid.max_x - grid.min_x) / max(size - 1, 1)
        cell_size_y = (grid.max_y - grid.min_y) / max(size - 1, 1)
        grid = ElevationGrid(
            values_m=resized_values,
            cell_size_x_m=float(cell_size_x),
            cell_size_y_m=float(cell_size_y),
            crs=grid.crs,
            bounds=grid.bounds,
        )

    config = config or TerrainConfig()
    costs = build_costmap(
        grid,
        PlannerConfig(
            max_slope_degrees=config.max_slope_degrees,
            slope_cost_weight=config.route_cost_weight,
        ),
    )
    obstacles = np.isnan(grid.values_m)
    costs[obstacles] = np.inf

    rows, cols = grid.values_m.shape
    start = start or (rows // 2, cols // 12)
    goal = goal or (rows // 2, cols - 1 - cols // 12)

    try:
        start = _nearest_traversable_cell(costs, start)
        goal = _nearest_traversable_cell(costs, goal)
        path, route_cost = find_route(
            costs,
            start,
            goal,
            cell_size_x=grid.cell_size_x_m,
            cell_size_y=grid.cell_size_y_m,
        )
    except ValueError:
        path, route_cost = [], float("inf")

    return TerrainScenario(
        grid=grid,
        x_m=grid.x_m,
        y_m=grid.y_m,
        obstacles=obstacles,
        start=start,
        goal=goal,
        path=path,
        route_cost=route_cost,
    )


def _box_faces(
    center: NDArray[np.float64],
    forward: NDArray[np.float64],
    lateral: NDArray[np.float64],
    up: NDArray[np.float64],
    length: float,
    width: float,
    bottom: float,
    top: float,
    x_offset: float = 0.0,
) -> list[NDArray[np.float64]]:
    half_length = length / 2.0
    half_width = width / 2.0
    local_vertices = np.array(
        [
            [x_offset - half_length, -half_width, bottom],
            [x_offset + half_length, -half_width, bottom],
            [x_offset + half_length, half_width, bottom],
            [x_offset - half_length, half_width, bottom],
            [x_offset - half_length, -half_width, top],
            [x_offset + half_length, -half_width, top],
            [x_offset + half_length, half_width, top],
            [x_offset - half_length, half_width, top],
        ],
        dtype=np.float64,
    )
    vertices = (
        center[None, :] +
        local_vertices[:, 0, None] * forward +
        local_vertices[:, 1, None] * lateral +
        local_vertices[:, 2, None] * up
    )
    face_indices = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    )
    return [vertices[list(indices)] for indices in face_indices]


def _vehicle_faces(
    position: NDArray[np.float64], heading: float, pitch: float
) -> list[NDArray[np.float64]]:
    forward = np.array(
        [np.cos(heading) * np.cos(pitch), np.sin(heading) * np.cos(pitch), np.sin(pitch)]
    )
    lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
    up = np.cross(forward, lateral)
    body = _box_faces(position, forward, lateral, up, 3.2, 1.8, 0.25, 1.05)
    cabin = _box_faces(position, forward, lateral, up, 1.45, 1.48, 1.05, 1.8, -0.15)
    return body + cabin


def animate_scenario(
    scenario: TerrainScenario,
    interval_ms: int = 40,
) -> FuncAnimation:
    """Open a Matplotlib 3D window and animate the routed vehicle."""
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    axes = figure.add_subplot(111, projection="3d", computed_zorder=False)
    elevations = scenario.grid.values_m
    surface = axes.plot_surface(
        scenario.x_m,
        scenario.y_m,
        elevations,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        rcount=min(elevations.shape[0], 120),
        ccount=min(elevations.shape[1], 120),
        alpha=0.94,
    )
    surface.set_zorder(1)

    path_rows = np.array([cell[0] for cell in scenario.path], dtype=int)
    path_columns = np.array([cell[1] for cell in scenario.path], dtype=int)
    path_x = scenario.x_m[path_rows, path_columns]
    path_y = scenario.y_m[path_rows, path_columns]
    path_z = elevations[path_rows, path_columns] + 0.08
    planned_route, = axes.plot(
        path_x,
        path_y,
        path_z,
        color="#f4d35e",
        linewidth=2.0,
        linestyle="--",
        zorder=5,
    )
    trail, = axes.plot([], [], [], color="#ef6a35", linewidth=3.0)
    trail.set_zorder(6)

    start = scenario.start
    goal = scenario.goal
    start_marker = axes.scatter(
        [scenario.x_m[start]],
        [scenario.y_m[start]],
        [elevations[start] + 0.15],
        color="#38d9c5",
        s=55,
        label="Start",
    )
    start_marker.set_zorder(7)
    goal_marker = axes.scatter(
        [scenario.x_m[goal]],
        [scenario.y_m[goal]],
        [elevations[goal] + 0.15],
        color="#f4d35e",
        s=55,
        label="Goal",
    )
    goal_marker.set_zorder(7)

    vehicle = Poly3DCollection(
        [],
        facecolors=["#ef6a35"] * 6 + ["#314c53"] * 6,
        edgecolors="#24343a",
        linewidths=0.6,
        alpha=0.98,
        zorder=8,
    )
    axes.add_collection3d(vehicle)

    axes.set(
        xlabel="East (m)",
        ylabel="North (m)",
        zlabel="Elevation (m)",
        xlim=(float(scenario.x_m.min()), float(scenario.x_m.max())),
        ylim=(float(scenario.y_m.min()), float(scenario.y_m.max())),
    )
    axes.xaxis.set_major_formatter(FormatStrFormatter("%.0f m"))
    axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f m"))
    axes.set_zlim(float(elevations.min() - 1.0), float(elevations.max() + 4.0))
    axes.set_box_aspect((1.0, 1.0, 0.42))
    axes.view_init(elev=34, azim=-58)
    axes.legend(loc="upper left")

    def update(frame: int) -> tuple[Poly3DCollection, object]:
        if not scenario.path:
            axes.set_title("No traversable route")
            vehicle.set_verts([])
            return vehicle, trail, planned_route

        row, column = scenario.path[frame]
        if frame < len(scenario.path) - 1:
            segment_start = (row, column)
            segment_end = scenario.path[frame + 1]
        else:
            segment_start = scenario.path[frame - 1]
            segment_end = (row, column)

        start_row, start_column = segment_start
        end_row, end_column = segment_end
        dx = float(
            scenario.x_m[end_row, end_column]
            - scenario.x_m[start_row, start_column]
        )
        dy = float(
            scenario.y_m[end_row, end_column]
            - scenario.y_m[start_row, start_column]
        )
        dz = float(
            elevations[end_row, end_column]
            - elevations[start_row, start_column]
        )
        horizontal_distance = max(float(np.hypot(dx, dy)), 1e-6)
        heading = float(np.arctan2(dy, dx))
        pitch = float(np.arctan2(dz, horizontal_distance))
        position = np.array(
            [scenario.x_m[row, column], scenario.y_m[row, column], elevations[row, column]],
            dtype=np.float64,
        )
        vehicle.set_verts(_vehicle_faces(position, heading, pitch))
        trail.set_data(path_x[: frame + 1], path_y[: frame + 1])
        trail.set_3d_properties(path_z[: frame + 1])
        axes.set_title(
            f"Terrain route | {frame + 1}/{len(scenario.path)} cells | "
            f"cost {scenario.route_cost:.1f}"
        )
        return vehicle, trail, planned_route

    return FuncAnimation(
        figure,
        update,
        frames=max(1, len(scenario.path)),
        interval=interval_ms,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate terrain-aware vehicle routing")
    parser.add_argument("--size", type=int, default=DEFAULT_GRID_SIZE, help="terrain width and height in cells")
    parser.add_argument("--interval", type=int, default=40, help="frame interval in milliseconds")
    parser.add_argument("--save", type=Path, help="save animation as a GIF instead of opening a window")
    parser.add_argument("--slope-steepness", type=float, default=1.0, help="base terrain slope multiplier")
    parser.add_argument("--height-variation", type=float, default=1.0, help="height amplitude multiplier")
    parser.add_argument("--obstacle-density", type=float, default=0.08, help="fraction of cells used as synthetic obstacles")
    parser.add_argument("--roughness", type=float, default=1.0, help="roughness intensity for local terrain noise")
    parser.add_argument("--heightmap", type=Path, help="optional grayscale elevation image or .npy heightmap")
    parser.add_argument("--dem", type=Path, help="optional path to real-world DEM or GeoTIFF file")
    args = parser.parse_args()

    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if not args.save and (
        not _has_desktop_display()
        or matplotlib.get_backend().lower() in noninteractive_backends
    ):
        parser.error(
            "an interactive Matplotlib backend and desktop display are required. On a desktop, "
            "install with `python -m pip install -e '.[gui]'` and unset MPLBACKEND; "
            "in a headless session, use `--save /path/to/terrain.gif`."
        )

    config = TerrainConfig(
        slope_steepness=args.slope_steepness,
        height_variation=args.height_variation,
        obstacle_density=args.obstacle_density,
        roughness=args.roughness,
    )
    if args.dem:
        scenario = build_dem_scenario(args.dem, config=config, size=args.size)
    else:
        scenario = build_synthetic_scenario(
            size=args.size,
            config=config,
            heightmap=args.heightmap,
        )
    if args.save:
        if args.save.suffix.lower() != ".gif":
            parser.error("--save currently supports .gif output")
        animation = animate_scenario(scenario, interval_ms=args.interval)
        animation.save(args.save, writer=PillowWriter(fps=max(1, 1000 // args.interval)))
        plt.close(animation._fig)
    else:
        from terrain_mapper.interactive import InteractiveTerrainEditor

        InteractiveTerrainEditor(scenario, config, interval_ms=args.interval).show()


if __name__ == "__main__":
    main()
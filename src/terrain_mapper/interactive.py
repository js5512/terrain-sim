"""Interactive 3D terrain viewer with a live top-down terrain editor."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from terrain_mapper.config import PlannerConfig
from terrain_mapper.costmap import build_costmap
from terrain_mapper.elevation import ElevationGrid
from terrain_mapper.planner import find_route
from terrain_mapper.simulation import (
    TerrainConfig,
    TerrainScenario,
    _nearest_traversable_cell,
    _vehicle_faces,
)


class InteractiveTerrainEditor:
    """Display terrain, animate a routed vehicle, and support brush-based edits."""

    def __init__(
        self,
        scenario: TerrainScenario,
        config: TerrainConfig,
        interval_ms: int = 60,
    ) -> None:
        self.config = config
        self.elevation = scenario.grid.values_m.copy()
        self.obstacles = scenario.obstacles.copy()
        self.x_m = scenario.x_m
        self.y_m = scenario.y_m
        self.cell_size_x_m = scenario.grid.cell_size_x_m
        self.cell_size_y_m = scenario.grid.cell_size_y_m
        self.start = scenario.start
        self.goal = scenario.goal
        self.path = list(scenario.path)
        self.route_cost = scenario.route_cost
        self.vehicle_cell = self.path[0] if self.path else self.start
        self.vehicle_index = 0
        self.mode = "Raise"
        self._dragging = False
        self._last_brush_cell: tuple[int, int] | None = None
        self._paused = False

        self.figure = plt.figure(figsize=(15, 9), constrained_layout=False)
        self.axes_3d = self.figure.add_axes([0.025, 0.10, 0.63, 0.84], projection="3d", computed_zorder=False)
        self.edit_axes = self.figure.add_axes([0.69, 0.43, 0.285, 0.51])
        self.mode_axes = self.figure.add_axes([0.69, 0.08, 0.14, 0.24])
        self.radius_axes = self.figure.add_axes([0.86, 0.275, 0.115, 0.025])
        self.strength_axes = self.figure.add_axes([0.86, 0.19, 0.115, 0.025])
        self.pause_axes = self.figure.add_axes([0.86, 0.09, 0.115, 0.05])
        self.status_text = self.figure.text(0.69, 0.37, "", va="center", ha="left", fontsize=9)
        self.figure.text(0.86, 0.31, "Brush radius (cells)", va="center", ha="left", fontsize=9)
        self.figure.text(0.86, 0.225, "Elevation strength (m)", va="center", ha="left", fontsize=9)

        self._draw_initial_scene()
        self.mode_widget = RadioButtons(
            self.mode_axes,
            ("Raise", "Lower", "Obstacle", "Erase", "Spawn", "Goal"),
            active=0,
        )
        self.mode_axes.set_title("Edit mode", loc="left", fontsize=10)
        self.mode_widget.on_clicked(self._select_mode)
        self.radius_widget = Slider(
            self.radius_axes,
            "",
            1,
            10,
            valinit=3,
            valstep=1,
        )
        self.strength_widget = Slider(
            self.strength_axes,
            "",
            0.1,
            3.0,
            valinit=0.8,
        )
        self.pause_widget = Button(self.pause_axes, "Pause")
        self.pause_widget.on_clicked(self._toggle_pause)

        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._replan()
        self.animation = FuncAnimation(
            self.figure,
            self._tick,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def _draw_initial_scene(self) -> None:
        self.axes_3d.set_title("Live terrain and vehicle")
        self.surface = self.axes_3d.plot_surface(
            self.x_m,
            self.y_m,
            self.elevation,
            cmap="terrain",
            linewidth=0,
            antialiased=True,
            rcount=min(self.elevation.shape[0], 120),
            ccount=min(self.elevation.shape[1], 120),
            alpha=0.95,
            zorder=1,
        )
        self.route_3d, = self.axes_3d.plot([], [], [], color="#f4d35e", linewidth=2.2, zorder=5)
        self.trail_3d, = self.axes_3d.plot([], [], [], color="#ef6a35", linewidth=3.0, zorder=6)
        self.start_3d = self.axes_3d.scatter([], [], [], color="#38d9c5", s=50, zorder=7, label="Start")
        self.goal_3d = self.axes_3d.scatter([], [], [], color="#f4d35e", s=50, zorder=7, label="Goal")
        self.obstacle_3d = self.axes_3d.scatter([], [], [], color="#6b3729", marker="^", s=30, zorder=7)
        self.vehicle = Poly3DCollection(
            [],
            facecolors=["#ef6a35"] * 6 + ["#314c53"] * 6,
            edgecolors="#24343a",
            linewidths=0.6,
            alpha=0.99,
            zorder=8,
        )
        self.axes_3d.add_collection3d(self.vehicle)
        self.axes_3d.set(
            xlabel="East (m)",
            ylabel="North (m)",
            zlabel="Elevation (m)",
            xlim=(float(self.x_m.min()), float(self.x_m.max())),
            ylim=(float(self.y_m.min()), float(self.y_m.max())),
        )
        self.axes_3d.set_box_aspect((1, 1, 0.42))
        self.axes_3d.view_init(elev=34, azim=-58)
        self.axes_3d.legend(loc="upper left")

        extent = (
            float(self.x_m.min()),
            float(self.x_m.max()),
            float(self.y_m.min()),
            float(self.y_m.max()),
        )
        self.height_image = self.edit_axes.imshow(
            self.elevation,
            origin="lower",
            extent=extent,
            cmap="terrain",
            interpolation="nearest",
            aspect="equal",
        )
        self.edit_axes.set_title("Terrain editor | drag to edit")
        self.edit_axes.set_xlabel("East (m)")
        self.edit_axes.set_ylabel("North (m)")
        self.route_2d, = self.edit_axes.plot([], [], color="#ffe56b", linewidth=1.5)
        self.vehicle_2d, = self.edit_axes.plot([], [], marker="o", color="#ef6a35", markersize=5)
        self.start_2d = self.edit_axes.scatter([], [], color="#38d9c5", s=35, zorder=4)
        self.goal_2d = self.edit_axes.scatter([], [], color="#f4d35e", s=35, zorder=4)
        self.obstacles_2d = self.edit_axes.scatter([], [], color="#6b3729", marker="s", s=18, zorder=4)
        self.edit_axes.grid(alpha=0.18)

    def _costmap(self) -> np.ndarray:
        grid = ElevationGrid(
            values_m=self.elevation,
            cell_size_x_m=self.cell_size_x_m,
            cell_size_y_m=self.cell_size_y_m,
            crs="synthetic-local-meters",
        )
        costs = build_costmap(
            grid,
            PlannerConfig(
                max_slope_degrees=self.config.max_slope_degrees,
                slope_cost_weight=self.config.route_cost_weight,
            ),
        )
        costs[self.obstacles] = np.inf
        return costs

    def _replan(self) -> None:
        costs = self._costmap()
        try:
            self.vehicle_cell = _nearest_traversable_cell(costs, self.vehicle_cell)
            self.goal = _nearest_traversable_cell(costs, self.goal)
            self.path, self.route_cost = find_route(costs, self.vehicle_cell, self.goal)
            self.vehicle_index = 0
            self.status_text.set_text(f"Route: {len(self.path)} cells\nCost: {self.route_cost:.1f}")
        except ValueError:
            self.path = []
            self.route_cost = float("inf")
            self.status_text.set_text("No route\nEdit terrain or clear obstacles")
        self._refresh_scene()

    def _refresh_scene(self) -> None:
        self.surface.remove()
        self.surface = self.axes_3d.plot_surface(
            self.x_m,
            self.y_m,
            self.elevation,
            cmap="terrain",
            linewidth=0,
            antialiased=False,
            rcount=min(self.elevation.shape[0], 120),
            ccount=min(self.elevation.shape[1], 120),
            alpha=0.95,
            zorder=1,
        )
        self.surface.set_zorder(1)
        z_min = float(np.min(self.elevation))
        z_max = float(np.max(self.elevation))
        self.axes_3d.set_zlim(z_min - 1.0, z_max + 4.0)
        self.height_image.set_data(self.elevation)
        self.height_image.set_clim(z_min, max(z_max, z_min + 1e-3))

        path_rows = np.array([cell[0] for cell in self.path], dtype=int)
        path_cols = np.array([cell[1] for cell in self.path], dtype=int)
        path_x = self.x_m[path_rows, path_cols] if len(self.path) else np.array([])
        path_y = self.y_m[path_rows, path_cols] if len(self.path) else np.array([])
        path_z = self.elevation[path_rows, path_cols] + 0.1 if len(self.path) else np.array([])
        self.route_3d.set_data_3d(path_x, path_y, path_z)
        self.route_2d.set_data(path_x, path_y)

        start_x, start_y, start_z = self._cell_world(self.vehicle_cell)
        goal_x, goal_y, goal_z = self._cell_world(self.goal)
        self.start_3d._offsets3d = ([start_x], [start_y], [start_z + 0.2])
        self.goal_3d._offsets3d = ([goal_x], [goal_y], [goal_z + 0.2])
        self.start_2d.set_offsets([[start_x, start_y]])
        self.goal_2d.set_offsets([[goal_x, goal_y]])

        obstacle_rows, obstacle_cols = np.nonzero(self.obstacles)
        obstacle_x = self.x_m[obstacle_rows, obstacle_cols]
        obstacle_y = self.y_m[obstacle_rows, obstacle_cols]
        obstacle_z = self.elevation[obstacle_rows, obstacle_cols] + 0.35
        self.obstacle_3d._offsets3d = (obstacle_x, obstacle_y, obstacle_z)
        self.obstacles_2d.set_offsets(np.column_stack((obstacle_x, obstacle_y)))

        current_x, current_y, current_z = self._cell_world(self.vehicle_cell)
        self.vehicle_2d.set_data([current_x], [current_y])
        self._update_vehicle_pose()
        self.figure.canvas.draw_idle()

    def _cell_world(self, cell: tuple[int, int]) -> tuple[float, float, float]:
        row, col = cell
        return (
            float(self.x_m[row, col]),
            float(self.y_m[row, col]),
            float(self.elevation[row, col]),
        )

    def _update_vehicle_pose(self) -> None:
        if not self.path:
            row, col = self.vehicle_cell
            self.vehicle.set_verts(_vehicle_faces(
                np.array([self.x_m[row, col], self.y_m[row, col], self.elevation[row, col]]),
                0.0,
                0.0,
            ))
            return

        index = min(self.vehicle_index, len(self.path) - 1)
        row, col = self.path[index]
        if index + 1 < len(self.path):
            next_row, next_col = self.path[index + 1]
            first, second = (row, col), (next_row, next_col)
        elif index > 0:
            prev_row, prev_col = self.path[index - 1]
            first, second = (prev_row, prev_col), (row, col)
        else:
            first = second = (row, col)

        dx = float(self.x_m[second] - self.x_m[first])
        dy = float(self.y_m[second] - self.y_m[first])
        dz = float(self.elevation[second] - self.elevation[first])
        heading = float(np.arctan2(dy, dx)) if dx or dy else 0.0
        pitch = float(np.arctan2(dz, max(float(np.hypot(dx, dy)), 1e-6)))
        position = np.array([self.x_m[row, col], self.y_m[row, col], self.elevation[row, col]])
        self.vehicle.set_verts(_vehicle_faces(position, heading, pitch))

    def _tick(self, _frame: int) -> tuple[object, ...]:
        if self.path:
            self.vehicle_index = min(self.vehicle_index + 1, len(self.path) - 1)
            self.vehicle_cell = self.path[self.vehicle_index]
            row, col = self.vehicle_cell
            self.vehicle_2d.set_data([self.x_m[row, col]], [self.y_m[row, col]])
            self.trail_3d.set_data(
                self.x_m[[cell[0] for cell in self.path[: self.vehicle_index + 1]],
                         [cell[1] for cell in self.path[: self.vehicle_index + 1]]],
                self.y_m[[cell[0] for cell in self.path[: self.vehicle_index + 1]],
                         [cell[1] for cell in self.path[: self.vehicle_index + 1]]],
            )
            trail_rows = [cell[0] for cell in self.path[: self.vehicle_index + 1]]
            trail_cols = [cell[1] for cell in self.path[: self.vehicle_index + 1]]
            self.trail_3d.set_3d_properties(self.elevation[trail_rows, trail_cols] + 0.14)
            self._update_vehicle_pose()
        self.figure.canvas.draw_idle()
        return (self.surface, self.route_3d, self.trail_3d, self.vehicle)

    def _select_mode(self, mode: str) -> None:
        self.mode = mode
        self.status_text.set_text(f"Mode: {mode}\nBrush: {self.radius_widget.val:.0f} cells")
        self.figure.canvas.draw_idle()

    def _toggle_pause(self, _event: object) -> None:
        if self._paused:
            self.animation.event_source.start()
            self.pause_widget.label.set_text("Pause")
        else:
            self.animation.event_source.stop()
            self.pause_widget.label.set_text("Resume")
        self._paused = not self._paused
        self.figure.canvas.draw_idle()

    def _on_press(self, event: object) -> None:
        if event.inaxes is not self.edit_axes or event.button != 1:
            return
        self._dragging = True
        self._last_brush_cell = None
        self._paint_at(event.xdata, event.ydata)

    def _on_release(self, _event: object) -> None:
        self._dragging = False
        self._last_brush_cell = None

    def _on_motion(self, event: object) -> None:
        if self._dragging and event.inaxes is self.edit_axes:
            self._paint_at(event.xdata, event.ydata)

    def _paint_at(self, x: float | None, y: float | None) -> None:
        if x is None or y is None:
            return
        col = int(round((x - float(self.x_m.min())) / self.cell_size_x_m))
        row = int(round((y - float(self.y_m.min())) / self.cell_size_y_m))
        row = int(np.clip(row, 0, self.elevation.shape[0] - 1))
        col = int(np.clip(col, 0, self.elevation.shape[1] - 1))
        if self._last_brush_cell == (row, col):
            return
        self._last_brush_cell = (row, col)
        self._paint_cell(row, col)

    def _paint_cell(self, row: int, col: int) -> None:
        radius = int(self.radius_widget.val)
        strength = float(self.strength_widget.val)
        row_min, row_max = max(0, row - radius), min(self.elevation.shape[0], row + radius + 1)
        col_min, col_max = max(0, col - radius), min(self.elevation.shape[1], col + radius + 1)
        rr, cc = np.ogrid[row_min:row_max, col_min:col_max]
        distance_squared = (rr - row) ** 2 + (cc - col) ** 2
        brush = distance_squared <= radius**2
        region = self.elevation[row_min:row_max, col_min:col_max]
        obstacle_region = self.obstacles[row_min:row_max, col_min:col_max]

        if self.mode == "Spawn":
            self.start = (row, col)
            self.vehicle_cell = (row, col)
            self._replan()
            return
        elif self.mode == "Goal":
            self.goal = (row, col)
            self._replan()
            return

        if self.mode in {"Raise", "Lower"}:
            falloff = np.exp(-distance_squared / max(1.0, radius**2 / 2.0))
            sign = 1.0 if self.mode == "Raise" else -1.0
            region[brush] += (sign * strength * falloff)[brush]
        elif self.mode == "Obstacle":
            obstacle_region[brush] = True
        elif self.mode == "Erase":
            obstacle_region[brush] = False
        self._replan()

    def show(self) -> None:
        plt.show()

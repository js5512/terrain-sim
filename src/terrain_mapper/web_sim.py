"""Browser-based live terrain pathfinding simulation."""

from __future__ import annotations

import argparse
import base64
import io
import os
import secrets
import webbrowser
from dataclasses import asdict
from pathlib import Path
from threading import Timer
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from PIL import Image

from terrain_mapper.config import PlannerConfig
from terrain_mapper.costmap import build_costmap, calculate_elevation_contours
from terrain_mapper.dem import parse_dem
from terrain_mapper.elevation import ElevationGrid
from terrain_mapper.planner import find_route_through_waypoints
from terrain_mapper.simulation import (
    DEFAULT_GRID_SIZE,
    TERRAIN_EXTENT_M,
    TerrainConfig,
    TerrainScenario,
    _nearest_traversable_cell,
    _resize_to_shape,
    build_dem_scenario,
    build_synthetic_scenario,
    load_heightmap,
)


TERRAIN_COLORS = [
    [0.0, "#204b59"],
    [0.28, "#34877c"],
    [0.55, "#91b77c"],
    [0.78, "#d3c17a"],
    [1.0, "#eee4ce"],
]


def _state_from_scenario(
    scenario: TerrainScenario,
    config: TerrainConfig,
    size: int,
    source_heightmap: np.ndarray | None = None,
) -> dict[str, Any]:
    min_x = float(scenario.grid.min_x)
    min_y = float(scenario.grid.min_y)
    max_x = float(scenario.grid.max_x)
    max_y = float(scenario.grid.max_y)
    extent_m = max_x - min_x
    return {
        "elevation": scenario.grid.values_m.tolist(),
        "obstacles": scenario.obstacles.tolist(),
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "waypoints": [],
        "vehicle_cell": list(scenario.path[0] if scenario.path else scenario.start),
        "path": [list(cell) for cell in scenario.path],
        "route_cost": scenario.route_cost if np.isfinite(scenario.route_cost) else None,
        "vehicle_index": 0,
        "route_visible": False,
        "paused": True,
        "size": size,
        "terrain_extent_m": extent_m if extent_m > 0 else TERRAIN_EXTENT_M,
        "bounds": [min_x, min_y, max_x, max_y],
        "crs": scenario.grid.crs,
        "cell_size_x_m": float(scenario.grid.cell_size_x_m),
        "cell_size_y_m": float(scenario.grid.cell_size_y_m),
        "config": asdict(config),
        "source_heightmap": source_heightmap.tolist() if source_heightmap is not None else None,
        "revision": 0,
        "is_real_dem": False,
    }


def _state_from_dem(
    grid: ElevationGrid,
    config: TerrainConfig,
    size: int | None = None,
    revision: int = 0,
) -> dict[str, Any]:
    scenario = build_dem_scenario(grid, config=config, size=size)
    actual_size = scenario.grid.values_m.shape[0]
    state = _state_from_scenario(scenario, config, actual_size)
    state["revision"] = revision
    state["is_real_dem"] = True
    return state


def _new_state(
    config: TerrainConfig,
    size: int,
    heightmap: np.ndarray | None = None,
    revision: int = 0,
) -> dict[str, Any]:
    scenario = build_synthetic_scenario(size=size, config=config, heightmap=heightmap)
    state = _state_from_scenario(scenario, config, size, heightmap)
    state["revision"] = revision
    return state


def _toggle_playback(state: dict[str, Any]) -> dict[str, Any]:
    if not state["path"]:
        state["paused"] = True
        return state
    if int(state["vehicle_index"]) >= len(state["path"]) - 1:
        state["vehicle_index"] = 0
        state["vehicle_cell"] = state["path"][0]
        state["paused"] = False
        return state
    state["paused"] = not bool(state["paused"])
    return state


def _randomize_state(
    state: dict[str, Any],
    slope: float,
    relief: float,
    density: float,
    roughness: float,
) -> dict[str, Any]:
    previous_seed = int(state["config"]["seed"])
    seed = (previous_seed + secrets.randbelow(2**32 - 1) + 1) % 2**32
    config = TerrainConfig(
        slope_steepness=float(slope),
        height_variation=float(relief),
        obstacle_density=float(density),
        roughness=float(roughness),
        seed=seed,
    )
    randomized = _new_state(
        config,
        int(state["size"]),
        revision=int(state["revision"]) + 1,
    )
    randomized["message"] = f"Randomized terrain · seed {seed}"
    return randomized


def _replan(state: dict[str, Any]) -> None:
    elevations = np.asarray(state["elevation"], dtype=np.float32)
    obstacles = np.asarray(state["obstacles"], dtype=bool)
    rows, cols = elevations.shape
    bounds = tuple(state.get("bounds", (0.0, 0.0, TERRAIN_EXTENT_M, TERRAIN_EXTENT_M)))
    min_x, min_y, max_x, max_y = bounds
    cell_size_x = float(state.get("cell_size_x_m", (max_x - min_x) / max(cols - 1, 1)))
    cell_size_y = float(state.get("cell_size_y_m", (max_y - min_y) / max(rows - 1, 1)))
    crs = state.get("crs", "synthetic-local-meters")

    grid = ElevationGrid(
        values_m=elevations,
        cell_size_x_m=cell_size_x,
        cell_size_y_m=cell_size_y,
        crs=crs,
        bounds=bounds,
    )
    config = TerrainConfig(**state["config"])
    costs = build_costmap(
        grid,
        PlannerConfig(
            max_slope_degrees=config.max_slope_degrees,
            slope_cost_weight=config.route_cost_weight,
        ),
    )
    costs[obstacles] = np.inf

    current = tuple(state.get("vehicle_cell", state["start"]))
    goal = tuple(state["goal"])
    try:
        current = _nearest_traversable_cell(costs, current)
        goal = _nearest_traversable_cell(costs, goal)
        waypoints = []
        for waypoint in state.get("waypoints", []):
            traversable_waypoint = _nearest_traversable_cell(costs, tuple(waypoint))
            if traversable_waypoint not in {current, goal, *waypoints}:
                waypoints.append(traversable_waypoint)
        state["waypoints"] = [list(cell) for cell in waypoints]
        path, route_cost = find_route_through_waypoints(
            costs,
            current,
            waypoints,
            goal,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
        )
        state["path"] = [list(cell) for cell in path]
        state["route_cost"] = route_cost
        state["vehicle_cell"] = list(current)
        state["goal"] = list(goal)
        state["vehicle_index"] = 0
    except ValueError:
        state["path"] = []
        state["route_cost"] = None
        state["vehicle_cell"] = list(current)
        state["vehicle_index"] = 0
    state["revision"] = int(state.get("revision", 0)) + 1


def _cell_for_xy(
    x: float,
    y: float,
    size: int,
    bounds: tuple[float, float, float, float] | list[float] | None = None,
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[int, int]:
    if bounds is None:
        min_x, min_y, max_x, max_y = 0.0, 0.0, TERRAIN_EXTENT_M, TERRAIN_EXTENT_M
    else:
        min_x, min_y, max_x, max_y = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    num_rows = rows if rows is not None else size
    num_cols = cols if cols is not None else size
    dx = (max_x - min_x) if max_x != min_x else 1.0
    dy = (max_y - min_y) if max_y != min_y else 1.0
    col = int(np.clip(round((x - min_x) * (num_cols - 1) / dx), 0, num_cols - 1))
    row = int(np.clip(round((y - min_y) * (num_rows - 1) / dy), 0, num_rows - 1))
    return row, col


def _add_waypoints(state: dict[str, Any], points: list[tuple[float, float]]) -> None:
    waypoints = [tuple(cell) for cell in state.get("waypoints", [])]
    start = tuple(state["start"])
    goal = tuple(state["goal"])
    bounds = state.get("bounds")
    elevations = state.get("elevation")
    rows = len(elevations) if elevations else int(state["size"])
    cols = len(elevations[0]) if elevations and elevations[0] else int(state["size"])
    for x, y in points:
        cell = _cell_for_xy(x, y, int(state["size"]), bounds=bounds, rows=rows, cols=cols)
        if cell not in {start, goal, *waypoints}:
            waypoints.append(cell)
    if len(waypoints) == len(state.get("waypoints", [])):
        return
    state["waypoints"] = [list(cell) for cell in waypoints]
    _replan(state)
    state["message"] = f"Mission updated: {len(state['waypoints'])} waypoint(s)"


def _clear_waypoints(state: dict[str, Any]) -> None:
    if not state.get("waypoints"):
        return
    state["waypoints"] = []
    _replan(state)
    state["message"] = "Mission waypoints cleared"


def _expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    expanded = mask.copy()
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            if row_offset * row_offset + col_offset * col_offset > radius * radius:
                continue
            source_rows = slice(max(0, -row_offset), min(mask.shape[0], mask.shape[0] - row_offset))
            source_cols = slice(max(0, -col_offset), min(mask.shape[1], mask.shape[1] - col_offset))
            target_rows = slice(max(0, row_offset), min(mask.shape[0], mask.shape[0] + row_offset))
            target_cols = slice(max(0, col_offset), min(mask.shape[1], mask.shape[1] + col_offset))
            expanded[target_rows, target_cols] |= mask[source_rows, source_cols]
    return expanded


def _apply_edit(
    state: dict[str, Any],
    points: list[tuple[float, float]],
    mode: str,
    radius: int,
    strength: float,
) -> None:
    bounds = state.get("bounds")
    elevations = np.asarray(state["elevation"], dtype=np.float32)
    obstacles = np.asarray(state["obstacles"], dtype=bool)
    rows, cols = elevations.shape
    selected = np.zeros((rows, cols), dtype=bool)
    for x, y in points:
        row, col = _cell_for_xy(x, y, int(state["size"]), bounds=bounds, rows=rows, cols=cols)
        selected[row, col] = True

    if not selected.any():
        return

    expanded = _expand_mask(selected, radius)
    if mode in {"Raise", "Lower"}:
        sign = 1.0 if mode == "Raise" else -1.0
        elevations[expanded] += sign * strength
        state["elevation"] = elevations.tolist()
    elif mode == "Obstacle":
        obstacles |= expanded
        state["obstacles"] = obstacles.tolist()
    elif mode == "Erase":
        obstacles[expanded] = False
        state["obstacles"] = obstacles.tolist()

    _replan(state)


def _decode_heightmap(contents: str, filename: str, size: int) -> np.ndarray:
    try:
        encoded = contents.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        if filename.lower().endswith(".npy"):
            source = np.load(io.BytesIO(raw), allow_pickle=False)
        else:
            source = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float32)
        return load_heightmap(np.asarray(source, dtype=np.float32), size=size)
    except Exception as error:
        raise ValueError(f"Could not read heightmap: {error}") from error


def _route_coordinates(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.asarray(state["path"], dtype=int).reshape((-1, 2))
    elevations = np.asarray(state["elevation"], dtype=np.float32)
    rows, cols = elevations.shape
    bounds = state.get("bounds", [0.0, 0.0, state.get("terrain_extent_m", TERRAIN_EXTENT_M), state.get("terrain_extent_m", TERRAIN_EXTENT_M)])
    min_x, min_y, max_x, max_y = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    x_coords = np.linspace(min_x, max_x, cols)
    y_coords = np.linspace(min_y, max_y, rows)
    if len(path):
        return x_coords[path[:, 1]], y_coords[path[:, 0]], path
    return np.array([]), np.array([]), path


def _surface_figure(state: dict[str, Any]) -> go.Figure:
    elevations = np.asarray(state["elevation"], dtype=np.float32)
    rows, cols = elevations.shape
    bounds = state.get("bounds", [0.0, 0.0, state.get("terrain_extent_m", TERRAIN_EXTENT_M), state.get("terrain_extent_m", TERRAIN_EXTENT_M)])
    min_x, min_y, max_x, max_y = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    x_coords = np.linspace(min_x, max_x, cols)
    y_coords = np.linspace(min_y, max_y, rows)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    route_x, route_y, path = _route_coordinates(state)
    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=elevations,
            colorscale=TERRAIN_COLORS,
            showscale=False,
            contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "#f1c75b", "project": {"z": False}}},
            lighting={"ambient": 0.72, "diffuse": 0.85, "roughness": 0.9, "specular": 0.08},
            name="Elevation",
            hovertemplate="E %{x:.1f} m<br>N %{y:.1f} m<br>Z %{z:.1f} m<extra></extra>",
        )
    )
    if len(path):
        route_length = (
            min(int(state["vehicle_index"]), len(path) - 1) + 1
            if state.get("route_visible")
            else 0
        )
        route_z = elevations[path[:route_length, 0], path[:route_length, 1]] + 0.15
        figure.add_trace(
            go.Scatter3d(
                x=route_x[:route_length],
                y=route_y[:route_length],
                z=route_z,
                mode="lines",
                line={"color": "#f1c75b", "width": 5},
                name="Planned route",
                visible=bool(state.get("route_visible", False)),
            )
        )
        index = min(int(state["vehicle_index"]), len(path) - 1)
        row, col = path[index]
        figure.add_trace(
            go.Scatter3d(
                x=[x_coords[col]],
                y=[y_coords[row]],
                z=[elevations[row, col] + 0.9],
                mode="markers",
                marker={"size": 8, "symbol": "diamond", "color": "#e97642", "line": {"color": "#213c37", "width": 2}},
                name="Vehicle",
            )
        )
    else:
        row, col = state["vehicle_cell"]
        figure.add_trace(
            go.Scatter3d(
                x=[x_coords[col]],
                y=[y_coords[row]],
                z=[elevations[row, col] + 0.9],
                mode="markers",
                marker={"size": 8, "symbol": "diamond", "color": "#e97642"},
                name="Vehicle",
            )
        )

    obstacle_rows, obstacle_cols = np.nonzero(np.asarray(state["obstacles"], dtype=bool))
    if len(obstacle_rows):
        figure.add_trace(
            go.Scatter3d(
                x=x_coords[obstacle_cols],
                y=y_coords[obstacle_rows],
                z=elevations[obstacle_rows, obstacle_cols] + 0.35,
                mode="markers",
                marker={"size": 3, "symbol": "diamond", "color": "#654236"},
                name="Obstacles",
            )
        )

    start_row, start_col = state["start"]
    goal_row, goal_col = state["goal"]
    figure.add_trace(
        go.Scatter3d(
            x=[x_coords[start_col], x_coords[goal_col]],
            y=[y_coords[start_row], y_coords[goal_row]],
            z=[elevations[start_row, start_col] + 0.3, elevations[goal_row, goal_col] + 0.3],
            mode="markers",
            marker={"size": 5, "color": ["#54cbb0", "#f1c75b"]},
            name="Start / final destination",
            showlegend=False,
        )
    )
    waypoints = np.asarray(state.get("waypoints", []), dtype=int).reshape((-1, 2))
    if len(waypoints):
        figure.add_trace(
            go.Scatter3d(
                x=x_coords[waypoints[:, 1]],
                y=y_coords[waypoints[:, 0]],
                z=elevations[waypoints[:, 0], waypoints[:, 1]] + 1.2,
                mode="markers+text",
                marker={"size": 7, "color": "#e97642", "symbol": "circle-open"},
                text=[f"W{index + 1}" for index in range(len(waypoints))],
                textposition="top center",
                name="Mission waypoints",
                showlegend=False,
            )
        )
    z_min, z_max = float(elevations.min()), float(elevations.max())
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        uirevision="terrain-camera",
        scene={
            "xaxis_title": "East (m)",
            "yaxis_title": "North (m)",
            "zaxis_title": "Elevation (m)",
            "xaxis": {"range": [min_x, max_x], "tickformat": ".0f", "ticksuffix": " m"},
            "yaxis": {"range": [min_y, max_y], "tickformat": ".0f", "ticksuffix": " m"},
            "zaxis": {"range": [z_min - 1, z_max + 5]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.42},
            "camera": {"eye": {"x": 1.55, "y": -1.7, "z": 1.25}},
        },
    )
    return figure


def _editor_figure(state: dict[str, Any]) -> go.Figure:
    elevations = np.asarray(state["elevation"], dtype=np.float32)
    obstacles = np.asarray(state["obstacles"], dtype=bool)
    rows, cols = elevations.shape
    bounds = state.get("bounds", [0.0, 0.0, state.get("terrain_extent_m", TERRAIN_EXTENT_M), state.get("terrain_extent_m", TERRAIN_EXTENT_M)])
    min_x, min_y, max_x, max_y = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
    x_coords = np.linspace(min_x, max_x, cols)
    y_coords = np.linspace(min_y, max_y, rows)
    route_x, route_y, path = _route_coordinates(state)
    figure = go.Figure(
        go.Heatmap(
            x=x_coords,
            y=y_coords,
            z=elevations,
            colorscale=TERRAIN_COLORS,
            showscale=False,
            hovertemplate="E %{x:.1f} m<br>N %{y:.1f} m<br>Height %{z:.1f} m<extra></extra>",
        )
    )
    if len(path):
        route_length = (
            min(int(state["vehicle_index"]), len(path) - 1) + 1
            if state.get("route_visible")
            else 0
        )
        figure.add_trace(
            go.Scatter(
                x=route_x[:route_length],
                y=route_y[:route_length],
                mode="lines",
                line={"color": "#fff0a3", "width": 2.5},
                name="Route",
                hoverinfo="skip",
                visible=bool(state.get("route_visible", False)),
            )
        )
        index = min(int(state["vehicle_index"]), len(path) - 1)
        row, col = path[index]
        figure.add_trace(
            go.Scatter(
                x=[x_coords[col]],
                y=[y_coords[row]],
                mode="markers",
                marker={"size": 12, "color": "#e97642", "line": {"color": "#ffffff", "width": 2}},
                name="Vehicle",
                hoverinfo="skip",
            )
        )
    obstacle_rows, obstacle_cols = np.nonzero(obstacles)
    if len(obstacle_rows):
        figure.add_trace(
            go.Scatter(
                x=x_coords[obstacle_cols],
                y=y_coords[obstacle_rows],
                mode="markers",
                marker={"size": 7, "symbol": "square", "color": "#654236"},
                name="Obstacles",
                hoverinfo="skip",
            )
        )
    start_row, start_col = state["start"]
    goal_row, goal_col = state["goal"]
    figure.add_trace(
        go.Scatter(
            x=[x_coords[start_col], x_coords[goal_col]],
            y=[y_coords[start_row], y_coords[goal_row]],
            mode="markers",
            marker={"size": 9, "color": ["#54cbb0", "#f1c75b"], "line": {"color": "#ffffff", "width": 1}},
            name="Start / final destination",
            hoverinfo="skip",
        )
    )
    waypoints = np.asarray(state.get("waypoints", []), dtype=int).reshape((-1, 2))
    if len(waypoints):
        figure.add_trace(
            go.Scatter(
                x=x_coords[waypoints[:, 1]],
                y=y_coords[waypoints[:, 0]],
                mode="markers+text",
                marker={"size": 11, "color": "#e97642", "symbol": "circle-open"},
                text=[f"W{index + 1}" for index in range(len(waypoints))],
                textposition="top center",
                name="Mission waypoints",
                hoverinfo="skip",
            )
        )

    # Calculate and overlay real-world elevation contours
    grid = ElevationGrid(
        values_m=elevations,
        cell_size_x_m=float(state.get("cell_size_x_m", (max_x - min_x) / max(cols - 1, 1))),
        cell_size_y_m=float(state.get("cell_size_y_m", (max_y - min_y) / max(rows - 1, 1))),
        crs=state.get("crs", "synthetic-local-meters"),
        bounds=tuple(bounds),
    )
    contours = calculate_elevation_contours(grid, num_levels=6)
    for contour in contours:
        for seg in contour.paths:
            figure.add_trace(
                go.Scatter(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    mode="lines",
                    line={"color": "rgba(255, 255, 255, 0.35)", "width": 1.0},
                    name="Contour",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Sparse-opacity cell targets make box-select drags return editable terrain cells.
    cell_x, cell_y = np.meshgrid(x_coords, y_coords)
    figure.add_trace(
        go.Scattergl(
            x=cell_x.ravel(),
            y=cell_y.ravel(),
            mode="markers",
            marker={"size": 8, "color": "rgba(0,0,0,0.015)"},
            name="Edit cells",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_layout(
        margin={"l": 5, "r": 5, "t": 8, "b": 5},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        dragmode="select",
        selectionrevision=int(state.get("revision", 0)),
        uirevision="terrain-editor",
        xaxis={"title": "East (m)", "range": [min_x, max_x], "tickformat": ".0f", "ticksuffix": " m", "constrain": "domain"},
        yaxis={"title": "North (m)", "range": [min_y, max_y], "tickformat": ".0f", "ticksuffix": " m", "scaleanchor": "x", "scaleratio": 1},
    )
    return figure


def _layout(state: dict[str, Any]) -> html.Div:
    config = state["config"]
    return html.Div(
        [
            html.Header(
                [
                    html.Div("FIELD / 01", className="brand-mark"),
                    html.Div(
                        [html.H1("Terrain Lab"), html.P("Autonomous route simulation")],
                        className="brand-title",
                    ),
                    html.Div("LIVE SIMULATION", className="live-indicator"),
                ],
                className="topbar",
            ),
            html.Section(
                [
                    html.Div([html.Label("Slope"), dcc.Slider(0.2, 3.0, 0.1, value=config["slope_steepness"], id="slope-slider", marks=None, tooltip={"placement": "bottom", "always_visible": True})], className="parameter"),
                    html.Div([html.Label("Relief"), dcc.Slider(0.2, 4.0, 0.1, value=config["height_variation"], id="relief-slider", marks=None, tooltip={"placement": "bottom", "always_visible": True})], className="parameter"),
                    html.Div([html.Label("Obstacles"), dcc.Slider(0.0, 0.3, 0.01, value=config["obstacle_density"], id="density-slider", marks=None, tooltip={"placement": "bottom", "always_visible": True})], className="parameter"),
                    html.Div([html.Label("Roughness"), dcc.Slider(0.1, 3.0, 0.1, value=config["roughness"], id="roughness-slider", marks=None, tooltip={"placement": "bottom", "always_visible": True})], className="parameter"),
                    html.Div([html.Label("Brush"), dcc.Slider(1, 10, 1, value=3, id="brush-slider", marks=None, tooltip={"placement": "bottom", "always_visible": True})], className="parameter brush-parameter"),
                    html.Div([html.Label("Strength (m)"), dcc.Slider(0.1, 3.0, 0.1, value=0.8, id="strength-slider", marks=None, tooltip={"placement": "bottom", "always_visible": True})], className="parameter brush-parameter"),
                ],
                className="parameter-bar",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Span("Edit", className="tool-label"),
                            dcc.RadioItems(
                                id="edit-mode",
                                options=[{"label": name, "value": name} for name in ("Raise", "Lower", "Obstacle", "Erase", "Waypoint")],
                                value="Raise",
                                inline=True,
                                className="mode-selector",
                            ),
                        ],
                        className="edit-tools",
                    ),
                    html.Div(
                        [
                            dcc.Upload(
                                id="heightmap-upload",
                                children=html.Button("Import heightmap", className="button button-secondary"),
                                accept=".png,.jpg,.jpeg,.npy,.tif,.tiff,.dem,.asc",
                                multiple=False,
                            ),
                            html.Button("Reset terrain", id="reset-button", className="button button-secondary"),
                            html.Button("Clear stops", id="clear-waypoints-button", className="button button-secondary"),
                            html.Button("Start simulation", id="pause-button", className="button button-primary"),
                            html.Button("Randomize terrain", id="randomize-button", className="button button-secondary"),
                        ],
                        className="action-tools",
                    ),
                ],
                className="tool-bar",
            ),
            html.Main(
                [
                    html.Section(
                        [
                            html.Div([html.H2("Terrain surface"), html.Span("3D / ROUTE", className="panel-kicker")], className="panel-heading"),
                            dcc.Graph(id="surface-graph", figure=_surface_figure(state), config={"displayModeBar": True, "scrollZoom": True, "responsive": True}),
                        ],
                        className="visual-panel surface-panel",
                    ),
                    html.Section(
                        [
                            html.Div([html.H2("Terrain editor"), html.Span("TOP-DOWN / EDITABLE", className="panel-kicker")], className="panel-heading"),
                            dcc.Graph(id="editor-graph", figure=_editor_figure(state), config={"displayModeBar": False, "scrollZoom": True, "responsive": True}),
                        ],
                        className="visual-panel editor-panel",
                    ),
                ],
                className="workspace",
            ),
            html.Footer(
                [
                    html.Div(id="route-status", className="route-status"),
                    html.Div("Click to stamp · drag to edit an area", className="footer-hint"),
                ],
                className="status-bar",
            ),
            dcc.Store(id="sim-state", data=state),
            dcc.Store(id="render-revision", data=state["revision"]),
            dcc.Store(id="playback-index", data={"index": 0, "revision": state["revision"]}),
            dcc.Interval(id="animation-tick", interval=100, n_intervals=0, disabled=True),
        ],
        className="app-shell",
    )


def create_app(
    size: int = DEFAULT_GRID_SIZE,
    config: TerrainConfig | None = None,
    heightmap: np.ndarray | None = None,
    dem: str | Path | ElevationGrid | None = None,
) -> Dash:
    config = config or TerrainConfig()
    if dem is not None:
        grid = dem if isinstance(dem, ElevationGrid) else parse_dem(dem)
        initial_state = _state_from_dem(grid, config, size=size)
    else:
        initial_state = _new_state(config, size, heightmap)

    app = Dash(__name__, title="Terrain Lab", update_title=None)
    app.layout = _layout(initial_state)

    @app.callback(
        Output("sim-state", "data"),
        Input("editor-graph", "clickData"),
        Input("editor-graph", "selectedData"),
        Input("slope-slider", "value"),
        Input("relief-slider", "value"),
        Input("density-slider", "value"),
        Input("roughness-slider", "value"),
        Input("reset-button", "n_clicks"),
        Input("randomize-button", "n_clicks"),
        Input("pause-button", "n_clicks"),
        Input("heightmap-upload", "contents"),
        Input("clear-waypoints-button", "n_clicks"),
        State("sim-state", "data"),
        State("edit-mode", "value"),
        State("brush-slider", "value"),
        State("strength-slider", "value"),
        State("heightmap-upload", "filename"),
        State("playback-index", "data"),
        prevent_initial_call=True,
    )
    def update_state(
        click_data: dict[str, Any] | None,
        selected_data: dict[str, Any] | None,
        slope: float,
        relief: float,
        density: float,
        roughness: float,
        _reset_clicks: int,
        _randomize_clicks: int,
        _pause_clicks: int,
        upload_contents: str | None,
        _clear_waypoints_clicks: int,
        current: dict[str, Any],
        mode: str,
        brush_radius: float,
        strength: float,
        upload_filename: str | None,
        playback_index: dict[str, int],
    ) -> dict[str, Any]:
        trigger_priority = (
            "pause-button",
            "randomize-button",
            "clear-waypoints-button",
            "reset-button",
            "heightmap-upload",
            "slope-slider",
            "relief-slider",
            "density-slider",
            "roughness-slider",
            "editor-graph",
        )
        triggered_props = ctx.triggered_prop_ids
        triggered = next(
            (
                prop_id
                for component_id in trigger_priority
                for prop_id in triggered_props
                if prop_id.startswith(f"{component_id}.")
            ),
            None,
        )
        if triggered is None:
            return no_update
        if (
            mode == "Waypoint"
            and triggered.startswith("editor-graph.")
            and "editor-graph.clickData" in triggered_props
        ):
            triggered = "editor-graph.clickData"
        component, prop = triggered.split(".", 1)
        updated = dict(current)
        if updated["path"]:
            index = min(
                max(int(playback_index.get("index", updated["vehicle_index"])), 0),
                len(updated["path"]) - 1,
            )
            updated["vehicle_index"] = index
            updated["vehicle_cell"] = updated["path"][index]

        if component == "pause-button":
            was_at_end = bool(updated["path"]) and (
                int(updated["vehicle_index"]) >= len(updated["path"]) - 1
            )
            was_paused = bool(updated["paused"])
            updated = _toggle_playback(updated)
            redraw = was_at_end
            if was_paused and not updated["paused"] and not updated["route_visible"]:
                updated["route_visible"] = True
                redraw = True
            if redraw:
                updated["revision"] = int(updated["revision"]) + 1
            return updated

        if component == "clear-waypoints-button":
            _clear_waypoints(updated)
            return updated

        config_now = TerrainConfig(
            slope_steepness=float(slope),
            height_variation=float(relief),
            obstacle_density=float(density),
            roughness=float(roughness),
        )
        source_heightmap = (
            np.asarray(current["source_heightmap"], dtype=np.float32)
            if current.get("source_heightmap") is not None
            else None
        )

        if component == "heightmap-upload" and upload_contents and upload_filename:
            ext = Path(upload_filename).suffix.lower()
            if ext in {".tif", ".tiff", ".dem", ".asc"}:
                try:
                    encoded = upload_contents.split(",", 1)[1]
                    raw_bytes = base64.b64decode(encoded)
                    dem_grid = parse_dem(raw_bytes)
                    updated = _state_from_dem(
                        dem_grid,
                        config_now,
                        size=size,
                        revision=int(current["revision"]) + 1,
                    )
                    updated["message"] = f"Loaded DEM: {upload_filename} ({dem_grid.crs})"
                    return updated
                except Exception as error:
                    updated["message"] = f"Could not parse DEM: {error}"
                    return updated
            else:
                try:
                    source_heightmap = _decode_heightmap(upload_contents, upload_filename, size)
                except ValueError as error:
                    updated["message"] = str(error)
                    return updated
                updated = _new_state(
                    config_now,
                    size,
                    source_heightmap,
                    revision=int(current["revision"]) + 1,
                )
                updated["message"] = f"Loaded {upload_filename}"
                return updated

        if component == "randomize-button":
            return _randomize_state(current, slope, relief, density, roughness)

        if component == "reset-button" or component.endswith("-slider"):
            updated = _new_state(
                config_now,
                size,
                source_heightmap,
                revision=int(current["revision"]) + 1,
            )
            updated["message"] = "Terrain reset" if component == "reset-button" else "Terrain regenerated"
            return updated

        points: list[tuple[float, float]] = []
        if component == "editor-graph":
            if mode == "Waypoint" and prop == "clickData" and click_data:
                points = [(float(click_data["points"][0]["x"]), float(click_data["points"][0]["y"]))]
                _add_waypoints(updated, points)
                return updated
            if mode != "Waypoint" and prop == "clickData" and click_data:
                points = [(float(click_data["points"][0]["x"]), float(click_data["points"][0]["y"]))]
            elif mode != "Waypoint" and prop == "selectedData" and selected_data:
                points = [
                    (float(point["x"]), float(point["y"]))
                    for point in selected_data.get("points", [])
                    if "x" in point and "y" in point
                ]
        if points:
            _apply_edit(updated, points, mode, int(brush_radius), float(strength))
            updated["message"] = "Route recalculated" if updated["path"] else "No traversable route"
            return updated
        return no_update

    @app.callback(
        Output("surface-graph", "figure"),
        Output("editor-graph", "figure"),
        Output("route-status", "children"),
        Output("render-revision", "data"),
        Input("sim-state", "data"),
        State("render-revision", "data"),
    )
    def render_state(
        state: dict[str, Any],
        rendered_revision: int,
    ) -> tuple[Any, ...]:
        if state["path"]:
            waypoint_count = len(state.get("waypoints", []))
            status = html.Div(
                [
                    html.Span(
                        "ROUTE ACTIVE" if state.get("route_visible") else "ROUTE READY",
                        className="status-label",
                    ),
                    html.Span(f"{len(state['path'])} cells", className="status-value"),
                    html.Span(f"Cost {state['route_cost']:.1f}", className="status-value"),
                    *(
                        [html.Span(f"{waypoint_count} waypoint(s)", className="status-value")]
                        if waypoint_count
                        else []
                    ),
                    html.Span(state.get("message", ""), className="status-message"),
                ]
            )
        else:
            status = html.Div(
                [
                    html.Span("NO ROUTE", className="status-label status-error"),
                    html.Span("Adjust terrain or clear obstacles", className="status-message"),
                ]
            )
        terrain_changed = int(state["revision"]) != int(rendered_revision)
        if terrain_changed:
            surface_update: go.Figure | Patch = _surface_figure(state)
            editor_update: go.Figure | Patch = _editor_figure(state)
            revision_update: int | Any = state["revision"]
        else:
            surface_update = no_update
            editor_update = no_update
            revision_update = no_update

        return surface_update, editor_update, status, revision_update

    app.clientside_callback(
        """
        function(state, _tick, playback) {
            const noUpdate = window.dash_clientside.no_update;
            if (!state || !state.path || state.path.length === 0) {
                return [noUpdate, noUpdate, {index: 0, revision: 0}, true, "Start simulation"];
            }

            const path = state.path;
            const lastIndex = path.length - 1;
            const revision = Number(state.revision) || 0;
            const stored = playback && typeof playback === "object"
                ? playback
                : {index: Number(state.vehicle_index) || 0, revision};
            let index = Math.min(Math.max(Number(stored.index) || 0, 0), lastIndex);
            const triggered = window.dash_clientside.callback_context.triggered_id;
            const labelFor = (vehicleIndex) => {
                if (vehicleIndex >= lastIndex) return "Restart simulation";
                if (!state.paused) return "Pause vehicle";
                return vehicleIndex > 0 ? "Resume vehicle" : "Start simulation";
            };
            const patchAt = (vehicleIndex) => {
                const vehicle = path[vehicleIndex];
                const bounds = (state.bounds && state.bounds.length === 4)
                    ? state.bounds
                    : [0, 0, Number(state.terrain_extent_m) || 1000, Number(state.terrain_extent_m) || 1000];
                const minX = Number(bounds[0]);
                const minY = Number(bounds[1]);
                const maxX = Number(bounds[2]);
                const maxY = Number(bounds[3]);
                const cols = (state.elevation && state.elevation[0]) ? state.elevation[0].length : (Number(state.size) || 160);
                const rows = state.elevation ? state.elevation.length : (Number(state.size) || 160);
                const coordinates = (cell) => [
                    minX + (maxX - minX) * cell[1] / Math.max(cols - 1, 1),
                    minY + (maxY - minY) * cell[0] / Math.max(rows - 1, 1)
                ];
                const prefix = path.slice(0, vehicleIndex + 1);
                const routeCoordinates = prefix.map(coordinates);
                const routeX = routeCoordinates.map((point) => point[0]);
                const routeY = routeCoordinates.map((point) => point[1]);
                const routeZ = prefix.map((cell) => state.elevation[cell[0]][cell[1]] + 0.15);
                const [vehicleX, vehicleY] = coordinates(vehicle);
                const vehicleZ = state.elevation[vehicle[0]][vehicle[1]] + 0.9;
                const assign = (trace, property, value) => ({
                    operation: "Assign",
                    location: ["data", trace, property],
                    params: {value}
                });
                const makePatch = (operations) => ({
                    __dash_patch_update: "__dash_patch_update",
                    operations
                });
                return [
                    makePatch([
                        assign(1, "x", routeX),
                        assign(1, "y", routeY),
                        assign(1, "z", routeZ),
                        assign(2, "x", [vehicleX]),
                        assign(2, "y", [vehicleY]),
                        assign(2, "z", [vehicleZ])
                    ]),
                    makePatch([
                        assign(1, "x", routeX),
                        assign(1, "y", routeY),
                        assign(2, "x", [vehicleX]),
                        assign(2, "y", [vehicleY])
                    ])
                ];
            };

            if (triggered === "sim-state") {
                const stateIndex = Math.min(Math.max(Number(state.vehicle_index) || 0, 0), lastIndex);
                const revisionChanged = Number(stored.revision) !== revision;
                const markerChanged = stateIndex !== index && !revisionChanged;
                index = stateIndex;
                const figures = markerChanged ? patchAt(index) : [noUpdate, noUpdate];
                return [
                    figures[0],
                    figures[1],
                    {index, revision},
                    Boolean(state.paused || index >= lastIndex),
                    labelFor(index)
                ];
            }

            if (state.paused) {
                return [noUpdate, noUpdate, {index, revision}, true, labelFor(index)];
            }

            index = Math.min(index + 1, lastIndex);
            const figures = patchAt(index);
            return [
                figures[0],
                figures[1],
                {index, revision},
                index >= lastIndex,
                labelFor(index)
            ];
        }
        """,
        Output("surface-graph", "figure", allow_duplicate=True),
        Output("editor-graph", "figure", allow_duplicate=True),
        Output("playback-index", "data"),
        Output("animation-tick", "disabled"),
        Output("pause-button", "children"),
        Input("sim-state", "data"),
        Input("animation-tick", "n_intervals"),
        State("playback-index", "data"),
        prevent_initial_call=True,
    )

    return app


def _load_cli_heightmap(path: str | None, size: int) -> np.ndarray | None:
    if path is None:
        return None
    return load_heightmap(path, size=size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the terrain simulation in a browser tab")
    parser.add_argument("--size", type=int, default=DEFAULT_GRID_SIZE, help="terrain width and height in cells")
    parser.add_argument("--port", type=int, default=8050, help="local web server port")
    parser.add_argument("--host", default="0.0.0.0", help="web server bind address")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab automatically")
    parser.add_argument("--slope-steepness", type=float, default=1.0)
    parser.add_argument("--height-variation", type=float, default=1.0)
    parser.add_argument("--obstacle-density", type=float, default=0.08)
    parser.add_argument("--roughness", type=float, default=1.0)
    parser.add_argument("--heightmap", help="initial grayscale image or .npy elevation map")
    parser.add_argument("--dem", help="path to real-world DEM or GeoTIFF (.tif, .tiff, .dem, .asc)")
    args = parser.parse_args()

    config = TerrainConfig(
        slope_steepness=args.slope_steepness,
        height_variation=args.height_variation,
        obstacle_density=args.obstacle_density,
        roughness=args.roughness,
    )
    app = create_app(
        args.size,
        config,
        _load_cli_heightmap(args.heightmap, args.size),
        dem=args.dem,
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Terrain Lab is starting at {url}")
    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open_new_tab(url)).start()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

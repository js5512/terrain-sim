 
import base64
import io
import json

import numpy as np

from terrain_mapper.simulation import TERRAIN_EXTENT_M, TerrainConfig
from terrain_mapper.web_sim import (
    DEFAULT_GRID_SIZE,
    _cell_for_xy,
    _apply_edit,
    _add_waypoints,
    _clear_waypoints,
    _decode_heightmap,
    _editor_figure,
    _new_state,
    _randomize_state,
    _surface_figure,
    _toggle_playback,
    create_app,
)


def test_dash_page_and_stylesheet_are_served() -> None:
    app = create_app(size=32)
    client = app.server.test_client()

    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200
    assert b"Terrain surface" in layout.data
    assert b"Start simulation" in layout.data
    assert b"Randomize terrain" in layout.data
    assert client.get("/assets/style.css").status_code == 200


def test_animation_interval_avoids_excessive_callback_frequency() -> None:
    app = create_app(size=32)
    animation_tick = next(
        component
        for component in app.layout.children
        if getattr(component, "id", None) == "animation-tick"
    )

    assert animation_tick.interval == 100


def test_numpy_heightmap_upload_is_normalized() -> None:
    stream = io.BytesIO()
    np.save(stream, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
    contents = "data:application/octet-stream;base64," + base64.b64encode(stream.getvalue()).decode()

    heightmap = _decode_heightmap(contents, "ridge.npy", size=32)

    assert heightmap.shape == (32, 32)
    assert np.isclose(heightmap.min(), 0.0)
    assert np.isclose(heightmap.max(), 1.0)


def test_browser_obstacle_edit_forces_route_detour() -> None:
    state = _new_state(TerrainConfig(), size=32)
    path = state["path"]
    row, col = path[len(path) // 2]
    coordinates = np.linspace(0.0, TERRAIN_EXTENT_M, 32)

    _apply_edit(
        state,
        [(float(coordinates[col]), float(coordinates[row]))],
        "Obstacle",
        radius=0,
        strength=0.8,
    )

    assert state["obstacles"][row][col]
    assert [row, col] not in state["path"]



def test_playback_starts_paused_and_restarts_after_finishing() -> None:
    state = _new_state(TerrainConfig(), size=32)

    assert state["paused"]
    _toggle_playback(state)
    assert not state["paused"]

    state["vehicle_index"] = len(state["path"]) - 1
    state["paused"] = True
    _toggle_playback(state)

    assert not state["paused"]
    assert state["vehicle_index"] == 0
    assert state["vehicle_cell"] == state["path"][0]


def test_waypoints_are_stitched_in_order_and_can_be_cleared() -> None:
    state = _new_state(TerrainConfig(), size=32)
    initial_path = [tuple(cell) for cell in state["path"]]
    waypoints = [initial_path[len(initial_path) // 3], initial_path[2 * len(initial_path) // 3]]
    coordinates = np.linspace(0.0, TERRAIN_EXTENT_M, 32)
    points = [(float(coordinates[col]), float(coordinates[row])) for row, col in waypoints]

    _add_waypoints(state, points)

    assert state["waypoints"] == [list(cell) for cell in waypoints]
    previous_index = 0
    for waypoint in waypoints:
        waypoint_index = state["path"].index(list(waypoint), previous_index + 1)
        previous_index = waypoint_index
    assert state["path"][-1] == state["goal"]
    marker = next(trace for trace in _editor_figure(state).data if trace.name == "Mission waypoints")
    assert list(marker.text) == ["W1", "W2"]

    _clear_waypoints(state)

    assert state["waypoints"] == []
    assert state["path"][-1] == state["goal"]


def test_waypoint_mode_and_clear_stops_control_are_available() -> None:
    app = create_app(size=32)
    layout = app.server.test_client().get("/_dash-layout").data

    assert b"Waypoint" in layout
    assert b"Clear stops" in layout


def test_planned_route_is_hidden_until_enabled() -> None:
    state = _new_state(TerrainConfig(), size=32)
    surface_route = next(trace for trace in _surface_figure(state).data if trace.name == "Planned route")
    editor_route = next(trace for trace in _editor_figure(state).data if trace.name == "Route")

    assert surface_route.visible is False
    assert editor_route.visible is False
    assert len(surface_route.x) == 0
    assert len(editor_route.x) == 0

    state["route_visible"] = True
    state["vehicle_index"] = 0
    surface_route = next(trace for trace in _surface_figure(state).data if trace.name == "Planned route")
    editor_route = next(trace for trace in _editor_figure(state).data if trace.name == "Route")
    assert surface_route.visible is True
    assert editor_route.visible is True
    assert len(surface_route.x) == len(editor_route.x) == 1

    state["vehicle_index"] = 4
    surface_route = next(trace for trace in _surface_figure(state).data if trace.name == "Planned route")
    editor_route = next(trace for trace in _editor_figure(state).data if trace.name == "Route")
    assert len(surface_route.x) == len(editor_route.x) == 5


def test_web_coordinates_and_axes_use_metric_map_extent() -> None:
    state = _new_state(TerrainConfig(), size=32)
    surface = _surface_figure(state)
    editor = _editor_figure(state)

    assert state["terrain_extent_m"] == TERRAIN_EXTENT_M
    assert DEFAULT_GRID_SIZE > 100
    assert _cell_for_xy(0.0, 0.0, 32) == (0, 0)
    assert _cell_for_xy(TERRAIN_EXTENT_M, TERRAIN_EXTENT_M, 32) == (31, 31)
    assert surface.data[0].x.min() == 0.0
    assert surface.data[0].x.max() == TERRAIN_EXTENT_M
    assert surface.layout.scene.xaxis.ticksuffix == " m"
    assert editor.layout.xaxis.range == (0, TERRAIN_EXTENT_M)
    assert editor.layout.xaxis.ticksuffix == " m"


def test_default_web_grid_uses_expanded_dimensions() -> None:
    app = create_app()
    state_store = next(
        component
        for component in app.layout.children
        if getattr(component, "id", None) == "sim-state"
    )

    assert state_store.data["size"] == DEFAULT_GRID_SIZE
    assert state_store.data["terrain_extent_m"] == TERRAIN_EXTENT_M


def test_randomize_state_changes_seed_and_terrain() -> None:
    state = _new_state(TerrainConfig(), size=32)
    randomized = _randomize_state(state, 1.0, 1.0, 0.08, 1.0)

    assert randomized["config"]["seed"] != state["config"]["seed"]
    assert randomized["elevation"] != state["elevation"]
    assert randomized["revision"] > state["revision"]


def test_animation_ticks_run_client_side() -> None:
    app = create_app(size=32)
    dependencies = app.server.test_client().get("/_dash-dependencies").get_json()
    state_callback = next(item for item in dependencies if item["output"] == "sim-state.data")
    animation_callback = next(
        item for item in dependencies
        if any(input_item["id"] == "animation-tick" for input_item in item["inputs"])
    )

    assert all(input_item["id"] != "animation-tick" for input_item in state_callback["inputs"])
    assert animation_callback.get("clientside_function")


def test_start_click_reveals_route_and_starts_playback() -> None:
    app = create_app(size=32)
    state = _new_state(TerrainConfig(), size=32)
    input_values = [
        ("editor-graph", "clickData", None),
        ("editor-graph", "selectedData", None),
        ("slope-slider", "value", 1.0),
        ("relief-slider", "value", 1.0),
        ("density-slider", "value", 0.08),
        ("roughness-slider", "value", 1.0),
        ("reset-button", "n_clicks", 0),
        ("randomize-button", "n_clicks", 0),
        ("pause-button", "n_clicks", 1),
        ("heightmap-upload", "contents", None),
        ("clear-waypoints-button", "n_clicks", 0),
    ]
    state_values = [
        ("sim-state", "data", state),
        ("edit-mode", "value", "Raise"),
        ("brush-slider", "value", 3),
        ("strength-slider", "value", 0.8),
        ("heightmap-upload", "filename", None),
        ("playback-index", "data", {"index": 0, "revision": 0}),
    ]
    payload = {
        "output": "sim-state.data",
        "outputs": {"id": "sim-state", "property": "data"},
        "inputs": [
            {"id": component, "property": prop, "value": value}
            for component, prop, value in input_values
        ],
        "state": [
            {"id": component, "property": prop, "value": value}
            for component, prop, value in state_values
        ],
        "changedPropIds": ["pause-button.n_clicks"],
    }

    response = app.server.test_client().post(
        "/_dash-update-component",
        data=json.dumps(payload),
        content_type="application/json",
    )

    assert response.status_code == 200
    result = response.get_json().get("response", {}).get("sim-state", {}).get("data")
    assert result is not None
    assert result["paused"] is False
    assert result["route_visible"] is True




    assert result["revision"] == 1

def test_waypoint_map_click_updates_mission_state() -> None:
    app = create_app(size=32)
    state = _new_state(TerrainConfig(), size=32)
    row, col = state["path"][len(state["path"]) // 2]
    coordinate = np.linspace(0.0, TERRAIN_EXTENT_M, 32)
    inputs = [
        ("editor-graph", "clickData", {"points": [{"x": coordinate[col], "y": coordinate[row]}]}),
        ("editor-graph", "selectedData", None),
        ("slope-slider", "value", 1.0),
        ("relief-slider", "value", 1.0),
        ("density-slider", "value", 0.08),
        ("roughness-slider", "value", 1.0),
        ("reset-button", "n_clicks", 0),
        ("randomize-button", "n_clicks", 0),
        ("pause-button", "n_clicks", 0),
        ("heightmap-upload", "contents", None),
        ("clear-waypoints-button", "n_clicks", 0),
    ]
    state_values = [
        ("sim-state", "data", state),
        ("edit-mode", "value", "Waypoint"),
        ("brush-slider", "value", 3),
        ("strength-slider", "value", 0.8),
        ("heightmap-upload", "filename", None),
        ("playback-index", "data", {"index": 0, "revision": 0}),
    ]
    payload = {
        "output": "sim-state.data",
        "outputs": {"id": "sim-state", "property": "data"},
        "inputs": [
            {"id": component, "property": prop, "value": value}
            for component, prop, value in inputs
        ],
        "state": [
            {"id": component, "property": prop, "value": value}
            for component, prop, value in state_values
        ],
        "changedPropIds": ["editor-graph.selectedData", "editor-graph.clickData"],
    }

    response = app.server.test_client().post(
        "/_dash-update-component",
        data=json.dumps(payload),
        content_type="application/json",
    )

    assert response.status_code == 200
    result = response.get_json()["response"]["sim-state"]["data"]
    assert result["waypoints"] == [[row, col]]
    assert result["path"].index([row, col]) > 0


def test_geotiff_upload_updates_real_world_state() -> None:
    import rasterio
    from rasterio.transform import from_origin

    app = create_app(size=32)
    state = _new_state(TerrainConfig(), size=32)

    # Create in-memory GeoTIFF
    elevations = np.ones((20, 20), dtype=np.float32) * 50.0
    transform = from_origin(500000.0, 4000000.0, 25.0, 25.0)
    stream = io.BytesIO()
    with rasterio.open(
        stream,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(elevations, 1)

    raw_bytes = stream.getvalue()
    encoded = "data:image/tiff;base64," + base64.b64encode(raw_bytes).decode("ascii")

    input_values = [
        ("editor-graph", "clickData", None),
        ("editor-graph", "selectedData", None),
        ("slope-slider", "value", 1.0),
        ("relief-slider", "value", 1.0),
        ("density-slider", "value", 0.08),
        ("roughness-slider", "value", 1.0),
        ("reset-button", "n_clicks", 0),
        ("randomize-button", "n_clicks", 0),
        ("pause-button", "n_clicks", 0),
        ("heightmap-upload", "contents", encoded),
        ("clear-waypoints-button", "n_clicks", 0),
    ]
    state_values = [
        ("sim-state", "data", state),
        ("edit-mode", "value", "Raise"),
        ("brush-slider", "value", 3),
        ("strength-slider", "value", 0.8),
        ("heightmap-upload", "filename", "alpine.tif"),
        ("playback-index", "data", {"index": 0, "revision": 0}),
    ]
    payload = {
        "output": "sim-state.data",
        "outputs": {"id": "sim-state", "property": "data"},
        "inputs": [
            {"id": component, "property": prop, "value": value}
            for component, prop, value in input_values
        ],
        "state": [
            {"id": component, "property": prop, "value": value}
            for component, prop, value in state_values
        ],
        "changedPropIds": ["heightmap-upload.contents"],
    }

    response = app.server.test_client().post(
        "/_dash-update-component",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    result = response.get_json()["response"]["sim-state"]["data"]
    assert result["is_real_dem"] is True
    assert result["bounds"][0] == 500000.0
    assert result["bounds"][3] == 4000000.0
    assert "alpine.tif" in result["message"]


def test_figures_use_real_world_dem_coordinates() -> None:
    from terrain_mapper.elevation import ElevationGrid
    from terrain_mapper.web_sim import _state_from_dem

    elevations = np.linspace(100, 200, 256, dtype=np.float32).reshape((16, 16))
    bounds = (600000.0, 3000000.0, 600300.0, 3000300.0)
    grid = ElevationGrid(
        values_m=elevations,
        cell_size_x_m=20.0,
        cell_size_y_m=20.0,
        crs="EPSG:32632",
        bounds=bounds,
    )

    state = _state_from_dem(grid, TerrainConfig())
    surface = _surface_figure(state)
    editor = _editor_figure(state)

    assert surface.data[0].x.min() == bounds[0]
    assert surface.data[0].x.max() == bounds[2]
    assert tuple(surface.layout.scene.xaxis.range) == (bounds[0], bounds[2])
    assert tuple(surface.layout.scene.yaxis.range) == (bounds[1], bounds[3])

    assert tuple(editor.layout.xaxis.range) == (bounds[0], bounds[2])
    assert tuple(editor.layout.yaxis.range) == (bounds[1], bounds[3])
    # Check that contours are present
    assert any(trace.name == "Contour" for trace in editor.data)

    # Test coordinate mapping
    r, c = _cell_for_xy(bounds[0], bounds[1], 16, bounds=bounds, rows=16, cols=16)
    assert (r, c) == (0, 0)
    r_end, c_end = _cell_for_xy(bounds[2], bounds[3], 16, bounds=bounds, rows=16, cols=16)
    assert (r_end, c_end) == (15, 15)
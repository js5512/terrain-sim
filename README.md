# Terrain Mapper

A small Python starter for planning over digital elevation models (DEMs). It loads a projected elevation raster, turns local slope into a traversal-cost surface, and routes between grid cells.

## Recommended stack

- **Python 3.11+** for data preparation, prototyping, and offline evaluation. NumPy handles elevation arrays, Rasterio reads GeoTIFF/COG DEMs, and scikit-image provides grid-cost routing. Use PyProj for explicit coordinate transformations when integrating GPS or other map sources; use SciPy for terrain derivatives and filtering as the pipeline grows.
- **C++ with ROS 2** for the on-vehicle runtime, sensor drivers, and deterministic integration. ROS 2 `nav_msgs/OccupancyGrid` is a useful interface for 2D cost layers; Nav2 costmaps and its Smac planners or OMPL are stronger deployment choices than a custom planner. Keep the Python pipeline for map generation and evaluation, and exchange maps and poses through well-defined ROS messages.
- **Elevation formats and coordinates:** GeoTIFF/Cloud Optimized GeoTIFF are practical raster formats. Reproject geographic (longitude/latitude) DEMs into a suitable local projected CRS before treating pixel spacing as meters. For larger worlds or changing terrain, add tiled storage (e.g. GDAL/COG) and a map-update layer.

The included planner is a prototype, not a vehicle safety system. It considers slope only; production planning should account for vehicle footprint, roll/pitch limits, surface condition, uncertainty, localization error, dynamic obstacles, and a validated stopping/safety policy.

## Project layout

```text
src/terrain_mapper/
	config.py       Planner thresholds and costs
	dem.py          Standard DEM and GeoTIFF parser
	elevation.py    ElevationGrid data model and coordinate mapping
	costmap.py      Slope steepness, elevation contours, and cost surface
	planner.py      A* pathfinding and waypoint routing in metric coordinates
	simulation.py   3D Matplotlib animation and scenario generation
	web_sim.py      Live browser simulation with real-world coordinate mapping
	cli.py          Pipeline initialization and command line
tests/
	test_costmap.py
	test_dem.py
	test_planner.py
	test_simulation.py
	test_web_sim.py
```

## Run

```bash
python -m pip install -e '.[dev]'
terrain-mapper dem.tif --start 120 80 --goal 260 310
```

Start and goal are `row column` cell indices. The input DEM must use a projected CRS with meter units. The command prints the resulting cell path as JSON.

Run the interactive terrain simulation in a browser tab with:

```bash
python -m pip install -e '.[dev]'
terrain-sim
```

The web app shows the moving vehicle in 3D alongside a top-down editor, terrain configuration controls, and heightmap upload. Select **Start simulation** to trace the route, pause or resume it, and use **Randomize terrain** to generate a fresh seeded landscape while retaining the selected terrain settings. Terrain edits immediately recalculate the route.

The browser simulation uses a 160 × 160 grid over a 1,000 m × 1,000 m local map, with roughly 6.3 m between grid samples. Axis labels and terrain edits use meters. Normalized heightmap images are scaled to a 60 m elevation range at the default height-variation setting. To add mission stops, choose **Waypoint** in the editor mode selector and click the map in the order the vehicle should visit them. Stops are numbered in order and the generated final destination remains last; **Clear stops** removes the intermediate waypoints.

The app listens on port 8050 and opens a browser tab automatically when possible. In a remote Codespace, open the forwarded port URL if the tab is not opened automatically. The simulated terrain and vehicle are illustrative; vehicle dynamics are not modeled.
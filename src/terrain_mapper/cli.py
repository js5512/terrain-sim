import argparse
import json

from terrain_mapper.config import PlannerConfig
from terrain_mapper.costmap import build_costmap
from terrain_mapper.elevation import load_dem
from terrain_mapper.planner import find_route


def main() -> None:
    parser = argparse.ArgumentParser(description="Route over a projected DEM")
    parser.add_argument("dem", help="path to a standard DEM or GeoTIFF file")
    parser.add_argument("--start", nargs=2, type=int, required=True, metavar=("ROW", "COL"))
    parser.add_argument("--goal", nargs=2, type=int, required=True, metavar=("ROW", "COL"))
    args = parser.parse_args()

    grid = load_dem(args.dem)
    costmap = build_costmap(grid, PlannerConfig())
    path, total_cost = find_route(
        costmap,
        tuple(args.start),
        tuple(args.goal),
        cell_size_x=grid.cell_size_x_m,
        cell_size_y=grid.cell_size_y_m,
    )
    world_path = [grid.grid_to_world(r, c) for r, c in path]
    print(json.dumps({"cost": total_cost, "path": path, "world_path": world_path, "crs": grid.crs}))


if __name__ == "__main__":
    main()
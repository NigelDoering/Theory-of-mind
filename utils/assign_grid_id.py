import pandas as pd
import numpy as np
import os
from pyproj import Transformer
from tqdm import tqdm

import folium
from folium.plugins import HeatMap
from shapely.geometry import box
import geopandas as gpd

# Grid resolution in meters
GRID_RES_METERS = 5000

# Define UTM projection (zone for Denmark is 32N)
transformer_wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
transformer_utm_to_wgs84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

def latlon_to_utm(lat, lon):
    return transformer_wgs84_to_utm.transform(lon, lat)

def utm_to_latlon(x, y):
    return transformer_utm_to_wgs84.transform(x, y)[::-1]

def build_grid(min_x, max_x, min_y, max_y, res):
    """
    Builds a dictionary mapping (x_idx, y_idx) to cell centroid in UTM and lat/lon.
    """
    grid = {}
    for i, x in enumerate(np.arange(min_x, max_x, res)):
        for j, y in enumerate(np.arange(min_y, max_y, res)):
            cx = x + res / 2
            cy = y + res / 2
            lat, lon = utm_to_latlon(cx, cy)
            grid[(i, j)] = {"centroid_x": cx, "centroid_y": cy, "lat": lat, "lon": lon}
    return grid

def assign_to_cell(x, y, min_x, min_y, res):
    ix = int((x - min_x) // res)
    iy = int((y - min_y) // res)
    return (ix, iy)

def visualize_grid_density(trip_df, grid, min_x, min_y, res, map_save_path="grid_density_map.html"):
    print("🗺️ Rendering grid density map...")

    # Prepare GeoDataFrame of all grid cells
    grid_boxes = []
    for (i, j), cell in grid.items():
        x0, y0 = min_x + i * res, min_y + j * res
        geom = box(x0, y0, x0 + res, y0 + res)
        lat, lon = utm_to_latlon(x0 + res / 2, y0 + res / 2)
        grid_boxes.append({"geometry": geom, "lat": lat, "lon": lon})
    grid_gdf = gpd.GeoDataFrame(grid_boxes, crs="EPSG:25832").to_crs(epsg=4326)

    # Create folium map centered over Denmark
    center_lat, center_lon = trip_df["start_lat"].mean(), trip_df["start_lon"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    # Add grid cells as rectangles
    for _, row in grid_gdf.iterrows():
        bounds = row["geometry"].bounds  # minx, miny, maxx, maxy
        lat1, lon1 = row["lat"], row["lon"]
        folium.Rectangle(
            bounds=[
                [row["geometry"].bounds[1], row["geometry"].bounds[0]],  # (miny, minx)
                [row["geometry"].bounds[3], row["geometry"].bounds[2]],  # (maxy, maxx)
            ],
            color="blue",
            weight=0.3,
            fill=False,
        ).add_to(fmap)

    # Add heatmap of start and end locations
    heat_data = trip_df[["start_lat", "start_lon"]].values.tolist() + trip_df[["end_lat", "end_lon"]].values.tolist()
    HeatMap(heat_data, radius=7, blur=15, min_opacity=0.2).add_to(fmap)

    # Save map
    fmap.save(map_save_path)
    print(f"🖼️ Map saved to {map_save_path}")

def main():
    # Load AIS data
    filepath = "./notebooks/data/trips/cleaned_combined_all_days.csv"
    df = pd.read_csv(filepath, parse_dates=["# Timestamp"])

    # Extract first and last points of each trip
    trip_extremes = (
        df.sort_values(["IMO", "trip_number", "# Timestamp"])
        .groupby(["IMO", "trip_number"])
        .agg(start_lat=("Latitude", "first"),
             start_lon=("Longitude", "first"),
             end_lat=("Latitude", "last"),
             end_lon=("Longitude", "last"))
        .reset_index()
    )

    # Convert all start/end lat/lons to UTM
    trip_extremes["start_x"], trip_extremes["start_y"] = latlon_to_utm(
        trip_extremes["start_lat"].values, trip_extremes["start_lon"].values
    )
    trip_extremes["end_x"], trip_extremes["end_y"] = latlon_to_utm(
        trip_extremes["end_lat"].values, trip_extremes["end_lon"].values
    )

    # Determine bounding box
    all_x = np.concatenate([trip_extremes["start_x"].values, trip_extremes["end_x"].values])
    all_y = np.concatenate([trip_extremes["start_y"].values, trip_extremes["end_y"].values])
    min_x, max_x = all_x.min() - 10000, all_x.max() + 10000
    min_y, max_y = all_y.min() - 10000, all_y.max() + 10000

    # Build grid
    print("🗺️ Building grid...")
    grid = build_grid(min_x, max_x, min_y, max_y, GRID_RES_METERS)

    # Assign each trip start and end to a grid cell
    print("📍 Assigning trips to grid nodes...")
    results = []
    flagged_trips = []

    for _, row in tqdm(trip_extremes.iterrows(), total=len(trip_extremes)):
        try:
            start_cell = assign_to_cell(row["start_x"], row["start_y"], min_x, min_y, GRID_RES_METERS)
            end_cell = assign_to_cell(row["end_x"], row["end_y"], min_x, min_y, GRID_RES_METERS)

            start_node = grid.get(start_cell)
            end_node = grid.get(end_cell)

            if not start_node or not end_node:
                flagged_trips.append((row["IMO"], row["trip_number"]))
                continue

            results.append({
                "IMO": row["IMO"],
                "trip_number": row["trip_number"],
                "start_node_id": f"{start_cell[0]}_{start_cell[1]}",
                "start_lat": start_node["lat"],
                "start_lon": start_node["lon"],
                "end_node_id": f"{end_cell[0]}_{end_cell[1]}",
                "end_lat": end_node["lat"],
                "end_lon": end_node["lon"],
            })
        except Exception as e:
            flagged_trips.append((row["IMO"], row["trip_number"]))

    # Save output
    output_df = pd.DataFrame(results)
    out_path = "./notebooks/data/trips/trip_node_assignments.csv"
    output_df.to_csv(out_path, index=False)
    print(f"✅ Trip node assignments saved to: {out_path} ({len(output_df)} trips assigned)")

    if flagged_trips:
        print(f"⚠️ {len(flagged_trips)} trips fell outside the grid and were skipped.")
        flagged_df = pd.DataFrame(flagged_trips, columns=["IMO", "trip_number"])
        flagged_df.to_csv("./notebooks/data/trips/trips_outside_grid.csv", index=False)

    # Add grid heatmap visualization
    visualize_grid_density(output_df, grid, min_x, min_y, GRID_RES_METERS)


if __name__ == "__main__":
    main()
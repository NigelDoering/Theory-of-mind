from datetime import timedelta
import pandas as pd
import numpy as np
import argparse
import os
import sys
import re
from tqdm import tqdm
from pyproj import Transformer
from haversine import haversine
from sklearn.cluster import DBSCAN

def extract_date_from_filename(filename):
    """Extracts the date from a filename in format 'YYYY-MM-DD' and returns it as 'YYYY_MM_DD'."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if not match:
        print("Error: Filename must include a date in format YYYY-MM-DD (e.g., aisdk-2025-01-01.csv)")
        sys.exit(1)
    return match.group(0).replace("-", "_")

def build_output_path(date_str):
    """Returns the output path for a cleaned AIS file based on a given date string."""
    return f"./notebooks/data/trips/cleaned_{date_str}.csv"

def read_data(filename, path="./notebooks/data/AIS_data/"):
    """Reads a CSV AIS file from the specified directory."""
    full_path = os.path.join(path, filename)
    return pd.read_csv(full_path)

def clean_mmsi_values(data):
    """
    Cleans the AIS data by:
    - Dropping rows with missing MMSI or IMO
    - Resolving cases where a single MMSI is linked to multiple IMOs
    Returns the cleaned DataFrame.
    """
    ais = data.dropna(subset=['MMSI', 'IMO'])
    ais["IMO"] = ais["IMO"].replace({"": None, "UNKNOWN": None, "Unknown": None, "unknown": None})
    mmsi_to_imo = (
        ais.dropna(subset=["IMO"])
        .groupby("MMSI")["IMO"]
        .agg(lambda s: s.unique().tolist())
    )
    def resolve_imo(imolist):
        return imolist[0] if len(imolist) == 1 else None
    mmsi_to_single_imo = mmsi_to_imo.apply(resolve_imo)
    impute_map = mmsi_to_single_imo.dropna()
    ais.loc[ais["IMO"].isna(), "IMO"] = ais.loc[ais["IMO"].isna(), "MMSI"].map(impute_map)
    return ais[ais["IMO"].notna()]

def filter_stationary_track(df):
    """
    Filters out tracks that are stationary or too short by computing:
    - Duration
    - Displacement (start to end)
    - Total track length
    Returns the DataFrame if it's valid, else None.
    """
    MIN_DURATION_MIN = 30
    MIN_DISPLACEMENT_KM = 5.0
    MIN_TRACK_LEN_KM = 5.0
    LAT_COL, LON_COL = "Latitude", "Longitude"
    TIME_COL = "# Timestamp"
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    t0, t1 = pd.to_datetime(df[TIME_COL].iloc[[0, -1]])
    duration_min = (t1 - t0).total_seconds() / 60.0
    start = (df[LAT_COL].iloc[0], df[LON_COL].iloc[0])
    end = (df[LAT_COL].iloc[-1], df[LON_COL].iloc[-1])
    displacement_km = haversine(start, end)
    coords = df[[LAT_COL, LON_COL]].to_numpy()
    track_len_km = sum(haversine(tuple(coords[i-1]), tuple(coords[i])) for i in range(1, len(coords)))
    if duration_min >= MIN_DURATION_MIN and displacement_km >= MIN_DISPLACEMENT_KM and track_len_km >= MIN_TRACK_LEN_KM:
        return df
    else:
        return None

def clean_latlon(df, lat_col='Latitude', lon_col='Longitude'):
    """Removes rows with invalid latitude or longitude values."""
    LAT_MIN, LAT_MAX = -85.05113, 85.05113
    LON_MIN, LON_MAX = -180.0, 180.0
    mask_lat = df[lat_col].between(LAT_MIN, LAT_MAX)
    mask_lon = df[lon_col].between(LON_MIN, LON_MAX)
    return df[mask_lat & mask_lon].copy()

def process_tracks(ais):
    """
    Processes AIS tracks:
    - Removes spatial outliers using DBSCAN
    - Filters out stationary tracks
    Returns a cleaned DataFrame of all valid tracks.
    """
    EPS_METERS = 300
    MIN_SAMPLES = 3
    MIN_VALID_PTS = MIN_SAMPLES + 1
    LAT_COL, LON_COL = "Latitude", "Longitude"
    proj = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    all_cleaned = []
    for imo, df in tqdm(ais.groupby("IMO"), desc="Processing Trajectories"):
        df = clean_latlon(df).dropna(subset=[LAT_COL, LON_COL])
        if len(df) < MIN_VALID_PTS:
            continue
        x, y = proj.transform(df[LON_COL].values, df[LAT_COL].values)
        coords = np.column_stack([x, y])
        labels = DBSCAN(eps=EPS_METERS, min_samples=MIN_SAMPLES).fit_predict(coords)
        df["label"] = labels
        df = df[df["label"] != -1].drop(columns="label")
        if df.empty:
            continue
        df = filter_stationary_track(df)
        if df is not None:
            all_cleaned.append(df)
    return pd.concat(all_cleaned, ignore_index=True) if all_cleaned else pd.DataFrame()

def assign_trip_numbers(df, timestamp_col="# Timestamp", gap_hours=4):
    """
    Adds a 'trip_number' column to a DataFrame containing AIS data for multiple IMOs.
    A new trip is defined when the time gap between consecutive rows exceeds `gap_hours`.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['IMO', '# Timestamp', 'Latitude', 'Longitude']
        timestamp_col (str): Name of the timestamp column
        gap_hours (int): Hour gap threshold to split trips
    
    Returns:
        pd.DataFrame: Input DataFrame with a new 'trip_number' column
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(["IMO", timestamp_col]).reset_index(drop=True)

    trip_numbers = []

    for imo, group in df.groupby("IMO"):
        group = group.sort_values(timestamp_col)
        time_diffs = group[timestamp_col].diff()
        new_trip_flags = (time_diffs > timedelta(hours=gap_hours)) | (time_diffs.isna())
        trip_ids = new_trip_flags.cumsum().astype(int)
        trip_numbers.extend(trip_ids)

    df["trip_number"] = trip_numbers
    return df

from tqdm import tqdm
from haversine import haversine
import pandas as pd

def bin_by_distance(df, distance_threshold_km=2.0):
    """
    Optimized binning of AIS rows by distance for each (IMO, trip_number) group.
    Rows within distance_threshold_km of the current bin start are grouped and averaged.

    Parameters:
        df (pd.DataFrame): AIS DataFrame with 'IMO', 'trip_number', '# Timestamp', 'Latitude', 'Longitude'
        distance_threshold_km (float): Max distance (in km) for points to be grouped together

    Returns:
        pd.DataFrame: Binned AIS data with averaged rows
    """
    binned_data = []

    # Iterate with tqdm for progress monitoring
    grouped = df.groupby(["IMO", "trip_number"])
    for (imo, trip), group in tqdm(grouped, desc="Binning by distance", unit="trip"):
        group = group.sort_values("# Timestamp").reset_index(drop=True)
        latitudes = group["Latitude"].values
        longitudes = group["Longitude"].values
        timestamps = group["# Timestamp"].values

        i = 0
        while i < len(group):
            bin_latitudes = [latitudes[i]]
            bin_longitudes = [longitudes[i]]
            bin_timestamps = [timestamps[i]]

            bin_start_coord = (latitudes[i], longitudes[i])
            j = i + 1

            while j < len(group):
                next_coord = (latitudes[j], longitudes[j])
                dist = haversine(bin_start_coord, next_coord)
                if dist < distance_threshold_km:
                    bin_latitudes.append(latitudes[j])
                    bin_longitudes.append(longitudes[j])
                    bin_timestamps.append(timestamps[j])
                    j += 1
                else:
                    break

            binned_data.append({
                "IMO": imo,
                "trip_number": trip,
                "# Timestamp": min(bin_timestamps),  # Can use mean if preferred
                "Latitude": sum(bin_latitudes) / len(bin_latitudes),
                "Longitude": sum(bin_longitudes) / len(bin_longitudes),
            })

            i = j

    return pd.DataFrame(binned_data)


def main():
    """Main script for cleaning and processing all AIS files in the /notebooks/data/AIS_data/ directory."""
    input_dir = "./notebooks/data/AIS_data/"
    combined_df = []

    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".csv") and re.match(r"aisdk-\d{4}-\d{2}-\d{2}\.csv", fname):
            print(f"📥 Reading file: {fname}")
            data = read_data(fname, path=input_dir)
            print("🧹 Cleaning MMSI and IMO values...")
            ais = clean_mmsi_values(data)
            print("🧭 Running outlier removal and stationary filtering...")
            cleaned = process_tracks(ais)
            if not cleaned.empty:
                combined_df.append(cleaned)

    if combined_df:
        all_cleaned = pd.concat(combined_df, ignore_index=True)
        print("🧮 Assigning trip numbers...")
        all_cleaned = assign_trip_numbers(all_cleaned)

        print("📏 Binning by distance...")
        all_cleaned = bin_by_distance(all_cleaned, distance_threshold_km=2.0)

        keep_cols = ["IMO", "# Timestamp", "Latitude", "Longitude", "trip_number"]
        all_cleaned = all_cleaned[keep_cols]

        out_path = "./notebooks/data/trips/cleaned_combined_all_days.csv"
        all_cleaned.to_csv(out_path, index=False)
        print(f"✅ Combined cleaned data saved to {out_path} ({len(all_cleaned)} rows)")
    else:
        print("⚠️ No valid tracks after cleaning.")

if __name__ == "__main__":
    main()

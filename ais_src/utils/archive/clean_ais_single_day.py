import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import random
import datetime
from pyproj import Transformer
import contextily as ctx
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import glob, os, pathlib, math
from haversine import haversine
import argparse
import os
import sys
import re

def extract_date_from_filename(filename):
    # Look for date pattern YYYY-MM-DD in filename
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if not match:
        print("Error: Filename must include a date in format YYYY-MM-DD (e.g., aisdk-2025-01-01.csv)")
        sys.exit(1)
    date_str = match.group(0).replace("-", "_")
    return date_str

def build_dir_name(date):
    return f"./notebooks/data/trips/{date}/"

def read_data(filename, path="./notebooks/data/AIS_data/"):
    """
    Takes filename and returns dataframe.
    """
    full_path = os.path.join(path, filename)
    return pd.read_csv(full_path)

def save_data(df, dst_dir):
    """
    """
    imo_number = df["IMO"].iloc[0]  # Assumes all rows have same IMO
    filename = f"IMO_{imo_number}.csv"
    out_fp = os.path.join(dst_dir, filename)
    df.to_csv(out_fp, index=False)
    
    
def clean_mmsi_values(data):
    """
    Takes in the original, not processed ais data, and removes all rows that are null or that has a MMSI that is
    associated with two or more IMOs making the ship unidentifiable.
    
    Args: data [pandas dataframe] the original ais data
    Returns: cleaned ais data with no conflicting MMSI values
    """
    ais = data.dropna(subset=['MMSI','IMO']) # Drop rows with missing IMO and MMSI because no way to identify ship
    mmsi_col = "MMSI"
    imo_col  = "IMO"

    # treat '' or 'UNKNOWN' (case-insensitive) as NA
    ais[imo_col] = ais[imo_col].replace(
        {"": None, "UNKNOWN": None, "Unknown": None, "unknown": None}
    )

    # map each MMSI to the set of *known* IMOs it appears with
    mmsi_to_imo = (
        ais.dropna(subset=[imo_col])
        .groupby(mmsi_col)[imo_col]
        .agg(lambda s: s.unique().tolist())
    )

    def resolve_imo(imolist):
        if len(imolist) == 1:
            return imolist[0]        # unique ⇒ OK
        else:
            print("Problem MMSI")
            return None              # 0 or >1 ⇒ unresolved

    mmsi_to_single_imo = mmsi_to_imo.apply(resolve_imo)

    # create a Series we can map with
    impute_map = mmsi_to_single_imo.dropna()  # keep only MMSI with a unique IMO

    # fill NA IMO values where possible
    mask_missing = ais[imo_col].isna()
    ais.loc[mask_missing, imo_col] = (
        ais.loc[mask_missing, mmsi_col]
        .map(impute_map)          # map returns NaN if MMSI not in `impute_map`
    )

    # Clean ais without ships that did not report an IMO
    ais = ais[ais[imo_col].notna()] 
    
    return ais

def filter_stationary_track(df):
    """
    Function to check if a track is mainly stationary (at anchor) or if it is actually traveling on a vogage.

    Args: df [Pandas dataframe] for a single track
    Returns: The same track if moving or None if not moving
    """
    MIN_DURATION_MIN  = 30       # track must span ≥ 30 min
    MIN_DISPLACEMENT_KM = 5.0    # start–end distance  ≥ 5 km
    MIN_TRACK_LEN_KM   = 5.0     # sum of segment lengths ≥ 5 km
    LAT_COL, LON_COL   = "Latitude", "Longitude"
    TIME_COL           = "# Timestamp"           # adjust to your column name

    def track_stats(df):
        """
        Returns duration (minutes), displacement_km, track_len_km
        `df` must be sorted by timestamp already.
        """
        # duration
        t0, t1 = pd.to_datetime(df[TIME_COL].iloc[[0, -1]])
        duration_min = (t1 - t0).total_seconds() / 60.0

        # displacement (great-circle between first & last point)
        start = (df[LAT_COL].iloc[0], df[LON_COL].iloc[0])
        end   = (df[LAT_COL].iloc[-1], df[LON_COL].iloc[-1])
        displacement_km = haversine(start, end)

        # track length = sum of distance between consecutive points
        coords = df[[LAT_COL, LON_COL]].to_numpy()
        track_len_km = sum(haversine(tuple(coords[i-1]), tuple(coords[i]))
                        for i in range(1, len(coords)))

        return duration_min, displacement_km, track_len_km
    
    # sort by time for safety
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # --------- compute stats ----------
    dur_min, disp_km, track_km = track_stats(df)

    # --------- keep / discard ----------
    if (dur_min >= MIN_DURATION_MIN and disp_km  >= MIN_DISPLACEMENT_KM and track_km >= MIN_TRACK_LEN_KM):
        return df
    else:
        return None


def remove_outliers_and_filter_stationary(ais, dst_dir):
    """
    """

    # Geodetic WGS84  →  metric UTM 32N
    to_utm = Transformer.from_crs("epsg:4326", "epsg:25832", always_xy=True).transform
    LAT_MIN, LAT_MAX = -85.05113, 85.05113     # Web-Mercator safe limits
    LON_MIN, LON_MAX = -180.0, 180.0

    def clean_latlon(df, lat_col='Latitude', lon_col='Longitude'):
        mask_lat = df[lat_col].between(LAT_MIN, LAT_MAX)
        mask_lon = df[lon_col].between(LON_MIN, LON_MAX)
        cleaned   = df[mask_lat & mask_lon].copy()
        dropped_n = len(df) - len(cleaned)
        if dropped_n:
            print(f"Dropped {dropped_n} rows with impossible lat/lon.")
        return cleaned
    
    LAT_COL, LON_COL = "Latitude", "Longitude"
    EPS_METERS       = 300          # DBSCAN ε
    MIN_SAMPLES      = 3            # DBSCAN minPts
    MIN_VALID_PTS    = MIN_SAMPLES+1  # skip smaller tracks

    proj = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)



    for imo, df in tqdm(ais.groupby('IMO'), desc="Processing Trajectories"):
        df = clean_latlon(df).dropna(subset=[LAT_COL, LON_COL])

        # if after cleaning we have almost no points → skip / drop
        if len(df) < MIN_VALID_PTS:
            # 1) skip:
            print("Trajectory too small")
            continue
            # project to metres
        x, y   = proj.transform(df[LON_COL].values, df[LAT_COL].values)
        coords = np.column_stack([x, y])

        # DBSCAN on tracks with enough points
        labels = DBSCAN(
            eps=EPS_METERS,
            min_samples=MIN_SAMPLES,
            metric="euclidean"
        ).fit_predict(coords)

        df["label"] = labels
        df_clean    = df[df["label"] != -1].drop(columns="label")

        # optional: if *all* points became noise, you might drop the file
        if df_clean.empty:
            continue

        # If track is good so far, check if it is stationary
        df = filter_stationary_track(df)
        if df is None: continue # If track is None then skip this track

        # Lastly, save track to file
        save_data(df, dst_dir)


def main():
    parser = argparse.ArgumentParser(description="Run AIS cleaning script.")
    parser.add_argument("filename", type=str, help="CSV filename in AIS_data/ (e.g., aisdk-2025-01-01.csv)")

    args = parser.parse_args()
    filename = args.filename

    date_str = extract_date_from_filename(filename)
    output_dir = build_dir_name(date_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading file: {args.filename}")
    data = read_data(args.filename)

    print("Cleaning MMSI values...")
    ais = clean_mmsi_values(data)

    print("Running outlier and stationary filter...")
    remove_outliers_and_filter_stationary(ais, output_dir)



if __name__ == "__main__":
    main()


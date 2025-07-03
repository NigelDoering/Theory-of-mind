import os
import pandas as pd
from datetime import timedelta
import re
from tqdm import tqdm

def get_all_imos(base_dir="./notebooks/data/trips"):
    imo_set = set()
    pattern = re.compile(r"IMO_(\d+)\.csv")

    for day_folder in os.listdir(base_dir):
        day_path = os.path.join(base_dir, day_folder)
        if not os.path.isdir(day_path):
            continue

        for filename in os.listdir(day_path):
            match = pattern.match(filename)
            if match:
                imo = match.group(1)
                imo_set.add(imo)

    return imo_set

def process_imo_voyages(imo_number, base_dir="./notebooks/data/trips", processed_dir="./notebooks/data/processed_trips", timestamp_col="# Timestamp", gap_hours=4):
    filename = f"IMO_{imo_number}.csv"
    all_dfs = []

    for day_folder in sorted(os.listdir(base_dir)):
        day_path = os.path.join(base_dir, day_folder)
        if not os.path.isdir(day_path):
            continue

        imo_path = os.path.join(day_path, filename)
        if os.path.exists(imo_path):
            df = pd.read_csv(imo_path)
            df["source_day"] = day_folder
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No data found for IMO {imo_number}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df[timestamp_col] = pd.to_datetime(full_df[timestamp_col])
    full_df = full_df.sort_values(timestamp_col).reset_index(drop=True)

    gap = timedelta(hours=gap_hours)
    full_df["TimeDiff"] = full_df[timestamp_col].diff()
    full_df["NewVoyage"] = (full_df["TimeDiff"] > gap) | (full_df["TimeDiff"].isna())
    full_df["VoyageID"] = full_df["NewVoyage"].cumsum().astype(int)

    os.makedirs(processed_dir, exist_ok=True)

    for i, voyage_id in enumerate(sorted(full_df["VoyageID"].unique()), start=1):
        voyage_df = full_df[full_df["VoyageID"] == voyage_id]
        out_filename = f"IMO_{imo_number}_trip_{i}.csv"
        out_path = os.path.join(processed_dir, out_filename)
        voyage_df.to_csv(out_path, index=False)


def main():
    imo_set = get_all_imos()
    for imo in tqdm(sorted(imo_set), desc="Processing IMOs"):
        try:
            process_imo_voyages(imo)
        except Exception as e:
            print(f"❌ Failed to process IMO {imo}: {e}")

if __name__ == "__main__":
    main()
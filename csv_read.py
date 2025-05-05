import os
import pandas as pd

META_DIR = "fma_metadata"  # adjust if needed
csv_files = ["tracks.csv", "features.csv", "echonest.csv", "genres.csv"]

for fname in csv_files:
    path = os.path.join(META_DIR, fname)
    # read only the header row
    df = pd.read_csv(path, nrows=0, low_memory=False)
    print(f"{fname} columns ({len(df.columns)}):")
    for col in df.columns:
        print("  -", col)
    print()
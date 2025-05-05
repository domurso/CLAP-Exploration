import os
import pandas as pd

feat_path = os.path.join("fma_metadata", "features.csv")

df0 = pd.read_csv(feat_path, nrows=0, low_memory=False)
print("features.csv columns:", df0.columns.tolist())

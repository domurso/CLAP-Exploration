import zipfile
import pandas as pd


# Load only the first 20 rows of genres.csv
with zipfile.ZipFile("fma_metadata.zip") as z:
    genre_path = next(n for n in z.namelist() if n.endswith("genres.csv"))
    df = pd.read_csv(z.open(genre_path))

print("genres.csv (first 20 rows)", df.to_string())

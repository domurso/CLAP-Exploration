# data_setup.py

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio

# Constants
ZIP_FILE = "fma_metadata.zip"
ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
TARGET_FOLDER = "data/fma_clap_with_local_genre_48K_50"

# Ensure metadata ZIP is available
if not os.path.exists(ZIP_FILE):
    print(f"💾 {ZIP_FILE} not found, downloading from {ZIP_URL} …")
    urllib.request.urlretrieve(ZIP_URL, ZIP_FILE)
    print("✅ Download complete.")

# Extract genres.csv and build genre_map assuming 'genre_title' column
# This turned out to be useless since the mappings for genres is incorrect in the metadata
with zipfile.ZipFile("fma_metadata.zip") as z:
    # tracks.csv has: track_id, title, artist_name, tags, listens, released, etc.
    with z.open("tracks.csv") as f:
        tracks_df = pd.read_csv(f, index_col="track_id")
    # features.csv has: track_id, tempo, key, loudness, etc.
    with z.open("features.csv") as f:
        feats_df = pd.read_csv(f, index_col="track_id")
    # echonest.csv has: track_id, energy, danceability, etc. (optional)
    with z.open("echonest.csv") as f:
        echo_df = pd.read_csv(f, index_col="track_id")

# Merge them into one big lookup
meta = tracks_df.join(feats_df, how="left").join(echo_df, how="left")

#updating to 48000K instead of 16000K sampling rate
def preprocess(batch, target_length=48_000 * 5):
    wav = batch["audio"]["array"]
    # … pad/truncate code unchanged …

    batch["input_values"] = wav.astype(np.float32)

    tid = batch["track_id"]
    row = meta.loc[tid]

    # pull out some fields
    title     = row["title"]
    artist    = row["artist_name"]
    tempo     = int(row.get("tempo", 0))
    loudness  = round(row.get("loudness", 0), 1)
    energy    = round(row.get("energy", 0), 2)           # from echonest
    tags      = batch.get("tags", [])[:3]
    tag_str   = ", ".join(tags) if tags else "no tags"

    # craft a richer caption
    prompt = (
        f"“{title}” by {artist}: "
        f"{tempo} BPM, {loudness} dB, energy {energy}. "
        f"Tags: {tag_str}."
    )
    batch["text"] = prompt
    return batch


def main():
    # Load a subset of examples
    ds = load_dataset(
        'benjamin-paine/free-music-archive-small',
        split='train[:500]'
    )

    # Decode MP3 to waveform arrays via FFmpeg
    #updating to 48000K instead of 16000K sampling rate
    ds = ds.cast_column('audio', Audio(sampling_rate=48_000, decode='ffmpeg'))

    # Preprocess to add input_values, text, and genre_title
    ds = ds.map(preprocess, num_proc=4)

    # Select only the fields needed for contrastive fine-tuning
    keep = [
        '',
        'audio',
        'title',
        'url',
        'artist',
        'genres',
        'tags',
        'album_title',
        # 'genre_title',
        'input_values',
        'text'
    ]
    ds = ds.select_columns(keep)

    # Save to disk
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    ds.save_to_disk(TARGET_FOLDER)
    print(f"✅ Dataset saved to '{TARGET_FOLDER}', columns: {ds.column_names}")

if __name__ == '__main__':
    main()

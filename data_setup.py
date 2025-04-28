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
TARGET_FOLDER = "data/fma_clap_with_local_genre"

# Ensure metadata ZIP is available
if not os.path.exists(ZIP_FILE):
    print(f"💾 {ZIP_FILE} not found, downloading from {ZIP_URL} …")
    urllib.request.urlretrieve(ZIP_URL, ZIP_FILE)
    print("✅ Download complete.")

# Extract genres.csv and build genre_map assuming 'genre_title' column
# This turned out to be useless since the mappings for genres is incorrect in the metadata
with zipfile.ZipFile(ZIP_FILE) as z:
    names = z.namelist()
    genre_path = next((n for n in names if n.endswith("genres.csv")), None)
    if genre_path is None:
        raise FileNotFoundError(
            f"Could not find 'genres.csv' in {ZIP_FILE}; contents:\n" + "\n".join(names)
        )
    with z.open(genre_path) as f:
        genre_df = pd.read_csv(f)

# Directly use the 'genre_title' column
if 'genre_title' not in genre_df.columns:
    raise KeyError("Expected 'genre_title' column in genres.csv")

genre_map = genre_df.set_index("genre_id")["genre_title"].to_dict()
print(f"🔖 Loaded genre map ({len(genre_map)} entries) using 'genre_title'.")


def preprocess(batch, target_length=16_000 * 5):
    # Audio: pad or truncate to fixed length
    wav = batch['audio']['array']
    if len(wav) < target_length:
        wav = np.pad(wav, (0, target_length - len(wav)))
    else:
        wav = wav[:target_length]
    batch['input_values'] = wav.astype(np.float32)

    # Build text prompt: title, artist, album, genre, tags
    title = batch.get('title', '')
    artist = batch.get('artist', '')
    album = batch.get('album_title', '')
    tags = batch.get('tags', [])[:3]
    genre_ids = batch.get('genres', [])
    genre_title = genre_map.get(genre_ids[0], '') if genre_ids else ''

    parts = [f"\"{title}\" by {artist}"]
    if album:
        parts.append(f"from the album \"{album}\"")
    # if genre_title:
    #     parts.append(f"Genre: {genre_title}")
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    prompt = ". ".join(parts) + "."

    batch['text'] = prompt
    batch['genre_title'] = genre_title
    return batch


def main():
    # Load a subset of examples
    ds = load_dataset(
        'benjamin-paine/free-music-archive-small',
        split='train[:100]'
    )

    # Decode MP3 to waveform arrays via FFmpeg
    ds = ds.cast_column('audio', Audio(sampling_rate=16_000, decode='ffmpeg'))

    # Preprocess to add input_values, text, and genre_title
    ds = ds.map(preprocess, num_proc=4)

    # Select only the fields needed for contrastive fine-tuning
    keep = [
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

#!/usr/bin/env python3
# data_setup.py

import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio

# VARS
META_DIR      = "fma_metadata"
DATASET_SPLIT = "train[:500]"
TARGET_DIR    = "data/fma_clap_basic_meta_48K_500"
SAMPLE_RATE   = 48_000
CLIP_SECONDS  = 5
TARGET_LEN    = SAMPLE_RATE * CLIP_SECONDS

# LOAD + FLATTEN
tracks = pd.read_csv(
    os.path.join(META_DIR, "tracks.csv"),
    header=[0, 1],
    index_col=0,
    low_memory=False,
)
flat = pd.DataFrame({
    "title":       tracks["track"]["title"],
    "artist_name": tracks["artist"]["name"],
    "genre_top":   tracks["track"]["genre_top"]
}, index=tracks.index)

# LOOKUP ARTIST + TITLE to Find GENRE
meta = flat.reset_index().rename(columns={"index": "track_id"})
meta["lookup_key"] = (
    meta["title"].str.lower().fillna("") + "||" +
    meta["artist_name"].str.lower().fillna("")
)

meta_map = {
    row.lookup_key: {
        "genre_top": row.genre_top or ""
    }
    for row in meta.itertuples(index=False)
}

# PREPROCESS 
def preprocess(batch):
    # a) pad/truncate audio
    wav = batch["audio"]["array"]
    if len(wav) < TARGET_LEN:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    else:
        wav = wav[:TARGET_LEN]
    batch["input_values"] = wav.astype(np.float32)

    # b) lookup genre
    title  = batch.get("title", "")  or ""
    artist = batch.get("artist", "") or ""
    key    = f"{title.strip().lower()}||{artist.strip().lower()}"
    genre  = meta_map.get(key, {}).get("genre_top", "")

    # c) build prompt
    tags    = batch.get("tags", [])[:3]
    tag_str = ", ".join(tags) if tags else "no tags"
    prompt = (
        f"“{title}” by {artist}. "
        f"Genre: {genre}. Tags: {tag_str}."
    )
    batch["text"] = prompt
    return batch

# MAIN 
def main():
    ds = load_dataset("benjamin-paine/free-music-archive-small", split=DATASET_SPLIT)
    print("Raw columns:", ds.column_names)

    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode="ffmpeg"))
    ds = ds.map(preprocess, num_proc=4)

    ds = ds.select_columns([
    "audio",
    "input_values",
    "text",
    "title",
    "artist",      # <-- add these two back
    ])
    os.makedirs(TARGET_DIR, exist_ok=True)
    ds.save_to_disk(TARGET_DIR)
    print("✅ Saved to", TARGET_DIR, "with columns", ds.column_names)

if __name__ == "__main__":
    main()

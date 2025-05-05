# data_setup.py

import zipfile
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio

# ——— Load genre_map  ———
with zipfile.ZipFile("fma_metadata.zip") as z:
    # find the path inside the ZIP that ends with 'genres.csv'
    names = z.namelist()
    genre_path = next((n for n in names if n.endswith("genres.csv")), None)
    if genre_path is None:
        raise FileNotFoundError(
            "Could not find 'genres.csv' in fma_metadata.zip; found:\n" +
            "\n".join(names)
        )

    # read the file once into a DataFrame (no index yet)
    with z.open(genre_path) as f:
        genre_df = pd.read_csv(f)

    # for debugging, you can print the actual columns:
    # print("genres.csv columns:", list(genre_df.columns))

    # pick the correct human‐readable column
    if "name" in genre_df.columns:
        human_col = "name"
    elif "title" in genre_df.columns:
        human_col = "title"
    else:
        # fallback to second column if neither exists
        human_col = genre_df.columns[1]

    # now build genre_id → human name map
    genre_map = genre_df.set_index("genre_id")[human_col].to_dict()

def preprocess(batch, target_length=16_000 * 5):
    # pad / truncate audio
    wav = batch["audio"]["array"]
    if len(wav) < target_length:
        wav = np.pad(wav, (0, target_length - len(wav)))
    else:
        wav = wav[:target_length]
    batch["input_values"] = wav.astype(np.float32)

    # build text prompt with title, genre, tags
    title = batch.get("title", "")
    tags  = batch.get("tags", []) or []
    genre_ids = batch.get("genres", [])
    genre_name = genre_map.get(genre_ids[0], "") if genre_ids else ""

    parts = [p for p in (title, genre_name) if p]
    prompt = " — ".join(parts)
    if tags:
        prompt += " — Tags: " + ", ".join(tags[:5])
    batch["text"] = prompt

    return batch

def main():
    ds = load_dataset(
        "benjamin-paine/free-music-archive-small",
        split="train[:100]"
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode="ffmpeg"))
    ds = ds.map(preprocess, num_proc=4)
    ds = ds.select_columns(["audio", "input_values", "text"])
    ds.save_to_disk("data/fma_clap_with_local_genre")
    print("✅ Saved; columns:", ds.column_names)

if __name__ == "__main__":
    main()

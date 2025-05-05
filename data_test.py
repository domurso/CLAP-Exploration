from datasets import load_from_disk

ds = load_from_disk("data/fma_clap_with_local_genre")
print(ds.column_names)

# first = ds[0]
# print(first)
# → {'track_id': 1234,
#    'title': 'Some Song',
#    'artist_name': 'Artist',
#    'audio': {'array': array([...]), 'sampling_rate': 16000},
#    'input_values': array([...], dtype=float32),
#    'labels': 70,
#    …}

df = ds.select([0]).to_pandas()
print(df)

for i, ex in enumerate(ds):
    print(f"{i:2d}: {ex['text']}")
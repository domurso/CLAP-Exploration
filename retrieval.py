#!/usr/bin/env python3
"""
retrieval.py

Given a text query, retrieves the best-matching audio clips
from your fine-tuned CLAP model + dataset.
"""

import torch
import torch.nn.functional as F
from transformers import ClapProcessor, ClapModel
from datasets import load_from_disk

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR    = "data/fma_clap_with_varied_prompts_48K"
MODEL_DIR   = "model_fma_clap_with_varied_prompts_48K"
SAMPLE_RATE = 48_000   # must match your preprocessed dataset
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1) Load the fine-tuned CLAP
    processor = ClapProcessor.from_pretrained(MODEL_DIR)
    model     = ClapModel.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    # 2) Load your dataset
    ds = load_from_disk(DATA_DIR)

    # 3) Precompute all audio embeddings
    audio_embs = []
    for ex in ds:
        wav = ex["input_values"]            # 48kHz, fixed length
        audio_in = processor.feature_extractor(
            [wav],
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            emb = model.get_audio_features(**audio_in)  # returns Tensor (1, D)
        audio_embs.append(emb.cpu())
    audio_embs = torch.cat(audio_embs, dim=0)  # (N, D)

    # 4) Define a simple retrieve function
    def retrieve(query: str, topk: int = 5):
        # Encode the text
        text_in = processor.tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            text_emb = model.get_text_features(**text_in).cpu()  # (1, D)

        # Cosine similarity between each audio emb and the single text emb
        sims = F.cosine_similarity(audio_embs, text_emb, dim=-1)  # (N,)

        # Grab top-k
        top_scores, top_idxs = sims.topk(topk)
        print(f"\nQuery: “{query}”\nTop {topk} matches:")
        for idx, score in zip(top_idxs.tolist(), top_scores.tolist()):
            meta = ds[int(idx)]
            print(f"  • [{idx:4d}] “{meta['title']}” by {meta['artist']} (score {score:.3f})")

    # 5) Example
    retrieve("Calm folk music", topk=5)

if __name__ == "__main__":
    main()

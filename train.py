#!/usr/bin/env python3
"""
train.py

Fine-tunes CLAP (audio↔text contrastive) on your preprocessed FMA dataset,
evaluates Recall@1/5/10 each epoch, and prints a quick sanity check.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ClapProcessor, ClapModel
from datasets import load_from_disk

# VARS
DATA_DIR      = "data/fma_clap_basic_meta_48K_500"
OUTPUT_DIR    = "model_fma_clap_with_varied_prompts_48K"
BATCH_SIZE    = 16
NUM_EPOCHS    = 3
LEARNING_RATE = 3e-5
SAMPLE_RATE   = 48_000   # must match your dataset’s audio sampling rate

# COLLATE
def collate_fn(batch):
    audios = [ex["input_values"] for ex in batch]
    texts  = [ex["text"]         for ex in batch]
    audio_inputs = processor.feature_extractor(
        audios,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt"
    )
    text_inputs = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return {**audio_inputs, **text_inputs}

# Retrieval metrics
def compute_recall_at_k(model, loader, device, ks=(1, 5, 10)):
    model.eval()
    audio_embs, text_embs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            audio_embs.append(out.audio_embeds)
            text_embs.append(out.text_embeds)

    A = torch.cat(audio_embs)  # (N, D)
    T = torch.cat(text_embs)   # (N, D)
    sims = F.cosine_similarity(A.unsqueeze(1), T.unsqueeze(0), dim=-1)  # (N, N)
    ranks = sims.argsort(dim=-1, descending=True)

    N = A.size(0)
    recalls = {}
    for k in ks:
        correct = sum((ranks[i, :k] == i).any().item() for i in range(N))
        recalls[f"R@{k}"] = correct / N
    return recalls

# MAIN
def main():
    # 1) Load & split
    ds = load_from_disk(DATA_DIR)
    N  = len(ds)
    split = int(0.8 * N)
    ds = load_from_disk(DATA_DIR)
    ds = ds.shuffle(seed=42)

    ds_train = ds.select(range(0, split))
    ds_val   = ds.select(range(split, N))

    # keep raw texts for the sanity check
    val_texts = ds_val["text"]

    # 2) Init processor & model
    global processor, model
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #DataLoaders
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    #Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fct  = torch.nn.CrossEntropyLoss()

    #Training + validation
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- training ---
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            logits_a = out.logits_per_audio
            logits_t = out.logits_per_text
            labels   = torch.arange(logits_a.size(0), device=device)
            loss = 0.5 * (loss_fct(logits_a, labels) + loss_fct(logits_t, labels))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # **compute avg_loss inside the loop so it's in scope**
        avg_loss = total_loss / len(train_loader)

        # --- validation metrics ---
        recalls = compute_recall_at_k(model, val_loader, device)
        metrics = " ".join(f"{k}={v:.3f}" for k, v in recalls.items())
        print(f"Epoch {epoch} — Train Loss: {avg_loss:.4f} — {metrics}")

        # --- sanity check: top-3 predictions for first 3 val clips ---
        model.eval()
        batch_feats = next(iter(val_loader))
        batch_feats = {k: v.to(device) for k, v in batch_feats.items()}
        with torch.no_grad():
            sims = model(**batch_feats).logits_per_audio  # (B, B)
        for i in range(min(3, sims.size(0))):
            top3 = sims[i].argsort(descending=True)[:3]
            print(f"\n Clip {i} — true text: {val_texts[i]}")
            for r in top3:
                print(f"    • \"{val_texts[r]}\" (score {sims[i,r]:.3f})")
        print("────────────────────────────────────────────────\n")

    # SAVE MODEL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ Saved fine-tuned model to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()

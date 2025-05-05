import torch
from torch.utils.data import DataLoader
from transformers import ClapProcessor, ClapModel
from datasets import load_from_disk

#USE 48K Sample rate
ds = load_from_disk("data/fma_clap_with_local_genre_48K")
#split into train/val
ds_train = ds.select(range(0, 80))
ds_val   = ds.select(range(80, 100))

#Instantiate processor & model
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
model.train()

# 3) Build collate_fn
def collate_fn(batch):
    # Gather raw waveforms and captions
    audios = [ex["input_values"] for ex in batch]
    texts  = [ex["text"]         for ex in batch]

    # Turn waveforms into model-ready audio tensors
    audio_inputs = processor.feature_extractor(
        audios,
        #48K sampling rate required
        sampling_rate=48_000,
        padding=True,
        return_tensors="pt",
    )

    # Turn strings into input IDs + attention masks
    text_inputs = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Merge into one dict
    inputs = {
        **audio_inputs,   # e.g. {"input_values": ..., "attention_mask": ...}
        **text_inputs,    # e.g. {"input_ids": ..., "attention_mask": ...}
    }

    return inputs


# DataLoader
train_loader = DataLoader(ds_train, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(ds_val,   batch_size=16, shuffle=False, collate_fn=collate_fn)

#optimizer & loss
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
loss_fct  = torch.nn.CrossEntropyLoss()

#Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 4):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        # move inputs to device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)

        # contrastive logits
        logits_a = outputs.logits_per_audio   # (B, B)
        logits_t = outputs.logits_per_text    # (B, B)
        labels   = torch.arange(logits_a.size(0), device=device)

        loss = 0.5 * (loss_fct(logits_a, labels) + loss_fct(logits_t, labels))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} — avg training loss: {avg_loss:.4f}")


model.save_pretrained("clap-finetuned-fma_small_batch")
processor.save_pretrained("clap-finetuned-fma_small_batch")
print("✅ Saved fine-tuned model to clap-finetuned-fma_small_batch/")

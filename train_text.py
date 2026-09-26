# train_text.py

import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer
from evaluate import evaluate

from utils.dataset import PhishingDataset
from models.text_model import TextModel

import os
import pandas as pd

# =========================
# PATHS (update if needed)
# =========================
DATA_DIR   = "data"
OUTPUT_DIR = "."

TEXT_DIR  = os.path.join(DATA_DIR, "text")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

SAMPLE_CSV = os.path.join(DATA_DIR, "sample.csv")

if not os.path.exists(SAMPLE_CSV):
    df = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))
    df = df.sample(5000, random_state=42)
    df.to_csv(SAMPLE_CSV, index=False)
    print(f"sample.csv created")
else:
    print("sample.csv already exists, skipping creation")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TOKENIZER
# =========================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# =========================
# DATASET + TRAIN/TEST SPLIT
# =========================
full_dataset = PhishingDataset(
    csv_file=SAMPLE_CSV,
    text_dir=TEXT_DIR,
    image_dir=IMAGE_DIR,
    tokenizer=tokenizer,
    use_text=True,
    use_image=False
)

train_size = int(0.8 * len(full_dataset))
test_size  = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train samples: {train_size} | Test samples: {test_size}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# =========================
# MODEL
# =========================
model = TextModel().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# =========================
# TRAINING + BEST MODEL SAVE
# =========================
EPOCHS    = 5
best_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device)
        )

        loss = criterion(outputs, batch['label'].to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f"Train Loss: {avg_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_text_model.pth"))
        print("✅ Best model saved!")

# =========================
# EVALUATION ON TEST SET
# =========================
print("\n--- Evaluation on held-out TEST set ---")
evaluate(model, test_loader, device)

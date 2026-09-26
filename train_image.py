# train_image.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluate import evaluate

from utils.dataset import PhishingDataset
from models.image_model import ImageModel

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
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# DATASET + TRAIN/TEST SPLIT
# =========================
full_dataset_train = PhishingDataset(
    csv_file=SAMPLE_CSV,
    text_dir=TEXT_DIR,
    image_dir=IMAGE_DIR,
    transform=train_transform,
    use_text=False,
    use_image=True
)

full_dataset_test = PhishingDataset(
    csv_file=SAMPLE_CSV,
    text_dir=TEXT_DIR,
    image_dir=IMAGE_DIR,
    transform=test_transform,
    use_text=False,
    use_image=True
)

generator  = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(full_dataset_train))
test_size  = len(full_dataset_train) - train_size

train_indices, test_indices = torch.utils.data.random_split(
    range(len(full_dataset_train)), [train_size, test_size], generator=generator
)

train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices.indices)
test_dataset  = torch.utils.data.Subset(full_dataset_test,  test_indices.indices)

print(f"Train samples: {train_size} | Test samples: {test_size}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# =========================
# MODEL
# =========================
model = ImageModel().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# =========================
# TRAINING + BEST MODEL SAVE
# =========================
EPOCHS    = 8
best_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(batch['image'].to(device))

        loss = criterion(outputs, batch['label'].to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f"Train Loss: {avg_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_image_model.pth"))
        print("✅ Best model saved!")

# =========================
# EVALUATION ON TEST SET
# =========================
print("\n--- Evaluation on held-out TEST set ---")
evaluate(model, test_loader, device, is_image=True)

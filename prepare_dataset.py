# prepare_dataset.py

import os
import shutil
import pandas as pd
from tqdm import tqdm
from utils.preprocessing import clean_html

# =========================
# CONFIG (UPDATE THIS PATH)
# =========================
SOURCE_DIR = "C:/phishpedia"   # <-- your extracted dataset path
OUTPUT_DIR = "data"

TEXT_DIR  = os.path.join(OUTPUT_DIR, "text")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(TEXT_DIR,  exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

rows = []

# =========================
# PROCESS EACH CLASS
# =========================
def process_folder(folder_path, label):
    count = 0

    for site in tqdm(os.listdir(folder_path), desc=f"Processing {folder_path}"):

        site_path = os.path.join(folder_path, site)

        if not os.path.isdir(site_path):
            continue

        html_path = os.path.join(site_path, "html.txt")
        img_path  = os.path.join(site_path, "shot.png")

        if not os.path.exists(html_path) or not os.path.exists(img_path):
            continue

        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_html = f.read()
        except Exception:
            continue

        text = clean_html(raw_html)[:5000]

        if len(text.strip()) == 0:
            continue

        text_file = f"{label}_{count}.txt"
        img_file  = f"{label}_{count}.png"

        with open(os.path.join(TEXT_DIR, text_file), "w", encoding="utf-8") as f:
            f.write(text)

        shutil.copy(img_path, os.path.join(IMAGE_DIR, img_file))

        rows.append({
            "text_file":  text_file,
            "image_file": img_file,
            "label":      label
        })

        count += 1


# =========================
# RUN PROCESSING
# =========================
print("Starting dataset preparation...")

process_folder(os.path.join(SOURCE_DIR, "phishing"), 1)
process_folder(os.path.join(SOURCE_DIR, "benign"), 0)

# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)

print("\nDataset prepared successfully!")
print(f"Total samples: {len(df)}")

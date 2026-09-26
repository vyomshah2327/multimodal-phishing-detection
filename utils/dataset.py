# utils/dataset.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class PhishingDataset(Dataset):
    def __init__(self, csv_file, text_dir, image_dir,
                 tokenizer=None, max_len=512, transform=None,
                 use_text=True, use_image=True):

        self.df = pd.read_csv(csv_file)
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        self.use_text = use_text
        self.use_image = use_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        data = {}

        # =========================
        # TEXT
        # =========================
        if self.use_text:
            text_path = os.path.join(self.text_dir, row['text_file'])
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )

            data['input_ids'] = encoding['input_ids'].squeeze(0)
            data['attention_mask'] = encoding['attention_mask'].squeeze(0)

        # =========================
        # IMAGE
        # =========================
        if self.use_image:
            img_path = os.path.join(self.image_dir, row['image_file'])
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            data['image'] = image

        # =========================
        # LABEL
        # =========================
        data['label'] = torch.tensor(row['label'], dtype=torch.long)

        return data

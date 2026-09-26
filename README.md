# Multimodal Phishing Webpage Detection

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-RoBERTa-FFD21E)
![License](https://img.shields.io/badge/License-MIT-blue)

Detecting phishing webpages from **what the page says** (HTML text → RoBERTa), **what it looks like** (screenshot → ResNet-50), and **both together** (late fusion) — and measuring which signal actually matters.

Trained on 5,000 labelled pages from the [Phishpedia](https://github.com/lindsey98/Phishpedia) dataset; evaluated on a held-out test set of 1,000 pages (582 phishing / 418 benign).

---

## Results (held-out test set)

| Model | Accuracy | Missed phishing (FNR) ↓ | False alarms (FPR) ↓ |
|---|---|---|---|
| **RoBERTa — HTML text only** | **97.5%** | **1.55%** (9 / 582) | **3.83%** |
| Late fusion — text + screenshot | 97.2% | 1.89% (11 / 582) | 4.07% |
| ResNet-50 — screenshot only | 94.0% | 5.84% (34 / 582) | 6.22% |

**Key finding:** adding the screenshot did **not** help. The text model alone missed the fewest phishing pages; fusion was slightly worse. Page text is the stronger signal — attackers can clone a site's look pixel-for-pixel far more easily than they can reproduce its real content — and naïve concatenation lets the weaker, higher-dimensional image features (2048-d vs 768-d) add noise rather than information.

In security, **FNR is the number that matters**: every missed phishing page is a potential stolen credential, so the models are compared on that first.

---

## Architecture

```
            HTML ──► cleaned text ──► RoBERTa-base ──► 768-d ─┐
Webpage ──►                                                   ├─► concat (2816) ──► MLP 512 ──► phishing / benign
            screenshot ──────────────► ResNet-50   ──► 2048-d ┘
```

- **Text branch:** HTML stripped with BeautifulSoup, truncated to 5,000 characters, RoBERTa pooled output; fine-tuned with Adam at 2e-5 and a cosine schedule.
- **Image branch:** ImageNet-pretrained ResNet-50 fine-tuned on page screenshots (lr 1e-4, step decay).
- **Fusion:** both encoders fine-tuned end-to-end with a 2-layer MLP head (dropout 0.3).

---

## Project structure

```
├── prepare_dataset.py      # Phishpedia folders → cleaned text + screenshots + labels.csv
├── train_text.py           # RoBERTa text classifier
├── train_image.py          # ResNet-50 screenshot classifier
├── train_multimodal.py     # late-fusion model
├── evaluate.py             # accuracy, FNR/FPR, confusion matrix
├── models/                 # text_model.py, image_model.py, multimodal_model.py
├── utils/                  # dataset + preprocessing helpers
└── outputs/                # training logs and test-set reports for all three models
```

## Run it

```bash
pip install -r requirements.txt
# download Phishpedia and set SOURCE_DIR in prepare_dataset.py
python prepare_dataset.py
python train_text.py
python train_image.py
python train_multimodal.py
```

---

**Tech:** Python · PyTorch · Hugging Face Transformers (RoBERTa) · torchvision (ResNet-50) · BeautifulSoup · scikit-learn · pandas

Built with [Yagni Patel](https://github.com/YagniPatel) · MIT License

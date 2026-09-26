# evaluate.py

import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, dataloader, device, is_multimodal=False, is_image=False):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:

            if is_multimodal:
                outputs = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['image'].to(device)
                )
            elif is_image:
                outputs = model(batch['image'].to(device))
            else:
                outputs = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device)
                )

            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(batch['label'].numpy())

    # ── Classification report ──────────────────────────────────────────────
    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=["Benign (0)", "Phishing (1)"]
    ))

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix:")
    print(f"                 Predicted Benign   Predicted Phishing")
    print(f"  Actual Benign       {tn:<10}         {fp:<10}")
    print(f"  Actual Phishing     {fn:<10}         {tp:<10}")

    # ── Security-relevant rates ────────────────────────────────────────────
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\nSecurity-Relevant Rates:")
    print(f"  False Negative Rate (FNR): {fnr:.4f}  ← phishing missed (lower is critical)")
    print(f"  False Positive Rate (FPR): {fpr:.4f}  ← benign flagged as phishing")

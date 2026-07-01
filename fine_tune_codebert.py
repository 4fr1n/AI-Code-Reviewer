"""
fine_tune_codebert.py
----------------------
Fine-tunes the CodeBERT classification head using Stratified K-Fold
cross-validation on the weakly-labelled CRAVE-derived dataset.

Why Stratified K-Fold:
  - Dataset is small (908 rows) and imbalanced (27-454 per class)
  - A single train/val/test split risks the minority classes (security,
    performance) being unevenly distributed or barely represented in
    whichever split they land in
  - K-Fold trains and evaluates on every row exactly once across folds,
    giving a much more reliable estimate of true model performance
  - StratifiedKFold specifically preserves the class ratio in every fold,
    so even the 27-sample performance class is represented in each fold

Usage:
    pip install scikit-learn pandas torch transformers
    python fine_tune_codebert.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModel
import time

# ── Config ──────────────────────────────────────────────────────────────────

CODEBERT_NAME = "microsoft/codebert-base"
CATEGORIES    = ["bug", "style", "performance", "security", "ok"]
LABEL_TO_IDX  = {c: i for i, c in enumerate(CATEGORIES)}
IDX_TO_LABEL  = {i: c for c, i in LABEL_TO_IDX.items()}

MAX_LEN     = 512
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 2e-5
N_FOLDS     = 5
SEED        = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class CodeReviewDataset(Dataset):
    def __init__(self, codes, labels, tokenizer):
        self.codes     = codes
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class CodeBERTClassifier(nn.Module):
    def __init__(self, num_labels: int = 5):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(CODEBERT_NAME)
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return self.classifier(cls_embedding)


# ── Training / evaluation for a single fold ──────────────────────────────────

def train_one_fold(train_loader, val_loader, class_weights, fold_num):
    model = CodeBERTClassifier(num_labels=len(CATEGORIES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Fold {fold_num} · Epoch {epoch+1}/{EPOCHS} · Loss: {avg_loss:.4f}")

    # ── Evaluate on validation fold ───────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds  = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")

    return model, acc, f1, all_labels, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading training data …")
    df = pd.read_csv("labeled_data_gemini.csv")
    df = df.dropna(subset=["code", "gemini_label"])
    print(f"  {len(df)} rows loaded\n")

    codes  = df["code"].tolist()
    labels = df["gemini_label"].map(LABEL_TO_IDX).tolist()

    # ── Class weights — inverse frequency, used in the loss function ────────
    # This is critical given the imbalance (ok=454 vs performance=27).
    # Without it, the model would learn to just predict 'ok' or 'bug' for
    # everything and still score deceptively high accuracy.
    label_counts  = pd.Series(labels).value_counts().sort_index()
    total         = len(labels)
    class_weights = torch.tensor(
        [total / (len(CATEGORIES) * label_counts[i]) for i in range(len(CATEGORIES))],
        dtype=torch.float,
    )
    print("Class weights (inverse frequency):")
    for i, w in enumerate(class_weights):
        print(f"  {IDX_TO_LABEL[i]:12s}: {w:.3f}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_NAME)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_accuracies = []
    fold_f1s        = []
    all_fold_labels = []
    all_fold_preds  = []

    start_time = time.time()

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(codes, labels), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{N_FOLDS}")
        print(f"{'='*60}")

        train_codes  = [codes[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_codes    = [codes[i] for i in val_idx]
        val_labels   = [labels[i] for i in val_idx]

        train_dataset = CodeReviewDataset(train_codes, train_labels, tokenizer)
        val_dataset   = CodeReviewDataset(val_codes, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model, acc, f1, fold_labels, fold_preds = train_one_fold(
            train_loader, val_loader, class_weights, fold_num
        )

        print(f"  Fold {fold_num} Results — Accuracy: {acc:.3f}  ·  Macro F1: {f1:.3f}")

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        all_fold_labels.extend(fold_labels)
        all_fold_preds.extend(fold_preds)

        # Save the last fold's model as the final model
        if fold_num == N_FOLDS:
            torch.save(model.state_dict(), "codebert_classifier_finetuned.pt")
            print(f"\n  Saved final model -> codebert_classifier_finetuned.pt")

    elapsed = time.time() - start_time

    # ── Aggregate results across all folds ───────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean Accuracy : {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
    print(f"  Mean Macro F1 : {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
    print(f"  Total time    : {elapsed:.1f}s ({elapsed/N_FOLDS:.1f}s per fold)")

    print(f"\n  Per-class report (aggregated across all {N_FOLDS} folds):")
    print(classification_report(
        all_fold_labels, all_fold_preds,
        target_names=CATEGORIES,
        zero_division=0,
    ))


if __name__ == "__main__":
    main()

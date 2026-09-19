"""
HFA Network Training Pipeline — Day 12
Trains HFANetwork on CASME II clips.
Saves best weights to mmbi/models/hfa_net.pt
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

from mmbi.models.hfa_network import HFANetwork, MODEL_PATH, N_EMOTION
from mmbi.models.hfa_dataset import CASMEHFADataset

# ─── CONFIGURE THESE ────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\datasets\casme2"
LABEL_XLSX   = os.path.join(DATASET_ROOT, "CASME2-coding-20140508.xlsx")
SAVE_PATH    = MODEL_PATH
# ─────────────────────────────────────────────────────────────────────────────

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 8     # clips are large — keep batch small on CPU
EPOCHS      = 40
LR          = 3e-4
WEIGHT_DECAY= 1e-4
PATIENCE    = 8


class HFALoss(nn.Module):
    """
    Combined loss for HFA training:
      - CrossEntropyLoss on emotion output (primary)
      - MSE on AU scores if AU labels are available (secondary)

    For CASME II we only have emotion labels so AU loss weight = 0.
    Once you collect your own AU-labeled data, set au_weight > 0.
    """

    def __init__(self, au_weight: float = 0.0):
        super().__init__()
        self.au_weight  = au_weight
        self.ce_loss    = nn.CrossEntropyLoss()
        self.mse_loss   = nn.MSELoss()

    def forward(self,
                au_pred:      torch.Tensor,
                emotion_pred: torch.Tensor,
                emotion_label:torch.Tensor,
                au_label:     torch.Tensor = None) -> torch.Tensor:

        emotion_loss = self.ce_loss(emotion_pred, emotion_label)

        au_loss = torch.tensor(0.0, device=emotion_pred.device)
        if self.au_weight > 0 and au_label is not None:
            au_loss = self.mse_loss(au_pred, au_label)

        return emotion_loss + self.au_weight * au_loss


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to handle CASME II class imbalance.
    'others' class dominates — sampler upweights minority classes.
    """
    counts  = Counter(labels)
    total   = len(labels)
    weights = [total / counts[l] for l in labels]
    return WeightedRandomSampler(
        weights     = torch.tensor(weights, dtype=torch.double),
        num_samples = len(weights),
        replacement = True
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            au_scores, emotion_out, _ = model(clips)
            loss          = criterion(au_scores, emotion_out, labels)
            total_loss   += loss.item() * len(clips)
            preds         = emotion_out.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += len(labels)
    return total_loss / total, correct / total


def train():
    print(f"Training HFA Network on: {DEVICE}")
    print(f"Loading dataset from:    {DATASET_ROOT}\n")

    # Build full dataset (train augment=False initially, set per split below)
    full_ds = CASMEHFADataset(
        dataset_root = DATASET_ROOT,
        label_xlsx   = LABEL_XLSX,
        augment      = False
    )

    if len(full_ds) == 0:
        print("ERROR: No clips loaded. Check DATASET_ROOT path.")
        return

    # Split indices
    indices = list(range(len(full_ds)))
    labels  = full_ds.labels
    tr_idx, val_idx = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=42
    )

    # Build train/val datasets from same full_ds but different augment
    from torch.utils.data import Subset
    train_ds       = Subset(full_ds, tr_idx)
    full_ds.augment = True    # enable augmentation for training samples
    val_ds          = Subset(full_ds, val_idx)

    # Weighted sampler for training
    train_labels  = [labels[i] for i in tr_idx]
    sampler       = make_weighted_sampler(train_labels)

    train_loader  = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=0, pin_memory=(DEVICE=='cuda')
    )
    val_loader    = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    # Model, optimizer, scheduler
    model     = HFANetwork().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = LR,
        steps_per_epoch = len(train_loader),
        epochs          = EPOCHS,
        pct_start       = 0.2
    )
    criterion = HFALoss(au_weight=0.0)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train clips: {len(tr_idx)}  |  Val clips: {len(val_idx)}")
    print("-" * 65)

    best_val_acc = 0.0
    patience_ctr = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for clips, labels_batch in train_loader:
            clips        = clips.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()
            au_scores, emotion_out, _ = model(clips)
            loss = criterion(au_scores, emotion_out, labels_batch)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(clips)
            preds       = emotion_out.argmax(dim=1)
            correct    += (preds == labels_batch).sum().item()
            total      += len(clips)

        train_loss /= total
        train_acc   = correct / total

        # ── Validate ───────────────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss':   round(val_loss,   4),
            'val_acc':    round(val_acc,     4)
        })

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        # ── Save best ──────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Best model saved  (val_acc={best_val_acc:.3f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val_acc: {best_val_acc:.3f}")
    print(f"Weights saved to:  {SAVE_PATH}")
    return history


if __name__ == '__main__':
    train()
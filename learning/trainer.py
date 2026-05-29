"""
Days 16-18: Online FusionMLP Trainer
======================================
Trains the FusionMLP on your collected real-face session data.
Run this after collecting at least one labelled session.

Usage:
  python -m mmbi.learning.trainer

What it does:
  1. Loads all .npz session files from mmbi/learning/data/sessions/
  2. Trains FusionMLP (20→64→32→2) with MSE loss on eng + stress targets
  3. Saves weights to mmbi/fusion/fusion_mlp.pt
  4. Next engine launch automatically loads these weights (rule-based OFF)

Save to:  mmbi/learning/trainer.py
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

from mmbi.fusion.fusion_layer import FusionMLP, FUSION_MODEL_PATH
from mmbi.learning.collector  import load_all_sessions

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 80
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 12
VAL_SPLIT    = 0.15
DEVICE       = 'cpu'    # FusionMLP is tiny — CPU is fast enough


class FusionLoss(nn.Module):
    """
    MSE on engagement + MSE on stress, weighted equally.
    Both targets are in [0,1] so losses are comparable.
    """
    def forward(self, pred: torch.Tensor,
                y_eng: torch.Tensor,
                y_str: torch.Tensor) -> torch.Tensor:
        eng_loss = nn.functional.mse_loss(pred[:, 0], y_eng)
        str_loss = nn.functional.mse_loss(pred[:, 1], y_str)
        return eng_loss + str_loss


def train():
    print("=" * 55)
    print("FusionMLP Trainer — Days 16-18")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    result = load_all_sessions()
    if result is None:
        print("\nNo session data found.")
        print("Run the engine, label some frames (keys 1-5), press W to save.")
        print("Then run this trainer again.")
        return

    X, y_eng, y_str = result
    print(f"\nDataset: {len(X)} frames  |  features: {X.shape[1]}")
    print(f"Engagement range: {y_eng.min():.2f} – {y_eng.max():.2f}")
    print(f"Stress range:     {y_str.min():.2f} – {y_str.max():.2f}")

    # ── Build tensors ─────────────────────────────────────────────────────────
    X_t     = torch.tensor(X,     dtype=torch.float32)
    y_eng_t = torch.tensor(y_eng, dtype=torch.float32)
    y_str_t = torch.tensor(y_str, dtype=torch.float32)

    dataset  = TensorDataset(X_t, y_eng_t, y_str_t)
    n_val    = max(1, int(len(dataset) * VAL_SPLIT))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = FusionMLP(input_dim=X.shape[1]).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = FusionLoss()

    print(f"\nTraining FusionMLP: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
    print(f"Train: {n_train} frames  |  Val: {n_val} frames")
    print("-" * 55)

    best_val_loss = float('inf')
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yeng, ystr in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yeng, ystr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yeng, ystr in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yeng, ystr).item() * len(xb)
        val_loss /= n_val
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}")

        # ── Save best ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(FUSION_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), FUSION_MODEL_PATH)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val loss: {best_val_loss:.5f}")
    print(f"FusionMLP saved → {FUSION_MODEL_PATH}")
    print("\nRestart the engine — it will now use your trained FusionMLP.")


if __name__ == '__main__':
    train()
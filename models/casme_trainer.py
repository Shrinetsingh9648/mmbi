"""
CASME II Training Pipeline for MicroLSTM.

CASME II dataset structure (download from: http://fu.psych.ac.cn/CASME/casme2.php)

After downloading and extracting:
  casme2/
  ├── Cropped/
  │   ├── sub01/
  │   │   ├── EP02_01f/
  │   │   │   ├── reg_img001.jpg
  │   │   │   └── ...
  │   │   └── ...
  │   └── ...
  └── CASME2-coding-20140508.xlsx

Set DATASET_ROOT below to your extraction path.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

from mmbi.models.lstm_temporal import MicroLSTM, MICRO_CLASSES, SEQ_LEN, FEATURE_DIM, MODEL_PATH
from mmbi.face.landmarks import LandmarkExtractor
from mmbi.face.action_units import AUCalculator
from mmbi.gaze.gaze_estimator import GazeEstimator
from mmbi.head.head_pose import HeadPoseEstimator
from mmbi.gaze.blink_detector import BlinkDetector
from mmbi.temporal.buffer import FEATURE_KEYS

# ─── CONFIGURE THESE PATHS ──────────────────────────────────────────────────
DATASET_ROOT  = r"C:\datasets\casme2"          # change to your path
CROPPED_DIR   = os.path.join(DATASET_ROOT, "Cropped")
LABEL_CSV     = os.path.join(DATASET_ROOT, "CASME2-coding-20140508.xlsx")
SAVE_PATH     = MODEL_PATH
# ─────────────────────────────────────────────────────────────────────────────

# CASME II emotion label mapping → our 5 classes
CASME_LABEL_MAP = {
    'happiness':   'happiness',
    'disgust':     'disgust',
    'repression':  'repression',
    'surprise':    'surprise',
    'fear':        'others',
    'sadness':     'others',
    'contempt':    'others',
    'others':      'others',
    'tense':       'others',
}

LABEL_TO_IDX = {v: k for k, v in MICRO_CLASSES.items()}
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training hyper-parameters
BATCH_SIZE    = 32
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10     # early stopping patience


# ── Step 1: Extract feature sequences from video frames ─────────────────────

def extract_sequence_from_folder(frame_folder: str,
                                  lm_extractor: LandmarkExtractor,
                                  au_calc: AUCalculator,
                                  gaze_est: GazeEstimator,
                                  head_est: HeadPoseEstimator) -> np.ndarray | None:
    """
    Reads all frames in a folder, extracts the 17-dim feature vector
    per frame, then resamples/pads to SEQ_LEN (60 frames).
    Returns numpy array of shape (SEQ_LEN, FEATURE_DIM) or None on failure.
    """
    frame_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.png', '.bmp'))
    ])

    if len(frame_files) < 3:
        return None

    au_calc.baseline = None   # reset per-sequence calibration

    features = []

    for fname in frame_files:
        path  = os.path.join(frame_folder, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm  = lm_extractor.extract(rgb)
        if lm is None:
            # Use zero vector for missing face frames
            features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue

        au_dict    = au_calc.compute(lm)
        gaze_dict  = gaze_est.compute(lm)
        head_dict  = head_est.compute(lm)

        # Build feature vector matching FEATURE_KEYS order
        all_data = {}
        all_data.update(au_dict)
        all_data.update(gaze_dict)
        all_data.update(head_dict)
        all_data['ear'] = 0.25   # static placeholder (no video EAR in CASME II)

        vec = np.array(
            [float(all_data.get(k, 0.0)) for k in FEATURE_KEYS],
            dtype=np.float32
        )
        features.append(vec)

    if len(features) < 3:
        return None

    seq = np.array(features, dtype=np.float32)   # (T, 17)

    # Resample to exactly SEQ_LEN frames using linear interpolation
    if len(seq) != SEQ_LEN:
        old_idx = np.linspace(0, len(seq) - 1, len(seq))
        new_idx = np.linspace(0, len(seq) - 1, SEQ_LEN)
        resampled = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for dim in range(FEATURE_DIM):
            resampled[:, dim] = np.interp(new_idx, old_idx, seq[:, dim])
        seq = resampled

    return seq   # (60, 17)


# ── Step 2: PyTorch Dataset ──────────────────────────────────────────────────

class CASMEDataset(Dataset):
    """
    Holds pre-extracted (sequence, label) pairs.
    Applies per-sequence normalization at __getitem__ time.
    """

    def __init__(self, sequences: list, labels: list, augment: bool = False):
        self.sequences = sequences   # list of (60, 17) arrays
        self.labels    = labels      # list of int class indices
        self.augment   = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = self.sequences[idx].copy()
        label = self.labels[idx]

        # Per-sequence normalization
        seq_min = seq.min(axis=0, keepdims=True)
        seq_max = seq.max(axis=0, keepdims=True)
        seq     = (seq - seq_min) / (seq_max - seq_min + 1e-6)

        # Data augmentation (training only)
        if self.augment:
            # Gaussian noise
            seq += np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            # Random time shift up to 5 frames
            shift = np.random.randint(-5, 6)
            if shift > 0:
                seq = np.concatenate([seq[shift:], seq[-shift:]], axis=0)
            elif shift < 0:
                seq = np.concatenate([seq[:shift], seq[:-shift]], axis=0)
            # Random feature dropout (zero out 1 random feature)
            drop_feat = np.random.randint(0, FEATURE_DIM)
            seq[:, drop_feat] = 0.0

        tensor = torch.tensor(seq, dtype=torch.float32)
        return tensor, torch.tensor(label, dtype=torch.long)


# ── Step 3: Build full dataset from CASME II folder ─────────────────────────

def build_dataset(verbose: bool = True) -> tuple[list, list]:
    """
    Reads the CASME II label spreadsheet and extracts feature sequences
    for every sample. Returns (sequences, labels).
    """
    print("Reading CASME II labels...")
    df = pd.read_excel(LABEL_CSV)
    # Column names in CASME2-coding-20140508.xlsx:
    # Subject, Filename, OnsetFrame, ApexFrame, OffsetFrame,
    # Action, Emotion, Notes
    df.columns = [c.strip() for c in df.columns]

    lm_extractor = LandmarkExtractor()
    au_calc      = AUCalculator()
    gaze_est     = GazeEstimator()
    head_est     = HeadPoseEstimator(frame_w=640, frame_h=480)

    sequences, labels = [], []
    skipped = 0

    for _, row in df.iterrows():
        subject  = str(row['Subject']).zfill(2)
        filename = str(row['Filename']).strip()
        emotion  = str(row.get('Emotion', 'others')).strip().lower()

        # Map CASME II label to our 5-class scheme
        mapped = CASME_LABEL_MAP.get(emotion, 'others')
        label  = LABEL_TO_IDX[mapped]

        folder = os.path.join(CROPPED_DIR, f"sub{subject}", filename)
        if not os.path.isdir(folder):
            skipped += 1
            continue

        seq = extract_sequence_from_folder(
            folder, lm_extractor, au_calc, gaze_est, head_est
        )
        if seq is None:
            skipped += 1
            continue

        sequences.append(seq)
        labels.append(label)

        if verbose:
            print(f"  [{len(sequences):03d}] sub{subject}/{filename}"
                  f"  → {mapped}  ({emotion})")

    print(f"\nTotal extracted: {len(sequences)}")
    print(f"Skipped (missing/no face): {skipped}")

    # Class distribution
    from collections import Counter
    dist = Counter([MICRO_CLASSES[l] for l in labels])
    print("Class distribution:", dict(dist))

    return sequences, labels


# ── Step 4: Training loop ────────────────────────────────────────────────────

def train(sequences: list, labels: list):
    """Full training loop with early stopping and LR scheduling."""
    print(f"\nTraining on device: {DEVICE}")

    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels,
        test_size=0.20, stratify=labels, random_state=42
    )

    train_ds = CASMEDataset(X_train, y_train, augment=True)
    val_ds   = CASMEDataset(X_val,   y_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = MicroLSTM().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Weighted loss for class imbalance (CASME II is heavily skewed)
    from collections import Counter
    counts = Counter(y_train)
    total  = sum(counts.values())
    weights = torch.tensor(
        [total / (len(counts) * counts.get(i, 1)) for i in range(len(MICRO_CLASSES))],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    patience_ctr = 0
    history      = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            emotion_logits, phase_logits, _ = model(xb)
            loss = criterion(emotion_logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss, correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                emotion_logits, _, _ = model(xb)
                loss = criterion(emotion_logits, yb)
                val_loss += loss.item() * len(xb)
                preds   = emotion_logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total_val += len(yb)
        val_loss /= len(val_ds)
        val_acc   = correct / total_val

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}")

        # ── Save best model ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Saved best model (val_acc={best_val_acc:.3f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {SAVE_PATH}")
    return history


# ── Step 5: Entry point ──────────────────────────────────────────────────────

if __name__ == '__main__':
    sequences, labels = build_dataset(verbose=True)
    if len(sequences) == 0:
        print("\nERROR: No sequences extracted.")
        print("Check that DATASET_ROOT and CROPPED_DIR paths are correct.")
    else:
        history = train(sequences, labels)
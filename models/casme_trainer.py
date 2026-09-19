"""
CASME II Training Pipeline for MicroLSTM — multi-task version.

CASME II dataset structure (download from: http://fu.psych.ac.cn/CASME/casme2.php)

After downloading and extracting:
  casme2/
  ├── Cropped/
  │   ├── sub01/
  │   │   ├── EP02_01f/
  │   │   │   ├── reg_img051.jpg      <- filename number == real video
  │   │   │   └── ...                    frame number, matches the
  │   │   └── ...                        OnsetFrame/ApexFrame/OffsetFrame
  │   └── ...                            columns in the coding sheet
  └── CASME2-coding-20140508.xlsx

Set DATASET_ROOT below to your extraction path.

WHAT CHANGED vs. the original single-task version (see technical audit,
§4/§7):
  1. Per-frame PHASE labels (neutral/onset/apex/offset) are now built
     from the OnsetFrame/ApexFrame/OffsetFrame columns and actually used
     in the loss (previously computed by the model and silently discarded).
  2. Per-frame binary SPOTTING labels (is this frame part of a ME) are
     built the same way and trained with BCEWithLogitsLoss.
  3. A per-frame INTENSITY proxy target (triangular 0->1->0 across
     onset->apex->offset) is built and trained with MSELoss.
     *** This is a PROXY target, NOT a clinically/FACS-validated
     intensity measurement. *** It assumes intensity ramps up linearly
     to the apex and back down, which is a simplification of real
     muscle activation dynamics. Treat any resulting "intensity" number
     as a rough relative indicator, not a calibrated physiological
     quantity, unless/until you have real graded AU-intensity ground
     truth (e.g. from SAMM's more complete AU coding) to train against.
  4. Multi-task loss with configurable weights (Phase 5 of the
     implementation spec).
  5. Subject-level (group) train/val split by default, replacing the
     original random frame/clip-level split, which leaked subject
     identity between train and val. A full Leave-One-Subject-Out (LOSO)
     driver is also provided (`run_loso`) for a properly rigorous
     evaluation, matching the standard CASME II/SAMM protocol.
  6. Feature vector is now 22-dim (was 17): the 5 new dims are regional
     optical-flow magnitude features from the FIXED
     face/microexpression.py (previously computed and discarded).

WHAT THIS FILE DOES NOT DO (documented limitations, not silently
skipped):
  - It does not train on continuous, multi-minute video with long
    neutral stretches — CASME II ships pre-cropped clips that mostly
    already bracket the onset..offset interval, so "neutral" per-frame
    labels mostly only come from any padding frames included in a given
    clip folder. For true continuous-video spotting (long neutral
    stretches interrupted by rare ME events), see the audit's dataset
    recommendation (SAMM/CASME III spotting protocol) — that requires a
    different data loader than this one, which is intentionally NOT
    fabricated here. See BLOCKED items in the final implementation report.
  - It does not verify that DATASET_ROOT actually contains real CASME II
    data rather than the output of generate_synthetic_data.py. Check
    this yourself before trusting any accuracy number this script prints.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

from mmbi.models.lstm_temporal import MicroLSTM, MICRO_CLASSES, PHASE_CLASSES, \
    SEQ_LEN, FEATURE_DIM, MODEL_PATH
from mmbi.face.landmarks import LandmarkExtractor
from mmbi.face.action_units import AUCalculator
from mmbi.face.microexpression import MicroExpressionDetector
from mmbi.gaze.gaze_estimator import GazeEstimator
from mmbi.head.head_pose import HeadPoseEstimator
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
PHASE_TO_IDX = {v: k for k, v in PHASE_CLASSES.items()}  # neutral=0 onset=1 apex=2 offset=3
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training hyper-parameters
BATCH_SIZE    = 32
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10     # early stopping patience

# ── Multi-task loss weights (Phase 5) ───────────────────────────────────────
# These are INITIAL, hand-picked starting values, not the result of any
# tuning sweep. Adjust based on your own validation curves; do not treat
# them as "optimal". `EMOTION_WEIGHT`/`PHASE_WEIGHT` are 1.0 because those
# are the two multi-class classification objectives that matter equally;
# `INTENSITY_WEIGHT` is lower because it's a noisier proxy target
# (triangular approximation, see module docstring) and we don't want its
# gradient to dominate the shared LSTM backbone early in training.
EMOTION_WEIGHT   = 1.0
PHASE_WEIGHT     = 1.0
SPOTTING_WEIGHT  = 1.0
INTENSITY_WEIGHT = 0.5

# Apex tolerance: frames within this many frame-numbers of the coded
# ApexFrame are also labeled "apex" (real annotators code a single apex
# frame, but the true peak often spans a couple of frames either side).
APEX_TOLERANCE_FRAMES = 1


# ── Step 1: Per-frame label construction from Onset/Apex/Offset ────────────

def _frame_number_from_filename(fname: str) -> int:
    """CASME II cropped filenames look like 'reg_img051.jpg' where 051 is
    the REAL frame number in the original video, matching the coding
    sheet's OnsetFrame/ApexFrame/OffsetFrame columns directly. We parse
    the trailing digits rather than assume positional (0,1,2,...)
    indexing, because clip folders do not necessarily start at frame 0."""
    match = re.search(r'(\d+)(?=\.\w+$)', fname)
    if not match:
        raise ValueError(f"Could not parse a frame number out of filename: {fname}")
    return int(match.group(1))


def build_frame_phase_labels(frame_numbers: list, onset: int, apex: int,
                              offset: int) -> tuple:
    """
    For each raw frame number in `frame_numbers`, returns:
      phase_labels : list[int]   0=neutral 1=onset 2=apex 3=offset
      is_micro     : list[int]   1 if onset <= frame <= offset else 0
      intensity    : list[float] triangular proxy target, 0 at onset/offset,
                                  1 at apex, linear ramp between (see module
                                  docstring: THIS IS A PROXY, NOT A
                                  CLINICALLY VALIDATED INTENSITY MEASURE)

    Frame labeling rule (matches the implementation spec exactly):
      before onset        -> NEUTRAL
      onset -> apex        -> ONSET
      apex (± tolerance)   -> APEX
      apex -> offset        -> OFFSET
      after offset          -> NEUTRAL
    """
    phase_labels, is_micro, intensity = [], [], []

    # Guard against malformed/missing coding-sheet values.
    if offset is None or offset <= onset:
        offset = max(frame_numbers) if frame_numbers else onset
    if apex is None or not (onset <= apex <= offset):
        apex = onset + (offset - onset) // 2

    for f in frame_numbers:
        if f < onset:
            phase_labels.append(PHASE_TO_IDX['neutral'])
            is_micro.append(0)
            intensity.append(0.0)
        elif abs(f - apex) <= APEX_TOLERANCE_FRAMES:
            phase_labels.append(PHASE_TO_IDX['apex'])
            is_micro.append(1)
            intensity.append(1.0)
        elif onset <= f < apex:
            phase_labels.append(PHASE_TO_IDX['onset'])
            is_micro.append(1)
            t = (f - onset) / max(apex - onset, 1)
            intensity.append(float(np.clip(t, 0.0, 1.0)))
        elif apex < f <= offset:
            phase_labels.append(PHASE_TO_IDX['offset'])
            is_micro.append(1)
            t = (offset - f) / max(offset - apex, 1)
            intensity.append(float(np.clip(t, 0.0, 1.0)))
        else:  # f > offset
            phase_labels.append(PHASE_TO_IDX['neutral'])
            is_micro.append(0)
            intensity.append(0.0)

    return phase_labels, is_micro, intensity


def _resample_categorical(labels: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbour resample for CATEGORICAL labels (phase, is_micro).
    Linear interpolation must never be used on integer class codes -- it
    can produce values like 1.5 that don't correspond to any real class."""
    src_len = len(labels)
    if src_len == target_len:
        return labels.copy()
    src_idx = np.linspace(0, src_len - 1, target_len)
    nearest = np.round(src_idx).astype(int)
    nearest = np.clip(nearest, 0, src_len - 1)
    return labels[nearest]


def _resample_continuous(values: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-interpolation resample for CONTINUOUS targets/features."""
    src_len = len(values)
    if src_len == target_len:
        return values.copy()
    old_idx = np.linspace(0, src_len - 1, src_len)
    new_idx = np.linspace(0, src_len - 1, target_len)
    return np.interp(new_idx, old_idx, values).astype(np.float32)


# ── Step 2: Extract feature + label sequences from a clip folder ───────────

def extract_sequence_from_folder(
    frame_folder: str,
    onset: int, apex: int, offset: int,
    lm_extractor: LandmarkExtractor,
    au_calc: AUCalculator,
    gaze_est: GazeEstimator,
    head_est: HeadPoseEstimator,
):
    """
    Reads all frames in a folder, extracts the FEATURE_DIM-dim feature
    vector per frame (AU proxies + gaze + head pose + regional optical
    flow), builds per-frame phase/spotting/intensity labels from the
    onset/apex/offset frame numbers, then resamples everything to
    SEQ_LEN frames (continuous features/intensity via linear
    interpolation, categorical phase/spotting via nearest-neighbour).

    Returns a dict with keys:
      'features'   : (SEQ_LEN, FEATURE_DIM) float32
      'phase'      : (SEQ_LEN,) int64
      'is_micro'   : (SEQ_LEN,) int64
      'intensity'  : (SEQ_LEN,) float32
    or None on failure (too few readable frames).
    """
    frame_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.png', '.bmp'))
    ])
    if len(frame_files) < 3:
        return None

    au_calc.baseline = None   # reset per-sequence calibration
    micro_det = MicroExpressionDetector()  # fresh optical-flow state per clip

    features, frame_numbers = [], []

    prev_frame_number = None
    for fname in frame_files:
        path  = os.path.join(frame_folder, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        try:
            frame_num = _frame_number_from_filename(fname)
        except ValueError:
            continue

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lm   = lm_extractor.extract(rgb)

        if lm is None:
            features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            frame_numbers.append(frame_num)
            prev_frame_number = frame_num
            continue

        au_dict    = au_calc.compute(lm)
        gaze_dict  = gaze_est.compute(lm)
        head_dict  = head_est.compute(lm)
        flow_res   = micro_det.update(gray, au_dict, landmarks=lm)
        flow_dict  = (flow_res or {}).get('flow_regions', {})

        all_data = {}
        all_data.update(au_dict)
        all_data.update(gaze_dict)
        all_data.update(head_dict)
        all_data.update(flow_dict)
        all_data['ear'] = 0.25   # static placeholder (no video EAR in CASME II)

        vec = np.array(
            [float(all_data.get(k, 0.0)) for k in FEATURE_KEYS],
            dtype=np.float32
        )
        features.append(vec)
        frame_numbers.append(frame_num)
        prev_frame_number = frame_num

    if len(features) < 3:
        return None

    seq = np.array(features, dtype=np.float32)          # (T, FEATURE_DIM)
    phase_raw, is_micro_raw, intensity_raw = build_frame_phase_labels(
        frame_numbers, onset, apex, offset
    )
    phase_raw    = np.array(phase_raw,    dtype=np.int64)
    is_micro_raw = np.array(is_micro_raw, dtype=np.int64)
    intensity_raw= np.array(intensity_raw, dtype=np.float32)

    if len(seq) != SEQ_LEN:
        resampled = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for dim in range(FEATURE_DIM):
            resampled[:, dim] = _resample_continuous(seq[:, dim], SEQ_LEN)
        seq = resampled

        phase_raw     = _resample_categorical(phase_raw, SEQ_LEN)
        is_micro_raw  = _resample_categorical(is_micro_raw, SEQ_LEN)
        intensity_raw = _resample_continuous(intensity_raw, SEQ_LEN)

    return {
        'features':  seq,
        'phase':     phase_raw,
        'is_micro':  is_micro_raw,
        'intensity': intensity_raw,
    }


# ── Step 3: PyTorch Dataset ──────────────────────────────────────────────────

class CASMEDataset(Dataset):
    """
    Holds pre-extracted (sequence, per-frame-label-arrays, emotion label,
    subject) tuples. Supervises phase/spotting/intensity using the LABEL
    OF THE LAST FRAME in each (resampled) window — this matches how the
    trained model is actually queried at real-time inference: a sliding
    window ending "now" produces one prediction "for now" (see engine.py
    and models/lstm_temporal.py's MicroLSTM docstring for the same design
    decision stated on the inference side).
    """

    def __init__(self, samples: list, augment: bool = False):
        # samples: list of dicts with 'features','phase','is_micro',
        # 'intensity','emotion_label','subject'
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq       = s['features'].copy()
        phase     = s['phase'].copy()
        is_micro  = s['is_micro'].copy()
        intensity = s['intensity'].copy()
        emotion_label = s['emotion_label']

        # Per-sequence feature normalization
        seq_min = seq.min(axis=0, keepdims=True)
        seq_max = seq.max(axis=0, keepdims=True)
        seq     = (seq - seq_min) / (seq_max - seq_min + 1e-6)

        if self.augment:
            # Gaussian noise (features only -- labels are ground truth,
            # must not be perturbed)
            seq += np.random.normal(0, 0.01, seq.shape).astype(np.float32)

            # Random time shift up to 5 frames -- shift features AND
            # labels together so they stay aligned; this changes which
            # frame is "last" so the phase/is_micro/intensity target for
            # this sample legitimately changes too, which is correct.
            shift = np.random.randint(-5, 6)
            if shift > 0:
                seq       = np.concatenate([seq[shift:], seq[-shift:]], axis=0)
                phase     = np.concatenate([phase[shift:], phase[-shift:]], axis=0)
                is_micro  = np.concatenate([is_micro[shift:], is_micro[-shift:]], axis=0)
                intensity = np.concatenate([intensity[shift:], intensity[-shift:]], axis=0)
            elif shift < 0:
                seq       = np.concatenate([seq[:shift], seq[:-shift]], axis=0)
                phase     = np.concatenate([phase[:shift], phase[:-shift]], axis=0)
                is_micro  = np.concatenate([is_micro[:shift], is_micro[:-shift]], axis=0)
                intensity = np.concatenate([intensity[:shift], intensity[:-shift]], axis=0)

            # Random feature dropout (zero out 1 random feature dim)
            drop_feat = np.random.randint(0, FEATURE_DIM)
            seq[:, drop_feat] = 0.0

        phase_target     = int(phase[-1])
        is_micro_target  = float(is_micro[-1])
        intensity_target = float(intensity[-1])

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(emotion_label, dtype=torch.long),
            torch.tensor(phase_target, dtype=torch.long),
            torch.tensor(is_micro_target, dtype=torch.float32),
            torch.tensor(intensity_target, dtype=torch.float32),
        )


# ── Step 4: Build full dataset from CASME II folder ─────────────────────────

def build_dataset(verbose: bool = True) -> list:
    """
    Reads the CASME II label spreadsheet and extracts feature+label
    sequences for every sample. Returns a list of sample dicts (see
    CASMEDataset docstring) including 'subject' for grouped splitting.
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

    samples = []
    skipped = 0

    for _, row in df.iterrows():
        subject  = str(row['Subject']).zfill(2)
        filename = str(row['Filename']).strip()
        emotion  = str(row.get('Emotion', 'others')).strip().lower()

        mapped = CASME_LABEL_MAP.get(emotion, 'others')
        emotion_label = LABEL_TO_IDX[mapped]

        def _to_int_or_none(v):
            try:
                iv = int(v)
                return iv if iv > 0 else None
            except (TypeError, ValueError):
                return None

        onset  = _to_int_or_none(row.get('OnsetFrame'))
        apex   = _to_int_or_none(row.get('ApexFrame'))
        offset = _to_int_or_none(row.get('OffsetFrame'))

        if onset is None:
            skipped += 1
            if verbose:
                print(f"  SKIP sub{subject}/{filename}: missing/invalid OnsetFrame")
            continue

        folder = os.path.join(CROPPED_DIR, f"sub{subject}", filename)
        if not os.path.isdir(folder):
            skipped += 1
            continue

        result = extract_sequence_from_folder(
            folder, onset, apex, offset,
            lm_extractor, au_calc, gaze_est, head_est
        )
        if result is None:
            skipped += 1
            continue

        result['emotion_label'] = emotion_label
        result['subject'] = subject
        samples.append(result)

        if verbose:
            print(f"  [{len(samples):03d}] sub{subject}/{filename}"
                  f"  emotion={mapped}  onset={onset} apex={apex} offset={offset}")

    print(f"\nTotal extracted: {len(samples)}")
    print(f"Skipped (missing/no face/bad labels): {skipped}")

    dist = Counter([MICRO_CLASSES[s['emotion_label']] for s in samples])
    print("Class distribution:", dict(dist))
    n_subjects = len(set(s['subject'] for s in samples))
    print(f"Unique subjects: {n_subjects}")

    return samples


# ── Step 5: Multi-task loss ──────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Total Loss = EMOTION_WEIGHT   * emotion_loss    (CrossEntropyLoss)
               + PHASE_WEIGHT     * phase_loss       (CrossEntropyLoss)
               + SPOTTING_WEIGHT  * spotting_loss     (BCEWithLogitsLoss)
               + INTENSITY_WEIGHT * intensity_loss     (MSELoss)

    Weights are configuration, not claimed-optimal constants -- see the
    module-level EMOTION_WEIGHT/PHASE_WEIGHT/SPOTTING_WEIGHT/
    INTENSITY_WEIGHT constants and their comments.
    """

    def __init__(self, emotion_class_weights: torch.Tensor = None,
                 emotion_weight: float = EMOTION_WEIGHT,
                 phase_weight: float = PHASE_WEIGHT,
                 spotting_weight: float = SPOTTING_WEIGHT,
                 intensity_weight: float = INTENSITY_WEIGHT,
                 spotting_pos_weight: torch.Tensor = None):
        super().__init__()
        self.emotion_weight   = emotion_weight
        self.phase_weight     = phase_weight
        self.spotting_weight  = spotting_weight
        self.intensity_weight = intensity_weight

        self.emotion_loss_fn   = nn.CrossEntropyLoss(weight=emotion_class_weights)
        self.phase_loss_fn     = nn.CrossEntropyLoss()
        self.spotting_loss_fn  = nn.BCEWithLogitsLoss(pos_weight=spotting_pos_weight)
        self.intensity_loss_fn = nn.MSELoss()

    def forward(self, emotion_logits, phase_logits, spotting_logits, intensity_pred,
                emotion_y, phase_y, is_micro_y, intensity_y):
        emotion_loss   = self.emotion_loss_fn(emotion_logits, emotion_y)
        phase_loss     = self.phase_loss_fn(phase_logits, phase_y)
        spotting_loss  = self.spotting_loss_fn(spotting_logits.squeeze(-1), is_micro_y)
        intensity_loss = self.intensity_loss_fn(intensity_pred.squeeze(-1), intensity_y)

        total = (self.emotion_weight   * emotion_loss
               + self.phase_weight     * phase_loss
               + self.spotting_weight  * spotting_loss
               + self.intensity_weight * intensity_loss)

        return total, {
            'emotion_loss':   emotion_loss.item(),
            'phase_loss':     phase_loss.item(),
            'spotting_loss':  spotting_loss.item(),
            'intensity_loss': intensity_loss.item(),
        }


# ── Step 6: Subject-level split (data-leakage prevention) ──────────────────

def subject_group_split(samples: list, test_size: float = 0.20, random_state: int = 42):
    """
    Splits `samples` into train/val indices such that NO subject appears
    in both splits (prevents the model from memorizing subject-specific
    facial geometry/lighting and inflating validation accuracy).

    Uses sklearn's GroupShuffleSplit with `subject` as the group key --
    NOT a random per-clip split (which was the original bug).
    """
    subjects = [s['subject'] for s in samples]
    labels   = [s['emotion_label'] for s in samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(samples, labels, groups=subjects))

    train_subjects = set(subjects[i] for i in train_idx)
    val_subjects   = set(subjects[i] for i in val_idx)
    overlap = train_subjects & val_subjects
    assert not overlap, f"Subject leakage detected: {overlap}"

    print(f"Subject-level split: {len(train_subjects)} train subjects, "
          f"{len(val_subjects)} val subjects, 0 overlapping subjects.")
    return list(train_idx), list(val_idx)


def leave_one_subject_out_splits(samples: list):
    """
    Generator yielding (train_idx, test_idx, held_out_subject) for a full
    Leave-One-Subject-Out cross-validation, the standard CASME II/SAMM
    evaluation protocol in the ME literature. This is the RIGOROUS option
    -- use `run_loso()` to actually execute it. It costs (n_subjects x
    normal training time), so it is opt-in via a CLI flag, not the
    default `train()` path (which uses the cheaper single grouped split
    above for fast iteration).
    """
    subjects = sorted(set(s['subject'] for s in samples))
    for held_out in subjects:
        train_idx = [i for i, s in enumerate(samples) if s['subject'] != held_out]
        test_idx  = [i for i, s in enumerate(samples) if s['subject'] == held_out]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield train_idx, test_idx, held_out


# ── Step 7: Training loop ────────────────────────────────────────────────────

def _make_loaders(samples, train_idx, val_idx):
    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]

    train_ds = CASMEDataset(train_samples, augment=True)
    val_ds   = CASMEDataset(val_samples,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_samples, val_samples


def _class_weights(train_samples):
    counts = Counter(s['emotion_label'] for s in train_samples)
    total  = sum(counts.values())
    weights = torch.tensor(
        [total / (len(counts) * counts.get(i, 1)) for i in range(len(MICRO_CLASSES))],
        dtype=torch.float32
    ).to(DEVICE)
    return weights


def _spotting_pos_weight(train_samples):
    """BCEWithLogitsLoss pos_weight to counter spotting class imbalance
    (most windows in a pre-cropped ME clip ARE positive -- see module
    docstring on continuous-video limitation)."""
    pos = sum(1 for s in train_samples if s['is_micro'][-1] == 1)
    neg = sum(1 for s in train_samples if s['is_micro'][-1] == 0)
    if pos == 0 or neg == 0:
        return None
    return torch.tensor(neg / pos, dtype=torch.float32).to(DEVICE)


def train(samples: list, save_path: str = SAVE_PATH, epochs: int = EPOCHS,
          verbose: bool = True):
    """Full multi-task training loop with early stopping and LR scheduling.
    Uses a single subject-disjoint train/val split (see subject_group_split).
    For a rigorous LOSO evaluation instead, use run_loso()."""
    print(f"\nTraining on device: {DEVICE}")

    train_idx, val_idx = subject_group_split(samples)
    train_loader, val_loader, train_samples, val_samples = _make_loaders(
        samples, train_idx, val_idx
    )

    model     = MicroLSTM().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion = MultiTaskLoss(
        emotion_class_weights=_class_weights(train_samples),
        spotting_pos_weight=_spotting_pos_weight(train_samples),
    )

    best_val_acc = 0.0
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_emotion_acc': [],
               'val_phase_acc': [], 'val_spotting_acc': []}

    print(f"\nStarting training: {epochs} epochs, batch={BATCH_SIZE}, lr={LR}")
    print(f"Loss weights: emotion={EMOTION_WEIGHT} phase={PHASE_WEIGHT} "
          f"spotting={SPOTTING_WEIGHT} intensity={INTENSITY_WEIGHT}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, y_emotion, y_phase, y_micro, y_intensity in train_loader:
            xb          = xb.to(DEVICE)
            y_emotion   = y_emotion.to(DEVICE)
            y_phase     = y_phase.to(DEVICE)
            y_micro     = y_micro.to(DEVICE)
            y_intensity = y_intensity.to(DEVICE)

            optimizer.zero_grad()
            emotion_logits, phase_logits, spotting_logits, intensity_pred, _ = model(xb)
            loss, parts = criterion(
                emotion_logits, phase_logits, spotting_logits, intensity_pred,
                y_emotion, y_phase, y_micro, y_intensity
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= max(len(train_loader.dataset), 1)

        val_loss, val_metrics = evaluate(model, val_loader, criterion)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_emotion_acc'].append(val_metrics['emotion_acc'])
        history['val_phase_acc'].append(val_metrics['phase_acc'])
        history['val_spotting_acc'].append(val_metrics['spotting_acc'])

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"emotion_acc={val_metrics['emotion_acc']:.3f}  "
                  f"phase_acc={val_metrics['phase_acc']:.3f}  "
                  f"spotting_acc={val_metrics['spotting_acc']:.3f}  "
                  f"intensity_mae={val_metrics['intensity_mae']:.3f}")

        # Model-selection criterion: emotion accuracy, matching the
        # original script's behaviour (kept unchanged rather than
        # silently redefined -- change this if you'd rather select on a
        # combined metric).
        if val_metrics['emotion_acc'] > best_val_acc:
            best_val_acc = val_metrics['emotion_acc']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val_emotion_acc={best_val_acc:.3f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val emotion accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {save_path}")
    return history


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct_emotion = 0
    correct_phase = 0
    correct_spotting = 0
    intensity_abs_err = 0.0

    for xb, y_emotion, y_phase, y_micro, y_intensity in loader:
        xb          = xb.to(DEVICE)
        y_emotion   = y_emotion.to(DEVICE)
        y_phase     = y_phase.to(DEVICE)
        y_micro     = y_micro.to(DEVICE)
        y_intensity = y_intensity.to(DEVICE)

        emotion_logits, phase_logits, spotting_logits, intensity_pred, _ = model(xb)
        loss, _ = criterion(
            emotion_logits, phase_logits, spotting_logits, intensity_pred,
            y_emotion, y_phase, y_micro, y_intensity
        )
        total_loss += loss.item() * len(xb)

        correct_emotion   += (emotion_logits.argmax(dim=1) == y_emotion).sum().item()
        correct_phase     += (phase_logits.argmax(dim=1) == y_phase).sum().item()
        spotting_pred_bin  = (torch.sigmoid(spotting_logits.squeeze(-1)) > 0.5).float()
        correct_spotting  += (spotting_pred_bin == y_micro).sum().item()
        intensity_abs_err += (intensity_pred.squeeze(-1) - y_intensity).abs().sum().item()

        total += len(xb)

    total = max(total, 1)
    return total_loss / total, {
        'emotion_acc':   correct_emotion / total,
        'phase_acc':     correct_phase / total,
        'spotting_acc':  correct_spotting / total,
        'intensity_mae': intensity_abs_err / total,
    }


def run_loso(samples: list, epochs: int = EPOCHS):
    """
    Full Leave-One-Subject-Out cross-validation (rigorous, expensive).
    Trains one model per held-out subject and reports mean +/- std of
    each metric across folds. Does NOT save a deployable checkpoint by
    itself (each fold's model is discarded after evaluation) -- run
    train() separately once you're happy with the LOSO numbers to
    produce the checkpoint you'll actually deploy.
    """
    fold_metrics = []
    for train_idx, test_idx, held_out_subject in leave_one_subject_out_splits(samples):
        print(f"\n=== LOSO fold: held-out subject {held_out_subject} "
              f"({len(test_idx)} clips) ===")
        train_loader, val_loader, train_samples, _ = _make_loaders(
            samples, train_idx, test_idx
        )
        model     = MicroLSTM().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = MultiTaskLoss(
            emotion_class_weights=_class_weights(train_samples),
            spotting_pos_weight=_spotting_pos_weight(train_samples),
        )

        for epoch in range(1, epochs + 1):
            model.train()
            for xb, y_emotion, y_phase, y_micro, y_intensity in train_loader:
                xb, y_emotion, y_phase = xb.to(DEVICE), y_emotion.to(DEVICE), y_phase.to(DEVICE)
                y_micro, y_intensity = y_micro.to(DEVICE), y_intensity.to(DEVICE)
                optimizer.zero_grad()
                out = model(xb)
                loss, _ = criterion(*out[:4], y_emotion, y_phase, y_micro, y_intensity)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        _, metrics = evaluate(model, val_loader, criterion)
        metrics['subject'] = held_out_subject
        fold_metrics.append(metrics)
        print(f"  fold result: {metrics}")

    if not fold_metrics:
        print("No LOSO folds could be run (not enough subjects/data).")
        return []

    for key in ['emotion_acc', 'phase_acc', 'spotting_acc', 'intensity_mae']:
        vals = [m[key] for m in fold_metrics]
        print(f"LOSO {key}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
              f"(n_folds={len(vals)})")

    return fold_metrics


# ── Step 8: Entry point ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train MicroLSTM on CASME II")
    parser.add_argument('--loso', action='store_true',
                         help="Run full Leave-One-Subject-Out cross-validation "
                              "instead of a single subject-grouped train/val split.")
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    samples = build_dataset(verbose=True)
    if len(samples) == 0:
        print("\nERROR: No sequences extracted.")
        print("Check that DATASET_ROOT and CROPPED_DIR paths are correct, "
              "and that they contain real CASME II data (see module "
              "docstring's warning about generate_synthetic_data.py).")
    elif args.loso:
        run_loso(samples, epochs=args.epochs)
    else:
        train(samples, epochs=args.epochs)

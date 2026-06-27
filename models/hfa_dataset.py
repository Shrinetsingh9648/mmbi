"""
HFA Dataset — builds (clip, label) pairs from CASME II / SAMM.
A clip is T=16 face patch frames extracted from a micro-expression video.
The face patch is cropped using MediaPipe landmarks and resized to 96×96.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mediapipe as mp

PATCH_SIZE = 96
T_FRAMES   = 16

# Reuse CASME II label map from casme_trainer.py
CASME_LABEL_MAP = {
    'happiness':  0,
    'disgust':    1,
    'repression': 2,
    'surprise':   3,
    'others':     4,
    'fear':       4,
    'sadness':    4,
    'contempt':   4,
    'tense':      4,
}


def crop_face_patch(frame_bgr: np.ndarray,
                    landmarks,
                    patch_size: int = PATCH_SIZE) -> np.ndarray | None:
    """
    Crops and resizes a face patch from a frame using landmark bounding box.
    Adds 20% padding around the bounding box.
    Returns (patch_size, patch_size, 3) RGB uint8 or None.
    """
    if landmarks is None:
        return None

    h, w = frame_bgr.shape[:2]
    lm   = landmarks   # (468, 3) normalized [0,1] from MediaPipe

    # Face bounding box from all landmarks
    xs = lm[:, 0]
    ys = lm[:, 1]
    x1 = int(xs.min() * w)
    x2 = int(xs.max() * w)
    y1 = int(ys.min() * h)
    y2 = int(ys.max() * h)

    # Add 20% padding
    pad_x = int((x2 - x1) * 0.20)
    pad_y = int((y2 - y1) * 0.20)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    patch = frame_bgr[y1:y2, x1:x2]
    patch = cv2.resize(patch, (patch_size, patch_size),
                       interpolation=cv2.INTER_LINEAR)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return patch


def load_frames_from_folder(folder: str,
                             onset: int,
                             offset: int) -> list:
    """
    Loads frames from onset to offset frame indices.
    Returns list of BGR numpy arrays.
    """
    all_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png', '.bmp'))
    ])
    selected = all_files[onset:offset + 1]
    frames   = []
    for fname in selected:
        img = cv2.imread(os.path.join(folder, fname))
        if img is not None:
            frames.append(img)
    return frames


def extract_clip(frames: list,
                 mp_face_mesh,
                 t_frames: int = T_FRAMES) -> np.ndarray | None:
    """
    Extracts face patches for T_FRAMES evenly sampled from the clip.
    Returns numpy array (T, 3, patch_size, patch_size) float32 in [0,1].
    """
    if len(frames) < 3:
        return None

    # Sample T_FRAMES evenly
    indices = np.linspace(0, len(frames) - 1, t_frames).astype(int)

    patches = []
    for idx in indices:
        frame = frames[idx]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mp_face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            # Use centre crop as fallback
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half   = min(cx, cy)
            patch  = frame[cy-half:cy+half, cx-half:cx+half]
            patch  = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
            patch  = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        else:
            lm_raw = results.multi_face_landmarks[0].landmark
            h, w   = frame.shape[:2]
            lm_arr = np.array([[p.x, p.y, p.z] for p in lm_raw])
            patch  = crop_face_patch(frame, lm_arr)
            if patch is None:
                continue

        # Normalize to [0,1] and convert to (3, H, W)
        patch_t = (patch.astype(np.float32) / 255.0).transpose(2, 0, 1)
        patches.append(patch_t)

    if len(patches) < t_frames:
        # Pad with last frame if not enough
        while len(patches) < t_frames:
            patches.append(patches[-1].copy())

    clip = np.stack(patches[:t_frames], axis=0)   # (T, 3, H, W)
    return clip


class CASMEHFADataset(Dataset):
    """
    Loads CASME II dataset as HFA clips.
    Each sample is (clip_tensor, label) where:
      clip_tensor : (T, 3, 96, 96) float32 in [0,1]
      label       : int 0–4
    """

    def __init__(
        self,
        dataset_root: str,
        label_xlsx:   str,
        augment:      bool = False,
        t_frames:     int  = T_FRAMES
    ):
        self.clips   = []
        self.labels  = []
        self.augment = augment

        cropped_dir = os.path.join(dataset_root, 'Cropped')
        df = pd.read_excel(label_xlsx)
        df.columns = [c.strip() for c in df.columns]

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode    = True,
            max_num_faces        = 1,
            refine_landmarks     = True,
            min_detection_confidence = 0.3
        )

        print("Building HFA dataset from CASME II...")
        skipped = 0

        for _, row in df.iterrows():
            subject  = str(row['Subject']).zfill(2)
            filename = str(row['Filename']).strip()
            emotion  = str(row.get('Emotion', 'others')).strip().lower()
            label    = CASME_LABEL_MAP.get(emotion, 4)

            # Onset / offset frame numbers
            try:
                onset  = int(row.get('OnsetFrame',  0))
                offset = int(row.get('OffsetFrame', -1))
            except (ValueError, TypeError):
                onset, offset = 0, -1

            folder = os.path.join(cropped_dir, f"sub{subject}", filename)
            if not os.path.isdir(folder):
                skipped += 1
                continue

            frames = load_frames_from_folder(folder, onset, offset)
            clip   = extract_clip(frames, mp_face_mesh, t_frames)
            if clip is None:
                skipped += 1
                continue

            self.clips.append(clip)
            self.labels.append(label)

        mp_face_mesh.close()
        print(f"Loaded {len(self.clips)} clips, skipped {skipped}")

        from collections import Counter
        dist = Counter([list(CASME_LABEL_MAP.keys())[l]
                        if l < len(CASME_LABEL_MAP) else 'others'
                        for l in self.labels])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip  = self.clips[idx].copy()    # (T, 3, 96, 96)
        label = self.labels[idx]

        if self.augment:
            clip = self._augment(clip)

        return torch.tensor(clip, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)

    def _augment(self, clip: np.ndarray) -> np.ndarray:
        """
        Augmentations that preserve the temporal structure of micro-expressions.
        Never reorder frames — onset→apex→offset order must be maintained.
        """
        T, C, H, W = clip.shape

        # Horizontal flip (left-right symmetric expressions)
        if np.random.rand() < 0.5:
            clip = clip[:, :, :, ::-1].copy()

        # Brightness jitter — uniform per clip (not per frame)
        brightness = np.random.uniform(0.8, 1.2)
        clip       = np.clip(clip * brightness, 0.0, 1.0)

        # Gaussian noise
        noise = np.random.normal(0, 0.01, clip.shape).astype(np.float32)
        clip  = np.clip(clip + noise, 0.0, 1.0)

        # Random time crop — keep contiguous T-2 frames, pad to T
        if T > 4 and np.random.rand() < 0.3:
            start = np.random.randint(0, 3)
            clip  = clip[start:start + T]
            if len(clip) < T:
                clip = np.concatenate(
                    [clip, np.repeat(clip[[-1]], T - len(clip), axis=0)],
                    axis=0
                )

        return clip.astype(np.float32)
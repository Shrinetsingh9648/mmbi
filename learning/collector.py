"""
Days 16-18: Self-Learning Data Collector
==========================================
Records labelled BehaviourState sessions to disk so the FusionMLP
can be trained on YOUR real face data instead of synthetic data.

How it works:
  - Every frame, the full fusion_vec (20 dims) is saved alongside
    a label that YOU provide via keyboard during the session.
  - Labels: E=engaged, D=disengaged, S=stressed, C=calm, N=neutral
  - Data is saved to mmbi/learning/data/sessions/  as .npz files
  - Each session file contains X (features) and y_eng, y_stress arrays

Save to:  mmbi/learning/collector.py
"""

from __future__ import annotations
import numpy as np
import os
import time
from datetime import datetime
from collections import deque

from mmbi.fusion.fusion_layer import BehaviourState

# ── Storage paths ─────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data', 'sessions')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Label keys (press during live session to label current state) ─────────────
KEY_LABELS = {
    ord('1'): ('engaged',     1.0, 0.1),   # key → (name, engagement, stress)
    ord('2'): ('neutral',     0.5, 0.2),
    ord('3'): ('disengaged',  0.1, 0.2),
    ord('4'): ('stressed',    0.4, 0.9),
    ord('5'): ('calm',        0.6, 0.05),
}

LABEL_HELP = (
    "LABEL KEYS:  1=Engaged  2=Neutral  3=Disengaged  4=Stressed  5=Calm"
)


class DataCollector:
    """
    Attach to engine. Call update() every frame, handle_key() on keypress.
    Call save_session() when done.
    """

    def __init__(self, buffer_size: int = 36000):  # 5 min at 120 FPS
        self._features:  list = []    # list of 20-dim vectors
        self._eng_labels:list = []    # float 0-1
        self._str_labels:list = []    # float 0-1
        self._names:     list = []    # string label names
        self._timestamps:list = []

        self._current_label      = ('neutral', 0.5, 0.2)
        self._current_label_name = 'neutral'
        self._collecting         = True
        self._frame_count        = 0
        self._session_start      = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, state: BehaviourState):
        """Call every frame after fusion.fuse()."""
        if not self._collecting or state is None:
            return

        vec = state.raw.get('fusion_vec', None)
        if vec is None:
            return

        _, eng_target, str_target = self._current_label

        self._features.append(vec)
        self._eng_labels.append(eng_target)
        self._str_labels.append(str_target)
        self._names.append(self._current_label_name)
        self._timestamps.append(time.time())
        self._frame_count += 1

    def handle_key(self, key: int) -> bool:
        """
        Call with cv2.waitKey() result. Returns True if key was consumed.
        """
        if key in KEY_LABELS:
            self._current_label      = KEY_LABELS[key]
            self._current_label_name = KEY_LABELS[key][0]
            print(f"[Collector] Label set → {self._current_label_name}")
            return True
        return False

    def save_session(self) -> str | None:
        """Saves current session to .npz file. Returns filepath."""
        if len(self._features) < 100:
            print("[Collector] Not enough data to save (need 100+ frames).")
            return None

        X       = np.array(self._features,   dtype=np.float32)
        y_eng   = np.array(self._eng_labels, dtype=np.float32)
        y_str   = np.array(self._str_labels, dtype=np.float32)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path      = os.path.join(DATA_DIR, f'session_{timestamp}.npz')

        np.savez_compressed(path,
                            X=X, y_eng=y_eng, y_str=y_str,
                            names=np.array(self._names))

        duration = time.time() - self._session_start
        print(f"[Collector] Saved {len(X)} frames ({duration:.0f}s) → {path}")
        return path

    def status(self) -> str:
        """One-line status string for display overlay."""
        return (f"REC {self._frame_count}f  "
                f"label={self._current_label_name}  "
                f"[1-5 to label]")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_label_name(self) -> str:
        return self._current_label_name


def load_all_sessions() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Loads and concatenates all saved session .npz files.
    Returns (X, y_eng, y_str) or None if no data found.
    """
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
    if not files:
        print("[Collector] No session files found.")
        return None

    all_X, all_eng, all_str = [], [], []
    for fname in sorted(files):
        path = os.path.join(DATA_DIR, fname)
        try:
            d = np.load(path)
            all_X.append(d['X'])
            all_eng.append(d['y_eng'])
            all_str.append(d['y_str'])
            print(f"[Collector] Loaded {len(d['X'])} frames from {fname}")
        except Exception as e:
            print(f"[Collector] Skipping {fname}: {e}")

    if not all_X:
        return None

    X     = np.concatenate(all_X,   axis=0)
    y_eng = np.concatenate(all_eng, axis=0)
    y_str = np.concatenate(all_str, axis=0)
    print(f"[Collector] Total: {len(X)} frames from {len(all_X)} sessions")
    return X, y_eng, y_str
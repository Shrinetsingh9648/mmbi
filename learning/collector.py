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
        self._session_records    = []

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

        # Granular frame-by-frame record for CSV timeline
        rec = {
            'timestamp': state.timestamp,
            'engagement_score': state.engagement_score,
            'engagement_label': state.engagement_label,
            'stress_index': state.stress_index,
            'stress_label': state.stress_label,
            'dominant_emotion': state.dominant_emotion,
            'emotion_confidence': state.emotion_confidence,
            'emotion_source': state.emotion_source,
            'macro_emotion': state.macro_emotion,
            'macro_confidence': state.macro_confidence,
            'macro_source': state.macro_source,
            'micro_emotion': state.micro_emotion,
            'micro_confidence': state.micro_confidence,
            'micro_duration_ms': state.micro_duration_ms,
            'micro_region': state.micro_region,
            'attention_zone': state.attention_zone,
            'eye_contact_score': state.eye_contact_score,
            'lean_signal': state.lean_signal,
            'blink_rate': state.blink_rate,
            'blink_stress': state.blink_stress,
            'active_aus': "|".join(state.active_aus) if state.active_aus else ""
        }
        
        # Add raw Action Units for deep analytics
        au_dict = state.raw.get('au', {}) or {}
        for au_name in ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU12', 'AU15', 'AU17', 'AU25', 'AU26']:
            rec[au_name] = round(float(au_dict.get(au_name, 0.0)), 4)
            
        self._session_records.append(rec)

    def handle_key(self, key: int) -> bool:
        """
        Call with cv2.waitKey() result. Returns True if key was consumed.
        """
        if key in KEY_LABELS:
            self._current_label      = KEY_LABELS[key]
            self._current_label_name = KEY_LABELS[key][0]
            print(f"[Collector] Label set -> {self._current_label_name}")
            return True
        return False

    def save_session(self, recorded_micro_events=None) -> tuple[str, str, str] | None:
        """
        Saves current session to:
          1. Binary training file (.npz)
          2. Detailed frame-by-frame timeline report (.csv)
          3. Granular aggregated analytical report (.json)
        Returns (npz_path, csv_path, json_path) or None.
        """
        if len(self._features) < 5:
            print("[Collector] Not enough data to save (need 5+ frames).")
            return None

        import csv
        import json

        # Paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        npz_path  = os.path.join(DATA_DIR, f'session_{timestamp}.npz')
        csv_path  = os.path.join(DATA_DIR, f'session_{timestamp}.csv')
        json_path = os.path.join(DATA_DIR, f'session_{timestamp}.json')

        # 1. Save binary NPZ
        X       = np.array(self._features,   dtype=np.float32)
        y_eng   = np.array(self._eng_labels, dtype=np.float32)
        y_str   = np.array(self._str_labels, dtype=np.float32)
        np.savez_compressed(npz_path, X=X, y_eng=y_eng, y_str=y_str, names=np.array(self._names))

        # 2. Save CSV Timeline
        if self._session_records:
            fields = list(self._session_records[0].keys())
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._session_records)

        # 3. Save JSON Analytical Assessment
        recorded_micro_events = recorded_micro_events or []
        duration = time.time() - self._session_start

        # Calculate averages
        avg_engagement = float(np.mean([r['engagement_score'] for r in self._session_records])) if self._session_records else 0.5
        avg_stress     = float(np.mean([r['stress_index'] for r in self._session_records])) if self._session_records else 0.0
        avg_gaze_ec    = float(np.mean([r['eye_contact_score'] for r in self._session_records])) if self._session_records else 0.5

        # High stress frames count
        high_stress_frames = sum(1 for r in self._session_records if r['stress_index'] > 0.70)
        high_stress_pct = float(high_stress_frames / len(self._session_records)) if self._session_records else 0.0

        # Macro emotions breakdown
        macro_counts = {}
        for r in self._session_records:
            m = r['macro_emotion']
            macro_counts[m] = macro_counts.get(m, 0) + 1
        
        total_rec = len(self._session_records) or 1
        macro_breakdown = {k: round(v / total_rec * 100, 1) for k, v in macro_counts.items()}

        # Expected hidden emotion
        suppressed_counts = {}
        for ev in recorded_micro_events:
            mic_emo = ev['micro_emotion']
            if mic_emo not in ['neutral', 'none']:
                suppressed_counts[mic_emo] = suppressed_counts.get(mic_emo, 0) + 1
                
        hidden_emotion = "NONE DETECTED"
        hidden_reason = "All involuntary micro-expressions were aligned with your baseline neutral facial state. No suppressed or hidden emotions were leaked during this session."
        
        if suppressed_counts:
            hidden_emotion = max(suppressed_counts, key=suppressed_counts.get).upper()
            hidden_reason = f"Leaks of involuntary micro-expression twitches caught by the sparse optical flow indicate a brief, transient state of [ {hidden_emotion} ] being leaked. The rapid duration suggests a suppressed or repressed emotional reaction."

        # Compile JSON report
        report_data = {
            'metadata': {
                'timestamp': timestamp,
                'session_start_time': datetime.fromtimestamp(self._session_start).strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': round(duration, 1),
                'total_frames': len(self._session_records)
            },
            'cognitive_summary': {
                'average_engagement': round(avg_engagement, 3),
                'average_stress_index': round(avg_stress, 3),
                'average_eye_contact_score': round(avg_gaze_ec, 3),
                'sustained_stress_load_pct': round(high_stress_pct * 100, 1)
            },
            'macro_facial_expressions': {
                'dominant_macro_breakdown_pct': macro_breakdown
            },
            'micro_expressions': {
                'total_events_detected': len(recorded_micro_events),
                'events_log': recorded_micro_events,
                'expected_hidden_emotion': hidden_emotion,
                'suppressed_emotion_analysis': hidden_reason
            }
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4)

        print(f"[Collector] Exported Session Logs:")
        print(f"  -> NPZ (Binary):  {npz_path}")
        print(f"  -> CSV (Timeline): {csv_path}")
        print(f"  -> JSON (Report):   {json_path}")
        return npz_path, csv_path, json_path

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
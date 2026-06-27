"""
Days 13–15: Multimodal Fusion Layer
=====================================
Combines signals from ALL subsystems into a single coherent
BehaviourState object every inference cycle.

Signal sources fused here:
  - AU values (10 dims)         from AUCalculator
  - Micro-expression spike      from MicroExpressionDetector
  - Gaze vector + eye contact   from GazeEstimator
  - Head pose (yaw/pitch/roll)  from HeadPoseEstimator
  - Blink rate + EAR            from BlinkDetector
  - LSTM prediction             from MicroLSTMInference
  - HFA prediction              from HFAInference

Fusion strategy:
  - Late fusion: each modality produces a scalar confidence score
  - Learned weights: a small MLP re-weights modalities per frame
  - Rule-based override: hard rules for high-confidence edge cases
    (e.g. eye contact = 0.0 always overrides engagement to LOW)

Output: BehaviourState dataclass with:
  - engagement_score   0.0 – 1.0
  - stress_index       0.0 – 1.0
  - dominant_emotion   str
  - emotion_confidence float
  - active_aus         list[str]
  - attention_zone     str
  - lean_signal        str
  - explanation        dict   (filled by ExplainabilityModule)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import os
import time

# ── BehaviourState ────────────────────────────────────────────────────────────

@dataclass
class BehaviourState:
    """
    Unified output of the MMBI engine for a single frame.
    All values are human-readable and serialisable to JSON.
    """
    timestamp:          float = 0.0

    # Engagement
    engagement_score:   float = 0.5
    engagement_label:   str   = "NEUTRAL"     # LOW / NEUTRAL / HIGH / PEAK

    # Stress
    stress_index:       float = 0.0
    stress_label:       str   = "CALM"        # CALM / MILD / MODERATE / HIGH

    # Emotion
    dominant_emotion:   str   = "neutral"
    emotion_confidence: float = 0.0
    emotion_source:     str   = "none"        # "lstm" | "hfa" | "rule"

    # Attention
    attention_zone:     str   = "CENTER"      # gaze direction
    eye_contact_score:  float = 0.5

    # AU activity
    active_aus:         list  = field(default_factory=list)

    # Body signals
    lean_signal:        str   = "NEUTRAL"
    blink_rate:         int   = 0
    blink_stress:       str   = "NORMAL"

    # Meta
    explanation:        dict  = field(default_factory=dict)
    raw:                dict  = field(default_factory=dict)


# ── Learned Fusion MLP ────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """
    Lightweight MLP that learns to combine per-modality features
    into engagement and stress scores.

    Input:  (batch, 20)  — flattened fusion feature vector (see build_fusion_vector)
    Output: engagement (batch, 1)  sigmoid
            stress     (batch, 1)  sigmoid

    This model is trained online from the collector (Days 16–18).
    Until trained, rule_based_fallback() is used instead.
    """

    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   # [engagement, stress]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 2)


FUSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fusion_mlp.pt')


# ── Feature vector builder ────────────────────────────────────────────────────

def build_fusion_vector(au:    dict,
                        gaze:  dict,
                        head:  dict,
                        blink: dict,
                        lstm:  dict,
                        hfa:   dict,
                        micro: dict) -> np.ndarray:
    """
    Assembles a 20-dimensional fusion feature vector from all module outputs.
    All values are clipped to [0, 1] for stable MLP input.

    Dimension map:
      0     eye_contact_score
      1     gaze_h (abs, normalised)
      2     gaze_v (abs, normalised)
      3     yaw    (abs, normalised to 0-1 over 90°)
      4     pitch  (normalised, -1=forward, +1=backward)
      5     roll   (abs, normalised)
      6     ear    (normalised: 0.15–0.40 range)
      7     blink_rate_norm (0–1 over 60 bpm)
      8     lstm_confidence
      9     hfa_confidence
      10    micro_magnitude (0–1 over 0.05 max)
      11    AU1  (inner brow raise)
      12    AU4  (brow lowerer / furrow)
      13    AU5  (upper lid raiser)
      14    AU6  (cheek raiser)
      15    AU12 (lip corner puller)
      16    AU15 (lip corner depressor)
      17    AU25 (lips part)
      18    lstm_emotion_onehot dominant index / 5  (rough emotion signal)
      19    hfa_emotion_onehot  dominant index / 5
    """
    def clip(v, lo=0.0, hi=1.0):
        try:
            return float(np.clip(float(v), lo, hi))
        except Exception:
            return 0.0

    vec = np.zeros(20, dtype=np.float32)

    # Gaze signals
    g = gaze or {}
    vec[0]  = clip(g.get('eye_contact',   0.5))
    vec[1]  = clip(abs(g.get('gaze_h',    0.0)))
    vec[2]  = clip(abs(g.get('gaze_v',    0.0)))

    # Head pose signals
    h = head or {}
    vec[3]  = clip(abs(h.get('yaw',   0.0)) / 90.0)
    raw_pitch = h.get('pitch', 0.0)
    vec[4]  = clip((raw_pitch + 30) / 60.0)  # forward=-30 → 0, backward=+30 → 1
    vec[5]  = clip(abs(h.get('roll',  0.0)) / 45.0)

    # Blink signals
    b = blink or {}
    ear = b.get('ear', 0.25)
    vec[6]  = clip((ear - 0.10) / 0.35)      # 0.10 closed → 0.45 open mapped to [0,1]
    vec[7]  = clip(b.get('blink_rate_per_min', 15) / 60.0)

    # LSTM / HFA confidence
    lp = lstm or {}
    vec[8]  = clip(lp.get('confidence', 0.0))
    hp = hfa or {}
    vec[9]  = clip(hp.get('confidence', 0.0))

    # Micro spike magnitude
    mi = micro or {}
    vec[10] = clip(mi.get('magnitude', 0.0) / 0.05) if mi else 0.0

    # AU values (already normalised by face_w in AUCalculator)
    au = au or {}
    vec[11] = clip(abs(au.get('AU1',  0.0)) * 20)
    vec[12] = clip(abs(au.get('AU4',  0.0)) * 20)
    vec[13] = clip(abs(au.get('AU5',  0.0)) * 20)
    vec[14] = clip(abs(au.get('AU6',  0.0)) * 20)
    vec[15] = clip(abs(au.get('AU12', 0.0)) * 20)
    vec[16] = clip(abs(au.get('AU15', 0.0)) * 20)
    vec[17] = clip(abs(au.get('AU25', 0.0)) * 20)

    # Emotion index signals
    vec[18] = clip(lp.get('emotion_idx', 4) / 5.0)
    vec[19] = clip(hp.get('emotion_idx', 4) / 5.0)

    return vec


# ── Rule-based fallback ───────────────────────────────────────────────────────

def rule_based_scores(vec: np.ndarray, gaze: dict, head: dict, blink: dict) -> tuple[float, float]:
    """
    Computes engagement and stress using interpretable heuristic rules.
    Used when FusionMLP has not been trained yet.

    Returns (engagement_score, stress_index) both in [0, 1].
    """
    g = gaze  or {}
    h = head  or {}
    b = blink or {}

    # ── Engagement score ──────────────────────────────────────
    # Start at 0.5 baseline and adjust with evidence
    engagement = 0.50

    # Eye contact: strongest single predictor of engagement
    eye_c = float(g.get('eye_contact', 0.5))
    engagement += (eye_c - 0.5) * 0.40   # ±0.20 contribution

    # Head lean: leaning forward = interest
    lean = h.get('lean_signal', 'NEUTRAL')
    if lean == 'FORWARD':
        engagement += 0.12
    elif lean == 'BACKWARD':
        engagement -= 0.10
    elif lean == 'AWAY':
        engagement -= 0.18

    # Blink rate: very low blinking = intense focus (positive for engagement)
    blink_rate = int(b.get('blink_rate_per_min', 15))
    if 8 <= blink_rate <= 20:
        engagement += 0.05    # normal, slight boost
    elif blink_rate < 5:
        engagement += 0.08    # staring = very focused
    elif blink_rate > 30:
        engagement -= 0.08    # excessive blinking = distraction/stress

    # AU6 (cheek raiser, genuine smile) and AU12 (lip corner puller)
    au6  = float(vec[14])
    au12 = float(vec[15])
    if au6 > 0.3 and au12 > 0.3:
        engagement += 0.08    # genuine positive expression

    # Absolute gaze deviation — looking away kills engagement
    gaze_dev = float(vec[1]) + float(vec[2])
    engagement -= gaze_dev * 0.15

    engagement = float(np.clip(engagement, 0.0, 1.0))

    # ── Stress index ──────────────────────────────────────────
    stress = 0.0

    # Blink stress signal
    blink_stress = b.get('stress_signal', 'NORMAL')
    stress_map   = {'NORMAL': 0.0, 'STARING': 0.2, 'ELEVATED': 0.5, 'HIGH_STRESS': 0.9}
    stress      += stress_map.get(blink_stress, 0.0) * 0.35

    # Brow furrow (AU4) = frown = stress/concentration
    au4 = float(vec[12])
    stress += au4 * 0.25

    # Head roll asymmetry = tension
    roll_abs = float(vec[5])
    stress  += roll_abs * 0.10

    # Micro spike presence (AU temporal spike = suppression attempt)
    micro_mag = float(vec[10])
    stress   += micro_mag * 0.20

    # AU15 (lip corner depressor) — sadness / distress
    au15 = float(vec[16])
    stress += au15 * 0.10

    stress = float(np.clip(stress, 0.0, 1.0))

    return engagement, stress


# ── Label helpers ─────────────────────────────────────────────────────────────

def engagement_label(score: float) -> str:
    if score < 0.30:   return "LOW"
    if score < 0.55:   return "NEUTRAL"
    if score < 0.78:   return "HIGH"
    return "PEAK"


def stress_label(score: float) -> str:
    if score < 0.20:   return "CALM"
    if score < 0.45:   return "MILD"
    if score < 0.70:   return "MODERATE"
    return "HIGH"


def dominant_emotion(lstm: dict, hfa: dict) -> tuple[str, float, str]:
    """
    Picks the most confident emotion from LSTM or HFA.
    Returns (emotion_name, confidence, source).
    """
    lp = lstm or {}
    hp = hfa  or {}

    lstm_conf = float(lp.get('confidence', 0.0))
    hfa_conf  = float(hp.get('confidence', 0.0))

    if lstm_conf == 0.0 and hfa_conf == 0.0:
        return 'neutral', 0.0, 'none'

    if hfa_conf >= lstm_conf:
        return hp.get('emotion', 'neutral'), hfa_conf, 'hfa'
    else:
        return lp.get('emotion', 'neutral'), lstm_conf, 'lstm'


# ── Main Fusion Layer ─────────────────────────────────────────────────────────

class FusionLayer:
    """
    Central fusion layer. Instantiate once in the engine.
    Call fuse() every frame with all module outputs.
    """

    def __init__(self, use_mlp: bool = True):
        self._use_mlp = use_mlp
        self._mlp     = None
        self._smoothed_engagement = 0.5
        self._smoothed_stress     = 0.0
        self._alpha               = 0.15   # EMA smoothing (lower = smoother)

        if use_mlp and os.path.exists(FUSION_MODEL_PATH):
            try:
                self._mlp = FusionMLP()
                state = torch.load(FUSION_MODEL_PATH, map_location='cpu')
                self._mlp.load_state_dict(state)
                self._mlp.eval()
                print("FusionLayer: loaded trained FusionMLP weights.")
            except Exception as e:
                print(f"FusionLayer: could not load FusionMLP ({e}). Using rules.")
                self._mlp = None
        else:
            print("FusionLayer: no FusionMLP weights found. Using rule-based fusion.")

    def fuse(self,
             au:    dict,
             gaze:  dict,
             head:  dict,
             blink: dict,
             lstm:  dict,
             hfa:   dict,
             micro: dict) -> BehaviourState:
        """
        Main fusion call. Returns a BehaviourState.
        Smooths engagement and stress with exponential moving average.
        """
        vec = build_fusion_vector(au, gaze, head, blink, lstm, hfa, micro)

        # ── Score computation ──────────────────────────────────────────
        if self._mlp is not None:
            with torch.no_grad():
                t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                out = self._mlp(t).squeeze(0).numpy()
            raw_engagement = float(out[0])
            raw_stress     = float(out[1])
        else:
            raw_engagement, raw_stress = rule_based_scores(vec, gaze, head, blink)

        # ── Exponential moving average smoothing ───────────────────────
        a = self._alpha
        self._smoothed_engagement = a * raw_engagement + (1 - a) * self._smoothed_engagement
        self._smoothed_stress     = a * raw_stress     + (1 - a) * self._smoothed_stress

        eng   = round(self._smoothed_engagement, 3)
        stress= round(self._smoothed_stress,     3)

        # ── Dominant emotion ───────────────────────────────────────────
        emotion, conf, source = dominant_emotion(lstm, hfa)

        # ── Active AUs (threshold 0.25 significance) ───────────────────
        au_names = ['AU1','AU2','AU4','AU5','AU6','AU12','AU15','AU17','AU25','AU26']
        active   = [
            name for name in au_names
            if abs(float((au or {}).get(name, 0.0))) * 20 > 0.25
        ]

        # ── Build state ────────────────────────────────────────────────
        g = gaze  or {}
        h = head  or {}
        b = blink or {}

        state = BehaviourState(
            timestamp          = round(time.time(), 3),
            engagement_score   = eng,
            engagement_label   = engagement_label(eng),
            stress_index       = stress,
            stress_label       = stress_label(stress),
            dominant_emotion   = emotion,
            emotion_confidence = round(conf, 3),
            emotion_source     = source,
            attention_zone     = g.get('direction', 'CENTER'),
            eye_contact_score  = round(float(g.get('eye_contact', 0.5)), 3),
            active_aus         = active,
            lean_signal        = h.get('lean_signal', 'NEUTRAL'),
            blink_rate         = int(b.get('blink_rate_per_min', 0)),
            blink_stress       = b.get('stress_signal', 'NORMAL'),
            raw                = {
                'au': au, 'gaze': g, 'head': h,
                'blink': b, 'lstm': lstm, 'hfa': hfa, 'micro': micro,
                'fusion_vec': vec.tolist()
            }
        )

        return state
"""
MMBI Interest State Classifier
==============================
Fuses multimodal behavioural signals into three high-level states:
  - INTERESTED
  - NEUTRAL
  - NOT INTERESTED

The decision combines:
  - Engagement score and trend
  - Eye contact score and gaze zone
  - Head pose (yaw, pitch, lean signal)
  - Action Units (AU1, AU4, AU6, AU12, AU15, AU17, AU25, AU26)
  - Blink behavior and stress signal
  - Dominant macro emotion and micro-expression activity
  - Facial dynamics and temporal behavior

Scientific Disclaimer:
  "Model-estimated interest state based on observed behavioural signals."
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

INTEREST_STATES = ["INTERESTED", "NEUTRAL", "NOT INTERESTED"]

class InterestClassifier:
    """
    Fuses multiple multimodal signals into a calibrated interest index [0.0, 1.0]
    and maps it to INTERESTED, NEUTRAL, or NOT INTERESTED with supporting evidence.
    """

    def __init__(self, smoothing_alpha: float = 0.25, window_size: int = 15):
        self.alpha = smoothing_alpha
        self.smoothed_score: float = 0.50
        self.history = deque(maxlen=window_size)
        self._current_state = "NEUTRAL"
        self._state_consecutive_frames = 0

    def compute(
        self,
        engagement_score: float,
        stress_index: float,
        eye_contact_score: float,
        attention_zone: str,
        head_lean: str,
        head_yaw: float,
        head_pitch: float,
        blink_rate: int,
        blink_stress: str,
        active_aus: List[str],
        dominant_emotion: str,
        macro_emotion: str,
        micro_info: Optional[Dict[str, Any]] = None,
        au_dict: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float, float, List[str]]:
        """
        Computes the current interest state, raw interest index, confidence,
        and supporting behavioral evidence.

        Returns:
            (interest_state, interest_index, confidence, supporting_signals)
        """
        supporting_signals: List[str] = []

        # Baseline starting score: 0.50 (Neutral)
        score = 0.50

        # ── 1. Engagement & Eye Contact (~35% weight) ─────────────────────────
        eng = float(np.clip(engagement_score, 0.0, 1.0))
        eye_c = float(np.clip(eye_contact_score, 0.0, 1.0))

        # Engagement contribution: deviation from neutral 0.50
        score += (eng - 0.50) * 0.35
        if eng >= 0.65:
            supporting_signals.append(f"Elevated cognitive engagement ({eng:.2f})")
        elif eng <= 0.35:
            supporting_signals.append(f"Depressed cognitive engagement ({eng:.2f})")

        # Eye contact contribution: deviation from neutral 0.50
        score += (eye_c - 0.50) * 0.25
        if eye_c >= 0.70:
            supporting_signals.append(f"High sustained eye contact ({eye_c:.2f})")
        elif eye_c <= 0.30:
            supporting_signals.append(f"Reduced eye contact ({eye_c:.2f})")

        # Gaze direction contribution (only penalize when diverted from center)
        att_zone = str(attention_zone).upper()
        if att_zone not in ["CENTER", ""]:
            score -= 0.15
            supporting_signals.append(f"Gaze directed away from center ({att_zone})")

        # ── 2. Head Pose & Postural Lean (~20% weight) ────────────────────────
        lean = str(head_lean).upper()
        if lean == "FORWARD":
            score += 0.12
            supporting_signals.append("Attentive forward head posture")
        elif lean == "BACKWARD":
            score -= 0.10
            supporting_signals.append("Reclined / disengaged head posture")
        elif lean == "AWAY":
            score -= 0.16
            supporting_signals.append("Head oriented away from interaction")

        # Head yaw deviation
        abs_yaw = abs(float(head_yaw))
        if abs_yaw > 25.0:
            score -= 0.12
            supporting_signals.append(f"Lateral head turn ({abs_yaw:.1f}° yaw)")

        # ── 3. Action Units & Facial Expression (~25% weight) ─────────────────
        au = au_dict or {}
        au6 = abs(float(au.get("AU6", 0.0))) * 20.0
        au12 = abs(float(au.get("AU12", 0.0))) * 20.0
        au4 = abs(float(au.get("AU4", 0.0))) * 20.0
        au15 = abs(float(au.get("AU15", 0.0))) * 20.0

        # Positive affective rapport: Duchenne smile / cheek raise + lip pull
        if au12 > 0.25 and au6 > 0.20:
            score += 0.12
            supporting_signals.append("Positive facial animation (AU6+AU12 smile)")
        elif au12 > 0.25:
            score += 0.06

        # Cognitive focus brow furrow (AU4) vs frustration
        if au4 > 0.30:
            if stress_index < 0.45:
                score += 0.05
                supporting_signals.append("Cognitive focus furrow (AU4)")
            else:
                score -= 0.08
                supporting_signals.append("Tension/distress furrow (AU4)")

        # Lip corner depressor (AU15)
        if au15 > 0.30:
            score -= 0.08
            supporting_signals.append("Downward lip angle (AU15)")

        # Macro emotion influence
        macro = str(macro_emotion).lower()
        if macro in ["happiness"]:
            score += 0.06
        elif macro in ["sadness", "disgust"]:
            score -= 0.10
            supporting_signals.append(f"Negative affective valence ({macro})")

        # ── 4. Blinking Behavior (~10% weight) ────────────────────────────────
        b_rate = int(blink_rate)
        b_stress = str(blink_stress).upper()
        if b_rate < 6 and att_zone == "CENTER":
            score += 0.08
            supporting_signals.append("Intense visual focus (low blink rate)")
        elif b_rate > 32 or b_stress in ["ELEVATED", "HIGH_STRESS"]:
            score -= 0.10
            supporting_signals.append(f"Elevated blink restlessness ({b_rate} bpm)")

        # ── 5. Micro-Expression Activity (~10% weight) ────────────────────────
        mi = micro_info or {}
        if mi.get("spike", False):
            m_emo = str(mi.get("micro_emotion", "")).lower()
            m_reg = str(mi.get("region", "facial")).replace("_", " ")
            m_dur = mi.get("duration_ms", 0.0)
            if m_emo in ["surprise", "happiness"]:
                score += 0.06
                supporting_signals.append(f"Micro-expression transient alert: {m_emo.upper()} ({m_reg}, {m_dur:.0f}ms)")
            elif m_emo in ["disgust", "sadness", "repression"]:
                score -= 0.06
                supporting_signals.append(f"Micro-expression suppression leakage: {m_emo.upper()} ({m_reg}, {m_dur:.0f}ms)")
            else:
                supporting_signals.append(f"Micro-expression detected in {m_reg} ({m_dur:.0f}ms)")

        # ── 6. Normalization & Exponential Moving Average ──────────────────────
        raw_score = float(np.clip(score, 0.0, 1.0))
        self.smoothed_score = self.alpha * raw_score + (1.0 - self.alpha) * self.smoothed_score
        final_index = float(np.clip(self.smoothed_score, 0.0, 1.0))
        self.history.append(final_index)

        # ── 7. Three-State Classification with Hysteresis ─────────────────────
        # INTERESTED: >= 0.58
        # NOT INTERESTED: <= 0.38
        # NEUTRAL: 0.38 < index < 0.58
        if final_index >= 0.58:
            target_state = "INTERESTED"
            confidence = float(np.clip((final_index - 0.50) / 0.50, 0.55, 0.98))
        elif final_index <= 0.38:
            target_state = "NOT INTERESTED"
            confidence = float(np.clip((0.50 - final_index) / 0.50, 0.55, 0.98))
        else:
            target_state = "NEUTRAL"
            confidence = float(np.clip(1.0 - abs(final_index - 0.50) * 3.0, 0.50, 0.92))

        # Hysteresis: require 3 consecutive frames to switch state unless extreme
        if target_state != self._current_state:
            if abs(final_index - 0.50) > 0.28:
                self._current_state = target_state
                self._state_consecutive_frames = 1
            else:
                self._state_consecutive_frames += 1
                if self._state_consecutive_frames >= 3:
                    self._current_state = target_state
                    self._state_consecutive_frames = 0
        else:
            self._state_consecutive_frames = 0

        # Ensure default supporting signal if empty
        if not supporting_signals:
            if self._current_state == "INTERESTED":
                supporting_signals.append("Positive multimodal engagement indicators")
                supporting_signals.append("Attentive frontal gaze alignment")
            elif self._current_state == "NOT INTERESTED":
                supporting_signals.append("Reduced gaze attention")
                supporting_signals.append("Subdued engagement indicators")
            else:
                supporting_signals.append("Stable resting facial baseline")
                supporting_signals.append("Balanced engagement and eye contact")

        return (
            self._current_state,
            round(final_index, 3),
            round(confidence, 3),
            supporting_signals[:4]
        )

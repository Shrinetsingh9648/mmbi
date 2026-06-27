"""
Explainability Module — Day 15
================================
Attaches a human-readable explanation to every BehaviourState.
Answers the question: "WHY did the engine produce this output?"

Three explanation levels:
  1. signal_report   — raw signal summary (what each sensor is reading)
  2. inference_chain — step-by-step reasoning (what was inferred from signals)
  3. natural_language— one-sentence English summary for UI display

Also produces:
  - confidence_breakdown  per-modality confidence contributions
  - conflict_flags        cases where signals disagree
  - attention_heatmap_note which face regions drove the prediction

Design principle: every output number must be traceable to a specific
sensor reading. This makes the system auditable and debuggable.
"""

from __future__ import annotations
from mmbi.fusion.fusion_layer import BehaviourState
import numpy as np


# ── Emotion descriptions for natural language output ──────────────────────────

EMOTION_DESCRIPTIONS = {
    'happiness':   "a positive/happy expression",
    'disgust':     "a disgust expression",
    'repression':  "an expression being suppressed or hidden",
    'surprise':    "a surprise expression",
    'others':      "a neutral or ambiguous expression",
    'neutral':     "a neutral expression"
}

AU_DESCRIPTIONS = {
    'AU1':  "inner brow raise (concern/surprise)",
    'AU2':  "outer brow raise (surprise)",
    'AU4':  "brow lowerer/furrow (concentration/anger)",
    'AU5':  "upper lid raiser (alertness/fear)",
    'AU6':  "cheek raiser (genuine positive emotion)",
    'AU12': "lip corner puller (smile)",
    'AU15': "lip corner depressor (sadness/distress)",
    'AU17': "chin raiser (doubt/disagreement)",
    'AU25': "lips part (mild surprise/speech)",
    'AU26': "jaw drop (strong surprise)"
}


class ExplainabilityModule:
    """
    Attach to the MMBI engine. Call explain() after every fuse() call.
    Mutates the state.explanation dict in-place.
    """

    def explain(self, state: BehaviourState) -> BehaviourState:
        """
        Fills state.explanation with:
          signal_report, inference_chain, natural_language,
          confidence_breakdown, conflict_flags
        """
        state.explanation = {
            'signal_report':       self._signal_report(state),
            'inference_chain':     self._inference_chain(state),
            'natural_language':    self._natural_language(state),
            'confidence_breakdown':self._confidence_breakdown(state),
            'conflict_flags':      self._conflict_flags(state)
        }
        return state

    # ── Signal report ─────────────────────────────────────────────────────────

    def _signal_report(self, s: BehaviourState) -> dict:
        """
        Plain summary of what each sensor is currently reporting.
        """
        raw   = s.raw or {}
        gaze  = raw.get('gaze',  {})
        head  = raw.get('head',  {})
        blink = raw.get('blink', {})
        lstm  = raw.get('lstm',  {})
        hfa   = raw.get('hfa',   {})

        au_active_str = ", ".join(s.active_aus) if s.active_aus else "none"
        au_descriptions = [
            f"{au} ({AU_DESCRIPTIONS.get(au, 'action unit')})"
            for au in s.active_aus
        ]

        return {
            'eye_contact':       f"{s.eye_contact_score:.2f} (1.0=direct camera gaze)",
            'gaze_direction':    s.attention_zone,
            'gaze_h':            f"{gaze.get('gaze_h', 0.0):+.3f} (negative=left, positive=right)",
            'gaze_v':            f"{gaze.get('gaze_v', 0.0):+.3f} (negative=up, positive=down)",
            'head_yaw':          f"{head.get('yaw',   0.0):+.1f}° (negative=left)",
            'head_pitch':        f"{head.get('pitch', 0.0):+.1f}° (negative=forward lean)",
            'head_roll':         f"{head.get('roll',  0.0):+.1f}° (0=upright)",
            'lean_signal':       s.lean_signal,
            'ear':               f"{blink.get('ear', 0.25):.3f} (>0.25=eyes open)",
            'blink_rate':        f"{s.blink_rate} per minute",
            'blink_stress':      s.blink_stress,
            'active_aus':        au_descriptions if au_descriptions else ["none active"],
            'lstm_emotion':      f"{lstm.get('emotion', 'none')} ({lstm.get('confidence', 0):.2f})",
            'hfa_emotion':       f"{hfa.get('emotion', 'none')} ({hfa.get('confidence', 0):.2f})",
        }

    # ── Inference chain ───────────────────────────────────────────────────────

    def _inference_chain(self, s: BehaviourState) -> list[str]:
        """
        Step-by-step reasoning trace showing HOW engagement and stress
        were derived from the raw signals. Written as numbered steps.
        """
        chain = []
        raw   = s.raw or {}
        gaze  = raw.get('gaze',  {})
        head  = raw.get('head',  {})
        blink = raw.get('blink', {})

        # Engagement reasoning
        chain.append(f"[1] Eye contact score = {s.eye_contact_score:.2f} "
                     f"→ {'strong positive' if s.eye_contact_score > 0.65 else 'weak/absent'} engagement signal")

        lean = s.lean_signal
        if lean == 'FORWARD':
            chain.append("[2] Head pitch forward detected → interest/lean-in signal (+engagement)")
        elif lean == 'BACKWARD':
            chain.append("[2] Head pitch backward → disengagement/lean-out signal (-engagement)")
        elif lean == 'AWAY':
            chain.append("[2] Head turned away → strong disengagement signal (-engagement)")
        else:
            chain.append("[2] Head pose neutral → no lean-in or lean-out detected")

        blink_rate = s.blink_rate
        if blink_rate < 5:
            chain.append(f"[3] Blink rate {blink_rate}/min (very low = fixed focus) → focus engaged")
        elif blink_rate > 25:
            chain.append(f"[3] Blink rate {blink_rate}/min (elevated) → stress/distraction signal")
        else:
            chain.append(f"[3] Blink rate {blink_rate}/min (normal range, 12–20)")

        if 'AU6' in s.active_aus and 'AU12' in s.active_aus:
            chain.append("[4] AU6 (cheek raise) + AU12 (lip pull) active → genuine smile (Duchenne marker)")
        elif 'AU4' in s.active_aus:
            chain.append("[4] AU4 (brow furrow) active → concentration or displeasure signal")
        elif not s.active_aus:
            chain.append("[4] No significant AUs active → neutral baseline face")

        chain.append(f"[5] Fused engagement score = {s.engagement_score:.3f} → {s.engagement_label}")

        # Stress reasoning
        chain.append(f"[6] Blink stress signal = '{s.blink_stress}' → "
                     f"{'elevated stress' if s.blink_stress in ('ELEVATED','HIGH_STRESS') else 'normal'}")

        micro = raw.get('micro', {})
        if micro and micro.get('magnitude', 0) > 0.015:
            chain.append(f"[7] Micro-expression spike detected (magnitude={micro['magnitude']:.3f}) "
                         f"→ possible emotion suppression (+stress)")
        else:
            chain.append("[7] No micro-expression spike detected this frame")

        chain.append(f"[8] Fused stress index = {s.stress_index:.3f} → {s.stress_label}")

        # Emotion reasoning
        if s.emotion_confidence > 0.0:
            em_desc = EMOTION_DESCRIPTIONS.get(s.dominant_emotion, s.dominant_emotion)
            chain.append(f"[9] Dominant emotion = '{s.dominant_emotion}' "
                         f"({em_desc}, confidence={s.emotion_confidence:.2f}, "
                         f"source={s.emotion_source})")
        else:
            chain.append("[9] No confident emotion prediction available yet "
                         "(buffer filling or model not trained)")

        return chain

    # ── Natural language summary ───────────────────────────────────────────────

    def _natural_language(self, s: BehaviourState) -> str:
        """
        One-sentence English summary suitable for a live UI overlay.
        Written to be informative but not clinical.
        """
        # Build phrase components
        eng_phrase = {
            'LOW':     "showing low engagement",
            'NEUTRAL': "showing neutral engagement",
            'HIGH':    "highly engaged",
            'PEAK':    "at peak engagement"
        }.get(s.engagement_label, "engaged")

        stress_phrase = {
            'CALM':     "relaxed",
            'MILD':     "mildly stressed",
            'MODERATE': "moderately stressed",
            'HIGH':     "under high stress"
        }.get(s.stress_label, "")

        # Attention phrase
        if s.attention_zone == 'CENTER' and s.eye_contact_score > 0.65:
            attention_phrase = "maintaining direct eye contact"
        elif s.attention_zone == 'CENTER':
            attention_phrase = "looking approximately at camera"
        else:
            attention_phrase = f"looking {s.attention_zone.lower()}"

        # Emotion phrase
        em_desc = EMOTION_DESCRIPTIONS.get(s.dominant_emotion, "")
        if s.emotion_confidence > 0.55 and em_desc:
            emotion_phrase = f" with {em_desc} detected"
        else:
            emotion_phrase = ""

        return (
            f"Subject is {eng_phrase} and {stress_phrase}, "
            f"{attention_phrase}{emotion_phrase}."
        )

    # ── Confidence breakdown ──────────────────────────────────────────────────

    def _confidence_breakdown(self, s: BehaviourState) -> dict:
        """
        Estimates how much each modality contributed to the final
        engagement and stress scores, as a percentage.
        For display in a breakdown widget.
        """
        raw   = s.raw or {}
        gaze  = raw.get('gaze',  {})
        blink = raw.get('blink', {})
        vec   = raw.get('fusion_vec', [0.0] * 20)

        # These weights mirror the rule-based logic in fusion_layer.py
        # so the breakdown is honest about what the engine actually used
        breakdown = {}

        # Eye contact contribution (largest single weight)
        ec = float(gaze.get('eye_contact', 0.5))
        breakdown['eye_contact']     = round(abs(ec - 0.5) * 40, 1)   # 0-20

        # Blink rate
        blink_rate = int(blink.get('blink_rate_per_min', 15))
        breakdown['blink_rate']      = round(min(abs(blink_rate - 15) / 15.0 * 15, 15), 1)

        # AU activity
        au_activity = float(np.mean(np.abs(vec[11:18]))) if len(vec) >= 18 else 0.0
        breakdown['au_activity']     = round(au_activity * 20, 1)

        # Head pose
        head_signal = float(vec[3]) + float(vec[4])
        breakdown['head_pose']       = round(head_signal * 10, 1)

        # LSTM / HFA model
        ml_conf = max(float(vec[8]), float(vec[9]))
        breakdown['ml_emotion_model']= round(ml_conf * 15, 1)

        # Micro spike
        breakdown['micro_expression']= round(float(vec[10]) * 10, 1)

        # Normalise to 100%
        total = sum(breakdown.values()) or 1.0
        breakdown = {k: round(v / total * 100, 1) for k, v in breakdown.items()}

        return breakdown

    # ── Conflict flags ────────────────────────────────────────────────────────

    def _conflict_flags(self, s: BehaviourState) -> list[str]:
        """
        Identifies cases where signals contradict each other.
        These are highlighted in the UI so the analyst knows
        the engine's confidence is lower than usual.
        """
        flags = []
        raw   = s.raw or {}
        lstm  = raw.get('lstm', {})
        hfa   = raw.get('hfa',  {})

        # LSTM and HFA disagree on emotion
        lstm_em = lstm.get('emotion', 'neutral')
        hfa_em  = hfa.get('emotion',  'neutral')
        if (lstm_em != hfa_em
                and lstm.get('confidence', 0) > 0.45
                and hfa.get('confidence',  0) > 0.45):
            flags.append(f"LSTM predicts '{lstm_em}' but HFA predicts '{hfa_em}' — models disagree")

        # High engagement but high stress simultaneously
        if s.engagement_score > 0.65 and s.stress_index > 0.55:
            flags.append("High engagement AND high stress simultaneously — "
                         "may indicate intense concentration or performance anxiety")

        # Low engagement but direct eye contact
        if s.engagement_score < 0.35 and s.eye_contact_score > 0.75:
            flags.append("Low engagement score but direct eye contact — "
                         "gaze is forward but other signals suggest disengagement")

        # Smiling but stressed
        if ('AU6' in s.active_aus or 'AU12' in s.active_aus) and s.stress_index > 0.55:
            flags.append("Positive AU (smile/cheek raise) active during elevated stress — "
                         "may indicate social masking")

        # Suppression signal with happy emotion
        if s.dominant_emotion == 'repression' and s.engagement_label == 'HIGH':
            flags.append("Repression/suppression emotion detected during high engagement — "
                         "subject may be concealing a reaction")

        return flags
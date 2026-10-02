"""
MMBI Session Management & Telemetry Architecture
================================================
Defines clean data structures and recorder logic for:
  - MicroExpressionEvent
  - InterestTimelineInterval
  - SessionResult
  - SessionRecorder
  - SessionManager
"""

from __future__ import annotations
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), 'learning', 'data', 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)

def format_timestamp(seconds: float, include_ms: bool = True) -> str:
    """Formats seconds into HH:MM:SS or HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    if include_ms:
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
        return f"{mins:02d}:{secs:02d}.{ms:03d}"
    else:
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

class SessionRecorder:
    """
    Unified telemetry recorder for both Live Webcam and Recorded Video sessions.
    Maintains interest state timeline, structured micro-expression events,
    and cumulative metric summaries.
    """

    def __init__(
        self,
        session_id: str,
        analysis_type: str = "Live",
        video_filename: Optional[str] = None
    ):
        self.session_id = session_id
        self.analysis_type = analysis_type
        self.video_filename = video_filename
        self.created_at = datetime.now()
        self.start_wall_time = time.time()
        
        # Timing
        self.first_frame_ts: Optional[float] = None
        self.last_frame_ts: Optional[float] = None
        self.total_frames = 0
        self.face_frames = 0
        self.no_face_frames = 0
        
        # State Accumulators
        self.micro_expressions: List[Dict[str, Any]] = []
        self.me_counter = 0
        
        # Interest State Tracking
        self.current_interest_state = "NEUTRAL"
        self.interest_durations = {
            "INTERESTED": 0.0,
            "NEUTRAL": 0.0,
            "NOT INTERESTED": 0.0
        }
        self.interest_timeline: List[Dict[str, Any]] = []
        self._current_interval_start = 0.0
        self._current_interval_eng_scores: List[float] = []
        self._current_interval_signals: List[str] = []
        self._current_interval_me_count = 0
        self.interest_transitions = 0
        
        # Metric Accumulators
        self.engagement_scores: List[float] = []
        self.stress_scores: List[float] = []
        self.eye_contact_scores: List[float] = []
        self.blink_rates: List[int] = []
        self.gaze_counts: Dict[str, int] = {}
        self.macro_emotion_counts: Dict[str, int] = {}
        self.head_pitch_scores: List[float] = []
        self.head_yaw_scores: List[float] = []
        self.head_lean_counts: Dict[str, int] = {}
        
        # Sampled Timeline for Graphs (max 300 points)
        self.graph_samples: List[Dict[str, Any]] = []
        self._last_sample_t = -1.0
        
        # Event Timeline (Significant Events)
        self.event_timeline: List[Dict[str, Any]] = []
        self._last_eng_event_val = 0.50

    def update(self, state: Any, timestamp: float, frame_idx: int):
        """
        Record a frame where a face was detected and processed.
        `timestamp` is seconds from session/video start.
        """
        self.total_frames += 1
        self.face_frames += 1
        
        if self.first_frame_ts is None:
            self.first_frame_ts = timestamp
            self._current_interval_start = timestamp
            self.current_interest_state = getattr(state, "interest_state", "NEUTRAL")
            
        dt = 0.0
        if self.last_frame_ts is not None:
            dt = max(0.0, timestamp - self.last_frame_ts)
        self.last_frame_ts = timestamp
        
        # Extract features
        interest_state = getattr(state, "interest_state", "NEUTRAL")
        interest_score = getattr(state, "interest_score", 0.5)
        eng_score = float(getattr(state, "engagement_score", 0.5))
        stress_idx = float(getattr(state, "stress_index", 0.0))
        eye_contact = float(getattr(state, "eye_contact_score", 0.5))
        gaze_zone = str(getattr(state, "attention_zone", "CENTER"))
        macro_emo = str(getattr(state, "macro_emotion", "neutral"))
        lean = str(getattr(state, "lean_signal", "NEUTRAL"))
        blink_r = int(getattr(state, "blink_rate", 0))
        signals = getattr(state, "supporting_signals", [])
        raw = getattr(state, "raw", {})
        head = raw.get("head", {})
        yaw = float(head.get("yaw", 0.0))
        pitch = float(head.get("pitch", 0.0))
        
        # Accumulate metrics
        self.engagement_scores.append(eng_score)
        self.stress_scores.append(stress_idx)
        self.eye_contact_scores.append(eye_contact)
        self.blink_rates.append(blink_r)
        self.gaze_counts[gaze_zone] = self.gaze_counts.get(gaze_zone, 0) + 1
        self.macro_emotion_counts[macro_emo] = self.macro_emotion_counts.get(macro_emo, 0) + 1
        self.head_pitch_scores.append(pitch)
        self.head_yaw_scores.append(yaw)
        self.head_lean_counts[lean] = self.head_lean_counts.get(lean, 0) + 1
        
        # Accumulate duration into current state
        if interest_state in self.interest_durations:
            self.interest_durations[interest_state] += dt
        else:
            self.interest_durations["NEUTRAL"] += dt
            
        self._current_interval_eng_scores.append(eng_score)
        if signals:
            self._current_interval_signals = signals

        # Compute session-relative time (prevents epoch display bugs like 497458:49:56)
        rel_ts = max(0.0, timestamp - self.first_frame_ts)
        rel_interval_start = max(0.0, self._current_interval_start - self.first_frame_ts)

        # Check Interest State Transition
        if interest_state != self.current_interest_state:
            # Seal previous interval using session-relative times
            dur = rel_ts - rel_interval_start
            if dur > 0.1:  # ignore micro glitches
                avg_eng = float(np.mean(self._current_interval_eng_scores)) if self._current_interval_eng_scores else 0.5
                self.interest_timeline.append({
                    "start_sec": round(rel_interval_start, 2),
                    "end_sec": round(rel_ts, 2),
                    "start_time": format_timestamp(rel_interval_start, include_ms=False),
                    "end_time": format_timestamp(rel_ts, include_ms=False),
                    "state": self.current_interest_state,
                    "duration_sec": round(dur, 2),
                    "avg_engagement": round(avg_eng, 3),
                    "supporting_signals": self._current_interval_signals[:3],
                    "micro_expressions_count": self._current_interval_me_count
                })
                self.interest_transitions += 1

                # Add transition to event timeline (session-relative)
                self.event_timeline.append({
                    "time_sec": round(rel_ts, 2),
                    "time": format_timestamp(rel_ts),
                    "event_type": "INTEREST_TRANSITION",
                    "title": f"Interest transitioned to {interest_state}",
                    "details": f"Previous: {self.current_interest_state} ({dur:.1f}s). Engagement: {eng_score:.2f}",
                    "severity": "info" if interest_state == "INTERESTED" else ("warning" if interest_state == "NOT INTERESTED" else "neutral")
                })

            self.current_interest_state = interest_state
            self._current_interval_start = timestamp  # keep absolute internally
            self._current_interval_eng_scores = [eng_score]
            self._current_interval_signals = signals
            self._current_interval_me_count = 0

        # Check for significant engagement shift (session-relative)
        if abs(eng_score - self._last_eng_event_val) >= 0.22:
            direction = "surged" if eng_score > self._last_eng_event_val else "dropped"
            self.event_timeline.append({
                "time_sec": round(rel_ts, 2),
                "time": format_timestamp(rel_ts),
                "event_type": "ENGAGEMENT_SHIFT",
                "title": f"Engagement {direction} to {eng_score:.2f}",
                "details": f"Shift of {abs(eng_score - self._last_eng_event_val):+.2f} while in {interest_state} state",
                "severity": "positive" if eng_score > self._last_eng_event_val else "warning"
            })
            self._last_eng_event_val = eng_score

        # Graph sample (session-relative, approx once every 0.3s)
        if rel_ts - self._last_sample_t >= 0.30 or self._last_sample_t < 0:
            self.graph_samples.append({
                "t": round(rel_ts, 2),
                "eng": round(eng_score, 3),
                "stress": round(stress_idx, 3),
                "interest_score": round(interest_score, 3),
                "state": interest_state
            })
            self._last_sample_t = rel_ts

    def record_no_face(self, timestamp: float, frame_idx: int):
        """Record a frame where no face was detected."""
        self.total_frames += 1
        self.no_face_frames += 1
        if self.first_frame_ts is None:
            self.first_frame_ts = timestamp
            self._current_interval_start = timestamp
        dt = 0.0
        if self.last_frame_ts is not None:
            dt = max(0.0, timestamp - self.last_frame_ts)
        self.last_frame_ts = timestamp
        # Accumulate no-face duration as NOT INTERESTED
        self.interest_durations["NOT INTERESTED"] += dt
        # Also advance last_sample_t tracker with relative time
        if self.first_frame_ts is not None:
            rel_ts = max(0.0, timestamp - self.first_frame_ts)
            self._last_sample_t = max(self._last_sample_t, rel_ts - 0.30)

    def record_micro_expression(self, me_data: Dict[str, Any], timestamp: float):
        """
        Record a detected micro-expression event with structured attributes.
        `timestamp` is the absolute timestamp; stored values are session-relative.
        """
        self.me_counter += 1
        self._current_interval_me_count += 1
        
        # Compute session-relative time
        first_ts = self.first_frame_ts if self.first_frame_ts is not None else timestamp
        rel_ts = max(0.0, timestamp - first_ts)
        
        event_id = f"ME-{self.me_counter:02d}"
        dur_ms = float(me_data.get("duration_ms", 150.0))
        magnitude = float(me_data.get("magnitude", 0.5))
        conf = float(me_data.get("confidence", 0.75))
        m_type = str(me_data.get("micro_emotion", me_data.get("type", "surprise"))).capitalize()
        region = str(me_data.get("region", "Mouth")).replace("_", " ").title()
        
        # Onset/apex/offset: convert from absolute to relative if they look like epoch times
        raw_onset = me_data.get("onset_time", timestamp - (dur_ms / 1000.0))
        raw_apex  = me_data.get("apex_time",  timestamp - (dur_ms / 2000.0))
        raw_offset = me_data.get("offset_time", timestamp)
        
        # If onset/apex/offset are absolute (epoch-scale), subtract first_frame_ts
        # Detect epoch: values > 1e7 are definitely epoch seconds
        def to_rel(t):
            if t is None:
                return rel_ts
            t = float(t)
            return max(0.0, t - first_ts) if t > 1e7 else max(0.0, t)
        
        onset_rel  = to_rel(raw_onset)
        apex_rel   = to_rel(raw_apex)
        offset_rel = to_rel(raw_offset)
        
        event = {
            "id": event_id,
            "timestamp": format_timestamp(rel_ts),
            "timestamp_sec": round(rel_ts, 3),
            "start_time": format_timestamp(onset_rel),
            "end_time": format_timestamp(offset_rel),
            "duration_ms": round(dur_ms, 1),
            "region": region,
            "type": m_type,
            "intensity": round(magnitude, 3),
            "confidence": round(conf, 3),
            "phase": "Apex reached",
            "onset_sec": round(onset_rel, 3),
            "apex_sec": round(apex_rel, 3),
            "offset_sec": round(offset_rel, 3)
        }
        self.micro_expressions.append(event)
        
        # Add to event timeline using session-relative time
        self.event_timeline.append({
            "time_sec": round(rel_ts, 2),
            "time": format_timestamp(rel_ts),
            "event_type": "MICRO_EXPRESSION",
            "title": f"Micro-expression #{self.me_counter:02d} ({m_type})",
            "details": f"Involuntary {region} twitch for {dur_ms:.0f}ms (Intensity: {magnitude:.2f})",
            "severity": "alert"
        })

    def end_session(self) -> Dict[str, Any]:
        """
        Finalizes the session and computes all aggregated summary analytics.
        """
        end_wall_time = time.time()
        session_duration = 0.0
        if self.first_frame_ts is not None and self.last_frame_ts is not None:
            session_duration = max(1.0, self.last_frame_ts - self.first_frame_ts)
        else:
            session_duration = max(1.0, end_wall_time - self.start_wall_time)
            
        # Seal last interest interval using session-relative times
        first_ts = self.first_frame_ts if self.first_frame_ts is not None else self.start_wall_time
        if self.last_frame_ts is not None and self._current_interval_start <= self.last_frame_ts:
            rel_end = max(0.0, self.last_frame_ts - first_ts)
            rel_start = max(0.0, self._current_interval_start - first_ts)
            dur = rel_end - rel_start
            if dur > 0.05 or not self.interest_timeline:
                avg_eng = float(np.mean(self._current_interval_eng_scores)) if self._current_interval_eng_scores else 0.5
                self.interest_timeline.append({
                    "start_sec": round(rel_start, 2),
                    "end_sec": round(rel_end, 2),
                    "start_time": format_timestamp(rel_start, include_ms=False),
                    "end_time": format_timestamp(rel_end, include_ms=False),
                    "state": self.current_interest_state,
                    "duration_sec": round(dur, 2),
                    "avg_engagement": round(avg_eng, 3),
                    "supporting_signals": self._current_interval_signals[:3],
                    "micro_expressions_count": self._current_interval_me_count
                })

        # Calculate Durations & Percentages
        total_tracked_time = sum(self.interest_durations.values())
        if total_tracked_time <= 0:
            total_tracked_time = session_duration
            self.interest_durations["NEUTRAL"] = session_duration
            
        interested_sec = self.interest_durations["INTERESTED"]
        neutral_sec = self.interest_durations["NEUTRAL"]
        not_interested_sec = self.interest_durations["NOT INTERESTED"]
        
        interested_pct = round((interested_sec / total_tracked_time) * 100, 1)
        neutral_pct = round((neutral_sec / total_tracked_time) * 100, 1)
        not_interested_pct = round((not_interested_sec / total_tracked_time) * 100, 1)
        
        # Round adjustments to sum to 100%
        diff = 100.0 - (interested_pct + neutral_pct + not_interested_pct)
        neutral_pct = round(neutral_pct + diff, 1)

        # Dominant State
        state_order = [
            ("INTERESTED", interested_sec),
            ("NEUTRAL", neutral_sec),
            ("NOT INTERESTED", not_interested_sec)
        ]
        dominant_state = max(state_order, key=lambda x: x[1])[0]

        # Engagement Analytics
        avg_eng = float(np.mean(self.engagement_scores)) if self.engagement_scores else 0.50
        min_eng = float(np.min(self.engagement_scores)) if self.engagement_scores else 0.50
        max_eng = float(np.max(self.engagement_scores)) if self.engagement_scores else 0.50

        # Stress Analytics
        avg_stress = float(np.mean(self.stress_scores)) if self.stress_scores else 0.0
        min_stress = float(np.min(self.stress_scores)) if self.stress_scores else 0.0
        max_stress = float(np.max(self.stress_scores)) if self.stress_scores else 0.0
        sustained_stress_count = sum(1 for s in self.stress_scores if s >= 0.50)
        sustained_stress_load_pct = round((sustained_stress_count / max(1, len(self.stress_scores))) * 100, 1)

        # Attention Analytics
        avg_eye_contact = float(np.mean(self.eye_contact_scores)) if self.eye_contact_scores else 0.50
        eye_contact_pct = round(avg_eye_contact * 100, 1)
        
        total_gaze = sum(self.gaze_counts.values()) or 1
        gaze_dist = {k: round((v / total_gaze) * 100, 1) for k, v in self.gaze_counts.items()}
        
        avg_yaw = float(np.mean(self.head_yaw_scores)) if self.head_yaw_scores else 0.0
        avg_pitch = float(np.mean(self.head_pitch_scores)) if self.head_pitch_scores else 0.0
        total_lean = sum(self.head_lean_counts.values()) or 1
        lean_dist = {k: round((v / total_lean) * 100, 1) for k, v in self.head_lean_counts.items()}
        
        avg_blink = int(np.mean(self.blink_rates)) if self.blink_rates else 14

        # Micro-Expression Analytics
        me_count = len(self.micro_expressions)
        rate_per_min = round((me_count / max(session_duration, 1.0)) * 60, 1)
        me_durations = [m["duration_ms"] for m in self.micro_expressions]
        avg_me_duration = round(float(np.mean(me_durations)), 1) if me_durations else 0.0
        me_intensities = [m["intensity"] for m in self.micro_expressions]
        avg_me_intensity = round(float(np.mean(me_intensities)), 3) if me_intensities else 0.0
        
        # Most frequent region
        region_counts: Dict[str, int] = {}
        for m in self.micro_expressions:
            reg = m["region"]
            region_counts[reg] = region_counts.get(reg, 0) + 1
        top_region = max(region_counts, key=region_counts.get) if region_counts else "None"

        # Emotion Analytics
        total_emotions = sum(self.macro_emotion_counts.values()) or 1
        macro_dist = {k: round((v / total_emotions) * 100, 1) for k, v in self.macro_emotion_counts.items()}
        dom_macro = max(self.macro_emotion_counts, key=self.macro_emotion_counts.get) if self.macro_emotion_counts else "neutral"

        # Processing FPS
        processing_fps = round(self.total_frames / max(end_wall_time - self.start_wall_time, 0.01), 1)

        # Assemble Supporting Evidence Explanation
        supporting_evidence = []
        if dominant_state == "INTERESTED":
            supporting_evidence.append("Sustained high cognitive engagement throughout session.")
            if avg_eye_contact >= 0.60:
                supporting_evidence.append(f"Strong direct eye contact maintained (average: {avg_eye_contact:.2f}).")
            if lean_dist.get("FORWARD", 0) > 20:
                supporting_evidence.append("Frequent forward head posture reflecting active curiosity.")
            if me_count > 0:
                supporting_evidence.append(f"Registered {me_count} micro-expression alert(s) including {top_region} responses.")
        elif dominant_state == "NOT INTERESTED":
            supporting_evidence.append("Reduced gaze attention and frequent gaze diversion from center.")
            supporting_evidence.append(f"Subdued overall engagement score (average: {avg_eng:.2f}).")
            if gaze_dist.get("CENTER", 0) < 60:
                supporting_evidence.append("Subject looked away from focal zone for prolonged durations.")
        else:
            supporting_evidence.append("Balanced resting facial baseline with moderate attention levels.")
            supporting_evidence.append(f"Engagement stabilized near baseline (average: {avg_eng:.2f}).")
            supporting_evidence.append("Standard cognitive blink rate and neutral head alignment.")

        session_result = {
            "session_id": self.session_id,
            "analysis_type": self.analysis_type,
            "start_time": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(session_duration, 1),
            "duration_formatted": format_timestamp(session_duration, include_ms=False),
            "video_filename": self.video_filename or ("Live Camera" if self.analysis_type == "Live" else "N/A"),
            "total_frames": self.total_frames,
            "face_frames": self.face_frames,
            "no_face_frames": self.no_face_frames,
            "processing_fps": processing_fps,
            
            "interest_summary": {
                "dominant_state": dominant_state,
                "interested_seconds": round(interested_sec, 1),
                "neutral_seconds": round(neutral_sec, 1),
                "not_interested_seconds": round(not_interested_sec, 1),
                "interested_percentage": interested_pct,
                "neutral_percentage": neutral_pct,
                "not_interested_percentage": not_interested_pct,
                "transition_count": self.interest_transitions,
                "average_engagement": round(avg_eng, 3),
                "supporting_signals": supporting_evidence
            },
            
            "engagement_summary": {
                "average_engagement": round(avg_eng, 3),
                "min_engagement": round(min_eng, 3),
                "max_engagement": round(max_eng, 3),
                "timeline_samples": self.graph_samples
            },
            
            "attention_summary": {
                "average_eye_contact": round(avg_eye_contact, 3),
                "eye_contact_percentage": eye_contact_pct,
                "gaze_distribution": gaze_dist,
                "head_pose": {
                    "average_yaw": round(avg_yaw, 1),
                    "average_pitch": round(avg_pitch, 1),
                    "lean_distribution": lean_dist
                },
                "blink_statistics": {
                    "average_blink_rate": avg_blink,
                    "stress_blink_rate": "NORMAL" if avg_blink <= 24 else "ELEVATED"
                }
            },
            
            "emotion_summary": {
                "dominant_macro": dom_macro,
                "macro_distribution": macro_dist
            },
            
            "stress_summary": {
                "average_stress": round(avg_stress, 3),
                "min_stress": round(min_stress, 3),
                "max_stress": round(max_stress, 3),
                "sustained_stress_load_pct": sustained_stress_load_pct
            },
            
            "micro_expressions": {
                "total_count": me_count,
                "rate_per_min": rate_per_min,
                "average_duration_ms": avg_me_duration,
                "average_intensity": avg_me_intensity,
                "most_frequent_region": top_region,
                "events_log": self.micro_expressions
            },
            
            "interest_timeline": self.interest_timeline,
            "event_timeline": self.event_timeline,
            "feedback": None,
            "metadata": {
                "platform": "MMBI Cognitive Perception Platform",
                "version": "2.4.0",
                "pipeline": "MediaPipe FaceMesh + FACS Action Units + Optical Flow FSM + MicroLSTM + Multimodal Late Fusion",
                "disclaimer": "Model-estimated interest state based on observed behavioural signals."
            }
        }

        # Save to disk
        SessionManager.save_session(session_result)
        return session_result


class SessionManager:
    """Utility class to manage session storage and retrieval."""

    @staticmethod
    def generate_session_id() -> str:
        """Generates sequential standard ID: MMBI-2026-XXXXX"""
        year = datetime.now().year
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        files = [f for f in os.listdir(SESSIONS_DIR) if f.startswith("MMBI-") and f.endswith(".json")]
        next_seq = len(files) + 1
        return f"MMBI-{year}-{next_seq:05d}"

    @staticmethod
    def save_session(session_data: Dict[str, Any]) -> str:
        """Saves session dict to JSON file."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        session_id = session_data.get("session_id", "MMBI-UNKNOWN")
        file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        return file_path

    @staticmethod
    def load_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Loads a session by ID."""
        file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    @staticmethod
    def update_feedback(session_id: str, feedback: Dict[str, Any]) -> bool:
        """Attaches feedback to an existing session."""
        session = SessionManager.load_session(session_id)
        if not session:
            return False
        feedback["submitted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session["feedback"] = feedback
        SessionManager.save_session(session)
        return True

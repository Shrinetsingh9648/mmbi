# mmbi/engine.py  ← MASTER CLASS (entry point)
# import cv2, numpy as np
# from mmbi.capture.camera import Camera
# from mmbi.face.landmarks import LandmarkExtractor
# from mmbi.face.action_units import AUCalculator
# from mmbi.face.microexpression import MicroExpressionDetector
#
# class BehaviourEngine:
#     def __init__(self, camera_index=1):
#         self.cam   = Camera(camera_index)
#         self.lm    = LandmarkExtractor()
#         self.au    = AUCalculator()
#         self.micro = MicroExpressionDetector()
#
#     def run(self):
#         while True:
#             frame = self.cam.read()
#             if frame is None:
#                 break
#             rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             landmarks = self.lm.extract(rgb)
#             result = {'au': {}, 'micro': None}
#             if landmarks is not None:
#                 result['au']    = self.au.compute(landmarks)
#                 result['micro'] = self.micro.update(gray, result['au'])
#             self._display(frame, result)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         self.cam.release()
#         cv2.destroyAllWindows()
#
#     def _display(self, frame, result):
#         y = 30
#         for k, v in result['au'].items():
#             bar = int(min(abs(v) * 2000, 200))
#             cv2.rectangle(frame, (10, y-12), (10+bar, y), (0,200,100), -1)
#             cv2.putText(frame, f"{k}: {v:.4f}", (215, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
#             y += 18
#         if result['micro']:
#             cv2.putText(frame, f"MICRO DETECTED {result['micro']['magnitude']}",
#                         (10, y+20), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8, (0,0,255), 2)
#         cv2.imshow('MMBI Engine', frame)
#
# if __name__ == '__main__':
#     BehaviourEngine(camera_index=0).run()

import cv2
import numpy as np
import time
from mmbi.learning.collector import DataCollector  # Day 16


from mmbi.capture.camera        import Camera
from mmbi.face.landmarks        import LandmarkExtractor
from mmbi.face.action_units     import AUCalculator
from mmbi.face.microexpression  import MicroExpressionDetector
from mmbi.gaze.gaze_estimator   import GazeEstimator
from mmbi.head.head_pose        import HeadPoseEstimator
from mmbi.gaze.blink_detector   import BlinkDetector
from mmbi.temporal.buffer       import TemporalBuffer
from mmbi.models.lstm_temporal  import MicroLSTMInference
from mmbi.models.hfa_inference  import HFAInference
from mmbi.fusion.fusion_layer   import FusionLayer          # Day 13
from mmbi.fusion.engagement     import EngagementScorer     # Day 13
from mmbi.fusion.stress         import StressScorer         # Day 14
from mmbi.fusion.explainability import ExplainabilityModule # Day 15


# ── Colour palette ────────────────────────────────────────────────────────────
C_WHITE   = (255, 255, 255)
C_GREY    = (160, 160, 160)
C_GREEN   = (0,   220,  80)
C_YELLOW  = (0,   200, 255)
C_ORANGE  = (0,   140, 255)
C_RED     = (0,    30, 220)
C_CYAN    = (255, 220,   0)
C_PANEL   = (18,   18,  18)


def _score_colour(score: float) -> tuple:
    """Green → yellow → red gradient for any 0-1 score."""
    if score < 0.35:  return C_GREEN
    if score < 0.65:  return C_YELLOW
    if score < 0.80:  return C_ORANGE
    return C_RED


def _draw_bar(frame, x1, y, bar_w, score, colour, label, w_frame):
    """Draws a labelled horizontal score bar."""
    filled = int(bar_w * score)
    cv2.rectangle(frame, (x1, y - 10), (x1 + bar_w, y + 2), (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y - 10), (x1 + filled, y + 2), colour,      -1)
    cv2.putText(frame, f"{label}: {score:.2f}",
                (x1, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Engine
# ═════════════════════════════════════════════════════════════════════════════

class BehaviourEngine:

    def __init__(self, camera_index: int = 1,
                 frame_w: int = 1280, frame_h: int = 720):
        print("Initialising MMBI Engine (Days 1–15)…")

        # ── Perception stack (Days 1-12) ──────────────────────────────────────
        self.cam    = Camera(camera_index, fps=60, width=frame_w, height=frame_h)
        self.lm     = LandmarkExtractor()
        self.au     = AUCalculator()
        self.micro  = MicroExpressionDetector()
        self.gaze   = GazeEstimator()
        self.head   = HeadPoseEstimator(frame_w, frame_h)
        self.blink  = BlinkDetector(fps=60)
        self.buffer = TemporalBuffer()
        self.lstm   = MicroLSTMInference()
        self.hfa    = HFAInference()

        # ── Fusion stack (Days 13-15) ─────────────────────────────────────────
        self.fusion     = FusionLayer()
        self.engagement = EngagementScorer(fps=60)
        self.stress_sc  = StressScorer(fps=60)
        self.explainer  = ExplainabilityModule()
        self.collector = DataCollector()
        self._recorded_micro_expressions = []
        self._macro_emotion_counts = {}

        # Auto-load calibration profile if exists
        import os
        profile_path = os.path.join(os.path.dirname(__file__), 'calibration_profile.json')
        if os.path.exists(profile_path):
            try:
                import json
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                self.gaze.import_calibration(profile_data.get('gaze', {}))
                self.au.import_calibration(profile_data.get('au', {}))
                self._calibrated = True
                print(f"[Engine] Persistent calibration profile auto-loaded from {profile_path}")
            except Exception as e:
                print(f"[Engine] Warning: Could not auto-load calibration profile: {e}")


        # ── State ─────────────────────────────────────────────────────────────
        self._frame_count       = 0
        self._lstm_interval     = 30        # run LSTM every 30 frames (4×/sec)
        self._last_lstm_result  = {}
        self._last_state        = None
        self._last_eng_info     = {}
        self._last_stress_info  = {}
        self._show_explain      = False
        self._frame_times       = []
        self._calibrated        = False
        self._calibrated_show_frames = 0
        self._calibration_failed_show_frames = 0
        self._last_micro_event   = None
        self._micro_event_show_frames = 0

        print("Engine ready. C=recalibrate  E=toggle explain  S=summary  Q=quit")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Thread-safe entry point to process a single BGR video frame from any source.
        Runs the full perception, buffer, fusion, tracker, and collection layers.
        Returns the updated BehaviourState, or None if no face is detected.
        """
        if frame is None:
            return None

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_count += 1
        now = time.time()

        # Dynamic real-time FPS calculation
        self._frame_times.append(now)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        
        actual_fps = 60.0
        if len(self._frame_times) > 1:
            duration = self._frame_times[-1] - self._frame_times[0]
            if duration > 0:
                actual_fps = (len(self._frame_times) - 1) / duration

        # Dynamically feed actual FPS back to perception and scoring systems
        self.blink.fps = actual_fps
        self.blink.consec_frames = max(1, int(0.04 * actual_fps))
        self.engagement.fps = actual_fps
        self.stress_sc.fps = actual_fps
        
        # Scale prediction intervals dynamically to run every ~150ms
        self._lstm_interval = max(1, int(actual_fps * 0.15))
        self.hfa.inference_interval = max(1, int(actual_fps * 0.15))

        landmarks = self.lm.extract(rgb)

        raw_result = {
            'au': {}, 'micro': None,
            'gaze': {}, 'head': {},
            'blink': {}, 'lstm': {}, 'hfa': {}
        }

        if landmarks is not None:
            raw_result['au']    = self.au.compute(landmarks)
            self.micro.fs = actual_fps
            raw_result['micro'] = self.micro.update(gray, raw_result['au'], landmarks[:, :2] if landmarks is not None else None)
            if raw_result['micro'] and raw_result['micro'].get('spike', False):
                self._last_micro_event = raw_result['micro']
                self._micro_event_show_frames = 90  # Show for 1.5 seconds
                print(f"MICRO-EXPRESSION EVENT: {self._last_micro_event}")
            raw_result['gaze']  = self.gaze.compute(landmarks)
            raw_result['head']  = self.head.compute(landmarks)
            raw_result['blink'] = self.blink.update(landmarks)
            raw_result['hfa']   = self.hfa.update(frame, landmarks)

            # Temporal buffer
            self.buffer.push(
                raw_result['au'],
                raw_result['gaze'],
                raw_result['head'],
                raw_result['blink']
            )

            # Dynamically size the LSTM temporal window to exactly 500ms of real camera data
            lstm_window_size = max(2, int(actual_fps * 0.5))
            
            # LSTM (every _lstm_interval frames)
            if (self._frame_count % self._lstm_interval == 0
                    and self.buffer.is_ready(lstm_window_size)):
                window = self.buffer.get_window(lstm_window_size)
                self._last_lstm_result = self.lstm.predict(window)
            raw_result['lstm'] = self._last_lstm_result

            # ── Fusion (Day 13) ───────────────────────────────────────────
            state = self.fusion.fuse(
                au    = raw_result['au'],
                gaze  = raw_result['gaze'],
                head  = raw_result['head'],
                blink = raw_result['blink'],
                lstm  = raw_result['lstm'],
                hfa   = raw_result['hfa'],
                micro = raw_result['micro'] or {}
            )

            # Record completed micro-expression events for post-session expected hiding emotion analysis
            if raw_result['micro'] and raw_result['micro'].get('spike', False):
                event = {
                    'timestamp': now,
                    'region': state.micro_region,
                    'duration_ms': state.micro_duration_ms,
                    'confidence': state.micro_confidence,
                    'micro_emotion': state.micro_emotion,
                    'macro_emotion': state.macro_emotion
                }
                self._recorded_micro_expressions.append(event)

            # ── Trackers (Days 13-14) ─────────────────────────────────────
            eng_info    = self.engagement.update(
                state.engagement_score, state.engagement_label, now
            )
            stress_info = self.stress_sc.update(
                state.stress_index, state.stress_label, state.raw, now
            )

            # ── Explainability (Day 15) ───────────────────────────────────
            state = self.explainer.explain(state)

            # Cache for display when face temporarily lost
            self._last_state       = state
            self._last_eng_info    = eng_info
            self._last_stress_info = stress_info
            self.collector.update(state)
            
            # Track macro expressions for post-session report card
            macro_emo = state.macro_emotion
            self._macro_emotion_counts[macro_emo] = self._macro_emotion_counts.get(macro_emo, 0) + 1

            # Embed 2D landmarks inside the state object for client-side drawing
            state.raw['landmarks_2d'] = landmarks[:, :2].tolist()

            return state

        return None

    def run(self):
        while True:
            frame = self.cam.read()
            if frame is None:
                break

            now = time.time()
            state = self.process_frame(frame)

            if state and 'landmarks_2d' in state.raw:
                # 3D axes + attention heatmap overlays
                frame = self.head.draw_axes(frame, state.raw['head'])
                
                # Reconstruct landmarks format for heatmap overlay
                lms = np.array(state.raw['landmarks_2d'])
                lms_3d = np.zeros((len(lms), 3))
                lms_3d[:, :2] = lms
                frame = self.hfa.overlay_heatmap(frame, lms_3d, alpha=0.28)

            # ── render ────────────────────────────────────────────────────────
            self._display(frame, self._last_state,
                          self._last_eng_info, self._last_stress_info)

            # ── Key handling ──────────────────────────────────────────────────
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     break
            # elif key == ord('c') and landmarks is not None:
            #     self.gaze.calibrate(landmarks)
            #     self.au.calibrate(landmarks)
            #     self.buffer.reset()
            #     self.engagement.reset()
            #     self.stress_sc.reset()
            #     print("Recalibrated.")
            # elif key == ord('e'):
            #     self._show_explain = not self._show_explain
            #     print(f"Explanation panel: {'ON' if self._show_explain else 'OFF'}")
            # elif key == ord('s'):
            #     self._print_summary()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.collector.save_session(self._recorded_micro_expressions)
                break
            elif key == ord('c'):
                if landmarks is not None:
                    self.gaze.calibrate(landmarks)
                    self.au.calibrate(landmarks)
                    self.buffer.reset()
                    self.engagement.reset()
                    self.stress_sc.reset()
                    self._calibrated = True
                    self._calibrated_show_frames = 90
                    print("Recalibrated.")
                    
                    # Persistent profile saving
                    import json
                    import os
                    profile_data = {
                        'gaze': self.gaze.export_calibration(),
                        'au': self.au.export_calibration()
                    }
                    profile_path = os.path.join(os.path.dirname(__file__), 'calibration_profile.json')
                    with open(profile_path, 'w', encoding='utf-8') as f:
                        json.dump(profile_data, f, indent=4)
                    print(f"[Engine] Calibration profile saved to {profile_path}")
                else:
                    self._calibration_failed_show_frames = 90
                    print("Calibration failed: No face detected.")
            elif key == ord('e'):
                self._show_explain = not self._show_explain
            elif key == ord('s'):
                self._print_summary()
            elif key == ord('w'):
                self.collector.save_session()
            elif self.collector.handle_key(key):
                pass  # label key consumed



        # ── Session end ───────────────────────────────────────────────────────
        self._print_summary()
        self._display_report_card()
        self.cam.release()
        cv2.destroyAllWindows()

    # ── Display ───────────────────────────────────────────────────────────────

    def _display(self, frame, state, eng_info, stress_info):
        if frame is None:
            return

        h, w = frame.shape[:2]

        # ── Dark panel backgrounds ────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0),        (260, h),    C_PANEL, -1)   # left (wider to fit descriptions)
        cv2.rectangle(frame, (w - 320, 0),  (w, 395),    C_PANEL, -1)   # top-right (taller for card)
        cv2.rectangle(frame, (0, h - 105),  (w, h),      C_PANEL, -1)   # bottom

        # ── Bridging Header Bar ───────────────────────────────────────────────
        cv2.rectangle(frame, (260, 0), (w - 320, 32), C_PANEL, -1)
        
        # Pulsing Calibration Status dot
        pulse = (time.time() % 1.0 > 0.5)
        dot_color = (0, 220, 80) if getattr(self, '_calibrated', False) else (0, 140, 255)
        if not getattr(self, '_calibrated', False) and not pulse:
            dot_color = (0, 70, 130)  # dim when flashing off
        cv2.circle(frame, (280, 16), 5, dot_color, -1)
        
        header_text = "● MMBI COGNITIVE ENGINE - LIVE ANALYST"
        if not getattr(self, '_calibrated', False):
            header_text = "● MMBI ENGINE - UNCALIBRATED (PRESS 'C')"
        cv2.putText(frame, header_text, (295, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_WHITE, 1)

        # ════════════════════════════════════════════════════════════════
        # LEFT PANEL — AU bars (Friendly Names!)
        # ════════════════════════════════════════════════════════════════
        y = 22
        cv2.putText(frame, "FACIAL MUSCLE SIGNALS", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREY, 1)
        y += 18

        au_dict = (state.raw.get('au', {}) if state else {}) or {}
        
        # Map raw AU names to friendly descriptions for normal people
        AU_FRIENDLY = {
            'AU1':  'Concern / Inner Brow',
            'AU2':  'Surprise / Outer Brow',
            'AU4':  'Brow Furrow (Frown)',
            'AU5':  'Wide Eyes (Alert)',
            'AU6':  'Cheek Raise (Squint)',
            'AU12': 'Smile (Lip Pull)',
            'AU15': 'Frown (Lip Drop)',
            'AU17': 'Chin Tension',
            'AU25': 'Lips Parted',
            'AU26': 'Jaw Drop'
        }
        au_names = ['AU1','AU2','AU4','AU5','AU6',
                    'AU12','AU15','AU17','AU25','AU26']

        for name in au_names:
            v     = float(au_dict.get(name, 0.0))
            # Draw smaller bar (width 80px) to fit friendly text next to it
            bar   = int(min(abs(v) * 2000, 80))
            colour = C_GREEN if v >= 0 else C_RED
            active = name in (state.active_aus if state else [])
            
            # Active background highlight
            if active:
                cv2.rectangle(frame, (10, y - 9), (250, y + 4), (40, 50, 40) if v >= 0 else (40, 40, 50), -1)
                
            # Draw bar
            cv2.rectangle(frame, (10, y - 9), (10 + bar, y + 4), colour, -1)
            
            # Draw friendly text
            friendly_name = AU_FRIENDLY.get(name, name)
            cv2.putText(frame, friendly_name, (96, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        C_YELLOW if active else C_WHITE, 1)
                        
            # Draw numerical deviation value
            cv2.putText(frame, f"{v:+.3f}", (215, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_GREY, 1)
            y += 17

        # Buffer fill bar at bottom of AU panel
        if self.buffer:
            fill = self.buffer.current_frame_count / 600
            bw   = int(240 * fill)
            cv2.rectangle(frame, (10, y + 4), (250, y + 10), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, y + 4), (10 + bw, y + 10), C_CYAN, -1)
            cv2.putText(frame, f"Calibration Buffer: {int(fill*100)}%", (10, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GREY, 1)

        if state is None:
            cv2.imshow('MMBI Engine — Days 1-15', frame)
            return

        # ════════════════════════════════════════════════════════════════
        # TOP CENTRE — Engagement + Stress bars
        # ════════════════════════════════════════════════════════════════
        cx = w // 2
        bar_x = cx - 150

        # Engagement bar (moved down 25px)
        eng_col = _score_colour(1.0 - state.engagement_score)  # green=high, red=low
        eng_col = C_GREEN if state.engagement_score > 0.65 else \
                  C_YELLOW if state.engagement_score > 0.45 else C_RED
        _draw_bar(frame, bar_x, 48, 300,
                  state.engagement_score, eng_col,
                  f"ENGAGEMENT [{state.engagement_label}]", w)

        # Stress bar (moved down 25px)
        st_col = _score_colour(state.stress_index)
        _draw_bar(frame, bar_x, 78, 300,
                  state.stress_index, st_col,
                  f"STRESS [{state.stress_label}]", w)

        # Trend indicators
        eng_trend   = eng_info.get('trend', 'STABLE')
        stress_trend= stress_info.get('trend', 'STABLE')
        trend_sym   = {'RISING': '▲', 'FALLING': '▼', 'STABLE': '—'}
        cv2.putText(frame,
                    f"eng {trend_sym.get(eng_trend, '—')}  "
                    f"str {trend_sym.get(stress_trend, '—')}",
                    (bar_x, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREY, 1)

        # Session averages
        cv2.putText(frame,
                    f"sess avg  eng={eng_info.get('session_avg',0):.2f}  "
                    f"str={stress_info.get('session_avg',0):.2f}",
                    (bar_x, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)

        # ════════════════════════════════════════════════════════════════
        # TOP RIGHT — Gaze + Head + Blink
        # ════════════════════════════════════════════════════════════════
        rx = w - 315
        ry = 22

        # Gaze
        gaze_col = C_GREEN if state.attention_zone == 'CENTER' else C_ORANGE
        cv2.putText(frame,
                    f"Gaze: {state.attention_zone}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.60, gaze_col, 2)
        ry += 22
        cv2.putText(frame,
                    f"Eye contact: {state.eye_contact_score:.2f}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1)
        ry += 18

        # Head pose
        raw = state.raw or {}
        h_dict = raw.get('head', {})
        cv2.putText(frame,
                    f"Y:{h_dict.get('yaw',0):+.1f}  "
                    f"P:{h_dict.get('pitch',0):+.1f}  "
                    f"R:{h_dict.get('roll',0):+.1f}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1)
        ry += 18
        lean_col = C_GREEN if state.lean_signal == 'FORWARD' else \
                   C_ORANGE if state.lean_signal == 'AWAY'   else C_WHITE
        cv2.putText(frame,
                    f"Lean: {state.lean_signal}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.54, lean_col, 2)
        ry += 22

        # Blink
        b_dict  = raw.get('blink', {})
        bl_col  = C_RED if state.blink_stress == 'HIGH_STRESS' else \
                  C_ORANGE if state.blink_stress == 'ELEVATED'  else C_WHITE
        cv2.putText(frame,
                    f"Blink: {state.blink_rate}/min  EAR={b_dict.get('ear',0):.3f}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.44, bl_col, 1)
        ry += 18
        cv2.putText(frame,
                    f"Blink stress: {state.blink_stress}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.44, bl_col, 1)
        ry += 22

        # ════════════════════════════════════════════════════════════════
        # PREMIUM LIVE EMOTION CARD (DUAL READOUT MACRO & MICRO)
        # ════════════════════════════════════════════════════════════════
        # Dynamic color & label mapping for universal emotions
        emotion_map = {
            'happiness': ( (0, 200, 255),   "JOYFUL / HAPPY" ),
            'sadness':   ( (255, 120, 0),   "SAD / DISTRESSED" ),
            'anger':     ( (0, 0, 220),     "ANGRY / FURIOUS" ),
            'surprise':  ( (255, 220, 0),   "SURPRISED / AMAZED" ),
            'fear':      ( (240, 32, 160),  "ANXIOUS / FEARFUL" ),
            'disgust':   ( (0, 128, 0),     "DISGUSTED" ),
            'contempt':  ( (0, 140, 255),   "CONTEMPTUOUS" ),
            'neutral':   ( (160, 160, 160), "NEUTRAL / RESTING" )
        }
        
        mac_color, mac_label = emotion_map.get(state.macro_emotion, ((160,160,160), state.macro_emotion.upper()))
        
        card_x1 = w - 310
        card_y1 = 145
        card_x2 = w - 10
        card_y2 = 355
        
        # Border color changes dynamically
        is_micro_active = (state.micro_emotion != 'none')
        if is_micro_active:
            mic_color, mic_label = emotion_map.get(state.micro_emotion, ((0,140,255), state.micro_emotion.upper()))
            border_color = mic_color
        else:
            border_color = mac_color
            
        # Draw background and dynamic border
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (25, 25, 25), -1)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), border_color, 1)
        
        # ── 1. Top Section: Macro Expression ──────────────────────────
        cv2.putText(frame, "MACRO (ongoing baseline)", (card_x1 + 15, card_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GREY, 1)
        cv2.putText(frame, f"[ {mac_label} ]", (card_x1 + 15, card_y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, mac_color, 2)
        cv2.putText(frame, f"Conf: {int(state.macro_confidence * 100)}% ({state.macro_source.upper()})",
                    (card_x1 + 15, card_y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1)
                    
        # Macro sleek progress bar
        cv2.rectangle(frame, (card_x1 + 15, card_y1 + 75), (card_x2 - 15, card_y1 + 80), (50, 50, 50), -1)
        bar_w = (card_x2 - 15) - (card_x1 + 15)
        filled_mac = int(bar_w * state.macro_confidence)
        cv2.rectangle(frame, (card_x1 + 15, card_y1 + 75), (card_x1 + 15 + filled_mac, card_y1 + 80), mac_color, -1)
        
        # Sleek separator line
        cv2.line(frame, (card_x1 + 10, card_y1 + 93), (card_x2 - 10, card_y1 + 93), (45, 45, 45), 1)
        
        # ── 2. Bottom Section: Micro Expression ───────────────────────
        cv2.putText(frame, "MICRO (transient twitch)", (card_x1 + 15, card_y1 + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GREY, 1)
                    
        if is_micro_active:
            # Highlight micro expression in card
            cv2.rectangle(frame, (card_x1 + 8, card_y1 + 97), (card_x2 - 8, card_y2 - 8), (20, 20, 45), -1)
            cv2.rectangle(frame, (card_x1 + 8, card_y1 + 97), (card_x2 - 8, card_y2 - 8), border_color, 1)
            # Re-draw the labels over the highlight
            cv2.putText(frame, "MICRO (transient twitch)", (card_x1 + 15, card_y1 + 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_YELLOW, 1)
            cv2.putText(frame, f"[ {mic_label} ]", (card_x1 + 15, card_y1 + 137),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, mic_color, 2)
            cv2.putText(frame, f"Reg: {state.micro_region.upper()} | Dur: {state.micro_duration_ms:.0f}ms",
                        (card_x1 + 15, card_y1 + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1)
                        
            # Duration progress bar (relative to 500ms max)
            cv2.rectangle(frame, (card_x1 + 15, card_y1 + 172), (card_x2 - 15, card_y1 + 177), (50, 50, 50), -1)
            dur_ratio = min(1.0, state.micro_duration_ms / 500.0)
            filled_mic = int(bar_w * dur_ratio)
            cv2.rectangle(frame, (card_x1 + 15, card_y1 + 172), (card_x1 + 15 + filled_mic, card_y1 + 177), mic_color, -1)
        else:
            cv2.putText(frame, "[ LISTENING / MONITORING ]", (card_x1 + 15, card_y1 + 137),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)
            cv2.putText(frame, "Signal telemetry: ACTIVE", (card_x1 + 15, card_y1 + 157),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GREEN, 1)
            
            # Oscilloscope signal telemetry wave (pulsing green sine wave)
            t_pulse = time.time()
            for i in range(12):
                wave_h = int(8 + 10 * np.sin(t_pulse * 12 + i * 0.8))
                x_pos = card_x1 + 18 + i * 22
                cv2.line(frame, (x_pos, card_y1 + 195), (x_pos, card_y1 + 195 - wave_h), C_GREEN, 2)

        # Sustained stress warning (moved below card)
        sustained = stress_info.get('sustained_stress_secs', 0)
        if sustained > 5:
            cv2.putText(frame,
                        f"⚠ Stress sustained {sustained:.0f}s",
                        (rx, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_RED, 2)

        # ════════════════════════════════════════════════════════════════
        # BOTTOM BAR — Natural language + conflict flags + stress drivers (Shifted x to 270)
        # ════════════════════════════════════════════════════════════════
        exp = state.explanation or {}
        nl  = exp.get('natural_language', '')
        if nl:
            cv2.putText(frame, nl,
                        (270, h - 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1)

        # Conflict flags (first one, red)
        flags = exp.get('conflict_flags', [])
        if flags:
            flag_text = f"⚑ {flags[0][:90]}"
            cv2.putText(frame, flag_text,
                        (270, h - 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ORANGE, 1)

        # Stress drivers (first two)
        drivers = stress_info.get('drivers', [])
        for i, drv in enumerate(drivers[:2]):
            cv2.putText(frame, f"• {drv[:95]}",
                        (270, h - 42 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GREY, 1)

        # Micro spike
        micro = raw.get('micro', {}) or {}
        
        # 1. Live FSM State & Local Motion Vector tracker rendering (Dynamic GUI updates!)
        if not micro.get('spike', False) and 'state' in micro:
            state_text = f"Micro State: {micro['state'].upper()}"
            if micro['state'] != "Baseline":
                # Draw dynamic state tracking dot
                pulse = (time.time() % 0.4 > 0.2)
                dot_color = C_RED if pulse else (0, 0, 180)
                cv2.circle(frame, (285, h - 14), 4, dot_color, -1)
                cv2.putText(frame, state_text, (295, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_YELLOW, 1)
            else:
                cv2.putText(frame, state_text, (270, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GREY, 1)
                            
        # 2. Prominent Micro-Expression Alarm Overlay (Segmented Event Telemetry!)
        if getattr(self, '_micro_event_show_frames', 0) > 0 and getattr(self, '_last_micro_event', None) is not None:
            self._micro_event_show_frames -= 1
            ev = self._last_micro_event
            
            # Draw semi-transparent alarm banner
            alarm_overlay = frame.copy()
            cx = w // 2
            x1, y1, x2, y2 = cx - 290, h - 150, cx + 290, h - 110
            cv2.rectangle(alarm_overlay, (x1, y1), (x2, y2), (10, 10, 40), -1)
            cv2.addWeighted(alarm_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_RED, 2)
            
            # Format: "⚠️ MICRO SPIKE: MOUTH TWITCH (220ms) Mag=0.21"
            region_name = ev.get('region', 'face').replace('_', ' ').upper()
            duration = ev.get('duration_ms', 0.0)
            mag = ev.get('magnitude', 0.0)
            msg = f"MICRO SPIKE: {region_name} TWITCH ({duration:.0f}ms) Mag={mag:.3f}"
            cv2.putText(frame, msg, (cx - 275, h - 124),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WHITE, 2)

        # ════════════════════════════════════════════════════════════════
        # EXPLANATION PANEL (toggle with E key, shifted x to 270)
        # ════════════════════════════════════════════════════════════════
        if self._show_explain:
            chain  = exp.get('inference_chain', [])
            conf_b = exp.get('confidence_breakdown', {})

            # Semi-transparent dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (270, 105), (w - 325, 105 + 20 + len(chain) * 17 + 80),
                          (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

            ey = 122
            cv2.putText(frame, "INFERENCE CHAIN",
                        (275, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1)
            ey += 14
            for step in chain:
                cv2.putText(frame, step[:95],
                            (275, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_WHITE, 1)
                ey += 16

            # Confidence breakdown
            ey += 6
            cv2.putText(frame, "CONFIDENCE BREAKDOWN",
                        (275, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
            ey += 14
            breakdown_line = "  ".join(
                [f"{k}: {v}%" for k, v in conf_b.items()]
            )
            cv2.putText(frame, breakdown_line[:110],
                        (275, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_GREY, 1)
            # Collector status bar
            cv2.putText(frame, self.collector.status(),
                        (270, h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (0, 180, 255), 1)

        # ════════════════════════════════════════════════════════════════
        # PULSING CALIBRATION NOTIFICATION OVERLAY
        # ════════════════════════════════════════════════════════════════
        # Pulsing banner if uncalibrated
        if not getattr(self, '_calibrated', False):
            cx = w // 2
            cy = h // 2
            x1 = cx - 300
            y1 = cy - 45
            x2 = cx + 300
            y2 = cy + 45
            
            # semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            pulse_color = (0, 140, 255) if pulse else (0, 60, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), pulse_color, 2)
            
            cv2.putText(frame, "⚠️ SYSTEM UNCALIBRATED", (x1 + 30, y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)
            cv2.putText(frame, "Look neutral and press 'C' to calibrate your face.", (x1 + 30, y1 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1)
                        
        # Temporary calibration successful overlay
        elif getattr(self, '_calibrated_show_frames', 0) > 0:
            self._calibrated_show_frames -= 1
            cx = w // 2
            cy = h // 2
            x1 = cx - 250
            y1 = cy - 35
            x2 = cx + 250
            y2 = cy + 35
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 25, 10), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
            
            cv2.putText(frame, "● CALIBRATION SUCCESSFUL ●", (x1 + 45, y1 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 80), 2)
                        
        # Temporary calibration failed overlay
        elif getattr(self, '_calibration_failed_show_frames', 0) > 0:
            self._calibration_failed_show_frames -= 1
            cx = w // 2
            cy = h // 2
            x1 = cx - 250
            y1 = cy - 35
            x2 = cx + 250
            y2 = cy + 35
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 25), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
            
            cv2.putText(frame, "❌ CALIBRATION FAILED: NO FACE DETECTED", (x1 + 20, y1 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 2)

        cv2.imshow('MMBI Engine — Days 1-15', frame)

    # ── Console summary ───────────────────────────────────────────────────────

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        eng_sum = self.engagement.session_summary()
        print(f"Engagement avg:  {eng_sum.get('avg_engagement', 0):.3f}")
        print(f"Engagement max:  {eng_sum.get('max_engagement', 0):.3f}")
        print(f"Engagement min:  {eng_sum.get('min_engagement', 0):.3f}")
        print(f"Notable events:  {eng_sum.get('total_events', 0)}")

        str_sum = self.stress_sc.session_summary()
        print(f"Stress avg:      {str_sum.get('avg_stress', 0):.3f}")
        print(f"Stress peak:     {str_sum.get('peak_stress', 0):.3f}")
        print(f"High stress %:   {str_sum.get('high_stress_pct', 0) * 100:.1f}%")
        print(f"Max sustained:   {str_sum.get('max_sustained_secs', 0):.1f}s")
        print(f"Duration:        {eng_sum.get('duration_secs', 0):.1f}s")
        print("=" * 60)

    def _display_report_card(self):
        import textwrap
        
        w, h = 800, 600
        report = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background (glassmorphism dark theme)
        cv2.rectangle(report, (0, 0), (w, h), (18, 18, 18), -1)
        
        # Outer double border
        cv2.rectangle(report, (15, 15), (w - 15, h - 15), (45, 45, 45), 2)
        cv2.rectangle(report, (10, 10), (w - 10, h - 10), (255, 140, 0), 1) # Orange outer glow
        
        # Title Block
        cv2.putText(report, "MMBI COGNITIVE ENGINE", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
        cv2.putText(report, "POST-SESSION DEEP ANALYST REPORT", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)
        cv2.line(report, (40, 95), (w - 40, 95), (45, 45, 45), 2)
        
        # Session Statistics Column (Left, x from 40 to 360)
        cv2.putText(report, "SESSION STATISTICS", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 80), 1)
        
        eng_sum = self.engagement.session_summary()
        str_sum = self.stress_sc.session_summary()
        
        total_macro_frames = sum(self._macro_emotion_counts.values()) if self._macro_emotion_counts else 1
        sorted_macros = sorted(self._macro_emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        dom_macro_text = "NEUTRAL (100%)"
        sec_macro_text = "NONE"
        if sorted_macros:
            dom_m, dom_c = sorted_macros[0]
            dom_macro_text = f"{dom_m.upper()} ({int((dom_c / total_macro_frames)*100)}%)"
            if len(sorted_macros) > 1:
                sec_m, sec_c = sorted_macros[1]
                sec_macro_text = f"{sec_m.upper()} ({int((sec_c / total_macro_frames)*100)}%)"
        
        stats = [
            f"Session Duration:  {eng_sum.get('duration_secs', 0):.1f} seconds",
            f"Total Frames:      {self._frame_count} frames",
            f"Engagement Avg:    {eng_sum.get('avg_engagement', 0):.3f}",
            f"Stress Average:    {str_sum.get('avg_stress', 0):.3f}",
            f"Micro Twitches:    {len(self._recorded_micro_expressions)} detected",
            f"Dominant Macro:    {dom_macro_text}",
            f"Secondary Macro:   {sec_macro_text}"
        ]
        sy = 160
        for stat in stats:
            cv2.putText(report, stat, (40, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)
            sy += 25
            
        # Vertical divider line
        cv2.line(report, (370, 115), (370, 340), (45, 45, 45), 1)
        
        # Micro-Expression Log Column (Right, x from 390 to 760)
        cv2.putText(report, "MICRO-EXPRESSION EVENTS LOG", (390, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 80), 1)
        
        if not self._recorded_micro_expressions:
            cv2.putText(report, "No transient twitches captured.", (390, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)
        else:
            my = 160
            for i, ev in enumerate(self._recorded_micro_expressions[:7]):
                reg = ev['region'].replace('_', ' ').upper()
                dur = ev['duration_ms']
                emo = ev['micro_emotion'].upper()
                text = f"{i+1}. {reg} ({emo}) - {dur:.0f}ms"
                cv2.putText(report, text, (390, my), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 220, 0), 1)
                my += 24
                
        # Suppressed Emotion Analysis Block (y from 350 to 520)
        cv2.line(report, (40, 350), (w - 40, 350), (45, 45, 45), 2)
        
        # Aggregate suppressed emotions: Any involuntary micro-expression that is not neutral/none represents a transient leak!
        suppressed_counts = {}
        for ev in self._recorded_micro_expressions:
            mic_emo = ev['micro_emotion']
            if mic_emo not in ['neutral', 'none']:
                suppressed_counts[mic_emo] = suppressed_counts.get(mic_emo, 0) + 1
                
        hidden_emotion = "NONE DETECTED"
        hidden_color = (0, 220, 80) # green
        hidden_reason = "All involuntary micro-expressions were aligned with your baseline neutral facial state. No suppressed or hidden emotions were leaked during this session."
        
        if suppressed_counts:
            hidden_emotion = max(suppressed_counts, key=suppressed_counts.get).upper()
            hidden_color = (0, 30, 220) # Bright Red/Orange
            hidden_reason = f"Leaks of involuntary micro-expression twitches caught by the sparse optical flow indicate a brief, transient state of [ {hidden_emotion} ] being leaked. The rapid duration suggests a suppressed or repressed emotional reaction."
            
        # Draw analysis sub-card
        cv2.rectangle(report, (40, 370), (w - 40, 520), (25, 25, 25), -1)
        cv2.rectangle(report, (40, 370), (w - 40, 520), hidden_color, 1)
        
        cv2.putText(report, "INTELLIGENT REPRESSED / SUPPRESSED EMOTION ANALYSIS", (55, 395),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
        
        cv2.putText(report, f"EXPECTED HIDDEN EMOTION: [ {hidden_emotion} ]", (55, 432),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, hidden_color, 2)
                    
        lines = textwrap.wrap(hidden_reason, width=82)
        ly = 465
        for line in lines:
            cv2.putText(report, line, (55, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1)
            ly += 18
            
        # Footer
        cv2.putText(report, "Press ANY KEY on your keyboard to close the analyst card and exit...", (40, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
                    
        cv2.imshow('MMBI Engine Report Card', report)
        cv2.waitKey(0)


if __name__ == '__main__':
    import sys

    # Auto-detect first working camera index
    def find_camera() -> int:
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"Using camera index {i}")
                    return i
        print("ERROR: No working camera found. Plug in a webcam and try again.")
        sys.exit(1)

    # Allow override: python -m mmbi.engine 0
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = find_camera()

    BehaviourEngine(camera_index=idx).run()
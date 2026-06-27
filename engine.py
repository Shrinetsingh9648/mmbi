import cv2
import numpy as np
import time
import sys

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
from mmbi.fusion.fusion_layer   import FusionLayer
from mmbi.fusion.engagement     import EngagementScorer
from mmbi.fusion.stress         import StressScorer
from mmbi.fusion.explainability import ExplainabilityModule
from mmbi.learning.collector    import DataCollector
from mmbi.dashboard.data_bridge import DataBridge

C_WHITE  = (255, 255, 255)
C_GREY   = (160, 160, 160)
C_GREEN  = (0,   220,  80)
C_YELLOW = (0,   200, 255)
C_ORANGE = (0,   140, 255)
C_RED    = (0,    30, 220)
C_CYAN   = (255, 220,   0)
C_PANEL  = (18,   18,  18)


def _eng_colour(score):
    if score > 0.65: return C_GREEN
    if score > 0.45: return C_YELLOW
    return C_RED


def _stress_colour(score):
    if score < 0.20: return C_GREEN
    if score < 0.45: return C_YELLOW
    if score < 0.70: return C_ORANGE
    return C_RED


def _draw_bar(frame, x1, y, bar_w, score, colour, label):
    filled = int(bar_w * max(0.0, min(1.0, score)))
    cv2.rectangle(frame, (x1, y-10), (x1+bar_w, y+2), (50,50,50), -1)
    cv2.rectangle(frame, (x1, y-10), (x1+filled, y+2), colour, -1)
    cv2.putText(frame, f"{label}: {score:.2f}",
                (x1, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1)


class BehaviourEngine:

    def __init__(self, camera_index=0, frame_w=1280, frame_h=720):
        print("Initialising MMBI Engine (Days 1-20)...")

        self.cam    = Camera(camera_index)
        self.lm     = LandmarkExtractor()
        self.au     = AUCalculator()
        self.micro  = MicroExpressionDetector()
        self.gaze   = GazeEstimator()
        self.head   = HeadPoseEstimator(frame_w, frame_h)
        self.blink  = BlinkDetector(fps=30)
        self.buffer = TemporalBuffer()
        self.lstm   = MicroLSTMInference()
        self.hfa    = HFAInference()

        self.fusion     = FusionLayer()
        self.engagement = EngagementScorer(fps=30)
        self.stress_sc  = StressScorer(fps=30)
        self.explainer  = ExplainabilityModule()
        self.collector  = DataCollector()
        self.bridge     = DataBridge()

        self._frame_count      = 0
        self._lstm_interval    = 60
        self._last_lstm_result = {}
        self._last_hfa_result  = {}
        self._last_state       = None
        self._last_eng_info    = {}
        self._last_stress_info = {}
        self._show_explain     = False

        print("Engine ready.")
        print("  1=Engaged  2=Neutral  3=Disengaged  4=Stressed  5=Calm")
        print("  W=Save  E=Explain  S=Summary  C=Recalibrate  Q=Quit")

    def run(self):
        while True:
            frame = self.cam.read()
            if frame is None:
                break

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._frame_count += 1
            now = time.time()

            landmarks = self.lm.extract(rgb)

            raw_result = {
                'au': {}, 'micro': None,
                'gaze': {}, 'head': {},
                'blink': {}, 'lstm': {}, 'hfa': {}
            }

            if landmarks is not None:
                raw_result['au'] = self.au.compute(landmarks)

                if self._frame_count % 4 == 0:
                    raw_result['micro'] = self.micro.update(
                        gray, raw_result['au'])
                else:
                    raw_result['micro'] = None

                raw_result['gaze']  = self.gaze.compute(landmarks)
                raw_result['head']  = self.head.compute(landmarks)
                raw_result['blink'] = self.blink.update(landmarks)

                if self._frame_count % 4 == 0:
                    self._last_hfa_result = self.hfa.update(
                        frame, landmarks)
                raw_result['hfa'] = self._last_hfa_result

                self.buffer.push(
                    raw_result['au'],
                    raw_result['gaze'],
                    raw_result['head'],
                    raw_result['blink']
                )

                if (self._frame_count % self._lstm_interval == 0
                        and self.buffer.is_ready()):
                    window = self.buffer.get_micro_window()
                    self._last_lstm_result = self.lstm.predict(window)
                raw_result['lstm'] = self._last_lstm_result

                frame = self.head.draw_axes(frame, raw_result['head'])
                if self._frame_count % 4 == 0:
                    frame = self.hfa.overlay_heatmap(
                        frame, landmarks, alpha=0.25)

                state = self.fusion.fuse(
                    au    = raw_result['au'],
                    gaze  = raw_result['gaze'],
                    head  = raw_result['head'],
                    blink = raw_result['blink'],
                    lstm  = raw_result['lstm'],
                    hfa   = raw_result['hfa'],
                    micro = raw_result['micro'] or {}
                )

                eng_info = self.engagement.update(
                    state.engagement_score, state.engagement_label, now)
                stress_info = self.stress_sc.update(
                    state.stress_index, state.stress_label, state.raw, now)

                state = self.explainer.explain(state)
                self.collector.update(state)
                self.bridge.write(state, eng_info, stress_info)

                self._last_state       = state
                self._last_eng_info    = eng_info
                self._last_stress_info = stress_info

            self._display(frame, self._last_state,
                          self._last_eng_info, self._last_stress_info)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.collector.save_session()
                self.bridge.clear()
                break
            elif key == ord('c') and landmarks is not None:
                self.gaze.calibrate(landmarks)
                self.au.calibrate(landmarks)
                self.buffer.reset()
                self.engagement.reset()
                self.stress_sc.reset()
                print("Recalibrated.")
            elif key == ord('e'):
                self._show_explain = not self._show_explain
                print(f"Explain: {'ON' if self._show_explain else 'OFF'}")
            elif key == ord('s'):
                self._print_summary()
            elif key == ord('w'):
                self.collector.save_session()
            elif self.collector.handle_key(key):
                pass

        self._print_summary()
        self.cam.release()
        cv2.destroyAllWindows()

    def _display(self, frame, state, eng_info, stress_info):
        if frame is None:
            return

        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0),     (200, h),  C_PANEL, -1)
        cv2.rectangle(frame, (w-320, 0), (w, 280),  C_PANEL, -1)
        cv2.rectangle(frame, (0, h-110), (w, h),    C_PANEL, -1)

        y = 20
        cv2.putText(frame, "ACTION UNITS", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)
        y += 14

        au_dict  = (state.raw.get('au', {}) if state else {}) or {}
        au_names = ['AU1','AU2','AU4','AU5','AU6',
                    'AU12','AU15','AU17','AU25','AU26']

        for name in au_names:
            v      = float(au_dict.get(name, 0.0))
            bar    = int(min(abs(v) * 2000, 160))
            col    = C_GREEN if v >= 0 else C_RED
            active = name in (state.active_aus if state else [])
            if active:
                cv2.rectangle(frame, (8, y-9), (172, y+3), (40,60,40), -1)
            cv2.rectangle(frame, (8, y-9), (8+bar, y+3), col, -1)
            cv2.putText(frame, f"{name} {v:+.3f}", (170, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        C_YELLOW if active else C_WHITE, 1)
            y += 15

        fill = self.buffer.current_frame_count / 600
        bw   = int(180 * fill)
        cv2.rectangle(frame, (8, y+4), (188, y+12), (40,40,40), -1)
        cv2.rectangle(frame, (8, y+4), (8+bw, y+12), C_CYAN, -1)
        cv2.putText(frame, f"buf {int(fill*100)}%", (8, y+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_GREY, 1)

        if state is None:
            cv2.putText(frame, "No face detected",
                        (w//2-80, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GREY, 1)
            cv2.putText(frame, self.collector.status(),
                        (205, h-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0,180,255), 1)
            cv2.imshow('MMBI Engine', frame)
            return

        cx    = w // 2
        bar_x = cx - 150

        _draw_bar(frame, bar_x, 22, 300,
                  state.engagement_score,
                  _eng_colour(state.engagement_score),
                  f"ENGAGEMENT [{state.engagement_label}]")

        _draw_bar(frame, bar_x, 55, 300,
                  state.stress_index,
                  _stress_colour(state.stress_index),
                  f"STRESS [{state.stress_label}]")

        trend_sym = {'RISING': '^ ', 'FALLING': 'v ', 'STABLE': '- '}
        eng_t = eng_info.get('trend', 'STABLE')
        str_t = stress_info.get('trend', 'STABLE')
        cv2.putText(frame,
                    f"eng {trend_sym.get(eng_t,'- ')} "
                    f"str {trend_sym.get(str_t,'- ')}",
                    (bar_x, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)
        cv2.putText(frame,
                    f"avg eng={eng_info.get('session_avg',0):.2f}  "
                    f"str={stress_info.get('session_avg',0):.2f}",
                    (bar_x, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GREY, 1)

        rx = w - 315
        ry = 22

        gaze_col = C_GREEN if state.attention_zone == 'CENTER' else C_ORANGE
        cv2.putText(frame, f"Gaze: {state.attention_zone}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.56, gaze_col, 2)
        ry += 20
        cv2.putText(frame, f"Eye contact: {state.eye_contact_score:.2f}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1)
        ry += 18

        raw    = state.raw or {}
        h_dict = raw.get('head', {})
        cv2.putText(frame,
                    f"Y:{h_dict.get('yaw',0):+.1f} "
                    f"P:{h_dict.get('pitch',0):+.1f} "
                    f"R:{h_dict.get('roll',0):+.1f}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_WHITE, 1)
        ry += 18

        lean_col = C_GREEN  if state.lean_signal == 'FORWARD' else \
                   C_ORANGE if state.lean_signal == 'AWAY'    else C_WHITE
        cv2.putText(frame, f"Lean: {state.lean_signal}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.50, lean_col, 2)
        ry += 22

        b_dict = raw.get('blink', {})
        bl_col = C_RED    if state.blink_stress == 'HIGH_STRESS' else \
                 C_ORANGE if state.blink_stress == 'ELEVATED'    else C_WHITE
        cv2.putText(frame,
                    f"Blink: {state.blink_rate}/min  "
                    f"EAR={b_dict.get('ear',0):.3f}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.40, bl_col, 1)
        ry += 17
        cv2.putText(frame, f"Blink stress: {state.blink_stress}",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.40, bl_col, 1)
        ry += 22

        em_col = C_CYAN if state.emotion_confidence > 0.60 else C_GREY
        cv2.putText(frame,
                    f"{state.dominant_emotion.upper()} "
                    f"{state.emotion_confidence:.2f} "
                    f"[{state.emotion_source}]",
                    (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.54, em_col, 2)
        ry += 22

        sustained = stress_info.get('sustained_stress_secs', 0)
        if sustained > 5:
            cv2.putText(frame,
                        f"! Stress {sustained:.0f}s sustained",
                        (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_RED, 2)

        exp     = state.explanation or {}
        nl      = exp.get('natural_language', '')
        flags   = exp.get('conflict_flags', [])
        drivers = stress_info.get('drivers', [])

        if nl:
            cv2.putText(frame, nl,
                        (205, h-90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_WHITE, 1)
        if flags:
            cv2.putText(frame, f"! {flags[0][:90]}",
                        (205, h-72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_ORANGE, 1)
        for i, drv in enumerate(drivers[:2]):
            cv2.putText(frame, f". {drv[:90]}",
                        (205, h-54+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GREY, 1)

        micro = raw.get('micro', {}) or {}
        mag   = micro.get('magnitude', 0)
        if mag and float(mag) > 0.015:
            cv2.putText(frame,
                        f"MICRO SPIKE {float(mag):.4f}",
                        (205, h-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, C_RED, 2)

        cv2.putText(frame, self.collector.status(),
                    (205, h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,180,255), 1)

        if self._show_explain:
            chain  = exp.get('inference_chain', [])
            conf_b = exp.get('confidence_breakdown', {})
            panel_h = 30 + len(chain) * 14 + 35
            overlay = frame.copy()
            cv2.rectangle(overlay, (205, 105), (w-325, 105+panel_h),
                          (10,10,10), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            ey = 120
            cv2.putText(frame, "INFERENCE CHAIN",
                        (210, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
            ey += 13
            for step in chain:
                cv2.putText(frame, step[:93],
                            (210, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_WHITE, 1)
                ey += 13
            ey += 4
            cv2.putText(frame, "BREAKDOWN",
                        (210, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_CYAN, 1)
            ey += 13
            bd = "  ".join([f"{k}:{v}%" for k, v in conf_b.items()])
            cv2.putText(frame, bd[:105],
                        (210, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_GREY, 1)

        cv2.imshow('MMBI Engine', frame)

    def _print_summary(self):
        print("\n" + "=" * 55)
        print("SESSION SUMMARY")
        print("=" * 55)
        e = self.engagement.session_summary()
        s = self.stress_sc.session_summary()
        print(f"Duration:         {e.get('duration_secs',0):.1f}s")
        print(f"Engagement avg:   {e.get('avg_engagement',0):.3f}")
        print(f"Engagement max:   {e.get('max_engagement',0):.3f}")
        print(f"Engagement min:   {e.get('min_engagement',0):.3f}")
        print(f"Notable events:   {e.get('total_events',0)}")
        print(f"Stress avg:       {s.get('avg_stress',0):.3f}")
        print(f"Stress peak:      {s.get('peak_stress',0):.3f}")
        print(f"High stress %:    {s.get('high_stress_pct',0)*100:.1f}%")
        print(f"Max sustained:    {s.get('max_sustained_secs',0):.1f}s")
        print(f"Collected frames: {self.collector.frame_count}")
        print("=" * 55)


if __name__ == '__main__':

    def find_camera():
        for i in range(5):
            cap = cv2.VideoCapture(i) # cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"Using camera index {i}")
                    return i
        print("ERROR: No camera found.")
        sys.exit(1)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else find_camera()
    BehaviourEngine(camera_index=idx).run()



"""
MMBI Test Suite
================
Tests:
  1. Multimodal InterestClassifier (INTERESTED, NEUTRAL, NOT INTERESTED)
  2. MicroExpressionDetector timestamp preservation and event formatting
  3. SessionRecorder metrics aggregation, durations, percentages, and transitions
  4. SessionManager JSON persistence and user feedback updating
  5. Standalone PDF Report generation via ReportLab & Matplotlib
  6. Recorded video processing pipeline with synthetic MP4
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmbi.fusion.interest_classifier import InterestClassifier
from mmbi.face.microexpression import MicroExpressionDetector
from mmbi.session_manager import SessionRecorder, SessionManager, format_timestamp
from mmbi.report_generator import generate_pdf_report
from mmbi.video_processor import VideoProcessor
from mmbi.fusion.fusion_layer import BehaviourState


class TestInterestClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = InterestClassifier(smoothing_alpha=0.99)  # high alpha for immediate response in unit test

    def test_interested_classification(self):
        """High eye contact, forward lean, high engagement, positive AUs -> INTERESTED"""
        state, score, conf, signals = self.clf.compute(
            engagement_score=0.85,
            stress_index=0.15,
            eye_contact_score=0.92,
            attention_zone="CENTER",
            head_lean="FORWARD",
            head_yaw=2.0,
            head_pitch=-5.0,
            blink_rate=14,
            blink_stress="NORMAL",
            active_aus=["AU6", "AU12"],
            dominant_emotion="happiness",
            macro_emotion="happiness",
            au_dict={"AU6": 0.03, "AU12": 0.04}
        )
        self.assertEqual(state, "INTERESTED")
        self.assertGreaterEqual(score, 0.58)
        self.assertTrue(any("engagement" in s.lower() or "eye contact" in s.lower() for s in signals))

    def test_not_interested_classification(self):
        """Low eye contact, looking away, backward lean, low engagement -> NOT INTERESTED"""
        state, score, conf, signals = self.clf.compute(
            engagement_score=0.20,
            stress_index=0.30,
            eye_contact_score=0.15,
            attention_zone="LEFT",
            head_lean="BACKWARD",
            head_yaw=32.0,
            head_pitch=15.0,
            blink_rate=38,
            blink_stress="ELEVATED",
            active_aus=["AU15"],
            dominant_emotion="neutral",
            macro_emotion="neutral",
            au_dict={"AU15": 0.02}
        )
        self.assertEqual(state, "NOT INTERESTED")
        self.assertLessEqual(score, 0.38)
        self.assertTrue(any("reduced" in s.lower() or "away" in s.lower() or "depressed" in s.lower() for s in signals))

    def test_neutral_classification(self):
        """Balanced baseline signals -> NEUTRAL"""
        state, score, conf, signals = self.clf.compute(
            engagement_score=0.50,
            stress_index=0.10,
            eye_contact_score=0.50,
            attention_zone="CENTER",
            head_lean="NEUTRAL",
            head_yaw=0.0,
            head_pitch=0.0,
            blink_rate=15,
            blink_stress="NORMAL",
            active_aus=[],
            dominant_emotion="neutral",
            macro_emotion="neutral",
            au_dict={}
        )
        self.assertEqual(state, "NEUTRAL")
        self.assertGreater(score, 0.38)
        self.assertLess(score, 0.58)


class TestMicroExpressionDetector(unittest.TestCase):
    def test_timestamp_preservation(self):
        """Ensure custom current_time is respected for video analysis."""
        detector = MicroExpressionDetector(target_fs=30.0)
        gray = np.zeros((200, 200), dtype=np.uint8)
        au_idle = {k: 0.0 for k in ['AU1','AU2','AU4','AU5','AU6','AU12','AU15','AU17','AU25','AU26']}
        
        # Feed baseline frames with synthetic timestamps
        for i in range(20):
            t = i * (1.0 / 30.0)
            res = detector.update(gray, au_idle, current_time=t)
            self.assertIsNotNone(res)
            self.assertFalse(res.get('spike', False))


class TestSessionRecorderAndManager(unittest.TestCase):
    def test_session_lifecycle_and_feedback(self):
        session_id = "MMBI-TEST-00001"
        recorder = SessionRecorder(session_id=session_id, analysis_type="Test")
        
        # Simulate 15 frames: 5 Neutral, 5 Interested, 5 Not Interested
        for i in range(5):
            s = BehaviourState(
                interest_state="NEUTRAL",
                interest_score=0.50,
                engagement_score=0.50,
                stress_index=0.10,
                eye_contact_score=0.50
            )
            recorder.update(s, timestamp=i * 0.1, frame_idx=i)
            
        for i in range(5, 10):
            s = BehaviourState(
                interest_state="INTERESTED",
                interest_score=0.75,
                engagement_score=0.80,
                stress_index=0.10,
                eye_contact_score=0.90
            )
            recorder.update(s, timestamp=i * 0.1, frame_idx=i)
            
        for i in range(10, 15):
            s = BehaviourState(
                interest_state="NOT INTERESTED",
                interest_score=0.25,
                engagement_score=0.20,
                stress_index=0.40,
                eye_contact_score=0.10
            )
            recorder.update(s, timestamp=i * 0.1, frame_idx=i)

        # Record a micro expression
        recorder.record_micro_expression({
            "region": "mouth",
            "micro_emotion": "surprise",
            "duration_ms": 180.0,
            "magnitude": 0.62,
            "confidence": 0.81
        }, timestamp=1.2)

        # End session
        result = recorder.end_session()
        self.assertEqual(result["session_id"], session_id)
        self.assertEqual(result["total_frames"], 15)
        self.assertEqual(len(result["micro_expressions"]["events_log"]), 1)
        self.assertEqual(result["micro_expressions"]["events_log"][0]["region"], "Mouth")
        self.assertEqual(result["micro_expressions"]["events_log"][0]["type"], "Surprise")
        
        # Check interest summary
        int_sum = result["interest_summary"]
        self.assertIn("interested_percentage", int_sum)
        self.assertIn("neutral_percentage", int_sum)
        self.assertIn("not_interested_percentage", int_sum)
        total_pct = int_sum["interested_percentage"] + int_sum["neutral_percentage"] + int_sum["not_interested_percentage"]
        self.assertAlmostEqual(total_pct, 100.0, delta=0.5)

        # Verify disk persistence
        loaded = SessionManager.load_session(session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["session_id"], session_id)

        # Verify feedback update
        feedback = {
            "rating": 5,
            "interest_accuracy": 5,
            "micro_accuracy": 4,
            "comment": "Accurately captured user state transitions."
        }
        success = SessionManager.update_feedback(session_id, feedback)
        self.assertTrue(success)
        
        reloaded = SessionManager.load_session(session_id)
        self.assertIsNotNone(reloaded["feedback"])
        self.assertEqual(reloaded["feedback"]["rating"], 5)
        self.assertEqual(reloaded["feedback"]["comment"], "Accurately captured user state transitions.")


class TestPDFReportGeneration(unittest.TestCase):
    def test_pdf_generation(self):
        session_id = "MMBI-TEST-PDF"
        recorder = SessionRecorder(session_id=session_id, analysis_type="Test")
        for i in range(10):
            s = BehaviourState(
                interest_state="INTERESTED",
                interest_score=0.72,
                engagement_score=0.75,
                stress_index=0.15,
                eye_contact_score=0.88
            )
            recorder.update(s, timestamp=i * 0.2, frame_idx=i)
            
        recorder.record_micro_expression({
            "region": "left_eye_eb",
            "micro_emotion": "surprise",
            "duration_ms": 145.0,
            "magnitude": 0.58,
            "confidence": 0.79
        }, timestamp=1.4)
        
        result = recorder.end_session()
        SessionManager.update_feedback(session_id, {
            "rating": 5,
            "interest_accuracy": 5,
            "micro_accuracy": 4,
            "comment": "Test PDF generated successfully."
        })
        result = SessionManager.load_session(session_id)

        pdf_path = generate_pdf_report(result, output_filename=f"{session_id}_report.pdf")
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)  # Should be substantial PDF (>5KB)
        print(f"Generated test PDF report: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")


class TestRecordedVideoAnalysis(unittest.TestCase):
    def test_synthetic_video_analysis(self):
        """Create a 30-frame synthetic MP4 video and run VideoProcessor."""
        tmp_dir = os.path.join(PROJECT_ROOT, "learning", "data", "uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        video_path = os.path.join(tmp_dir, "test_synthetic.mp4")
        
        # Write 30 frames at 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (320, 240))
        for _ in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Draw a simple circle
            cv2.circle(frame, (160, 120), 40, (200, 200, 200), -1)
            out.write(frame)
        out.release()
        
        session_id = "MMBI-TEST-VIDEO-001"
        # Run synchronous worker directly for test
        VideoProcessor._process_video_worker(video_path, session_id)
        
        status = VideoProcessor.get_job_status(session_id)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["total_frames"], 30)
        self.assertIn("session_result", status)
        self.assertEqual(status["session_result"]["total_frames"], 30)
        print("Synthetic video analysis successfully completed.")


if __name__ == '__main__':
    unittest.main()

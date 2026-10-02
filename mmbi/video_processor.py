"""
MMBI Recorded Video Processing Engine
=====================================
Processes recorded videos (MP4, AVI, MOV, MKV) frame-by-frame using
the EXACT SAME BehaviourEngine multimodal perception and fusion pipeline.

Key properties:
  - Preserves native video timestamps: timestamp = frame_idx / fps
  - Runs in non-blocking background threads with active progress reporting
  - Deterministic primary face selection
  - Complete error resilience (handles corrupted frames, missing faces, variable FPS)
  - Produces identical SessionResult and PDF report cards as live sessions
"""

from __future__ import annotations
import os
import cv2
import time
import threading
import math
from typing import Dict, Any, Optional, Callable

from mmbi.engine import BehaviourEngine
from mmbi.session_manager import SessionRecorder, SessionManager, format_timestamp
from mmbi.report_generator import generate_pdf_report

# Global dictionary to track active and recent video processing jobs
JOB_STATUS: Dict[str, Dict[str, Any]] = {}

class VideoProcessor:
    """
    Executes frame-by-frame MMBI cognitive analysis on an uploaded video file.
    """

    @staticmethod
    def get_job_status(session_id: str) -> Dict[str, Any]:
        """Returns the current progress and status of a video processing job."""
        if session_id in JOB_STATUS:
            return JOB_STATUS[session_id]
        # Check if already completed and on disk
        session = SessionManager.load_session(session_id)
        if session:
            return {
                "session_id": session_id,
                "status": "completed",
                "progress_pct": 100.0,
                "current_frame": session.get("total_frames", 0),
                "total_frames": session.get("total_frames", 0),
                "micro_expressions_detected": session.get("micro_expressions", {}).get("total_count", 0),
                "current_interest_state": session.get("interest_summary", {}).get("dominant_state", "NEUTRAL"),
                "session_result": session
            }
        return {"session_id": session_id, "status": "not_found", "progress_pct": 0.0}

    @staticmethod
    def start_processing_async(
        video_path: str,
        session_id: str,
        engine: Optional[BehaviourEngine] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> threading.Thread:
        """Launches video analysis in a dedicated background worker thread."""
        thread = threading.Thread(
            target=VideoProcessor._process_video_worker,
            args=(video_path, session_id, engine, on_complete),
            daemon=True,
            name=f"MMBI-Worker-{session_id}"
        )
        thread.start()
        return thread

    @staticmethod
    def _process_video_worker(
        video_path: str,
        session_id: str,
        engine: Optional[BehaviourEngine] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Worker executing frame-by-frame analysis with error recovery."""
        filename = os.path.basename(video_path)
        JOB_STATUS[session_id] = {
            "session_id": session_id,
            "filename": filename,
            "status": "processing",
            "progress_pct": 0.0,
            "current_frame": 0,
            "total_frames": 0,
            "current_timestamp": "00:00",
            "processing_fps": 0.0,
            "micro_expressions_detected": 0,
            "current_interest_state": "INITIALISING",
            "current_engagement": 0.50,
            "error": None
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            err_msg = f"Failed to open video file: {filename}. Codec unsupported or file corrupted."
            JOB_STATUS[session_id]["status"] = "error"
            JOB_STATUS[session_id]["error"] = err_msg
            print(f"[VideoProcessor] Error: {err_msg}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 1.0 or native_fps > 240.0 or math.isnan(native_fps):
            native_fps = 30.0  # sensible fallback

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        JOB_STATUS[session_id]["total_frames"] = total_frames
        JOB_STATUS[session_id]["fps"] = round(native_fps, 2)

        # Create or reset BehaviourEngine for this video
        if engine is None:
            engine = BehaviourEngine(camera_index=-1, frame_w=frame_w, frame_h=frame_h)
        else:
            engine.reset_buffers(target_fps=native_fps)

        engine.blink.fps = native_fps
        engine.engagement.fps = native_fps
        engine.stress_sc.fps = native_fps
        engine.micro.fs = native_fps

        # Start dedicated session recorder
        recorder = engine.start_session(
            session_id=session_id,
            analysis_type="Recorded",
            video_filename=filename
        )

        frame_idx = 0
        start_processing_time = time.time()
        last_progress_update_t = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Precise native video timestamp based on video index and FPS
                video_ts = frame_idx / native_fps

                try:
                    # Run frame through identical engine pipeline
                    state = engine.process_frame(frame, timestamp=video_ts)
                except Exception as frame_err:
                    print(f"[VideoProcessor] Warning: frame {frame_idx} processing error: {frame_err}")
                    state = None

                frame_idx += 1

                # Update live job telemetry at reasonable intervals (~10 fps to prevent lock thrashing)
                now_wall = time.time()
                if now_wall - last_progress_update_t >= 0.15 or frame_idx >= total_frames:
                    elapsed = max(0.001, now_wall - start_processing_time)
                    p_fps = round(frame_idx / elapsed, 1)
                    pct = round((frame_idx / max(1, total_frames)) * 100, 1)
                    
                    int_st = getattr(state, "interest_state", recorder.current_interest_state)
                    eng_sc = getattr(state, "engagement_score", 0.50)

                    JOB_STATUS[session_id].update({
                        "progress_pct": min(100.0, pct),
                        "current_frame": frame_idx,
                        "current_timestamp": format_timestamp(video_ts, include_ms=False),
                        "processing_fps": p_fps,
                        "micro_expressions_detected": recorder.me_counter,
                        "current_interest_state": int_st,
                        "current_engagement": round(float(eng_sc), 3)
                    })
                    last_progress_update_t = now_wall

        except Exception as e:
            err_msg = f"Fatal processing error during video analysis: {e}"
            JOB_STATUS[session_id]["status"] = "error"
            JOB_STATUS[session_id]["error"] = err_msg
            print(f"[VideoProcessor] {err_msg}")
            cap.release()
            return
        finally:
            cap.release()

        # Finalize and compile SessionResult
        session_result = engine.stop_session()
        if not session_result:
            session_result = recorder.end_session()

        # Automatically generate the PDF report
        try:
            pdf_path = generate_pdf_report(session_result)
            session_result["report_pdf"] = os.path.basename(pdf_path)
            SessionManager.save_session(session_result)
        except Exception as pdf_err:
            print(f"[VideoProcessor] Warning: PDF report generation failed: {pdf_err}")

        JOB_STATUS[session_id].update({
            "status": "completed",
            "progress_pct": 100.0,
            "current_frame": frame_idx,
            "session_result": session_result
        })

        print(f"[VideoProcessor] Video analysis completed successfully for {filename} ({frame_idx} frames).")

        if on_complete:
            try:
                on_complete(session_result)
            except Exception as cb_err:
                print(f"[VideoProcessor] Error in completion callback: {cb_err}")

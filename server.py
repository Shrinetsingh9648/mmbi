# mmbi/server.py
import os
import cv2
import numpy as np
import json
import time
import asyncio
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from mmbi.engine import BehaviourEngine
from mmbi.session_manager import SessionRecorder, SessionManager, SESSIONS_DIR
from mmbi.report_generator import generate_pdf_report, REPORTS_DIR
from mmbi.video_processor import VideoProcessor, JOB_STATUS

app = FastAPI(title="MMBI Enterprise Cognitive & Interest Analysis Platform")

# Enable CORS for cross-origin web client dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploads directory
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'learning', 'data', 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Initialize global engine instance (without triggering cv2 camera read)
engine = BehaviourEngine(camera_index=-1)
# Create a dedicated single-threaded executor for frame-safe serial inference
executor = ThreadPoolExecutor(max_workers=1)

# Keep track of active video paths for session analysis
UPLOADED_VIDEOS: Dict[str, str] = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[Server] Client connected to live analyst WebSocket stream.")
    loop = asyncio.get_event_loop()
    
    # Auto-initialize a live session if not already tracking
    if engine.active_recorder is None:
        engine.start_session(analysis_type="Live")

    try:
        while True:
            # 1. Receive binary frame blob (JPEG compressed) from client
            data = await websocket.receive_bytes()
            if not data:
                continue

            # 2. Decode JPEG bytes in memory
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 3. Offload heavy CV/ML computation to background worker thread
            state = await loop.run_in_executor(executor, engine.process_frame, frame)
            
            if state is None:
                # No face detected
                await websocket.send_json({
                    'face_detected': False,
                    'session_id': engine.active_recorder.session_id if engine.active_recorder else None
                })
                continue

            rec = engine.active_recorder
            me_count = rec.me_counter if rec else len(engine._recorded_micro_expressions)
            last_me = rec.micro_expressions[-1] if (rec and rec.micro_expressions) else None
            sess_dur = round(time.time() - rec.first_frame_ts, 1) if (rec and rec.first_frame_ts) else 0.0

            # 4. Serialize comprehensive telemetry map and send back to client
            telemetry = {
                'face_detected': True,
                'session_id': rec.session_id if rec else None,
                'session_duration': sess_dur,
                
                # High-Level Interest State Analysis
                'interest_state': state.interest_state,
                'interest_score': round(float(state.interest_score), 3),
                'interest_confidence': round(float(state.interest_confidence), 3),
                'supporting_signals': state.supporting_signals,
                
                # Cognitive Summary
                'cognitive_summary': {
                    'engagement_score': round(float(state.engagement_score), 3),
                    'engagement_label': state.engagement_label,
                    'stress_index': round(float(state.stress_index), 3),
                    'stress_label': state.stress_label
                },
                
                # Emotion & Micro-expression Analysis
                'emotion_analysis': {
                    'dominant_emotion': state.dominant_emotion,
                    'confidence': round(float(state.emotion_confidence), 3),
                    'source': state.emotion_source,
                    'macro_emotion': state.macro_emotion,
                    'macro_confidence': round(float(state.macro_confidence), 3),
                    'micro_emotion': state.micro_emotion,
                    'micro_confidence': round(float(state.micro_confidence), 3),
                    'micro_duration_ms': round(float(state.micro_duration_ms), 1),
                    'micro_region': state.micro_region,
                    'micro_expression_count': me_count,
                    'last_micro_event': last_me
                },
                
                # Body Signals
                'body_signals': {
                    'attention_zone': state.attention_zone,
                    'eye_contact_score': round(float(state.eye_contact_score), 3),
                    'lean_signal': state.lean_signal,
                    'blink_rate': state.blink_rate,
                    'blink_stress': state.blink_stress,
                    'active_aus': state.active_aus
                },
                
                # Stream 2D landmarks coordinates for canvas rendering
                'landmarks_2d': state.raw.get('landmarks_2d', [])
            }

            await websocket.send_json(telemetry)

    except WebSocketDisconnect:
        print("[Server] Client disconnected from WebSocket stream.")
    except Exception as e:
        print(f"[Server] Connection error: {e}")


# ── Session Lifecycle Endpoints ──────────────────────────────────────────────

@app.post("/api/session/start")
def start_session(data: Optional[Dict[str, Any]] = None):
    """Explicitly initiates a new live session."""
    session_id = (data or {}).get("session_id")
    analysis_type = (data or {}).get("analysis_type", "Live")
    video_filename = (data or {}).get("video_filename")
    
    recorder = engine.start_session(
        session_id=session_id,
        analysis_type=analysis_type,
        video_filename=video_filename
    )
    return {
        "status": "success",
        "message": "Session started.",
        "session_id": recorder.session_id,
        "analysis_type": recorder.analysis_type
    }


@app.post("/api/session/stop")
def stop_session():
    """Stops the active live session, finalizes metrics, and auto-generates report."""
    try:
        session_result = engine.stop_session()
        if not session_result:
            return {"status": "error", "message": "No active session in progress to stop."}
        
        # Generate PDF report
        try:
            pdf_path = generate_pdf_report(session_result)
            session_result["report_pdf"] = os.path.basename(pdf_path)
            SessionManager.save_session(session_result)
        except Exception as pe:
            print(f"[Server] PDF generation error: {pe}")

        return {
            "status": "success",
            "message": "Session finalized successfully.",
            "session_id": session_result.get("session_id"),
            "session_result": session_result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Retrieves stored session result JSON."""
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_data


@app.get("/api/session/{session_id}/report.pdf")
def download_pdf_report(session_id: str):
    """Generates (if needed) and serves self-contained PDF report."""
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session record not found.")

    pdf_filename = f"{session_id}_report.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    # If PDF does not exist yet or was regenerated, generate now
    if not os.path.exists(pdf_path):
        try:
            pdf_path = generate_pdf_report(session_data, output_filename=pdf_filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to compile PDF report: {e}")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_filename,
        headers={"Content-Disposition": f"attachment; filename={pdf_filename}"}
    )


@app.post("/api/session/{session_id}/feedback")
async def submit_feedback(session_id: str, request: Request):
    """Attaches user feedback to session and updates the PDF report."""
    try:
        feedback = await request.json()
        success = SessionManager.update_feedback(session_id, feedback)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found to attach feedback.")
        
        # Regenerate PDF report so feedback is included
        session_data = SessionManager.load_session(session_id)
        if session_data:
            try:
                generate_pdf_report(session_data, output_filename=f"{session_id}_report.pdf")
            except Exception as e:
                print(f"[Server] Warning: Failed to re-render PDF with feedback: {e}")

        return {"status": "success", "message": "Feedback submitted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Recorded Video Endpoints ─────────────────────────────────────────────────

@app.post("/api/video/upload")
async def upload_video(video: UploadFile = File(...)):
    """Receives and stores an uploaded video file (MP4, AVI, MOV, MKV)."""
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {ext}. Supported formats: MP4, AVI, MOV, MKV."
        )

    session_id = SessionManager.generate_session_id()
    saved_filename = f"{session_id}_{video.filename}"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)

    with open(saved_path, "wb") as f:
        content = await video.read()
        f.write(content)

    UPLOADED_VIDEOS[session_id] = saved_path
    print(f"[Server] Video uploaded: {video.filename} -> {saved_path} (Session: {session_id})")

    return {
        "status": "success",
        "message": "Video uploaded successfully.",
        "session_id": session_id,
        "filename": video.filename,
        "filesize_bytes": len(content)
    }


@app.post("/api/video/analyze/{session_id}")
def start_video_analysis(session_id: str):
    """Starts asynchronous frame-by-frame analysis of an uploaded video."""
    video_path = UPLOADED_VIDEOS.get(session_id)
    if not video_path or not os.path.exists(video_path):
        # Check if file exists matching session_id in UPLOADS_DIR
        matches = [os.path.join(UPLOADS_DIR, f) for f in os.listdir(UPLOADS_DIR) if f.startswith(session_id)]
        if matches:
            video_path = matches[0]
            UPLOADED_VIDEOS[session_id] = video_path
        else:
            raise HTTPException(status_code=404, detail="Uploaded video file for this session was not found.")

    # Launch background processing worker
    VideoProcessor.start_processing_async(
        video_path=video_path,
        session_id=session_id,
        engine=engine
    )

    return {
        "status": "success",
        "message": "Video analysis started in background worker.",
        "session_id": session_id
    }


@app.get("/api/video/progress/{session_id}")
def get_video_progress(session_id: str):
    """Polls progress of a video analysis job."""
    status = VideoProcessor.get_job_status(session_id)
    return status


# ── Legacy Endpoints (Maintained for Backward Compatibility) ──────────────────

@app.get("/api/calibration/load")
def load_calibration():
    profile_path = os.path.join(os.path.dirname(__file__), 'calibration_profile.json')
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            engine.gaze.import_calibration(profile_data.get('gaze', {}))
            engine.au.import_calibration(profile_data.get('au', {}))
            engine._calibrated = True
            return {'status': 'success', 'message': 'Persistent profile loaded successfully.'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    return {'status': 'error', 'message': 'No calibration profile file found.'}


@app.post("/api/session/save")
def save_session():
    try:
        # If active recorder exists, finalize it
        if engine.active_recorder:
            session_result = engine.stop_session()
            try:
                generate_pdf_report(session_result)
            except Exception:
                pass
            return {
                'status': 'success',
                'message': 'Session analytics generated successfully.',
                'files': {
                    'json': f"{session_result['session_id']}.json",
                    'pdf': f"{session_result['session_id']}_report.pdf"
                }
            }
        
        # Fallback to data collector
        paths = engine.collector.save_session(engine._recorded_micro_expressions)
        if paths:
            return {
                'status': 'success',
                'message': 'Session analytics generated successfully.',
                'files': {
                    'npz': os.path.basename(paths[0]),
                    'csv': os.path.basename(paths[1]),
                    'json': os.path.basename(paths[2])
                }
            }
        return {'status': 'error', 'message': 'Insufficient frame data to save session.'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ── Static File Mounts ────────────────────────────────────────────────────────

app.mount("/sessions", StaticFiles(directory=SESSIONS_DIR), name="sessions")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')
os.makedirs(WEB_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

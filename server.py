# mmbi/server.py
import os
import cv2
import numpy as np
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mmbi.engine import BehaviourEngine

app = FastAPI(title="MMBI Enterprise Cognitive Server")

# Enable CORS for cross-origin web client dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global engine instance (without triggering cv2 camera read)
engine = BehaviourEngine(camera_index=-1)
# Create a dedicated single-threaded executor for frame-safe serial inference
executor = ThreadPoolExecutor(max_workers=1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[Server] Client connected to live analyst WebSocket stream.")
    loop = asyncio.get_event_loop()
    
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
                await websocket.send_json({'face_detected': False})
                continue

            # 4. Serialize telemetry map and send back to client
            telemetry = {
                'face_detected': True,
                'cognitive_summary': {
                    'engagement_score': round(float(state.engagement_score), 3),
                    'engagement_label': state.engagement_label,
                    'stress_index': round(float(state.stress_index), 3),
                    'stress_label': state.stress_label
                },
                'emotion_analysis': {
                    'dominant_emotion': state.dominant_emotion,
                    'confidence': round(float(state.emotion_confidence), 3),
                    'source': state.emotion_source,
                    'macro_emotion': state.macro_emotion,
                    'macro_confidence': round(float(state.macro_confidence), 3),
                    'micro_emotion': state.micro_emotion,
                    'micro_confidence': round(float(state.micro_confidence), 3),
                    'micro_duration_ms': round(float(state.micro_duration_ms), 1),
                    'micro_region': state.micro_region
                },
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

# Expose REST endpoint to load calibration profile
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

# Expose REST endpoint to save session reports (CSV/JSON)
@app.post("/api/session/save")
def save_session():
    try:
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

# Serve static Web Analyst files from mounted directory
SESSIONS_DIR = os.path.join(os.path.dirname(__file__), 'learning', 'data', 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)
app.mount("/sessions", StaticFiles(directory=SESSIONS_DIR), name="sessions")

WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')
os.makedirs(WEB_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

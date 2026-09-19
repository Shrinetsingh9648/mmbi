import numpy as np
from collections import deque

# ── IMPORTANT: FPS is now a RUNTIME value, not a hardcoded constant ────────
# Previously this file assumed a fixed 120 FPS camera to convert frame
# counts into durations ("60 frames = 500ms at 120 FPS"). Most consumer
# webcams cannot sustain 120 FPS, and `cv2.CAP_PROP_FPS` requests are
# frequently ignored by the driver. `Camera.measure_actual_fps()`
# (capture/camera.py) measures real throughput at startup; pass that
# value into `TemporalBuffer(fps=...)` / `set_fps()` so window-duration
# math (and any logging/UI that reports "Xms window") is honest about
# what the hardware is actually delivering. The DEFAULT_FPS constant
# below is only a fallback used before a real measurement is available.
DEFAULT_FPS  = 30.0
WINDOW_SECS  = 5

# Each frame has a feature vector combining:
#   AU1, AU2, AU4, AU5, AU6, AU12, AU15, AU17, AU25, AU26   → 10 dims  (geometric AU proxies)
#   gaze_h, gaze_v, eye_contact                              →  3 dims
#   yaw, pitch, roll                                         →  3 dims
#   ear                                                      →  1 dim
#   flow_brow, flow_eye, flow_nose, flow_mouth, flow_cheek   →  5 dims  (regional optical-flow
#                                                                        magnitude, see
#                                                                        face/microexpression.py)
# Total: 22 dims
#
# NOTE: this is 5 dims larger than the original 17-dim vector the shipped
# micro_lstm.pt was (if ever) trained with. Any checkpoint trained on the
# 17-dim scheme will NOT load its input_proj layer cleanly against this
# feature set — see models/lstm_temporal.py MicroLSTMInference, which
# loads with strict=False and prints exactly which layers came up
# randomly-initialized as a result. This is a genuine architecture
# change, not a compatible extension, and requires retraining.
FEATURE_KEYS = [
    'AU1','AU2','AU4','AU5','AU6',
    'AU12','AU15','AU17','AU25','AU26',
    'gaze_h','gaze_v','eye_contact',
    'yaw','pitch','roll',
    'ear',
    'flow_brow', 'flow_eye', 'flow_nose', 'flow_mouth', 'flow_cheek',
]
FEATURE_DIM  = len(FEATURE_KEYS)   # 22
BUFFER_SIZE  = int(DEFAULT_FPS * WINDOW_SECS)   # 150 frames @ 30fps fallback

# Micro-expression window: true ME duration is 40-500ms. The frame COUNT
# that corresponds to that duration depends on the real camera FPS (see
# above) -- MICRO_WIN_MAX below is expressed in FRAMES because that's
# what the trained model's fixed input shape needs, but its real-world
# duration must be computed from the measured FPS, not assumed. Use
# TemporalBuffer.window_duration_ms() to get the honest number.
MICRO_WIN_MIN = 5
MICRO_WIN_MAX = 60


class TemporalBuffer:
    """
    Stores a rolling window of per-frame feature vectors.
    Provides sliding windows of any length for the LSTM model.
    All values are stored as float32 numpy arrays.
    """

    def __init__(self, buffer_size=BUFFER_SIZE, feature_dim=FEATURE_DIM,
                 fps: float = DEFAULT_FPS):
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        self.fps = float(fps)
        self._buf = deque(maxlen=buffer_size)

    def set_fps(self, fps: float) -> None:
        """Call once real FPS has been measured (Camera.measure_actual_fps()).
        Only affects duration math (window_duration_ms); does not resize
        the buffer or change which frames are stored."""
        if fps and fps > 0:
            self.fps = float(fps)

    def window_duration_ms(self, n_frames: int) -> float:
        """Honest duration of an n_frames window given the current fps."""
        if self.fps <= 0:
            return float('nan')
        return (n_frames / self.fps) * 1000.0

    def push(self, au_dict: dict, gaze_dict: dict,
             head_dict: dict, blink_dict: dict, flow_dict: dict = None):
        """
        Call every frame with the output dicts from each module.
        Builds a flat feature vector and appends to buffer.
        Missing keys default to 0.0 so partial frames never crash.
        `flow_dict` (optional, new) supplies the regional optical-flow
        magnitude features produced by MicroExpressionDetector.update();
        omit it (or pass None) and those dims simply default to 0.0,
        which keeps this method backward compatible with existing callers.
        """
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        all_data = {}
        all_data.update(au_dict    or {})
        all_data.update(gaze_dict  or {})
        all_data.update(head_dict  or {})
        all_data.update(blink_dict or {})
        all_data.update(flow_dict  or {})

        for idx, key in enumerate(FEATURE_KEYS):
            val = all_data.get(key, 0.0)
            # Convert to float safely — some values may be strings
            try:
                vec[idx] = float(val)
            except (TypeError, ValueError):
                vec[idx] = 0.0

        self._buf.append(vec)

    def get_window(self, n_frames: int) -> np.ndarray | None:
        """
        Returns the last n_frames as shape (n_frames, feature_dim).
        Returns None if buffer does not have enough frames yet.
        """
        if len(self._buf) < n_frames:
            return None
        arr = np.array(list(self._buf)[-n_frames:], dtype=np.float32)
        return arr   # shape: (n_frames, 17)

    def get_micro_window(self) -> np.ndarray | None:
        """Returns last MICRO_WIN_MAX frames for micro-expression model."""
        return self.get_window(MICRO_WIN_MAX)

    def get_full_window(self) -> np.ndarray | None:
        """Returns full 5-second buffer for macro emotion model."""
        return self.get_window(self.buffer_size)

    def is_ready(self, n_frames: int = MICRO_WIN_MAX) -> bool:
        """True once buffer has collected at least n_frames."""
        return len(self._buf) >= n_frames

    def get_delta(self, n_frames: int = 10) -> np.ndarray | None:
        """
        Returns frame-to-frame AU deltas for the last n_frames.
        Shape: (n_frames-1, feature_dim).
        Used as additional input signal for spike detection.
        """
        window = self.get_window(n_frames)
        if window is None:
            return None
        return np.diff(window, axis=0)   # (n_frames-1, 17)

    def smoothed(self, n_frames: int = 30, alpha: float = 0.3) -> np.ndarray | None:
        """
        Exponentially smoothed version of the last n_frames.
        alpha=0.3 keeps fast movements while reducing noise.
        Shape: (n_frames, feature_dim).
        """
        window = self.get_window(n_frames)
        if window is None:
            return None
        smoothed = window.copy()
        for t in range(1, len(smoothed)):
            smoothed[t] = alpha * window[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed

    @property
    def current_frame_count(self) -> int:
        return len(self._buf)

    def reset(self):
        self._buf.clear()
import numpy as np
from collections import deque

# At 120 FPS, 5 seconds = 600 frames
# Each frame has a 17-dimensional feature vector:
#   AU1, AU2, AU4, AU5, AU6, AU12, AU15, AU17, AU25, AU26  → 10 dims
#   gaze_h, gaze_v, eye_contact                             →  3 dims
#   yaw, pitch, roll                                        →  3 dims
#   ear                                                     →  1 dim
# Total: 17 dims

FEATURE_KEYS = [
    'AU1','AU2','AU4','AU5','AU6',
    'AU12','AU15','AU17','AU25','AU26',
    'gaze_h','gaze_v','eye_contact',
    'yaw','pitch','roll',
    'ear'
]
FEATURE_DIM  = len(FEATURE_KEYS)   # 17
FPS          = 120
WINDOW_SECS  = 5
BUFFER_SIZE  = FPS * WINDOW_SECS   # 600 frames

# Micro-expression window: 5 to 60 frames (40ms–500ms)
MICRO_WIN_MIN = 5
MICRO_WIN_MAX = 60


class TemporalBuffer:
    """
    Stores the last 5 seconds of per-frame feature vectors.
    Provides sliding windows of any length for the LSTM model.
    All values are stored as float32 numpy arrays.
    """

    def __init__(self, buffer_size=BUFFER_SIZE, feature_dim=FEATURE_DIM):
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        # Deque is O(1) append and popleft at 120 FPS
        self._buf = deque(maxlen=buffer_size)

    def push(self, au_dict: dict, gaze_dict: dict,
             head_dict: dict, blink_dict: dict):
        """
        Call every frame with the output dicts from each module.
        Builds a flat 17-dim feature vector and appends to buffer.
        Missing keys default to 0.0 so partial frames never crash.
        """
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        all_data = {}
        all_data.update(au_dict    or {})
        all_data.update(gaze_dict  or {})
        all_data.update(head_dict  or {})
        all_data.update(blink_dict or {})

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
# mmbi/face/microexpression.py
import cv2
import numpy as np
from collections import deque
import time
from scipy.signal import butter, filtfilt

FPS = 120
MIN_FRAMES = 5    # 40 ms minimum
MAX_FRAMES = 60   # 500 ms maximum

class MicroExpressionDetector:
    def __init__(self, method='sparse', threshold=0.0025, k_threshold=3.0, 
                 weight_au=0.6, weight_flow=0.4, target_fs=60.0):
        """
        High-Performance Landmark-based ROI Sparse/Dense Optical Flow & FSM Phase Segmenter
        for Micro-Expression Detection.
        
        Co-designed with Senior Research Head and Senior Developer.
        """
        self.au_buffer = deque(maxlen=MAX_FRAMES)
        self.score_buffer = deque(maxlen=MAX_FRAMES)
        self.prev_gray = None
        self.prev_landmarks = None
        self.method = method
        self.threshold = threshold
        self.k_threshold = k_threshold
        self.weight_au = weight_au
        self.weight_flow = weight_flow
        
        # FSM State variables for micro-expression phase tracking
        self.state = "Baseline"
        self.onset_start_idx = 0
        self.onset_start_t = 0.0
        self.apex_start_idx = 0
        self.apex_start_t = 0.0
        self.apex_max_val = 0.0
        self.offset_start_idx = 0
        self.offset_start_t = 0.0
        
        self.frame_count = 0
        self.fs = target_fs
        
        # Design Butterworth filter
        self._design_filter()

    def _design_filter(self):
        nyq = 0.5 * self.fs
        low = 1.5 / nyq
        high = min(12.0, self.fs * 0.45) / nyq
        # Ensure cutoffs are valid in case of extremely low FPS
        if low >= 1.0:
            low = 0.1
        if high >= 1.0:
            high = 0.9
        try:
            self.b, self.a = butter(2, [low, high], btype='band')
        except Exception:
            # Fallback if designing fails
            self.b, self.a = butter(2, 0.5, btype='low')

    def update(self, frame_gray, au_values, landmarks=None, current_time=None):
        self.frame_count += 1
        self.au_buffer.append(au_values)
        
        if frame_gray is None:
            return None
            
        h, w = frame_gray.shape[:2]
        flow_magnitude = 0.0
        region_magnitudes = {}

        # 1. Physical motion via Landmark-guided Optical Flow
        if landmarks is not None and self.prev_gray is not None and self.prev_landmarks is not None:
            # If coordinates are normalized, scale them to pixel dimensions
            lms_pixel = np.array(landmarks, dtype=np.float32)
            if np.max(lms_pixel) <= 1.0:
                lms_pixel[:, 0] *= w
                lms_pixel[:, 1] *= h
                
            if len(lms_pixel) == len(self.prev_landmarks):
                if self.method == 'sparse':
                    flow_magnitude, region_magnitudes = self._compute_sparse_lk_flow(frame_gray, lms_pixel)
                elif self.method == 'dense':
                    flow_magnitude, region_magnitudes = self._compute_dense_roi_flow(frame_gray, lms_pixel)

        # Update references
        self.prev_gray = frame_gray.copy() if frame_gray is not None else None
        if landmarks is not None:
            lms_pixel = np.array(landmarks, dtype=np.float32)
            if np.max(lms_pixel) <= 1.0:
                lms_pixel[:, 0] *= w
                lms_pixel[:, 1] *= h
            self.prev_landmarks = lms_pixel
        else:
            self.prev_landmarks = None

        # 2. Compute semantic AU delta
        au_delta = 0.0
        if len(self.au_buffer) >= 2:
            recent = list(self.au_buffer)[-2:]
            deltas = [abs(recent[1][k] - recent[0].get(k, 0.0)) for k in recent[1]]
            au_delta = np.mean(deltas) if deltas else 0.0

        # 3. Gated Fusion Math
        scaled_flow = flow_magnitude * 0.1
        gated_flow = min(scaled_flow, au_delta * 2.0 + 0.005)
        raw_score = self.weight_au * au_delta + self.weight_flow * gated_flow
        
        self.score_buffer.append(raw_score)

        # 4. Zero-phase bandpass filtering
        f_score = raw_score
        n_scores = len(self.score_buffer)
        if n_scores >= 15:
            try:
                # Apply filtfilt on rolling buffer (zero phase shift)
                filtered = filtfilt(self.b, self.a, list(self.score_buffer))
                f_score = filtered[-1]
            except Exception:
                pass

        # 5. Dynamic threshold calculation via rolling MAD
        threshold = self.threshold
        med = 0.0
        if n_scores >= 15:
            scores_arr = np.array(list(self.score_buffer), dtype=np.float32)
            med = np.median(scores_arr)
            mad = 1.4826 * np.median(np.abs(scores_arr - med))
            threshold = med + self.k_threshold * max(mad, 0.0003)

        # 6. Finite State Machine Phase Segmentation
        return self._run_phase_fsm(f_score, threshold, region_magnitudes, current_time=current_time)

    def _get_landmark_regions(self, num_pts, landmarks):
        if num_pts == 68:
            return {
                'left_eye_eb': list(range(17, 22)) + list(range(36, 42)),
                'right_eye_eb': list(range(22, 27)) + list(range(42, 48)),
                'mouth': list(range(48, 68)),
                'nose': list(range(27, 31))
            }
        elif num_pts >= 468:
            return {
                'left_eye_eb': [33, 133, 159, 145, 70, 63, 105, 66, 107],
                'right_eye_eb': [362, 263, 386, 374, 336, 296, 334, 293, 300],
                'mouth': [61, 291, 78, 308, 95, 324, 88, 318, 178, 402, 13, 14, 321, 91, 375, 146, 292, 407, 271, 315],
                'nose': [4, 6, 168, 197]
            }
        else:
            pts = np.array(landmarks)
            mean_x, mean_y = np.mean(pts, axis=0)[:2]
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            dist_to_mean = np.linalg.norm(pts[:, :2] - [mean_x, mean_y], axis=1)
            nose = np.where(dist_to_mean < 0.15 * (max_y - min_y))[0].tolist()
            
            left_eye_eb = np.where((pts[:, 0] < mean_x) & (pts[:, 1] < mean_y))[0].tolist()
            right_eye_eb = np.where((pts[:, 0] >= mean_x) & (pts[:, 1] < mean_y))[0].tolist()
            mouth = np.where(pts[:, 1] >= mean_y)[0].tolist()
            
            left_eye_eb = [i for i in left_eye_eb if i not in nose]
            right_eye_eb = [i for i in right_eye_eb if i not in nose]
            mouth = [i for i in mouth if i not in nose]
            
            return {
                'left_eye_eb': left_eye_eb,
                'right_eye_eb': right_eye_eb,
                'mouth': mouth,
                'nose': nose
            }

    def _compute_sparse_lk_flow(self, frame_gray, landmarks):
        curr_landmarks = np.array(landmarks, dtype=np.float32)
        num_pts = len(curr_landmarks)
        prev_pts = self.prev_landmarks[:, :2].reshape(-1, 1, 2)
        
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        status = status.flatten()
        regions = self._get_landmark_regions(num_pts, curr_landmarks)
        
        # 1. Rigid translation (nose bridge)
        nose_disps = []
        for idx in regions['nose']:
            if idx < len(status) and status[idx] == 1:
                disp = next_pts[idx, 0] - prev_pts[idx, 0]
                nose_disps.append(disp)
                
        rigid_translation = np.mean(nose_disps, axis=0) if nose_disps else np.array([0.0, 0.0], dtype=np.float32)
        
        # 2. Local deformation
        def get_region_mag(indices):
            mags = []
            for idx in indices:
                if idx < len(status) and status[idx] == 1:
                    disp = next_pts[idx, 0] - prev_pts[idx, 0]
                    relative_disp = disp - rigid_translation
                    mags.append(np.linalg.norm(relative_disp))
            return np.mean(mags) if mags else 0.0
            
        region_mags = {
            'left_eye_eb': get_region_mag(regions['left_eye_eb']),
            'right_eye_eb': get_region_mag(regions['right_eye_eb']),
            'mouth': get_region_mag(regions['mouth'])
        }
        
        overall_flow_mag = max(region_mags.values()) if region_mags else 0.0
        return overall_flow_mag, region_mags

    def _compute_dense_roi_flow(self, frame_gray, landmarks):
        curr_landmarks = np.array(landmarks, dtype=np.int32)
        num_pts = len(curr_landmarks)
        regions = self._get_landmark_regions(num_pts, curr_landmarks)
        
        h, w = frame_gray.shape
        
        def get_roi_bbox(indices, padding=15):
            pts = curr_landmarks[indices]
            if len(pts) == 0:
                return None
            x_min, y_min = np.min(pts, axis=0)[:2]
            x_max, y_max = np.max(pts, axis=0)[:2]
            return (
                max(0, x_min - padding),
                max(0, y_min - padding),
                min(w - 1, x_max + padding),
                min(h - 1, y_max + padding)
            )
            
        rois = {name: get_roi_bbox(indices) for name, indices in regions.items()}
        
        rigid_translation = np.array([0.0, 0.0], dtype=np.float32)
        nose_roi = rois.get('nose')
        if nose_roi:
            x1, y1, x2, y2 = nose_roi
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                prev_crop = self.prev_gray[y1:y2, x1:x2]
                curr_crop = frame_gray[y1:y2, x1:x2]
                flow_nose = cv2.calcOpticalFlowFarneback(
                    prev_crop, curr_crop, None, 0.5, 2, 10, 2, 5, 1.1, 0
                )
                rigid_translation = np.mean(flow_nose, axis=(0, 1))
                
        region_mags = {}
        for name in ['left_eye_eb', 'right_eye_eb', 'mouth']:
            roi = rois.get(name)
            if roi:
                x1, y1, x2, y2 = roi
                if (x2 - x1) > 4 and (y2 - y1) > 4:
                    prev_crop = self.prev_gray[y1:y2, x1:x2]
                    curr_crop = frame_gray[y1:y2, x1:x2]
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_crop, curr_crop, None, 0.5, 2, 10, 2, 5, 1.1, 0
                    )
                    rel_flow = flow - rigid_translation
                    mags = np.linalg.norm(rel_flow, axis=2)
                    region_mags[name] = float(np.mean(mags))
                else:
                    region_mags[name] = 0.0
            else:
                region_mags[name] = 0.0
                
        overall_flow_mag = max(region_mags.values()) if region_mags else 0.0
        return overall_flow_mag, region_mags

    def _run_phase_fsm(self, val, thresh, region_mags, current_time=None):
        n_scores = len(self.score_buffer)
        vel = 0.0
        if n_scores >= 2:
            scores_arr = list(self.score_buffer)
            vel = (scores_arr[-1] - scores_arr[-2]) * self.fs

        epsilon_v = 0.015  # velocity trigger threshold
        now_t = time.time() if current_time is None else float(current_time)
        
        trigger_val = max(thresh, self.threshold)
        ret_result = None

        if self.state == "Baseline":
            if val > trigger_val and vel > epsilon_v:
                self.state = "Onset"
                self.onset_start_idx = self.frame_count
                self.onset_start_t = now_t
                self.apex_max_val = val
                
        elif self.state == "Onset":
            self.apex_max_val = max(self.apex_max_val, val)
            if vel <= 0 and val >= 0.80 * self.apex_max_val:
                self.state = "Apex"
                self.apex_start_idx = self.frame_count
                self.apex_start_t = now_t
                
        elif self.state == "Apex":
            self.apex_max_val = max(self.apex_max_val, val)
            if val < 0.85 * self.apex_max_val and vel < -epsilon_v:
                self.state = "Offset"
                self.offset_start_idx = self.frame_count
                self.offset_start_t = now_t
                
        elif self.state == "Offset":
            if val <= trigger_val or (abs(vel) < epsilon_v and val < trigger_val + 0.005):
                self.state = "Baseline"
                duration = (now_t - self.onset_start_t) * 1000.0
                if duration <= 0:
                    duration = max(1.0, (self.frame_count - self.onset_start_idx) / max(self.fs, 1.0) * 1000.0)
                
                primary_region = "unknown"
                if region_mags:
                    primary_region = max(region_mags, key=region_mags.get)
                
                ret_result = {
                    'spike': True,
                    'magnitude': round(self.apex_max_val, 4),
                    'duration_ms': round(duration, 1),
                    'region': primary_region,
                    'onset_time': round(self.onset_start_t, 3),
                    'apex_time': round(self.apex_start_t, 3),
                    'offset_time': round(now_t, 3)
                }
                
            elif val > trigger_val and vel > epsilon_v:
                duration = (now_t - self.onset_start_t) * 1000.0
                if duration <= 0:
                    duration = max(1.0, (self.frame_count - self.onset_start_idx) / max(self.fs, 1.0) * 1000.0)
                primary_region = "unknown"
                if region_mags:
                    primary_region = max(region_mags, key=region_mags.get)
                    
                ret_result = {
                    'spike': True,
                    'magnitude': round(self.apex_max_val, 4),
                    'duration_ms': round(duration, 1),
                    'region': primary_region,
                    'onset_time': round(self.onset_start_t, 3),
                    'apex_time': round(self.apex_start_t, 3),
                    'offset_time': round(now_t, 3)
                }
                
                self.state = "Onset"
                self.onset_start_idx = self.frame_count
                self.onset_start_t = now_t
                self.apex_max_val = val

        if ret_result:
            return ret_result
            
        return {
            'spike': False,
            'magnitude': round(val, 4),
            'threshold': round(trigger_val, 4),
            'state': self.state,
            'region_magnitudes': {k: round(v, 4) for k, v in region_mags.items()} if region_mags else {}
        }
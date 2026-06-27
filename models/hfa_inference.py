"""
HFA Inference — real-time HFA prediction on live camera frames.
Maintains a sliding clip buffer of T=16 face patches.
Runs HFANetwork every 16 frames (7.5x per second at 120 FPS).
Also exports attention heatmaps for live visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from collections import deque

from mmbi.models.hfa_network import HFANetwork, MODEL_PATH, T_FRAMES, PATCH_SIZE, N_EMOTION
from mmbi.models.hfa_dataset import crop_face_patch

EMOTION_NAMES = {
    0: 'happiness',
    1: 'disgust',
    2: 'repression',
    3: 'surprise',
    4: 'others'
}

AU_NAMES = [
    'AU1','AU2','AU4','AU5','AU6','AU7','AU9','AU10','AU11',
    'AU12','AU13','AU14','AU15','AU16','AU17','AU18','AU20',
    'AU22','AU23','AU24','AU25','AU26','AU27','AU28',
    'AU41','AU42','AU43','AU44','AU45','AU46',
    'AU61','AU62','AU63','AU64','AU65','AU66','AU67','AU68','AU69',
    'AD19','AD29','AD30','AD31','AD32','AD33','AD34','AD35','AD36'
]


class HFAInference:
    """
    Real-time HFA Network inference wrapper.
    Feed one face patch per frame using update().
    Prediction fires automatically every T_FRAMES patches.
    """

    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cpu'):
        self.device       = torch.device(device)
        self.model        = HFANetwork().to(self.device)
        self.model.eval()
        self._patch_buf   = deque(maxlen=T_FRAMES)
        self._last_result = {}
        self._frame_count = 0

        # if os.path.exists(model_path):
        #     state = torch.load(model_path, map_location=self.device)
        #     self.model.load_state_dict(state)
        #     print(f"HFA: loaded weights from {model_path}")
        # else:
        #     print("HFA: no weights found — using untrained model.")

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state)
                print(f"HFA: loaded weights from {model_path}")
            except Exception as e:
                print(f"WARNING: Corrupt HFA weights ({e}). Skipping — using random weights.")
                os.remove(model_path)
        else:
            print("HFA: no weights found — using untrained model.")



    def update(self, frame_bgr: np.ndarray,
               landmarks: np.ndarray) -> dict:
        """
        Call every frame.

        frame_bgr  : (H, W, 3) BGR numpy array
        landmarks  : (468, 3) from LandmarkExtractor (pixel coords)
                     OR None if no face detected

        Returns the latest prediction dict (updated every T_FRAMES frames).
        """
        self._frame_count += 1

        # Extract face patch for this frame
        patch = self._extract_patch(frame_bgr, landmarks)
        self._patch_buf.append(patch)

        # Run inference when buffer is full
        # if (len(self._patch_buf) == T_FRAMES
        #         and self._frame_count % T_FRAMES == 0):

        if (len(self._patch_buf) == T_FRAMES
                and self._frame_count % 60 == 0):  # run HFA every 60 frames, not 16
            self._last_result = self._predict()

        return self._last_result

    def _extract_patch(self, frame_bgr: np.ndarray,
                       landmarks: np.ndarray) -> np.ndarray:
        """
        Returns (3, 96, 96) float32 patch in [0,1].
        Falls back to centre crop if landmarks unavailable.
        """
        if landmarks is not None:
            # Convert pixel landmarks to normalized [0,1] for crop_face_patch
            h, w = frame_bgr.shape[:2]
            lm_norm = landmarks.copy()
            lm_norm[:, 0] /= w
            lm_norm[:, 1] /= h
            patch_rgb = crop_face_patch(frame_bgr, lm_norm, PATCH_SIZE)
        else:
            patch_rgb = None

        if patch_rgb is None:
            # Fallback: centre crop
            h, w   = frame_bgr.shape[:2]
            half   = min(h, w) // 2
            cy, cx = h // 2, w // 2
            crop   = frame_bgr[cy-half:cy+half, cx-half:cx+half]
            crop   = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))
            patch_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        patch = (patch_rgb.astype(np.float32) / 255.0)
        return patch.transpose(2, 0, 1)   # (3, 96, 96)

    @torch.no_grad()
    def _predict(self) -> dict:
        """Runs HFANetwork on the current T_FRAMES buffer."""
        # Build clip tensor: (1, T, 3, 96, 96)
        clip = np.stack(list(self._patch_buf), axis=0)   # (T, 3, 96, 96)
        clip_t = torch.tensor(clip, dtype=torch.float32) \
                      .unsqueeze(0).to(self.device)

        au_scores, emotion_out, spatial_maps = self.model(clip_t)

        # Probabilities
        emotion_probs = F.softmax(emotion_out, dim=1) \
                          .squeeze(0).cpu().numpy()
        au_vals       = au_scores.squeeze(0).cpu().numpy()
        smap          = spatial_maps.squeeze(0).cpu().numpy()  # (T, 1, 12, 12)

        emotion_idx  = int(emotion_probs.argmax())
        confidence   = float(emotion_probs.max())

        # Top AU activations (threshold 0.3)
        active_aus = {
            AU_NAMES[i]: round(float(au_vals[i]), 3)
            for i in range(min(len(AU_NAMES), len(au_vals)))
            if au_vals[i] > 0.30
        }

        # Attention heatmap from last frame's spatial map
        attn_last = smap[-1, 0]                            # (12, 12)
        attn_norm = (attn_last - attn_last.min()) / \
                    (attn_last.max() - attn_last.min() + 1e-6)
        attn_vis  = cv2.resize(
            (attn_norm * 255).astype(np.uint8),
            (PATCH_SIZE, PATCH_SIZE),
            interpolation=cv2.INTER_CUBIC
        )
        attn_heat = cv2.applyColorMap(attn_vis, cv2.COLORMAP_JET)

        return {
            'emotion':       EMOTION_NAMES[emotion_idx],
            'emotion_idx':   emotion_idx,
            'confidence':    round(confidence, 3),
            'emotion_probs': emotion_probs.tolist(),
            'active_aus':    active_aus,
            'attn_heatmap':  attn_heat,     # (96, 96, 3) BGR for cv2.imshow
            'attn_raw':      attn_norm      # (12, 12) for custom rendering
        }

    def overlay_heatmap(self, frame_bgr: np.ndarray,
                        landmarks: np.ndarray,
                        alpha: float = 0.45) -> np.ndarray:
        """
        Overlays the attention heatmap onto the face region of the frame.
        Call after update() to get a live visualization.
        """
        if not self._last_result or 'attn_heatmap' not in self._last_result:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        result = frame_bgr.copy()

        if landmarks is not None:
            lm_norm = landmarks.copy()
            lm_norm[:, 0] /= w
            lm_norm[:, 1] /= h

            xs = lm_norm[:, 0]
            ys = lm_norm[:, 1]
            x1 = int(max(0, xs.min() * w - w * 0.05))
            y1 = int(max(0, ys.min() * h - h * 0.05))
            x2 = int(min(w, xs.max() * w + w * 0.05))
            y2 = int(min(h, ys.max() * h + h * 0.05))

            heat = cv2.resize(
                self._last_result['attn_heatmap'],
                (x2 - x1, y2 - y1)
            )
            roi = result[y1:y2, x1:x2]
            if roi.shape == heat.shape:
                result[y1:y2, x1:x2] = cv2.addWeighted(
                    roi, 1 - alpha, heat, alpha, 0
                )

        return result
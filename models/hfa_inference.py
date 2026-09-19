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

    Prediction cadence (FIXED — see technical audit §4/§1): this used to
    be hardcoded to `frame_count % 60 == 0` despite the docstring/deque
    claiming "every T_FRAMES(=16) frames", which meant ~44 of every 60
    camera frames were captured into the buffer but never actually
    predicted on -- a periodic, non-sliding-window inference pattern.
    `predict_every` now defaults to T_FRAMES (restoring the originally
    intended cadence) and is configurable if you need to trade accuracy
    for CPU headroom; it must never silently regress back to a large
    interval like 60.
    """

    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cpu',
                 predict_every: int = T_FRAMES):
        self.device        = torch.device(device)
        self.model         = HFANetwork().to(self.device)
        self.model.eval()
        self._patch_buf    = deque(maxlen=T_FRAMES)
        self._last_result  = {}
        self._frame_count  = 0
        self.predict_every = max(1, int(predict_every))

        self.checkpoint_loaded = False
        # AU head is NEVER trained by the current hfa_trainer.py
        # (HFALoss is constructed with au_weight=0.0) -- its sigmoid
        # output is therefore not a real AU detector regardless of
        # whether emotion weights were loaded. Callers must not present
        # `active_aus` from this class as validated AU detections.
        self.au_head_trained = False

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state)
                self.checkpoint_loaded = True
                print(f"HFA: loaded weights from {model_path}")
                print("HFA: NOTE — the AU head in this checkpoint was trained "
                      "with au_weight=0.0 (see models/hfa_trainer.py) unless "
                      "you have since retrained with real AU labels. Its "
                      "'active_aus' output is NOT a validated AU detector.")
            except Exception as e:
                print(f"WARNING: Corrupt HFA weights ({e}). Skipping — using random weights.")
        else:
            print("HFA: no weights found — using untrained model (all outputs, "
                  "including emotion, are random until models/hfa_trainer.py is run).")

    def update(self, frame_bgr: np.ndarray,
               landmarks: np.ndarray) -> dict:
        """
        Call every frame.

        frame_bgr  : (H, W, 3) BGR numpy array
        landmarks  : (468, 3) from LandmarkExtractor (pixel coords)
                     OR None if no face detected

        Returns the latest prediction dict, refreshed every
        `self.predict_every` frames (default: every T_FRAMES frames,
        i.e. as close to the 16-frame clip length as CPU budget allows —
        NOT every 60 frames).
        """
        self._frame_count += 1

        # Extract face patch for this frame
        patch = self._extract_patch(frame_bgr, landmarks)
        self._patch_buf.append(patch)

        if (len(self._patch_buf) == T_FRAMES
                and self._frame_count % self.predict_every == 0):
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
            'emotion':          EMOTION_NAMES[emotion_idx],
            'emotion_idx':      emotion_idx,
            'confidence':       round(confidence, 3),
            'emotion_probs':    emotion_probs.tolist(),
            'active_aus':       active_aus,
            'au_head_trained':  self.au_head_trained,  # always False today — see __init__
            'attn_heatmap':     attn_heat,     # (96, 96, 3) BGR for cv2.imshow
            'attn_raw':         attn_norm      # (12, 12) for custom rendering
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
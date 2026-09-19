import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from mmbi.temporal.buffer import FEATURE_DIM as BUFFER_FEATURE_DIM

# Micro-expression classes from CASME II dataset
MICRO_CLASSES = {
    0: 'happiness',
    1: 'disgust',
    2: 'repression',
    3: 'surprise',
    4: 'others'
}

# Phase labels — onset / apex / offset detection
PHASE_CLASSES = {
    0: 'neutral',
    1: 'onset',
    2: 'apex',
    3: 'offset'
}

# FEATURE_DIM is derived from temporal/buffer.py's FEATURE_KEYS so the two
# never silently drift apart. It is now 22 (was 17): AU + gaze + head +
# blink + 5 regional optical-flow magnitude features (see
# face/microexpression.py / temporal/buffer.py for what changed and why).
FEATURE_DIM  = BUFFER_FEATURE_DIM
# SEQ_LEN is a FRAME COUNT, not a fixed duration. The original comment
# here ("60 frames = 500ms at 120 FPS") assumed a camera speed that most
# webcams cannot sustain. Use Camera.measure_actual_fps() /
# TemporalBuffer.window_duration_ms(SEQ_LEN) to find out what SEQ_LEN
# actually represents in milliseconds on your hardware. If real FPS is
# far from 120, this window is not really covering a 40-500ms micro-
# expression and SEQ_LEN should be revisited (and the model retrained)
# with a value derived from the true FPS.
SEQ_LEN      = 60
N_EMOTIONS   = len(MICRO_CLASSES)
N_PHASES     = len(PHASE_CLASSES)
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'micro_lstm.pt')


class MicroLSTM(nn.Module):
    """
    Two-layer bidirectional LSTM for micro-expression classification,
    onset/apex/offset phase, binary spotting, and intensity estimation.

    Input:  (batch_size, seq_len, feature_dim)  ->  (B, 60, 22)
    Output: emotion_logits (B, 5), phase_logits (B, 4),
            spotting_logits (B, 1), intensity (B, 1)

    Bidirectional because micro-expressions need context from both
    directions — the apex is identifiable by what comes before AND after.

    Design note on spotting/intensity vs. emotion/phase: all four heads
    share the SAME pooled context vector (temporal-attention-weighted
    sum over the window). This mirrors how the window is used at
    inference time — one sliding window -> one set of predictions "as of
    the last frame in the window" (see models/casme_trainer.py for how
    per-frame phase/spotting/intensity labels are aligned to windows for
    training, and engine.py for how windows are queried every frame in
    real-time inference).
    """

    def __init__(
        self,
        feature_dim: int  = FEATURE_DIM,
        hidden_size: int  = 128,
        num_layers:  int  = 2,
        dropout:     float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        # Input projection: expand feature_dim to 64 before LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size   = 64,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        lstm_out_dim = hidden_size * self.directions   # 256 if bidirectional

        # Temporal attention — learns WHICH frames matter most
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Output heads
        self.dropout     = nn.Dropout(dropout)

        self.emotion_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_EMOTIONS)
        )
        self.phase_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, N_PHASES)
        )
        # NEW: binary "is this window a micro-expression" spotting head.
        # Outputs a single logit; apply sigmoid for a probability.
        self.spotting_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        # NEW: continuous intensity regression head, 0 (neutral) .. 1 (apex).
        # Sigmoid-activated so the output is bounded in [0, 1] by
        # construction, matching the proxy target built in
        # casme_trainer.py (triangular onset->apex->offset curve).
        self.intensity_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, feature_dim)
        Returns:
          emotion_logits:  (batch, N_EMOTIONS)
          phase_logits:    (batch, N_PHASES)
          spotting_logits: (batch, 1)   -- raw logit, apply sigmoid for prob
          intensity:       (batch, 1)   -- already in [0, 1]
          attn_weights:    (batch, seq_len)  — for visualization
        """
        # Project input
        x = self.input_proj(x)           # (B, T, 64)

        # LSTM
        lstm_out, _ = self.lstm(x)       # (B, T, 256)

        # Temporal attention
        attn_scores  = self.attention(lstm_out)          # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)     # (B, T, 1)
        context      = (lstm_out * attn_weights).sum(1)  # (B, 256)

        context = self.dropout(context)

        emotion_logits  = self.emotion_head(context)    # (B, 5)
        phase_logits    = self.phase_head(context)       # (B, 4)
        spotting_logits = self.spotting_head(context)     # (B, 1)
        intensity       = self.intensity_head(context)    # (B, 1)
        attn_out        = attn_weights.squeeze(-1)         # (B, T)

        return emotion_logits, phase_logits, spotting_logits, intensity, attn_out


class MicroLSTMInference:
    """
    Wrapper around MicroLSTM for real-time inference.
    Loads saved weights and runs prediction on a numpy window.

    Honesty guarantee: this class loads checkpoints with strict=False and
    inspects exactly which parameter tensors came from the checkpoint
    file vs. which were left at their random nn.Module initialization
    (because the checkpoint predates a head, or has an incompatible
    shape e.g. from the FEATURE_DIM 17->22 change). Untrained heads are
    reported in `self.untrained_heads` and echoed into every prediction
    dict under the `warnings` key -- callers (engine.py, dashboard) MUST
    check this before presenting phase/spotting/intensity output as real.
    """

    ALL_HEADS = ['emotion_head', 'phase_head', 'spotting_head', 'intensity_head', 'input_proj']

    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model  = MicroLSTM().to(self.device)
        self.model.eval()

        self.checkpoint_loaded = False
        self.untrained_heads = list(self.ALL_HEADS)  # assume nothing loaded until proven otherwise

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=False)
                own_state = self.model.state_dict()

                loaded_keys = []
                skipped_keys = []
                for k, v in state.items():
                    if k in own_state and own_state[k].shape == v.shape:
                        own_state[k] = v
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append(k)
                missing_keys = [k for k in own_state.keys() if k not in loaded_keys]

                self.model.load_state_dict(own_state)
                self.checkpoint_loaded = True

                # Work out which whole HEADS ended up with at least one
                # missing/randomly-initialized parameter tensor.
                self.untrained_heads = sorted({
                    head for head in self.ALL_HEADS
                    if any(k.startswith(head + '.') for k in missing_keys)
                })

                print(f"Loaded MicroLSTM weights from {model_path}")
                print(f"  Loaded tensors:  {len(loaded_keys)}")
                print(f"  Skipped/missing: {len(missing_keys)} "
                      f"(shape mismatch or new parameter)")
                if self.untrained_heads:
                    print(f"  WARNING: the following heads are NOT fully "
                          f"covered by this checkpoint and are therefore "
                          f"UNTRAINED (random weights): {self.untrained_heads}")
                else:
                    print("  All heads fully covered by checkpoint.")

            except Exception as e:
                print(f"WARNING: Corrupt/incompatible LSTM weights ({e}). "
                      f"Using random weights for ALL heads.")
        else:
            print("No saved weights found. Using random weights for ALL heads "
                  "until trained — see models/casme_trainer.py.")

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> dict:
        """
        window: numpy array of shape (SEQ_LEN, FEATURE_DIM) from TemporalBuffer.
        Returns dict with emotion, phase, spotting probability, intensity,
        confidence, attention weights, and an explicit `warnings` list
        naming any head whose output should NOT be trusted because its
        weights are untrained.
        """
        if window is None or window.shape != (SEQ_LEN, FEATURE_DIM):
            return {}

        # Normalize input per feature (simple min-max)
        w = window.copy()
        w_min = w.min(axis=0, keepdims=True)
        w_max = w.max(axis=0, keepdims=True)
        w = (w - w_min) / (w_max - w_min + 1e-6)

        tensor = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(self.device)

        emotion_logits, phase_logits, spotting_logits, intensity, attn = self.model(tensor)

        emotion_probs  = F.softmax(emotion_logits, dim=1).squeeze(0).cpu().numpy()
        phase_probs    = F.softmax(phase_logits,   dim=1).squeeze(0).cpu().numpy()
        spotting_prob  = float(torch.sigmoid(spotting_logits).squeeze().cpu().item())
        intensity_val  = float(intensity.squeeze().cpu().item())
        attn_weights   = attn.squeeze(0).cpu().numpy()

        emotion_idx    = int(emotion_probs.argmax())
        phase_idx      = int(phase_probs.argmax())
        confidence     = float(emotion_probs.max())

        warnings = [
            f"{head} weights are UNTRAINED (random init) — its output "
            f"below is not meaningful until models/casme_trainer.py is "
            f"run and this checkpoint is regenerated."
            for head in self.untrained_heads
        ]

        return {
            'emotion':          MICRO_CLASSES[emotion_idx],
            'emotion_idx':      emotion_idx,
            'emotion_probs':    emotion_probs.tolist(),
            'phase':            PHASE_CLASSES[phase_idx],
            'phase_probs':      phase_probs.tolist(),
            'spotting_prob':    round(spotting_prob, 4),
            'intensity':        round(intensity_val, 4),
            'confidence':       round(confidence, 3),
            'attn_weights':     attn_weights.tolist(),
            'untrained_heads':  list(self.untrained_heads),
            'warnings':         warnings,
        }

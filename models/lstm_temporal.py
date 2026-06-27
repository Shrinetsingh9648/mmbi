import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

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

FEATURE_DIM  = 17    # matches buffer.py FEATURE_KEYS
SEQ_LEN      = 60    # 60 frames = 500ms at 120 FPS (max micro-expression)
N_EMOTIONS   = len(MICRO_CLASSES)
N_PHASES     = len(PHASE_CLASSES)
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'micro_lstm.pt')


class MicroLSTM(nn.Module):
    """
    Two-layer bidirectional LSTM for micro-expression classification.
    Input:  (batch_size, seq_len, feature_dim)  →  (B, 60, 17)
    Output: emotion_logits (B, 5), phase_logits (B, 4)

    Bidirectional because micro-expressions need context from both
    directions — the apex is identifiable by what comes before AND after.
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

        # Input projection: expand 17 dims to 64 before LSTM
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

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, feature_dim)
        Returns:
          emotion_logits: (batch, N_EMOTIONS)
          phase_logits:   (batch, N_PHASES)
          attn_weights:   (batch, seq_len)  — for visualization
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

        emotion_logits = self.emotion_head(context)   # (B, 5)
        phase_logits   = self.phase_head(context)     # (B, 4)
        attn_out       = attn_weights.squeeze(-1)     # (B, T)

        return emotion_logits, phase_logits, attn_out


class MicroLSTMInference:
    """
    Wrapper around MicroLSTM for real-time inference.
    Loads saved weights and runs prediction on a numpy window.
    """

    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model  = MicroLSTM().to(self.device)
        self.model.eval()

        # if os.path.exists(model_path):
        #     state = torch.load(model_path, map_location=self.device)
        #     self.model.load_state_dict(state)
        #     print(f"Loaded MicroLSTM weights from {model_path}")
        # else:
        #     print("No saved weights found. Model will use random weights until trained.")
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state)
                print(f"Loaded MicroLSTM weights from {model_path}")
            except Exception as e:
                print(f"WARNING: Corrupt LSTM weights ({e}). Skipping — using random weights.")
                os.remove(model_path)
        else:
            print("No saved weights found. Using random weights until trained.")

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> dict:
        """
        window: numpy array of shape (60, 17) from TemporalBuffer.
        Returns dict with emotion, phase, confidence, attention weights.
        """
        if window is None or window.shape != (SEQ_LEN, FEATURE_DIM):
            return {}

        # Normalize input per feature (simple min-max)
        w = window.copy()
        w_min = w.min(axis=0, keepdims=True)
        w_max = w.max(axis=0, keepdims=True)
        w = (w - w_min) / (w_max - w_min + 1e-6)

        tensor = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(self.device)

        emotion_logits, phase_logits, attn = self.model(tensor)

        emotion_probs  = F.softmax(emotion_logits, dim=1).squeeze(0).cpu().numpy()
        phase_probs    = F.softmax(phase_logits,   dim=1).squeeze(0).cpu().numpy()
        attn_weights   = attn.squeeze(0).cpu().numpy()

        emotion_idx    = int(emotion_probs.argmax())
        phase_idx      = int(phase_probs.argmax())
        confidence     = float(emotion_probs.max())

        return {
            'emotion':        MICRO_CLASSES[emotion_idx],
            'emotion_idx':    emotion_idx,
            'emotion_probs':  emotion_probs.tolist(),
            'phase':          PHASE_CLASSES[phase_idx],
            'confidence':     round(confidence, 3),
            'attn_weights':   attn_weights.tolist()
        }
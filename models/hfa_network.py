"""
HFA Network — High-Frequency Attention Network
Day 10: HighFrequencyFilter + CNN Backbone
Day 11: SpatialAttention + ChannelAttention (CBAM-style)
Day 12: TemporalTransformer + AU head + Emotion head

Input  : (batch, T, 3, 96, 96)  — T=16 face patch frames per clip
Output : au_scores   (batch, 46)   — sigmoid 0–1 per AU
         emotion_out (batch, 5)    — softmax micro-expression class
         spatial_attn(batch, T, 1, H, W) — for visualization

Why 6 channels after HF filter:
  The Laplacian of an RGB image gives 3 edge maps (one per channel).
  Concatenating [RGB | Laplacian] gives 6 channels.
  This forces the CNN to see BOTH the smooth face structure (RGB)
  AND the high-frequency wrinkle/edge signal (Laplacian) at the same time.
  Standard CNNs trained on RGB alone learn to ignore fine texture.
  This is the core insight of the HFA architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ── Constants ────────────────────────────────────────────────────────────────
PATCH_SIZE   = 96        # face patch height and width in pixels
T_FRAMES     = 16        # frames per clip (133 ms at 120 FPS)
N_AU         = 46        # FACS action units
N_EMOTION    = 5         # micro-expression classes (CASME II)
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'hfa_net.pt')


# ═════════════════════════════════════════════════════════════════════════════
# DAY 10 — High-Frequency Filter
# ═════════════════════════════════════════════════════════════════════════════

class HighFrequencyFilter(nn.Module):
    """
    Applies a fixed Laplacian convolution to each input channel.
    The Laplacian kernel amplifies edges, wrinkles, and skin texture —
    exactly where micro-expressions appear.

    Input : (B, 3, H, W)  — standard RGB
    Output: (B, 6, H, W)  — RGB concatenated with 3 Laplacian maps

    The kernel is NOT learned. It is a fixed mathematical operator.
    This is intentional: we do not want the network to learn to ignore
    high-frequency signals during early training.
    """

    def __init__(self):
        super().__init__()

        # Standard discrete Laplacian kernel
        # Positive centre, negative neighbours = highlights intensity changes
        laplacian = torch.tensor([
            [ 0., -1.,  0.],
            [-1.,  4., -1.],
            [ 0., -1.,  0.]
        ], dtype=torch.float32)

        # Shape: (out_channels, in_channels/groups, kH, kW)
        # groups=3 means each channel filtered independently (depthwise)
        kernel = laplacian.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # register_buffer: saved with model state but NOT trained
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        Returns: (B, 6, H, W)
        """
        # Depthwise convolution — one kernel per channel
        hf = F.conv2d(x, self.kernel, padding=1, groups=3)

        # Normalize high-frequency maps to [0, 1]
        hf = torch.clamp(hf, 0.0, 1.0)

        # Concatenate original RGB with high-frequency maps
        return torch.cat([x, hf], dim=1)   # (B, 6, H, W)


# ═════════════════════════════════════════════════════════════════════════════
# DAY 10 — CNN Backbone
# ═════════════════════════════════════════════════════════════════════════════

class CNNBackbone(nn.Module):
    """
    Three-stage convolutional backbone.
    Input : (B, 6, 96, 96)
    Output: (B, 128, 12, 12)  — 128 feature maps at 12×12 spatial resolution

    Spatial resolution trace:
      96×96 → MaxPool → 48×48 → MaxPool → 24×24 → MaxPool → 12×12
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.stage1 = conv_block(6,   32)    # 96→48,  6 ch → 32 ch
        self.stage2 = conv_block(32,  64)    # 48→24, 32 ch → 64 ch
        self.stage3 = conv_block(64, 128)    # 24→12, 64 ch → 128 ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)   # (B, 32,  48, 48)
        x = self.stage2(x)   # (B, 64,  24, 24)
        x = self.stage3(x)   # (B, 128, 12, 12)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# DAY 11 — Spatial Attention
# ═════════════════════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    """
    Spatial attention module — learns WHERE to look on the face.
    On a 12×12 feature map from a 96×96 face patch, each cell
    corresponds to an 8×8 pixel region.

    The model learns to assign high weights to the eye, brow and
    mouth corner regions — exactly where AUs are expressed.

    Input : (B, C, 12, 12)
    Output: (B, C, 12, 12)  — spatially reweighted features
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Compress channels → 1 attention map
        self.attention_map = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
          out      : (B, C, H, W) — weighted feature map
          attn_map : (B, 1, H, W) — attention weights (for visualization)
        """
        attn_map = self.attention_map(x)   # (B, 1, 12, 12)
        out      = x * attn_map            # broadcast multiply
        return out, attn_map


# ═════════════════════════════════════════════════════════════════════════════
# DAY 11 — Channel Attention (CBAM-style)
# ═════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    Channel attention — learns WHICH feature maps matter.
    Some CNN filters will learn to detect wrinkles, others eye openness,
    others lip tension. Channel attention weights which of these signals
    to emphasize for the current frame.

    Input : (B, C, H, W)
    Output: (B, C, H, W)  — channel-reweighted features
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Global average pool + global max pool
        avg_pool = x.mean(dim=[2, 3])    # (B, C)
        max_pool = x.amax(dim=[2, 3])    # (B, C)
        # MLP on both, sum, sigmoid
        weights  = self.sigmoid(
            self.mlp(avg_pool) + self.mlp(max_pool)
        )                                # (B, C)
        # Reweight channels
        return x * weights.unsqueeze(-1).unsqueeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# DAY 12 — Temporal Transformer
# ═════════════════════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    """
    Transformer encoder over the T=16 frame sequence.
    Each frame's CNN feature vector becomes one token.
    The Transformer learns which frames in the sequence correspond
    to onset, apex, and offset of the micro-expression.

    Input : (B, T, feature_dim)    — T=16 frame tokens
    Output: (B, feature_dim)       — aggregated sequence representation

    Why Transformer instead of LSTM here:
      The LSTM (Days 7-9) operates on lightweight AU vectors (17 dims).
      This Transformer operates on rich CNN feature maps (18432 dims
      projected to 256). The Transformer's attention mechanism is better
      at finding the apex frame within 16 frames than LSTM because it
      can directly compare any two frames regardless of distance.
    """

    def __init__(
        self,
        feature_dim:  int = 18432,
        proj_dim:     int = 256,
        n_heads:      int = 4,
        n_layers:     int = 2,
        dropout:      float = 0.1
    ):
        super().__init__()

        # Project high-dim CNN features down to manageable size
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # Learnable positional encoding — each frame position gets
        # a unique embedding so the Transformer knows frame order
        self.pos_embed = nn.Parameter(
            torch.randn(1, T_FRAMES, proj_dim) * 0.02
        )

        # Standard Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = proj_dim,
            nhead           = n_heads,
            dim_feedforward = proj_dim * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True    # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, feature_dim)
        Returns: (B, proj_dim) — last-frame representation used for heads
        """
        x = self.input_proj(x)            # (B, T, 256)
        x = x + self.pos_embed            # add positional encoding
        x = self.transformer(x)           # (B, T, 256)
        x = self.norm(x)
        # Use the last frame token as the sequence summary
        # (corresponds to the most recent frame — onset→apex→offset read left to right)
        return x[:, -1, :]                # (B, 256)


# ═════════════════════════════════════════════════════════════════════════════
# FULL HFA NETWORK — wires all components together
# ═════════════════════════════════════════════════════════════════════════════

class HFANetwork(nn.Module):
    """
    Complete High-Frequency Attention Network.

    Forward pass for a single clip of T=16 frames:
      1. For each frame: HF filter → CNN → spatial attn → channel attn → flatten
      2. Stack T frame vectors → Temporal Transformer
      3. Two output heads: AU scores (46) + emotion (5)

    Input : (B, T, 3, 96, 96)   T=16 frames, 3-channel RGB, 96×96 patch
    Output:
      au_scores    (B, 46)       sigmoid — per-AU activation 0–1
      emotion_out  (B, 5)        logits  — cross-entropy loss target
      spatial_maps (B, T, 1, 12, 12) — for visualization only
    """

    def __init__(self):
        super().__init__()
        self.hf_filter      = HighFrequencyFilter()
        self.cnn            = CNNBackbone()
        self.spatial_attn   = SpatialAttention(128)
        self.channel_attn   = ChannelAttention(128)
        self.temporal       = TemporalTransformer(
            feature_dim = 128 * 12 * 12,   # 18432
            proj_dim    = 256
        )
        self.dropout        = nn.Dropout(0.3)

        # AU head: 256 → 128 → 46 (sigmoid output)
        self.au_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, N_AU),
            nn.Sigmoid()
        )

        # Emotion head: 256 → 64 → 5 (logits, softmax at inference)
        self.emotion_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_EMOTION)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for Conv2d, Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, 3, H, W)
        """
        B, T, C, H, W = x.shape
        spatial_maps   = []
        frame_features = []

        # Process each frame through the per-frame stack
        for t in range(T):
            frame = x[:, t]                      # (B, 3, H, W)

            # Step 1: High-frequency filter
            frame = self.hf_filter(frame)         # (B, 6, H, W)

            # Step 2: CNN backbone
            feat  = self.cnn(frame)               # (B, 128, 12, 12)

            # Step 3: Spatial attention
            feat, smap = self.spatial_attn(feat)  # (B, 128, 12, 12), (B, 1, 12, 12)
            spatial_maps.append(smap)

            # Step 4: Channel attention
            feat  = self.channel_attn(feat)       # (B, 128, 12, 12)

            # Step 5: Flatten to vector
            feat  = feat.flatten(1)               # (B, 18432)
            frame_features.append(feat)

        # Stack into sequence: (B, T, 18432)
        sequence = torch.stack(frame_features, dim=1)

        # Step 6: Temporal Transformer
        context  = self.temporal(sequence)        # (B, 256)
        context  = self.dropout(context)

        # Step 7: Output heads
        au_scores   = self.au_head(context)       # (B, 46)
        emotion_out = self.emotion_head(context)  # (B, 5)

        # Stack spatial maps: (B, T, 1, 12, 12)
        spatial_out = torch.stack(spatial_maps, dim=1)

        return au_scores, emotion_out, spatial_out
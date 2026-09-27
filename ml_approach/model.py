"""
Physics-Guided Mass Spectrometry Predictor — Model v4

Key redesign: Per-channel sigmoid output (not softmax).
- Softmax forces a "probability budget" across all 500 channels → every channel
  gets a non-zero value → noisy, over-predicted spectrum.
- Sigmoid treats each m/z channel INDEPENDENTLY: 0 = no peak, 1 = full peak.
- This is the approach used in NEIMS (Wei et al. 2019, ACS Cent. Sci.) and other
  state-of-the-art EI-MS predictors.
- Result: naturally sparse predictions matching real spectral data.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block with GELU and bottleneck expansion."""
    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.10):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim * expand)
        self.fc2   = nn.Linear(dim * expand, dim)
        self.drop  = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.8)
        nn.init.zeros_(self.fc1.bias)
        # Small init for fc2 so residuals start as identity
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(self.norm1(x)))
        h = self.drop(self.fc2(h))
        return x + h


class ResNetMassNet(nn.Module):
    """
    EI-MS Predictor with per-channel sigmoid output.

    Training (return_spectrum=False):
        Returns raw logits → BCEWithLogitsLoss handles sigmoid internally.

    Inference (return_spectrum=True):
        sigmoid(logits) × 100 → zero out < THRESHOLD% of base peak → normalise.
        Produces clean, sparse spectra matching the NIST style.
    """
    # Threshold: only report peaks ≥ 2% of base peak (industry standard noise floor)
    SPARSE_THRESHOLD = 2.0  # percent of base peak

    def __init__(self, in_features: int = 2238, hidden_dim: int = 512,
                 num_blocks: int = 5, max_mz: int = 500, dropout: float = 0.10):
        super().__init__()
        self.max_mz = max_mz

        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_mz),
        )
        self.register_buffer("mz_coords", torch.arange(1, max_mz + 1, dtype=torch.float32))
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                # Bias init: push initial sigmoid outputs toward 0 (sparse start)
                # sigmoid(-4) ≈ 0.018 — model starts predicting nearly all zeros
                nn.init.constant_(m.bias, -4.0)

    def _physics_mask(self, logits: torch.Tensor, mw: torch.Tensor) -> torch.Tensor:
        """Zero-mask (via large negative bias) channels beyond MW+2 Da."""
        mask = self.mz_coords.unsqueeze(0) > (torch.floor(mw.unsqueeze(-1)) + 2.0)
        return logits.masked_fill(mask, -10.0)

    def forward(self, x: torch.Tensor, mw: Optional[torch.Tensor] = None,
                return_spectrum: bool = False) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(h)                        # (B, max_mz) — raw

        if mw is not None:
            logits = self._physics_mask(logits, mw)

        if return_spectrum:
            # Convert to intensity space: sigmoid → [0, 1] → scale to [0, 100]
            probs = torch.sigmoid(logits) * 100.0    # (B, max_mz)

            # Sparsify: zero out channels below THRESHOLD% of base peak
            base = probs.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            probs = probs * (probs >= self.SPARSE_THRESHOLD / 100.0 * base).float()

            # Re-normalise: base peak = 100
            base2 = probs.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            return 100.0 * probs / base2

        # Training: return raw logits for BCEWithLogitsLoss
        return logits

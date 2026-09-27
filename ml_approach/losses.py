"""
Loss functions — v4

Primary loss: Weighted BCEWithLogitsLoss
  - Each m/z channel predicted independently (no softmax competition)
  - High-intensity peaks receive up to 10× more weight than background
  - Zero-intensity channels are pushed toward 0 by BCE → natural sparsity

Secondary loss: Stein-Scott Cosine Similarity
  - Directly optimises the evaluation metric
  - Mass-weighted (heavier ions get more importance)
  - Applied in intensity space after sigmoid

Combined: total = BCE + alpha * (1 - cosine_sim)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for sparse spectral prediction.

    logits : (B, max_mz) — raw model output (before sigmoid)
    target : (B, max_mz) — intensities in [0, 100] base-peak scale

    Peak importance weighting:
        w = 1 + peak_boost * (target / 100)
    So zero channels get weight=1, base peak channel gets weight=(1 + peak_boost).
    Default peak_boost=9 → 10× more emphasis on high-intensity peaks.
    """
    def __init__(self, peak_boost: float = 9.0):
        super().__init__()
        self.peak_boost = peak_boost

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_norm = target / 100.0                          # [0, 1]
        weight      = 1.0 + self.peak_boost * target_norm    # [1, 10]
        return F.binary_cross_entropy_with_logits(logits, target_norm, weight=weight)


class SteinScottCosLoss(nn.Module):
    """
    Stein-Scott mass-weighted cosine loss in intensity space.
    Works directly on sigmoid outputs (scale-invariant by normalising to [0,1]).

    Encourages correct relative peak heights independently of overall scale.
    """
    def __init__(self, max_mz: int = 500, mass_power: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        mz = torch.arange(1, max_mz + 1, dtype=torch.float32)
        self.register_buffer("weights", torch.pow(mz, mass_power).unsqueeze(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred  = torch.sigmoid(logits) * 100.0      # [0,100] scale
        w = self.weights

        p_mod = w * torch.sqrt(pred.clamp(min=0))
        t_mod = w * torch.sqrt(target.clamp(min=0))

        dot   = (p_mod * t_mod).sum(dim=-1)
        n_p   = torch.sqrt((p_mod ** 2).sum(dim=-1) + self.eps)
        n_t   = torch.sqrt((t_mod ** 2).sum(dim=-1) + self.eps)

        cosine = dot / (n_p * n_t)
        return (1.0 - cosine).mean()


class HybridSpectralLoss(nn.Module):
    """
    Combined loss: Weighted BCE (sparsity) + Stein-Scott Cosine (accuracy).

    BCE ensures zero-intensity channels stay zero → no overprediction.
    Cosine loss ensures correct relative peak heights.
    """
    def __init__(self, max_mz: int = 500, peak_boost: float = 9.0,
                 cos_weight: float = 0.5):
        super().__init__()
        self.bce = WeightedBCELoss(peak_boost=peak_boost)
        self.cos = SteinScottCosLoss(max_mz=max_mz)
        self.cos_weight = cos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.cos_weight * self.cos(logits, target)


# Backward-compat alias used by old imports
SpectralKLLoss = HybridSpectralLoss


def compute_spectral_metrics(pred_vec: np.ndarray, target_vec: np.ndarray,
                             threshold_pct: float = 15.0) -> dict:
    """
    Quantitative metrics between predicted and experimental spectra.
    Both inputs in [0, 100] base-peak normalised scale.
    """
    pred_vec   = np.asarray(pred_vec,   dtype=np.float64)
    target_vec = np.asarray(target_vec, dtype=np.float64)

    p_max = np.max(pred_vec);   t_max = np.max(target_vec)
    p_norm = (pred_vec   / p_max * 100.0) if p_max > 0 else pred_vec.copy()
    t_norm = (target_vec / t_max * 100.0) if t_max > 0 else target_vec.copy()

    mz_arr = np.arange(1, len(pred_vec) + 1, dtype=np.float64)
    w = np.sqrt(mz_arr)

    p_mod  = w * np.sqrt(np.maximum(p_norm, 0.0))
    t_mod  = w * np.sqrt(np.maximum(t_norm, 0.0))
    denom  = np.linalg.norm(p_mod) * np.linalg.norm(t_mod)
    cosine = float(np.dot(p_mod, t_mod) / denom) if denom > 0 else 0.0

    pred_base_mz   = int(np.argmax(p_norm) + 1)
    target_base_mz = int(np.argmax(t_norm) + 1)
    base_match     = (pred_base_mz == target_base_mz)

    major_peaks = np.where(t_norm >= threshold_pct)[0]
    recall = float(np.sum(p_norm[major_peaks] >= 1.0) / len(major_peaks) * 100.0) \
             if len(major_peaks) > 0 else 100.0

    r = float(np.corrcoef(p_norm, t_norm)[0, 1]) \
        if (np.std(p_norm) > 0 and np.std(t_norm) > 0) else 0.0

    active = np.where((p_norm > 0.5) | (t_norm > 0.5))[0]
    if len(active) > 0:
        mae  = float(np.mean(np.abs(p_norm[active] - t_norm[active])))
        rmse = float(np.sqrt(np.mean((p_norm[active] - t_norm[active]) ** 2)))
    else:
        mae = rmse = 0.0

    return {
        "cosine_similarity": cosine,
        "peak_recall_pct":   recall,
        "pred_base_mz":      pred_base_mz,
        "ref_base_mz":       target_base_mz,
        "base_match":        base_match,
        "pearson_r":         r,
        "mae_pct":           mae,
        "rmse_pct":          rmse,
    }

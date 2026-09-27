"""
Evaluation, Side-by-Side Spectrogram Generator, and Benchmark Analytics — v4

Key fixes:
- NIST reference peaks are explicitly normalised to base peak = 100 before plotting
  (raw intensities from different DB sources can have arbitrary scale)
- Both panels now apply a 2% intensity threshold — matches standard spectral display
- Shared x-axis range derived from BOTH spectra (not just ML predictions)
- Cleaner layout: matched axis limits, explicit major peak labels on both sides
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

from ml_approach.dataset import extract_molecular_features, peaks_to_vector
from ml_approach.model import ResNetMassNet
from ml_approach.losses import compute_spectral_metrics

# Minimum intensity (% of base peak) to display a line in the plot
_DISPLAY_THRESHOLD = 2.0


def predict_spectrum(model: ResNetMassNet, smiles: str,
                     device: torch.device) -> Optional[np.ndarray]:
    """Predicts a sparse [0,100]-normalised mass spectrum for a SMILES string."""
    feat_info = extract_molecular_features(smiles)
    if feat_info is None:
        return None
    feat_vec, exact_mw = feat_info

    x   = torch.tensor(feat_vec,   dtype=torch.float32).unsqueeze(0).to(device)
    mw  = torch.tensor([exact_mw], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x, mw=mw, return_spectrum=True)
    return pred.squeeze(0).cpu().numpy()


def _normalize_peaks(peaks: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Normalise a peak list so that the base peak = 100."""
    if not peaks:
        return peaks
    max_int = max(p[1] for p in peaks)
    if max_int <= 0:
        return peaks
    return [(mz, intensity / max_int * 100.0) for mz, intensity in peaks]


def _draw_spectrum(ax, mz_values: np.ndarray, intensities: np.ndarray,
                   base_mz: int, color_peak: str, color_bar: str,
                   label_threshold: float = 8.0):
    """Helper: draw a stem-style mass spectrum on ax."""
    for idx in range(len(mz_values)):
        mz_val = mz_values[idx]
        int_val = intensities[idx]
        if int_val < _DISPLAY_THRESHOLD:
            continue
        is_base = (int(round(mz_val)) == base_mz)
        color = color_peak if is_base else color_bar
        lw    = 2.4 if is_base else 1.5
        ax.vlines(x=mz_val, ymin=0, ymax=int_val, color=color, linewidth=lw, alpha=0.93)

        if int_val >= label_threshold or is_base:
            lbl = f"m/z {int(round(mz_val))}" + ("\n(Base)" if is_base else "")
            ax.text(mz_val, int_val + 2.5, lbl,
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold" if is_base else "normal",
                    color=color,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.75))


def plot_ml_side_by_side(
    mol_name:       str,
    smiles:         str,
    pred_vec:       np.ndarray,
    ref_peaks:      List[Tuple[float, float]],
    output_path:    str,
    formula:        str  = "",
    chemical_class: str  = "",
    key_pathway:    str  = "",
    db_name:        str  = "NIST Chemistry WebBook SRD 69",
) -> Dict:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ── Normalise NIST reference to base peak = 100 ───────────────────────────
    ref_peaks_norm = _normalize_peaks(ref_peaks)
    ref_vec        = peaks_to_vector(ref_peaks_norm, max_mz=500)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_spectral_metrics(pred_vec, ref_vec)

    # ── Shared m/z display range ──────────────────────────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    mw  = Descriptors.ExactMolWt(mol) if mol else 200.0

    pred_active_mz = (np.where(pred_vec >= _DISPLAY_THRESHOLD)[0] + 1).tolist()
    ref_active_mz  = [p[0] for p in ref_peaks_norm if p[1] >= _DISPLAY_THRESHOLD]
    all_mz = pred_active_mz + ref_active_mz + [mw]

    if all_mz:
        min_x = max(10,  int(min(all_mz)) - 8)
        max_x = min(500, int(max(all_mz)) + 12)
    else:
        min_x, max_x = 10, int(mw) + 12

    # ── Figure layout ─────────────────────────────────────────────────────────
    try:
        style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
        plt.style.use(style)
    except Exception:
        pass

    fig = plt.figure(figsize=(16, 9), dpi=300)
    plt.suptitle(
        f"Mass Spectrometry Side-by-Side: {mol_name}  ({formula})\n"
        f"Left: ML Prediction (ResNetMassNet v4) │ Right: Experimental Reference ({db_name})",
        fontsize=14, fontweight="bold", y=0.97, color="#0d2137"
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[0.20, 0.80], hspace=0.30, wspace=0.12)

    # ── Info panel (top-left) ─────────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis("off")
    if mol:
        try:
            img = Draw.MolToImage(mol, size=(200, 90))
            ax_info.imshow(img, aspect="equal", extent=[0, 30, 10, 90])
        except Exception:
            pass
    ax_info.text(
        32, 50,
        f"{mol_name}  |  MW ≈ {round(mw)} Da\nFormula: {formula}\n"
        f"Class: {chemical_class}\nKey fragment pathways:\n  {key_pathway}",
        fontsize=9, va="center", ha="left", color="#182838",
        family="sans-serif", linespacing=1.4
    )
    ax_info.set_xlim(0, 100); ax_info.set_ylim(0, 100)

    # ── Metrics panel (top-right) ─────────────────────────────────────────────
    ax_met = fig.add_subplot(gs[0, 1])
    ax_met.axis("off")
    match_str = "✓ MATCH" if metrics["base_match"] else "✗ MISMATCH"
    match_col = "#1a7a2a" if metrics["base_match"] else "#b02020"
    metric_txt = (
        f"  QUANTITATIVE METRICS (70 eV EI)\n"
        f"  ─────────────────────────────────────────\n"
        f"  Stein-Scott Cosine Similarity : {metrics['cosine_similarity']:.3f} / 1.000\n"
        f"  Peak Recall (≥ 15% intensity) : {metrics['peak_recall_pct']:.1f}%\n"
        f"  Pearson Correlation (r)       : {metrics['pearson_r']:.3f}\n"
        f"  Mean Absolute Error (MAE)     : {metrics['mae_pct']:.2f}%\n"
        f"  Base Peak — Pred: m/z {metrics['pred_base_mz']:<4}  Ref: m/z {metrics['ref_base_mz']:<4} {match_str}"
    )
    ax_met.text(
        0.03, 0.50, metric_txt,
        fontsize=9, va="center", ha="left", color="#162435",
        family="monospace", linespacing=1.35,
        transform=ax_met.transAxes,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#edf4fb",
                  edgecolor="#9ab8d8", linewidth=1.3)
    )

    # ── ML Predicted Spectrum (bottom-left) ───────────────────────────────────
    ax_pred = fig.add_subplot(gs[1, 0])
    pred_mz_arr   = np.arange(1, 501, dtype=float)
    pred_int_arr  = pred_vec.copy()
    _draw_spectrum(ax_pred, pred_mz_arr, pred_int_arr,
                   metrics["pred_base_mz"], "#c0392b", "#154f7a", label_threshold=8.0)

    n_pred_peaks = int(np.sum(pred_vec >= _DISPLAY_THRESHOLD))
    ax_pred.set_title(
        f"PREDICTED  (ResNetMassNet v4)\n"
        f"{n_pred_peaks} peaks above {_DISPLAY_THRESHOLD:.0f}% threshold",
        fontsize=11, fontweight="bold", color="#0e3a66", pad=8
    )
    ax_pred.set_xlabel("m/z", fontsize=11, fontweight="bold")
    ax_pred.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight="bold")
    ax_pred.set_xlim(min_x, max_x); ax_pred.set_ylim(0, 118)
    ax_pred.xaxis.set_major_locator(ticker.AutoLocator())
    ax_pred.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_pred.grid(True, linestyle="--", alpha=0.3)

    # ── NIST Reference Spectrum (bottom-right) ────────────────────────────────
    ax_ref = fig.add_subplot(gs[1, 1])
    ref_mz_arr  = np.array([p[0] for p in ref_peaks_norm])
    ref_int_arr = np.array([p[1] for p in ref_peaks_norm])
    _draw_spectrum(ax_ref, ref_mz_arr, ref_int_arr,
                   metrics["ref_base_mz"], "#8b0000", "#2c4a6e", label_threshold=8.0)

    n_ref_peaks = int(np.sum(ref_int_arr >= _DISPLAY_THRESHOLD))
    ax_ref.set_title(
        f"REFERENCE  ({db_name})\n"
        f"{n_ref_peaks} peaks reported",
        fontsize=11, fontweight="bold", color="#6b0f0f", pad=8
    )
    ax_ref.set_xlabel("m/z", fontsize=11, fontweight="bold")
    ax_ref.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight="bold")
    ax_ref.set_xlim(min_x, max_x); ax_ref.set_ylim(0, 118)
    ax_ref.xaxis.set_major_locator(ticker.AutoLocator())
    ax_ref.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_ref.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {output_path}")
    return metrics

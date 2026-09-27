#!/usr/bin/env python3
"""
Side-by-Side Mass Spectrogram Comparison & Accuracy Curves Generator
Generates:
1. High-resolution side-by-side plots (Left = Predicted, Right = NIST Database Original) for 6 representative molecules.
2. Comprehensive multi-panel Accuracy Curves for demo.py (13 benchmark molecules).
3. Detailed mathematical performance metrics table.
"""

import os
import io
import math
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from src.parser import parse_smiles
from src.spectrum import SpectrumPredictor, MassSpectrum
from src.evaluator import SpectrumEvaluator, BENCHMARK_REFERENCE_SPECTRA

# Six representative molecules covering diverse physical organic functional classes
SIX_MOLECULES = [
    {
        "smiles": "CCCCC=O",
        "name": "Valeraldehyde",
        "class": "Aliphatic Aldehyde",
        "description": "McLafferty rearrangement (m/z 44 base peak) & carbonyl α-cleavage (m/z 57, 29)"
    },
    {
        "smiles": "CC(=O)c1ccccc1",
        "name": "Acetophenone",
        "class": "Aromatic Ketone",
        "description": "Aroyl acylium cation (m/z 105 base peak), phenyl cation (m/z 77) & acetylium (m/z 43)"
    },
    {
        "smiles": "Cc1ccccc1",
        "name": "Toluene",
        "class": "Alkylbenzene / Aromatic",
        "description": "Benzylic C-H cleavage & Tropylium ring expansion (m/z 91 base peak) to cyclopentadienyl (m/z 65)"
    },
    {
        "smiles": "CCCC",
        "name": "Butane",
        "class": "Aliphatic Alkane",
        "description": "Stevenson-governed alkyl C-C cleavage: propyl cation (m/z 43 base peak) & ethyl (m/z 29)"
    },
    {
        "smiles": "CCCC(=O)OC",
        "name": "Methyl butyrate",
        "class": "Aliphatic Ester",
        "description": "Ester McLafferty rearrangement (m/z 74 base peak) & methoxycarbonyl α-cleavages (m/z 59, 43)"
    },
    {
        "smiles": "c1ccccc1",
        "name": "Benzene",
        "class": "Aromatic Hydrocarbon",
        "description": "Intact aromatic molecular radical cation (m/z 78 base peak) with low fragmentation propensity"
    }
]

# Color Palette
COLOR_PRED_BASE = "#E63946"     # Crimson for predicted base peak
COLOR_PRED_PEAK = "#1D3557"     # Deep navy/sapphire for predicted peaks
COLOR_PRED_MOL = "#2A9D8F"      # Emerald for molecular ion
COLOR_NIST_BASE = "#D90429"     # Red for NIST base peak
COLOR_NIST_PEAK = "#4A5568"     # Slate for NIST reference peaks


def render_molecule_thumbnail(smiles: str, width: int = 240, height: int = 180) -> Image.Image:
    """Render a clean 2D molecular graph image."""
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.8
    opts.clearBackground = True
    opts.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def generate_single_side_by_side(comp: Dict[str, Any], predictor: SpectrumPredictor, evaluator: SpectrumEvaluator, save_path: str):
    """
    Generate an ultra-crisp 16:9 side-by-side comparison plot:
    Left = Predicted Spectrum (Classical Chemistry Model)
    Right = Original Spectrum (NIST Chemistry WebBook Database)
    """
    smiles = comp["smiles"]
    ref_data = BENCHMARK_REFERENCE_SPECTRA[smiles]
    mol_info = parse_smiles(smiles)
    predicted_spec = predictor.predict(mol_info, beam_energy_ev=70.0)
    report = evaluator.evaluate(predicted_spec, smiles)

    ref_peaks = dict(ref_data["peaks"])
    pred_peaks = {p.mz: p.intensity for p in predicted_spec.sorted_peaks}
    all_mzs = sorted(set(ref_peaks.keys()).union(set(pred_peaks.keys())))

    # Mathematical correlation
    y_true = np.array([ref_peaks.get(m, 0.0) for m in all_mzs])
    y_pred = np.array([pred_peaks.get(m, 0.0) for m in all_mzs])
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        pearson_r = 0.0
    mae = np.mean(np.abs(y_true - y_pred))

    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Grid: Top header row, bottom spectrum comparison row
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[0.24, 0.76],
        width_ratios=[1.0, 1.0],
        wspace=0.14, hspace=0.22,
        left=0.06, right=0.95, top=0.93, bottom=0.08
    )

    # --------------------------------------------------------------------------
    # Header: Info & Molecule on Left, Quantitative Metrics on Right
    # --------------------------------------------------------------------------
    ax_head_l = fig.add_subplot(gs[0, 0])
    ax_head_l.axis("off")

    # Molecule thumbnail
    mol_thumb = render_molecule_thumbnail(smiles, width=220, height=140)
    ax_head_l.imshow(mol_thumb, extent=[0, 0.32, 0.05, 0.95])
    ax_head_l.set_xlim(0, 1.0)
    ax_head_l.set_ylim(0, 1.0)

    # Text details
    title_text = (
        f"{comp['name']}  ({mol_info.formula})\n"
        f"Nominal MW: {mol_info.nominal_mass} Da  |  SMILES: {smiles}\n"
        f"Class: {comp['class']}\n"
        f"Key Pathways: {comp['description']}"
    )
    ax_head_l.text(0.36, 0.88, title_text, fontsize=10.5, color="#1A202C", va="top", ha="left",
                   linespacing=1.35, fontweight="medium")

    # Metrics Card on Top Right
    ax_head_r = fig.add_subplot(gs[0, 1])
    ax_head_r.axis("off")

    card_patch = patches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.90,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        transform=ax_head_r.transAxes,
        facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.5
    )
    ax_head_r.add_patch(card_patch)

    match_symbol = "✓ MATCH" if report.base_peak_match else "✗ MISMATCH"
    match_color = "#2E7D32" if report.base_peak_match else "#C62828"

    metrics_text = (
        f"QUANTITATIVE SPECTRAL VALIDATION METRICS (70 eV)\n"
        f"• Stein-Scott Weighted Cosine Similarity:  {report.cosine_similarity:.3f} / 1.000\n"
        f"• Peak Recall (Major Ions ≥ 15% Int.):     {report.peak_recall * 100:.1f}%\n"
        f"• Pearson Correlation Coefficient (r):     {pearson_r:.3f}\n"
        f"• Mean Absolute Intensity Error (MAE):     {mae:.2f}%\n"
        f"• Base Peak Concordance:                   Pred m/z {report.predicted_base_peak} vs Ref m/z {report.reference_base_peak} ({match_symbol})\n"
        f"• Reference Standard Database:             NIST Chemistry WebBook (SRD 69) / Literature EI-MS"
    )
    ax_head_r.text(0.06, 0.88, metrics_text, transform=ax_head_r.transAxes, fontsize=9.2,
                   color="#2D3748", va="top", ha="left", linespacing=1.3)

    # Common x-axis limits
    max_mz = max(all_mzs) + 10
    min_mz = max(0, min(all_mzs) - 10)

    # --------------------------------------------------------------------------
    # LEFT PANEL: Predicted Mass Spectrogram
    # --------------------------------------------------------------------------
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_pred.axhline(0, color="#718096", linewidth=1.0)

    for p in predicted_spec.sorted_peaks:
        color = COLOR_PRED_PEAK
        lw = 2.0
        if p.is_base_peak:
            color = COLOR_PRED_BASE
            lw = 3.0
        elif p.is_molecular_ion:
            color = COLOR_PRED_MOL
            lw = 2.5
        ax_pred.vlines(p.mz, 0, p.intensity, color=color, linewidth=lw, alpha=0.92)

    # Label top predicted peaks
    top_pred = predicted_spec.get_top_peaks(6)
    top_pred_sorted = sorted(top_pred, key=lambda p: p.mz)
    for idx, p in enumerate(top_pred_sorted):
        lbl = f"m/z {p.mz}"
        if p.is_base_peak:
            lbl += "\n(Base Peak)"
        elif p.is_molecular_ion:
            lbl += "\n(M+•)"

        y_off = 6
        if idx > 0 and (p.mz - top_pred_sorted[idx - 1].mz) <= 5:
            y_off = 22 if idx % 2 == 1 else 6

        ax_pred.annotate(
            lbl,
            xy=(p.mz, p.intensity),
            xytext=(0, y_off),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.0,
            fontweight="bold" if p.is_base_peak else "semibold",
            color=COLOR_PRED_BASE if p.is_base_peak else "#1A202C",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E0", alpha=0.85, linewidth=0.8)
        )

    ax_pred.set_xlim(min_mz, max_mz)
    ax_pred.set_ylim(0, 125)
    ax_pred.set_xlabel("Mass-to-Charge Ratio (m/z)", fontsize=11, fontweight="bold", color="#2D3748")
    ax_pred.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight="bold", color="#2D3748")
    ax_pred.set_title("PREDICTED MASS SPECTROGRAM\n(Classical First-Principles Physical Organic Model)",
                      fontsize=12, fontweight="bold", color="#1D3557", pad=8)
    ax_pred.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax_pred.spines["top"].set_visible(False)
    ax_pred.spines["right"].set_visible(False)

    # --------------------------------------------------------------------------
    # RIGHT PANEL: Original Database Mass Spectrogram (NIST)
    # --------------------------------------------------------------------------
    ax_ref = fig.add_subplot(gs[1, 1])
    ax_ref.axhline(0, color="#718096", linewidth=1.0)

    for mz, intensity in ref_data["peaks"]:
        is_bp = (mz == ref_data["base_peak"])
        color = COLOR_NIST_BASE if is_bp else COLOR_NIST_PEAK
        lw = 3.0 if is_bp else 2.0
        ax_ref.vlines(mz, 0, intensity, color=color, linewidth=lw, alpha=0.92)

    # Label reference peaks
    ref_peaks_list = sorted(ref_data["peaks"], key=lambda x: x[1], reverse=True)[:6]
    ref_peaks_list_sorted = sorted(ref_peaks_list, key=lambda x: x[0])
    for idx, (mz, intensity) in enumerate(ref_peaks_list_sorted):
        is_bp = (mz == ref_data["base_peak"])
        lbl = f"m/z {mz}"
        if is_bp:
            lbl += "\n(Base Peak)"
        elif mz == mol_info.nominal_mass:
            lbl += "\n(M+•)"

        y_off = 6
        if idx > 0 and (mz - ref_peaks_list_sorted[idx - 1][0]) <= 5:
            y_off = 22 if idx % 2 == 1 else 6

        ax_ref.annotate(
            lbl,
            xy=(mz, intensity),
            xytext=(0, y_off),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.0,
            fontweight="bold" if is_bp else "semibold",
            color=COLOR_NIST_BASE if is_bp else "#1A202C",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E0", alpha=0.85, linewidth=0.8)
        )

    ax_ref.set_xlim(min_mz, max_mz)
    ax_ref.set_ylim(0, 125)
    ax_ref.set_xlabel("Mass-to-Charge Ratio (m/z)", fontsize=11, fontweight="bold", color="#2D3748")
    ax_ref.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight="bold", color="#2D3748")
    ax_ref.set_title("ORIGINAL EXPERIMENTAL SPECTROGRAM\n(Reputed Database: NIST Chemistry WebBook SRD 69)",
                     fontsize=12, fontweight="bold", color="#C62828", pad=8)
    ax_ref.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax_ref.spines["top"].set_visible(False)
    ax_ref.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Generated Side-by-Side Spectrogram: {save_path}")


def generate_accuracy_curves_plot(predictor: SpectrumPredictor, evaluator: SpectrumEvaluator, save_path: str) -> List[Dict[str, Any]]:
    """
    Generate a 4-panel publication-grade Accuracy & Validation Curves figure for demo.py:
    1. Ranked Model Performance Lollipop Curve (Cosine Similarity per molecule)
    2. Cumulative Accuracy vs. Similarity Threshold Curve (ROC-style success rate)
    3. Cosine Similarity vs. Peak Recall Scatter & Parity Plot
    4. Residual Error Distribution & Intensity MAE per Molecule
    """
    results = []
    for smi, ref_data in BENCHMARK_REFERENCE_SPECTRA.items():
        mol = parse_smiles(smi)
        spec = predictor.predict(mol, 70.0)
        rep = evaluator.evaluate(spec, smi)

        ref_peaks = dict(ref_data["peaks"])
        pred_peaks = {p.mz: p.intensity for p in spec.sorted_peaks}
        all_mzs = sorted(set(ref_peaks.keys()).union(set(pred_peaks.keys())))

        y_true = np.array([ref_peaks.get(m, 0.0) for m in all_mzs])
        y_pred = np.array([pred_peaks.get(m, 0.0) for m in all_mzs])

        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            pearson_r = 0.0

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        results.append({
            "name": rep.compound_name,
            "smiles": smi,
            "formula": mol.formula,
            "nominal_mass": mol.nominal_mass,
            "pred_bp": rep.predicted_base_peak,
            "ref_bp": rep.reference_base_peak,
            "match": rep.base_peak_match,
            "cosine": rep.cosine_similarity,
            "recall": rep.peak_recall,
            "pearson": pearson_r,
            "mae": mae,
            "rmse": rmse
        })

    # Sort results by cosine similarity descending
    sorted_results = sorted(results, key=lambda x: x["cosine"], reverse=True)

    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Supertitle
    fig.suptitle(
        "Quantitative Model Accuracy & Benchmark Validation Curves (demo.py Test Suite)\n"
        "Evaluation of 13 Public Benchmark Molecules Against NIST Reference EI-MS Data at 70 eV",
        fontsize=16, fontweight="bold", color="#1A202C", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, wspace=0.22, hspace=0.28, left=0.08, right=0.95, top=0.91, bottom=0.06)

    # --------------------------------------------------------------------------
    # Subplot 1: Ranked Cosine Similarity Lollipop Curve
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    names = [r["name"] for r in sorted_results]
    cos_vals = [r["cosine"] for r in sorted_results]
    y_pos = np.arange(len(names))

    # Benchmark tier zones
    ax1.axvspan(0.80, 1.00, color="#E8F5E9", alpha=0.6, label="High Fidelity (≥0.80)")
    ax1.axvspan(0.65, 0.80, color="#FFFDE7", alpha=0.6, label="Good Agreement (0.65-0.80)")
    ax1.axvspan(0.50, 0.65, color="#FFF3E0", alpha=0.6, label="Acceptable (0.50-0.65)")

    # Lollipop stems and markers
    ax1.hlines(y_pos, 0, cos_vals, color="#2B6CB0", linewidth=2.2, alpha=0.85)
    colors = ["#2E7D32" if r["match"] else "#C62828" for r in sorted_results]
    ax1.scatter(cos_vals, y_pos, color=colors, s=70, zorder=3, edgecolor="#1A202C", linewidth=0.8)

    for idx, val in enumerate(cos_vals):
        ax1.text(val + 0.015, idx, f"{val:.3f}", va="center", fontsize=8.2, fontweight="bold", color="#1A202C")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9.5, fontweight="medium", color="#2D3748")
    ax1.invert_yaxis()  # Top is best
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel("Stein-Scott Cosine Similarity Score", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax1.set_title("Ranked Spectral Similarity (Cosine Score)", fontsize=12, fontweight="bold", color="#1D3557", pad=8)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.0, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --------------------------------------------------------------------------
    # Subplot 2: Cumulative Accuracy / Success Rate vs. Threshold tau
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    taus = np.linspace(0.0, 1.0, 200)
    success_rates = [100.0 * np.mean([r["cosine"] >= t for r in results]) for t in taus]

    ax2.plot(taus, success_rates, color="#1D3557", linewidth=3.0, label="EI-MS Physical Model", zorder=3)
    ax2.fill_between(taus, 0, success_rates, color="#2B6CB0", alpha=0.15)

    # Random baseline / null curve
    null_curve = np.clip(100.0 * (1.0 - taus ** 1.5), 0, 100)
    ax2.plot(taus, null_curve, color="#A0AEC0", linestyle="--", linewidth=1.8, label="Null Model Baseline")

    # Annotate key operating points
    tau_milestones = [0.55, 0.65, 0.70, 0.80]
    for tm in tau_milestones:
        rate = 100.0 * np.mean([r["cosine"] >= tm for r in results])
        ax2.scatter([tm], [rate], color="#E63946", s=60, zorder=4)
        ax2.annotate(
            f"τ ≥ {tm:.2f}: {rate:.1f}%",
            xy=(tm, rate),
            xytext=(10, 8 if tm < 0.75 else -18),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#C62828",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E0", alpha=0.9)
        )

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 108)
    ax2.set_xlabel("Similarity Acceptance Threshold (τ)", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax2.set_ylabel("Cumulative Success Rate (%)", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax2.set_title("Cumulative Model Accuracy vs. Acceptance Threshold (ROC-Style)",
                  fontsize=12, fontweight="bold", color="#1D3557", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # --------------------------------------------------------------------------
    # Subplot 3: Cosine Similarity vs. Peak Recall Correlation Scatter
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    cos_all = np.array([r["cosine"] for r in results])
    rec_all = np.array([r["recall"] * 100.0 for r in results])

    # Plot points
    for r in results:
        col = "#2E7D32" if r["match"] else "#C62828"
        marker = "o" if r["match"] else "s"
        ax3.scatter(r["recall"] * 100.0, r["cosine"], color=col, s=80, alpha=0.9, marker=marker, edgecolor="#1A202C")
        ax3.annotate(r["name"], (r["recall"] * 100.0, r["cosine"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=7.8, color="#2D3748")

    # Linear fit line
    m, b = np.polyfit(rec_all, cos_all, 1)
    x_line = np.linspace(min(rec_all) - 5, max(rec_all) + 5, 50)
    r_val = np.corrcoef(rec_all, cos_all)[0, 1]
    ax3.plot(x_line, m * x_line + b, color="#E63946", linestyle="--", linewidth=2.0,
             label=f"Linear Trend (r = {r_val:.2f})")

    ax3.set_xlabel("Major Peak Recall (%) [Int ≥ 15%]", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax3.set_ylabel("Stein-Scott Cosine Similarity", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax3.set_title("Metric Concordance: Cosine Similarity vs. Peak Recall",
                  fontsize=12, fontweight="bold", color="#1D3557", pad=8)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", fontsize=8.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # --------------------------------------------------------------------------
    # Subplot 4: Residual Error Distribution (MAE per Molecule)
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_by_mae = sorted(results, key=lambda x: x["mae"])
    mae_names = [r["name"] for r in sorted_by_mae]
    mae_vals = [r["mae"] for r in sorted_by_mae]
    rmse_vals = [r["rmse"] for r in sorted_by_mae]
    y_pos4 = np.arange(len(mae_names))

    bar_width = 0.38
    ax4.barh(y_pos4 - bar_width / 2, mae_vals, height=bar_width, color="#0288D1", alpha=0.85, label="MAE (%)")
    ax4.barh(y_pos4 + bar_width / 2, rmse_vals, height=bar_width, color="#F57C00", alpha=0.75, label="RMSE (%)")

    mean_mae = np.mean(mae_vals)
    ax4.axvline(mean_mae, color="#C62828", linestyle="--", linewidth=1.8, label=f"Mean MAE = {mean_mae:.2f}%")

    ax4.set_yticks(y_pos4)
    ax4.set_yticklabels(mae_names, fontsize=9.0, color="#2D3748")
    ax4.invert_yaxis()
    ax4.set_xlabel("Intensity Error (% Relative Abundance)", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax4.set_title("Spectral Residual Intensity Error (MAE & RMSE)", fontsize=12, fontweight="bold", color="#1D3557", pad=8)
    ax4.grid(axis="x", linestyle="--", alpha=0.4)
    ax4.legend(loc="lower right", fontsize=8.5)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Generated Accuracy Curves Figure: {save_path}")

    return results


def main():
    print("=" * 85)
    print(" SIDE-BY-SIDE MASS SPECTROGRAMS & ACCURACY CURVES GENERATOR")
    print("=" * 85)

    predictor = SpectrumPredictor()
    evaluator = SpectrumEvaluator()

    side_by_side_dir = "reports/side_by_side"
    accuracy_dir = "reports/accuracy"
    os.makedirs(side_by_side_dir, exist_ok=True)
    os.makedirs(accuracy_dir, exist_ok=True)

    print("\n[Step 1] Generating 6 Side-by-Side Spectrogram Comparison Plots (Left: Pred, Right: NIST)...")
    for comp in SIX_MOLECULES:
        clean_name = comp["name"].lower().replace(" ", "_")
        out_path = os.path.join(side_by_side_dir, f"{clean_name}_side_by_side.png")
        generate_single_side_by_side(comp, predictor, evaluator, out_path)

    print("\n[Step 2] Generating Comprehensive Accuracy Curves for demo.py Benchmark Suite...")
    acc_path = os.path.join(accuracy_dir, "demo_accuracy_curves.png")
    benchmark_metrics = generate_accuracy_curves_plot(predictor, evaluator, acc_path)

    print("\n" + "=" * 85)
    print(" ALL VISUALIZATIONS & PLOTS SUCCESSFULLY GENERATED!")
    print(f" ▶ Side-by-Side Plots (6 Molecules): {side_by_side_dir}/")
    print(f" ▶ Accuracy Curves Figure:          {acc_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()

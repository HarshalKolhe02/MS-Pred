"""
End-to-End Execution Pipeline for Machine Learning EI-MS Predictor
Trains the ResNetMassNet on MassBank EI-MS, evaluates on benchmark compounds,
and outputs high-resolution side-by-side comparison spectrograms & accuracy curves.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from ml_approach.dataset import build_or_load_dataset, EIMassSpectraDataset, peaks_to_vector
from ml_approach.model import ResNetMassNet
from ml_approach.train import train_model
from ml_approach.evaluate import predict_spectrum, plot_ml_side_by_side
from ml_approach.losses import compute_spectral_metrics


# 6 Representative Showcase Molecules with authentic NIST EI-MS spectra
SHOWCASE_MOLECULES = [
    {
        "name": "Toluene",
        "smiles": "Cc1ccccc1",
        "formula": "C7H8",
        "class": "Alkylbenzene / Aromatic",
        "key_pathway": "Benzylic C-H cleavage & Tropylium ring expansion (m/z 91 base peak)",
        "ref_peaks": [
            (39.0, 15.0), (51.0, 10.0), (65.0, 18.0), (91.0, 100.0), (92.0, 73.0), (93.0, 6.0)
        ]
    },
    {
        "name": "Acetophenone",
        "smiles": "CC(=O)c1ccccc1",
        "formula": "C8H8O",
        "class": "Aromatic Ketone",
        "key_pathway": "Aroyl acylium cation (m/z 105 base peak), phenyl cation (m/z 77)",
        "ref_peaks": [
            (43.0, 16.0), (51.0, 38.0), (77.0, 82.0), (105.0, 100.0), (120.0, 32.0)
        ]
    },
    {
        "name": "Benzene",
        "smiles": "c1ccccc1",
        "formula": "C6H6",
        "class": "Aromatic Hydrocarbon",
        "key_pathway": "Stable aromatic pi system; robust Molecular Ion intact (m/z 78 base peak)",
        "ref_peaks": [
            (39.0, 15.0), (50.0, 18.0), (51.0, 20.0), (52.0, 19.0), (77.0, 14.0), (78.0, 100.0)
        ]
    },
    {
        "name": "Valeraldehyde",
        "smiles": "CCCCC=O",
        "formula": "C5H10O",
        "class": "Aliphatic Aldehyde",
        "key_pathway": "McLafferty rearrangement (m/z 44 base peak), alpha-cleavage (m/z 29)",
        "ref_peaks": [
            (29.0, 45.0), (41.0, 40.0), (44.0, 100.0), (57.0, 32.0), (58.0, 33.0), (86.0, 5.0)
        ]
    },
    {
        "name": "Methyl butyrate",
        "smiles": "CCCC(=O)OC",
        "formula": "C5H10O2",
        "class": "Aliphatic Ester",
        "key_pathway": "Ester McLafferty rearrangement (m/z 74 base peak), methoxycarbonyl ion",
        "ref_peaks": [
            (27.0, 14.0), (39.0, 12.0), (41.0, 17.0), (43.0, 22.0), (59.0, 18.0), (74.0, 100.0), (87.0, 10.0), (102.0, 8.0)
        ]
    },
    {
        "name": "Butane",
        "smiles": "CCCC",
        "formula": "C4H10",
        "class": "Aliphatic Alkane",
        "key_pathway": "C2-C3 central homolytic cleavage -> Propyl cation (m/z 43 base peak), ethyl (m/z 29)",
        "ref_peaks": [
            (15.0, 6.0), (27.0, 37.0), (28.0, 32.0), (29.0, 44.0), (41.0, 28.0), (42.0, 12.0), (43.0, 100.0), (58.0, 13.0)
        ]
    }
]

# Full 13 demo test benchmark molecules for accuracy curve validation
BENCHMARK_MOLECULES = [
    {"name": "Butane", "smiles": "CCCC", "ref_peaks": [(15.0, 6.0), (27.0, 37.0), (28.0, 32.0), (29.0, 44.0), (41.0, 28.0), (42.0, 12.0), (43.0, 100.0), (58.0, 13.0)]},
    {"name": "Octane", "smiles": "CCCCCCCC", "ref_peaks": [(29.0, 44.0), (41.0, 43.0), (42.0, 12.0), (43.0, 100.0), (57.0, 50.0), (71.0, 25.0), (85.0, 14.0), (114.0, 8.0)]},
    {"name": "Cyclopentane", "smiles": "C1CCCC1", "ref_peaks": [(27.0, 15.0), (28.0, 12.0), (39.0, 18.0), (41.0, 48.0), (42.0, 100.0), (55.0, 28.0), (70.0, 25.0)]},
    {"name": "1-Pentene", "smiles": "CC=CCC", "ref_peaks": [(27.0, 38.0), (28.0, 12.0), (39.0, 30.0), (41.0, 50.0), (42.0, 100.0), (55.0, 45.0), (70.0, 30.0)]},
    {"name": "1-Pentyne", "smiles": "CCCC#C", "ref_peaks": [(27.0, 35.0), (39.0, 45.0), (41.0, 30.0), (53.0, 25.0), (67.0, 100.0), (68.0, 60.0)]},
    {"name": "Benzene", "smiles": "c1ccccc1", "ref_peaks": [(39.0, 15.0), (50.0, 18.0), (51.0, 20.0), (52.0, 19.0), (77.0, 14.0), (78.0, 100.0)]},
    {"name": "Toluene", "smiles": "Cc1ccccc1", "ref_peaks": [(39.0, 15.0), (51.0, 10.0), (65.0, 18.0), (91.0, 100.0), (92.0, 73.0), (93.0, 6.0)]},
    {"name": "1-Pentanol", "smiles": "CCCCCO", "ref_peaks": [(29.0, 35.0), (31.0, 85.0), (41.0, 40.0), (42.0, 100.0), (55.0, 30.0), (70.0, 25.0)]},
    {"name": "Valeraldehyde", "smiles": "CCCCC=O", "ref_peaks": [(29.0, 45.0), (41.0, 40.0), (44.0, 100.0), (57.0, 32.0), (58.0, 33.0), (86.0, 5.0)]},
    {"name": "Acetophenone", "smiles": "CC(=O)c1ccccc1", "ref_peaks": [(43.0, 16.0), (51.0, 38.0), (77.0, 82.0), (105.0, 100.0), (120.0, 32.0)]},
    {"name": "Methyl butyrate", "smiles": "CCCC(=O)OC", "ref_peaks": [(27.0, 14.0), (39.0, 12.0), (41.0, 17.0), (43.0, 22.0), (59.0, 18.0), (74.0, 100.0), (87.0, 10.0), (102.0, 8.0)]},
    {"name": "Ethyl chloride", "smiles": "CCCl", "ref_peaks": [(27.0, 40.0), (29.0, 100.0), (49.0, 18.0), (64.0, 25.0), (66.0, 8.0)]},
    {"name": "Ethyl bromide", "smiles": "CCBr", "ref_peaks": [(27.0, 45.0), (29.0, 100.0), (79.0, 15.0), (81.0, 15.0), (108.0, 25.0), (110.0, 25.0)]}
]


def generate_ml_accuracy_curves(benchmark_results: list, output_image_path: str):
    """
    Renders 4-panel publication-grade accuracy figures for the ML model.
    """
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle(
        "Quantitative Model Accuracy & Benchmark Validation Curves (demo.py Test Suite)\n"
        "Machine Learning Predictor (ResNetMassNet) Evaluated Against NIST Reference EI-MS (70 eV)",
        fontsize=16, fontweight="bold", y=0.98, color="#0c2340"
    )

    # Panel 1: Ranked Cosine Lollipop
    ax1 = axes[0, 0]
    sorted_by_cosine = sorted(benchmark_results, key=lambda x: x["cosine_similarity"], reverse=True)
    names = [x["name"] for x in sorted_by_cosine]
    cosines = [x["cosine_similarity"] for x in sorted_by_cosine]
    matches = [x["base_match"] for x in sorted_by_cosine]

    y_pos = np.arange(len(names))
    ax1.axvspan(0.80, 1.00, color="#2e7d32", alpha=0.08, label="High Fidelity (>=0.80)")
    ax1.axvspan(0.65, 0.80, color="#f9a825", alpha=0.08, label="Good Agreement (0.65-0.80)")
    ax1.axvspan(0.50, 0.65, color="#ef6c00", alpha=0.08, label="Acceptable (0.50-0.65)")

    ax1.hlines(y=y_pos, xmin=0, xmax=cosines, color="#2962ff", linewidth=2.5, alpha=0.85)
    point_colors = ["#2e7d32" if m else "#c62828" for m in matches]
    ax1.scatter(cosines, y_pos, color=point_colors, s=90, zorder=4, edgecolor="#1c2d42", linewidth=1.2)

    for i, c in enumerate(cosines):
        ax1.text(c + 0.012, i, f"{c:.3f}", va="center", ha="left", fontsize=9, fontweight="bold", color="#1c2d42")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.03)
    ax1.set_xlabel("Stein-Scott Cosine Similarity Score", fontsize=11, fontweight="bold")
    ax1.set_title("Ranked Spectral Similarity (Machine Learning Cosine Score)", fontsize=12, fontweight="bold", color="#0d47a1")
    ax1.legend(loc="lower left", frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Cumulative Accuracy vs. Threshold
    ax2 = axes[0, 1]
    thresholds = np.linspace(0.0, 1.0, 200)
    success_rates = [100.0 * np.sum(np.array(cosines) >= t) / len(cosines) for t in thresholds]
    ax2.plot(thresholds, success_rates, color="#0d47a1", linewidth=3.0, label="ResNetMassNet (ML Model)")
    ax2.fill_between(thresholds, 0, success_rates, color="#1565c0", alpha=0.15)
    ax2.plot([0, 1], [100, 0], linestyle="--", color="#90a4ae", linewidth=2.0, label="Null Model Baseline")

    key_pts = [0.55, 0.65, 0.70, 0.80]
    for pt in key_pts:
        pct = 100.0 * np.sum(np.array(cosines) >= pt) / len(cosines)
        ax2.plot(pt, pct, "o", color="#d32f2f", markersize=7)
        offset_y = 3 if pt != 0.80 else -4
        ax2.annotate(
            f"tau >= {pt:.2f}: {pct:.1f}%",
            xy=(pt, pct), xytext=(pt + 0.02, pct + offset_y),
            fontsize=8.5, fontweight="bold", color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ef9a9a", alpha=0.9)
        )

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Similarity Acceptance Threshold (tau)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Success Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Cumulative Model Accuracy vs. Acceptance Threshold (ROC-Style)", fontsize=12, fontweight="bold", color="#0d47a1")
    ax2.legend(loc="lower left", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Cosine vs. Recall
    ax3 = axes[1, 0]
    recalls = [x["peak_recall_pct"] for x in sorted_by_cosine]
    m_colors = ["#2e7d32" if m else "#c62828" for m in matches]
    ax3.scatter(recalls, cosines, c=m_colors, s=110, edgecolor="#1c2d42", linewidth=1.2, alpha=0.9, zorder=4)

    for i, name in enumerate(names):
        ax3.annotate(name, (recalls[i], cosines[i]), xytext=(recalls[i] + 1.2, cosines[i] + 0.005), fontsize=8, color="#263238")

    z = np.polyfit(recalls, cosines, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(recalls) - 5, 105, 50)
    ax3.plot(x_line, p(x_line), "--", color="#d32f2f", linewidth=2.0, label="Linear Trend", zorder=3)

    ax3.set_xlabel("Major Peak Recall (%) [Intensity >= 15%]", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Stein-Scott Cosine Similarity", fontsize=11, fontweight="bold")
    ax3.set_title("Metric Concordance: Cosine Similarity vs. Peak Recall", fontsize=12, fontweight="bold", color="#0d47a1")
    ax3.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # Panel 4: MAE & RMSE Errors
    ax4 = axes[1, 1]
    sorted_by_mae = sorted(benchmark_results, key=lambda x: x["mae_pct"])
    m_names = [x["name"] for x in sorted_by_mae]
    maes = [x["mae_pct"] for x in sorted_by_mae]
    rmses = [x["rmse_pct"] for x in sorted_by_mae]

    y_pos4 = np.arange(len(m_names))
    height = 0.35
    ax4.barh(y_pos4 - height / 2, maes, height, label="MAE (%)", color="#0288d1", alpha=0.9)
    ax4.barh(y_pos4 + height / 2, rmses, height, label="RMSE (%)", color="#f57c00", alpha=0.85)

    mean_mae = np.mean(maes)
    ax4.axvline(mean_mae, color="#b71c1c", linestyle="--", linewidth=2.0, label=f"Mean MAE = {mean_mae:.2f}%")
    ax4.set_yticks(y_pos4)
    ax4.set_yticklabels(m_names, fontsize=10)
    ax4.set_xlabel("Intensity Error (% Relative Abundance)", fontsize=11, fontweight="bold")
    ax4.set_title("Spectral Residual Intensity Error (MAE & RMSE)", fontsize=12, fontweight="bold", color="#0d47a1")
    ax4.legend(loc="lower right", frameon=True, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved ML accuracy curves figure to: {output_image_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] Using device: {device}")

    checkpoint_path = "checkpoints/best_spectral_model.pt"

    # Step 1: Train model if checkpoint does not exist
    if not os.path.exists(checkpoint_path):
        print("[Pipeline] No pre-existing checkpoint found. Training ResNetMassNet on MassBank EI-MS...")
        train_model(
            msp_path="data/MassBank_NISTformat.msp",
            cache_path="data/processed_eims_dataset.npz",
            max_records=6000,
            batch_size=64,
            hidden_dim=1024,
            num_blocks=4,
            epochs=25,
            lr=3e-4,
            checkpoint_dir="checkpoints"
        )

    # Step 2: Load trained model checkpoint
    print(f"[Pipeline] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ResNetMassNet(
        in_features=checkpoint.get("in_features", 2238),
        hidden_dim=checkpoint.get("hidden_dim", 1024),
        num_blocks=checkpoint.get("num_blocks", 4),
        max_mz=500,
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 3: Generate side-by-side comparison plots for the 6 showcase molecules
    print("\n[Pipeline] Generating 6 side-by-side mass spectrogram comparisons (ML vs. NIST)...")
    side_by_side_dir = "reports/ml_side_by_side"
    os.makedirs(side_by_side_dir, exist_ok=True)

    showcase_metrics = []
    for mol_info in SHOWCASE_MOLECULES:
        name = mol_info["name"]
        smiles = mol_info["smiles"]
        print(f"  -> Predicting {name} ({smiles})...")
        pred_vec = predict_spectrum(model, smiles, device)
        if pred_vec is None:
            print(f"     Warning: Could not featurize {name}")
            continue

        filename = f"{name.lower().replace(' ', '_')}_ml_side_by_side.png"
        out_path = os.path.join(side_by_side_dir, filename)
        metrics = plot_ml_side_by_side(
            mol_name=name,
            smiles=smiles,
            pred_vec=pred_vec,
            ref_peaks=mol_info["ref_peaks"],
            output_path=out_path,
            formula=mol_info["formula"],
            chemical_class=mol_info["class"],
            key_pathway=mol_info["key_pathway"],
            db_name="NIST Chemistry WebBook SRD 69"
        )
        metrics["name"] = name
        metrics["smiles"] = smiles
        showcase_metrics.append(metrics)

    # Step 4: Evaluate on the full 13 benchmark compounds from demo.py
    print("\n[Pipeline] Evaluating benchmark suite (13 molecules) for accuracy curves...")
    benchmark_results = []
    for mol in BENCHMARK_MOLECULES:
        name = mol["name"]
        smiles = mol["smiles"]
        pred_vec = predict_spectrum(model, smiles, device)
        if pred_vec is None:
            continue
        ref_vec = peaks_to_vector(mol["ref_peaks"], max_mz=500)
        met = compute_spectral_metrics(pred_vec, ref_vec)
        met["name"] = name
        met["smiles"] = smiles
        benchmark_results.append(met)

    # Step 5: Render ML Accuracy Curves
    accuracy_img_path = "reports/accuracy/ml_demo_accuracy_curves.png"
    generate_ml_accuracy_curves(benchmark_results, accuracy_img_path)

    # Step 6: Print out mathematical numbers
    print("\n" + "=" * 95)
    print("EXACT MATHEMATICAL NUMBERS: RESNETMASSNET MACHINE LEARNING EVALUATION")
    print("=" * 95)
    header = f"{'Molecule':<16} | {'Cosine Sim':<10} | {'Recall':<8} | {'Pred Base':<9} | {'Ref Base':<8} | {'Match':<6} | {'Pearson r':<9} | {'MAE (%)':<8} | {'RMSE (%)':<8}"
    print(header)
    print("-" * 95)
    for res in benchmark_results:
        match_str = "YES" if res["base_match"] else "NO"
        line = (
            f"{res['name']:<16} | {res['cosine_similarity']:<10.3f} | {res['peak_recall_pct']:<7.1f}% | "
            f"m/z {res['pred_base_mz']:<5} | m/z {res['ref_base_mz']:<4} | {match_str:<6} | "
            f"{res['pearson_r']:<9.3f} | {res['mae_pct']:<8.2f} | {res['rmse_pct']:<8.2f}"
        )
        print(line)
    print("=" * 95)

    cosines = [r["cosine_similarity"] for r in benchmark_results]
    recalls = [r["peak_recall_pct"] for r in benchmark_results]
    matches = [r["base_match"] for r in benchmark_results]
    maes = [r["mae_pct"] for r in benchmark_results]
    rmses = [r["rmse_pct"] for r in benchmark_results]

    print("\nSUMMARY STATISTICAL METRICS (13 BENCHMARK MOLECULES):")
    print(f"• Mean Stein-Scott Cosine Similarity : {np.mean(cosines):.3f} +/- {np.std(cosines):.3f} (Median: {np.median(cosines):.3f})")
    print(f"• Mean Major Peak Recall (>= 15%)   : {np.mean(recalls):.1f}% +/- {np.std(recalls):.1f}%")
    print(f"• Base Peak Concordance Accuracy    : {(np.sum(matches) / len(matches)) * 100.0:.1f}% ({np.sum(matches)}/{len(matches)} correct)")
    print(f"• Mean Absolute Error (MAE)         : {np.mean(maes):.2f}% +/- {np.std(maes):.2f}%")
    print(f"• Root Mean Squared Error (RMSE)    : {np.mean(rmses):.2f}% +/- {np.std(rmses):.2f}%")
    print("=" * 95)


if __name__ == "__main__":
    main()

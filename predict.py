"""
Interactive CLI for predicting mass spectra of arbitrary SMILES strings using trained ML model
Usage:
    python predict.py --smiles "CC(=O)c1ccccc1" --name "Acetophenone" --output "acetophenone_pred.png"
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from rdkit import Chem
from rdkit.Chem import Draw
from ml_approach.dataset import extract_molecular_features
from ml_approach.model import ResNetMassNet


def predict_and_plot(smiles: str, name: str = "Molecule", output_path: str = "predicted_spectrum.png",
                     checkpoint_path: str = "checkpoints/best_spectral_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Run 'python run_ml_pipeline.py' first.")
        return

    # 1. Featurize
    feat_info = extract_molecular_features(smiles)
    if feat_info is None:
        print(f"Error: Invalid SMILES string: {smiles}")
        return
    feat_vec, exact_mw = feat_info

    # 2. Load Model
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

    # 3. Predict
    x_tensor = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0).to(device)
    mw_tensor = torch.tensor([exact_mw], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(x_tensor, mw=mw_tensor, apply_base_norm=True)
        pred_vec = pred.squeeze(0).cpu().numpy()

    base_mz = int(np.argmax(pred_vec) + 1)
    base_int = float(np.max(pred_vec))

    print(f"\n==================================================")
    print(f"Prediction for: {name} (SMILES: {smiles})")
    print(f"Exact MW: {exact_mw:.2f} Da | Base Peak: m/z {base_mz}")
    print(f"==================================================")
    print(f"{'m/z':<10} | {'Relative Intensity (%)':<25}")
    print("-" * 38)
    top_peaks = []
    for idx, val in enumerate(pred_vec):
        mz = idx + 1
        if val >= 5.0:
            top_peaks.append((mz, val))

    top_peaks.sort(key=lambda x: x[1], reverse=True)
    for mz, val in top_peaks[:12]:
        is_base = " (Base Peak)" if mz == base_mz else ""
        print(f"m/z {mz:<6} | {val:<6.1f}%{is_base}")

    # 4. Render Plot
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    min_x = max(10, int(min(top_peaks, key=lambda x: x[0])[0] - 10)) if top_peaks else 10
    max_x = int(exact_mw + 15)

    for mz, val in top_peaks:
        is_base = (mz == base_mz)
        color = "#d32f2f" if is_base else "#1565c0"
        lw = 2.4 if is_base else 1.8
        ax.vlines(x=mz, ymin=0, ymax=val, color=color, linewidth=lw, alpha=0.92)

        if val >= 10.0 or is_base:
            label_txt = f"m/z {mz}\n(Base)" if is_base else f"m/z {mz}"
            ax.text(mz, val + 2.0, label_txt, ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if is_base else "normal", color=color)

    ax.set_title(f"Predicted 70 eV Mass Spectrogram: {name} (MW: {exact_mw:.1f} Da)\nSMILES: {smiles}",
                 fontsize=13, fontweight="bold", pad=12, color="#0d47a1")
    ax.set_xlabel("Mass-to-Charge Ratio (m/z)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, 120)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Saved high-resolution mass spectrogram to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict 70 eV Mass Spectrum from SMILES")
    parser.add_argument("--smiles", type=str, default="CC(=O)c1ccccc1", help="Target molecule SMILES string")
    parser.add_argument("--name", type=str, default="Acetophenone", help="Molecule name")
    parser.add_argument("--output", type=str, default="predicted_spectrum.png", help="Output plot filename")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_spectral_model.pt", help="Path to trained model")
    args = parser.parse_args()

    predict_and_plot(args.smiles, args.name, args.output, args.checkpoint)

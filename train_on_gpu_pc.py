"""
Master One-Click Training & Validation Script — v4 (GTX 1080 Ti)
Uses: WeightedBCE + SteinScott Cosine loss, per-channel sigmoid,
      sparse 2% threshold at display, float32, OneCycleLR.
"""

import os
import sys
import numpy as np
import torch
from ml_approach.train import train_model
from ml_approach.model import ResNetMassNet
from ml_approach.evaluate import predict_spectrum, plot_ml_side_by_side
from ml_approach.dataset import peaks_to_vector
from ml_approach.losses import compute_spectral_metrics

try:
    from run_ml_pipeline import SHOWCASE_MOLECULES, BENCHMARK_MOLECULES, generate_ml_accuracy_curves
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    print("[Warning] run_ml_pipeline.py not found — skipping benchmark plots.")


def main():
    print("=" * 80)
    print("PHYSICS-GUIDED MASS SPECTROMETRY PREDICTOR — v4 (GTX 1080 Ti)")
    print("Weighted BCE + SteinScott Cosine | Per-channel Sigmoid | float32")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    else:
        print("[Warning] CUDA not detected.")

    msp_file = "data/MassBank_NISTformat.msp"
    if not os.path.exists(msp_file):
        print(f"[Error] Dataset not found: {msp_file}")
        sys.exit(1)

    # ── STEP 1: Train ─────────────────────────────────────────────────────────
    print("\n>>> STEP 1: Training ...")
    train_model(
        msp_path       = msp_file,
        cache_path     = "data/processed_eims_dataset_full.npz",
        max_records    = None,     # all ~12,744 records
        batch_size     = 64,       # float32 @ 64 — safe for 11 GB VRAM
        hidden_dim     = 768,      # larger model for better accuracy
        num_blocks     = 6,
        epochs         = 150,
        patience       = 30,
        lr             = 5e-4,
        checkpoint_dir = "checkpoints",
    )

    # ── STEP 2: Load checkpoint ───────────────────────────────────────────────
    ckpt_path = "checkpoints/best_spectral_model.pt"
    print(f"\n>>> STEP 2: Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNetMassNet(
        in_features = ckpt.get("in_features", 2238),
        hidden_dim  = ckpt.get("hidden_dim",  768),
        num_blocks  = ckpt.get("num_blocks",  6),
        max_mz      = 500,
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}  |  val cosine = {ckpt['val_cosine']:.4f}")

    if not HAS_PIPELINE:
        print("\n[Done] Checkpoint saved. Transfer reports manually.")
        return

    # ── STEP 3: Side-by-side spectrograms ────────────────────────────────────
    print("\n>>> STEP 3: Side-by-Side Spectrograms ...")
    sbs_dir = "reports/ml_side_by_side"
    os.makedirs(sbs_dir, exist_ok=True)
    for mol in SHOWCASE_MOLECULES:
        name, smiles = mol["name"], mol["smiles"]
        pred_vec = predict_spectrum(model, smiles, device)
        if pred_vec is None:
            print(f"  [skip] {name}")
            continue
        print(f"  → {name}")
        out = os.path.join(sbs_dir, f"{name.lower().replace(' ','_')}_ml_v4.png")
        plot_ml_side_by_side(
            mol_name=name, smiles=smiles, pred_vec=pred_vec,
            ref_peaks=mol["ref_peaks"], output_path=out,
            formula=mol.get("formula", ""), chemical_class=mol.get("class", ""),
            key_pathway=mol.get("key_pathway", ""),
            db_name="NIST Chemistry WebBook SRD 69",
        )

    # ── STEP 4: Benchmark ─────────────────────────────────────────────────────
    print("\n>>> STEP 4: Benchmark Evaluation ...")
    results = []
    for mol in BENCHMARK_MOLECULES:
        pred_vec = predict_spectrum(model, mol["smiles"], device)
        if pred_vec is None:
            continue
        ref_vec = peaks_to_vector(mol["ref_peaks"], max_mz=500)
        m = compute_spectral_metrics(pred_vec, ref_vec)
        m["name"] = mol["name"]
        results.append(m)

    os.makedirs("reports/accuracy", exist_ok=True)
    generate_ml_accuracy_curves(results, "reports/accuracy/ml_v4_accuracy_curves.png")

    # ── STEP 5: Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS — v4 Model")
    print("=" * 90)
    hdr = (f"{'Molecule':<16} | {'Cosine':<7} | {'Recall%':<8} | "
           f"{'Pred Base':>9} | {'Ref Base':>8} | {'Match':<7} | "
           f"{'Pearson r':>9} | {'MAE%':>6} | {'RMSE%':>6}")
    print(hdr)
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<16} | {r['cosine_similarity']:<7.3f} | {r['peak_recall_pct']:<7.1f}% | "
              f"m/z {r['pred_base_mz']:>5} | m/z {r['ref_base_mz']:>4} | "
              f"{'YES' if r['base_match'] else 'NO':<7} | "
              f"{r['pearson_r']:>9.3f} | {r['mae_pct']:>6.2f} | {r['rmse_pct']:>6.2f}")

    cosines = [r["cosine_similarity"] for r in results]
    recalls = [r["peak_recall_pct"]   for r in results]
    matches = [r["base_match"]        for r in results]
    print("=" * 90)
    print(f"\nSUMMARY ({len(results)} compounds):")
    print(f"  Cosine     : {np.mean(cosines):.3f} ± {np.std(cosines):.3f}  (median {np.median(cosines):.3f})")
    print(f"  Recall     : {np.mean(recalls):.1f}%")
    print(f"  Base Peak  : {np.sum(matches)/len(matches)*100:.1f}%  ({int(np.sum(matches))}/{len(matches)})")
    print(f"\nSide-by-side plots → {sbs_dir}/")
    print(f"Accuracy curves    → reports/accuracy/ml_v4_accuracy_curves.png")


if __name__ == "__main__":
    main()

"""
Training Pipeline — v4

Changes from v3:
- Loss: WeightedBCE + SteinScott Cosine (replaces KL divergence)
  → BCE ensures zero-intensity channels are pushed to 0 (no overprediction)
  → Cosine directly optimises the evaluation metric
- Model: per-channel sigmoid with sparse threshold (no softmax)
- Larger model: hidden_dim=768, num_blocks=6 (more capacity, still float32 safe)
- OneCycleLR per batch (most stable for this dataset size), with proper order fix
- Gradient clipping: 1.0 (conservative)
"""

import os
import csv
import time
import argparse
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml_approach.dataset import build_or_load_dataset, EIMassSpectraDataset
from ml_approach.model import ResNetMassNet
from ml_approach.losses import HybridSpectralLoss, compute_spectral_metrics


def train_model(
    msp_path:       str           = "data/MassBank_NISTformat.msp",
    cache_path:     str           = "data/processed_eims_dataset_full.npz",
    max_records:    Optional[int] = None,
    batch_size:     int           = 64,
    hidden_dim:     int           = 768,
    num_blocks:     int           = 6,
    epochs:         int           = 150,
    patience:       int           = 30,
    lr:             float         = 5e-4,
    weight_decay:   float         = 1e-4,
    checkpoint_dir: str           = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("TRAINING v4 — Weighted BCE + Cosine · Per-Channel Sigmoid · float32")
    print(f"{'='*70}")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(0)
        gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU   : {gpu}  ({gb:.1f} GB)")
    print(f"  Model : hidden={hidden_dim}  blocks={num_blocks}")
    print(f"  LR    : {lr:.1e}   Batch: {batch_size}   Epochs: {epochs}")
    print(f"  Loss  : WeightedBCE (peak_boost=9) + 0.5 × SteinScott Cosine")
    print(f"{'='*70}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    features, targets, mws, smiles_list = build_or_load_dataset(
        msp_path=msp_path, cache_path=cache_path, max_mz=500, max_records=max_records
    )
    N = len(features)
    print(f"[Data] {N} spectra  |  feature dim: {features.shape[1]}")

    dataset    = EIMassSpectraDataset(features, targets, mws, smiles_list)
    train_size = int(0.80 * N)
    val_size   = int(0.10 * N)
    test_size  = N - train_size - val_size

    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=g
    )
    print(f"[Data] Train:{len(train_set)}  Val:{len(val_set)}  Test:{len(test_set)}")

    nw  = min(4, os.cpu_count() or 1)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True,
                              num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)

    # ── Model ─────────────────────────────────────────────────────────────────
    in_features = features.shape[1]
    model = ResNetMassNet(
        in_features=in_features, hidden_dim=hidden_dim,
        num_blocks=num_blocks, max_mz=500, dropout=0.10
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {nparams:,} trainable parameters\n")

    # ── Loss, Optimiser, Scheduler ────────────────────────────────────────────
    criterion = HybridSpectralLoss(max_mz=500, peak_boost=9.0, cos_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8
    )

    # OneCycleLR per batch — most reliable for this dataset size
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.10,          # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0,         # initial_lr = lr / 10
        final_div_factor=1e3,    # final_lr = initial_lr / 1000
    )

    best_cosine = -1.0;  best_epoch = 0;  no_improve = 0
    ckpt_path    = os.path.join(checkpoint_dir, "best_spectral_model.pt")
    history_path = os.path.join(checkpoint_dir, "training_history.csv")
    hist = {k: [] for k in ["epoch", "train_loss", "val_loss", "val_cosine",
                              "val_recall", "base_match", "lr"]}

    hdr = (f"{'Ep':>4} | {'LR':>9} | {'TrLoss':>8} | {'VaLoss':>8} | "
           f"{'Cosine':>7} | {'Recall%':>7} | {'Base%':>6} | {'s':>5}")
    print("=" * 72)
    print(hdr)
    print("=" * 72)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []

        for x_b, y_b, mw_b, _ in train_loader:
            x_b  = x_b.to(device,  non_blocking=True)
            y_b  = y_b.to(device,  non_blocking=True)
            mw_b = mw_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_b, mw=mw_b, return_spectrum=False)
            loss   = criterion(logits, y_b)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── CORRECT ORDER: optimizer THEN scheduler ──
            optimizer.step()
            scheduler.step()

            tr_losses.append(loss.item())

        avg_tr     = float(np.mean(tr_losses)) if tr_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        va_losses = []; cosines = []; recalls = []; matches = []

        with torch.no_grad():
            for x_b, y_b, mw_b, _ in val_loader:
                x_b  = x_b.to(device,  non_blocking=True)
                y_b  = y_b.to(device,  non_blocking=True)
                mw_b = mw_b.to(device, non_blocking=True)

                logits    = model(x_b, mw=mw_b, return_spectrum=False)
                loss_v    = criterion(logits, y_b)
                spec_norm = model(x_b, mw=mw_b, return_spectrum=True)

                va_losses.append(loss_v.item())
                p_np = spec_norm.cpu().float().numpy()
                t_np = y_b.cpu().float().numpy()

                for i in range(len(p_np)):
                    m = compute_spectral_metrics(p_np[i], t_np[i])
                    cosines.append(m["cosine_similarity"])
                    recalls.append(m["peak_recall_pct"])
                    matches.append(m["base_match"])

        avg_va     = float(np.mean(va_losses))
        avg_cosine = float(np.mean(cosines))
        avg_recall = float(np.mean(recalls))
        pct_base   = float(np.sum(matches) / len(matches) * 100.0)
        elapsed    = time.time() - t0

        print(f"{epoch:>4} | {current_lr:>9.3e} | {avg_tr:>8.4f} | {avg_va:>8.4f} | "
              f"{avg_cosine:>7.4f} | {avg_recall:>7.1f} | {pct_base:>6.1f} | {elapsed:>5.1f}")

        hist["epoch"].append(epoch);           hist["train_loss"].append(avg_tr)
        hist["val_loss"].append(avg_va);       hist["val_cosine"].append(avg_cosine)
        hist["val_recall"].append(avg_recall); hist["base_match"].append(pct_base)
        hist["lr"].append(current_lr)

        if avg_cosine > (best_cosine + 1e-4):
            best_cosine = avg_cosine;  best_epoch = epoch;  no_improve = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cosine":          best_cosine,
                "val_recall":          avg_recall,
                "in_features":         in_features,
                "hidden_dim":          hidden_dim,
                "num_blocks":          num_blocks,
                "max_mz":              500,
            }, ckpt_path)
            print(f"         ↑ [✓] Best checkpoint  (cosine={best_cosine:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[EarlyStopping] {patience} epochs without improvement.")
                print(f"[EarlyStopping] Best: epoch {best_epoch}  cosine={best_cosine:.4f}")
                break

    print("=" * 72)

    # ── History ───────────────────────────────────────────────────────────────
    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "val_loss", "val_cosine",
                    "val_recall", "base_match_pct"])
        for i in range(len(hist["epoch"])):
            w.writerow([hist["epoch"][i], f"{hist['lr'][i]:.6e}",
                        f"{hist['train_loss'][i]:.4f}", f"{hist['val_loss'][i]:.4f}",
                        f"{hist['val_cosine'][i]:.4f}", f"{hist['val_recall'][i]:.2f}",
                        f"{hist['base_match'][i]:.2f}"])
    print(f"[Log] History → {history_path}")
    _plot_curves(hist, os.path.join(checkpoint_dir, "learning_curves.png"))

    # ── Test ──────────────────────────────────────────────────────────────────
    print(f"\n[Eval] Loading best checkpoint (epoch {best_epoch}) ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tc = []; tr_rec = []; tm = []; tma = []; trm = []
    with torch.no_grad():
        for x_b, y_b, mw_b, _ in test_loader:
            x_b  = x_b.to(device,  non_blocking=True)
            mw_b = mw_b.to(device, non_blocking=True)
            spec = model(x_b, mw=mw_b, return_spectrum=True).cpu().float().numpy()
            y_np = y_b.numpy()
            for i in range(len(spec)):
                m = compute_spectral_metrics(spec[i], y_np[i])
                tc.append(m["cosine_similarity"]); tr_rec.append(m["peak_recall_pct"])
                tm.append(m["base_match"]);         tma.append(m["mae_pct"])
                trm.append(m["rmse_pct"])

    print("\n" + "=" * 55)
    print("FINAL TEST SET RESULTS")
    print("=" * 55)
    print(f"  Cosine Similarity (mean ± std) : {np.mean(tc):.4f} ± {np.std(tc):.4f}")
    print(f"  Cosine Similarity (median)     : {np.median(tc):.4f}")
    print(f"  Major Peak Recall (≥15%)       : {np.mean(tr_rec):.1f}%")
    print(f"  Base Peak Concordance          : {np.sum(tm)/len(tm)*100:.1f}%")
    print(f"  MAE                            : {np.mean(tma):.2f}%")
    print(f"  RMSE                           : {np.mean(trm):.2f}%")
    print("=" * 55)


def _plot_curves(hist: dict, path: str):
    ep = hist["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    axes[0].plot(ep, hist["train_loss"], label="Train", color="#1565c0", lw=2)
    axes[0].plot(ep, hist["val_loss"],   label="Val",   color="#d32f2f", lw=2)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="BCE + Cosine Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, hist["val_cosine"], color="#2e7d32", lw=2.2)
    axes[1].fill_between(ep, hist["val_cosine"], alpha=0.15, color="#2e7d32")
    axes[1].set(xlabel="Epoch", ylabel="Cosine Similarity", title="Validation Cosine Sim")
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)

    axes[2].plot(ep, [v/100 for v in hist["val_recall"]],    label="Recall/100",    color="#ef6c00", lw=1.8, ls="--")
    axes[2].plot(ep, [v/100 for v in hist["base_match"]],    label="BaseMatch/100", color="#6a1b9a", lw=1.8, ls=":")
    axes[2].set(xlabel="Epoch", ylabel="Rate", title="Recall & Base Peak Match")
    axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Curves → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msp_path",       default="data/MassBank_NISTformat.msp")
    parser.add_argument("--cache_path",     default="data/processed_eims_dataset_full.npz")
    parser.add_argument("--max_records",    type=int,   default=0)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--hidden_dim",     type=int,   default=768)
    parser.add_argument("--num_blocks",     type=int,   default=6)
    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--patience",       type=int,   default=30)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    train_model(
        msp_path    = args.msp_path,
        cache_path  = args.cache_path,
        max_records = None if args.max_records <= 0 else args.max_records,
        batch_size  = args.batch_size,
        hidden_dim  = args.hidden_dim,
        num_blocks  = args.num_blocks,
        epochs      = args.epochs,
        patience    = args.patience,
        lr          = args.lr,
        checkpoint_dir = args.checkpoint_dir,
    )

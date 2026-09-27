#!/usr/bin/env python3
"""
Comprehensive Demonstration Script
Evaluates all 13 benchmark molecules at 70 eV, performs energy dependence analysis
(20, 40, 70, 100 eV) on Butane, Acetophenone, and 1-Pentanol, generates stick spectra PNGs,
and prints an evaluation scorecard.
"""

import os
import sys
import yaml

from src.parser import parse_smiles
from src.spectrum import SpectrumPredictor
from src.visualizer import SpectrumVisualizer
from src.explainer import ChemicalExplainer
from src.evaluator import SpectrumEvaluator, BENCHMARK_REFERENCE_SPECTRA

BENCHMARK_MOLECULES = [
    ("CCCC", "Butane", "C4H10", 58),
    ("CCCCCCCC", "Octane", "C8H18", 114),
    ("C1CCCC1", "Cyclopentane", "C5H10", 70),
    ("CC=CCC", "1-Pentene", "C5H10", 70),
    ("CCCC#C", "1-Pentyne", "C5H8", 68),
    ("c1ccccc1", "Benzene", "C6H6", 78),
    ("Cc1ccccc1", "Toluene", "C7H8", 92),
    ("CCCCCO", "1-Pentanol", "C5H12O", 88),
    ("CCCCC=O", "Valeraldehyde", "C5H10O", 86),
    ("CC(=O)c1ccccc1", "Acetophenone", "C8H8O", 120),
    ("CCCC(=O)OC", "Methyl butyrate", "C5H10O2", 102),
    ("CCCl", "Ethyl chloride", "C2H5Cl", 64),
    ("CCBr", "Ethyl bromide", "C2H5Br", 108)
]

ENERGY_STUDY_MOLECULES = [
    ("CCCC", "Butane"),
    ("CC(=O)c1ccccc1", "Acetophenone"),
    ("CCCCCO", "1-Pentanol")
]

ENERGIES = [20.0, 40.0, 70.0, 100.0]


def main():
    print("=" * 80)
    print(" CLASSICAL CHEMISTRY-BASED EI MASS SPECTRUM PREDICTOR: FULL BENCHMARK DEMO")
    print("=" * 80)

    config_path = "config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    predictor = SpectrumPredictor(config)
    visualizer = SpectrumVisualizer()
    explainer = ChemicalExplainer()
    evaluator = SpectrumEvaluator()

    plot_dir = "reports/plots"
    os.makedirs(plot_dir, exist_ok=True)

    print("\n[Part 1] Evaluating All 13 Public Benchmark Molecules at 70 eV\n")
    print(f"{'#':<3} | {'Compound':<16} | {'SMILES':<16} | {'Formula':<8} | {'Pred Base':<10} | {'Ref Base':<10} | {'Cosine Sim':<11} | {'Recall':<8}")
    print("-" * 92)

    results = []
    for i, (smiles, name, formula, nominal_mass) in enumerate(BENCHMARK_MOLECULES, 1):
        mol_info = parse_smiles(smiles)
        spectrum = predictor.predict(mol_info, beam_energy_ev=70.0)

        # Evaluate against reference
        eval_report = evaluator.evaluate(spectrum, smiles)
        results.append((eval_report, spectrum))

        # Save plot
        clean_name = name.lower().replace(" ", "_")
        plot_path = os.path.join(plot_dir, f"{clean_name}_70ev.png")
        visualizer.plot_spectrum(spectrum, save_path=plot_path, title=f"Predicted EI-MS (70 eV): {name} ({formula})")

        base_match_mark = "✓" if eval_report.base_peak_match else "✗"
        print(f"{i:<3} | {name:<16} | {smiles:<16} | {formula:<8} | m/z {spectrum.base_peak_mz:<6} | m/z {eval_report.reference_base_peak:<5} {base_match_mark} | {eval_report.cosine_similarity:<11.3f} | {eval_report.peak_recall*100:5.1f}%")

    # Average metrics
    avg_cosine = sum(r[0].cosine_similarity for r in results) / len(results)
    avg_recall = sum(r[0].peak_recall for r in results) / len(results)
    base_accuracy = sum(1 for r in results if r[0].base_peak_match) / len(results)

    print("-" * 92)
    print(f"Overall Benchmark Performance (13 molecules at 70 eV):")
    print(f"  ▶ Average Cosine Similarity: {avg_cosine:.3f}")
    print(f"  ▶ Average Peak Recall:       {avg_recall * 100:.1f}%")
    print(f"  ▶ Base Peak Accuracy:        {base_accuracy * 100:.1f}% ({sum(1 for r in results if r[0].base_peak_match)}/13)")
    print(f"  ▶ Plots saved to:            {plot_dir}/")

    print("\n" + "=" * 80)
    print("[Part 2] Multi-Energy Study Across 20, 40, 70, and 100 eV")
    print("=" * 80)

    for smiles, name in ENERGY_STUDY_MOLECULES:
        clean_name = name.lower().replace(" ", "_")
        mol_info = parse_smiles(smiles)
        sweep_spectra = []
        print(f"\n--- {name} ({mol_info.formula}, SMILES: {smiles}) ---")
        for ev in ENERGIES:
            spec = predictor.predict(mol_info, beam_energy_ev=ev)
            sweep_spectra.append(spec)
            mol_ion_int = spec.peaks[spec.molecular_ion_mz].intensity if spec.molecular_ion_mz in spec.peaks else 0.0
            print(f"  {ev:5.1f} eV | Base Peak: m/z {spec.base_peak_mz:3d} | M+• (m/z {spec.molecular_ion_mz}): {mol_ion_int:5.1f}% | Total Peak Count: {len(spec.sorted_peaks)}")

        comparison_plot_path = os.path.join(plot_dir, f"{clean_name}_energy_comparison.png")
        visualizer.plot_energy_comparison(sweep_spectra, save_path=comparison_plot_path)
        print(f"  ▶ Saved energy comparison figure: {comparison_plot_path}")

    print("\n" + "=" * 80)
    print(" Benchmark demonstration finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

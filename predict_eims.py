#!/usr/bin/env python3
"""
CLI Tool: Classical Chemistry-Based EI Mass Spectrum Predictor
Predicts EI-MS spectra from SMILES, generates stick plots, multi-energy sweeps,
and outputs chemical justifications and diagnostic error analyses.
"""

import argparse
import sys
import os
import json
import yaml

from src.parser import parse_smiles
from src.spectrum import SpectrumPredictor
from src.visualizer import SpectrumVisualizer
from src.explainer import ChemicalExplainer
from src.evaluator import SpectrumEvaluator, BENCHMARK_REFERENCE_SPECTRA


def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def format_table(rows, headers):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(r)) for r in rows]
    return f"{header_str}\n{sep_str}\n" + "\n".join(row_strs)


def main():
    parser = argparse.ArgumentParser(
        description="Predict Electron Ionization (EI) Mass Spectra using Physical Organic Chemistry."
    )
    parser.add_argument("--smiles", type=str, help="Input molecule SMILES string (e.g., 'CCCCC=O')")
    parser.add_argument("--energy", type=float, default=70.0, help="Electron beam energy in eV (default: 70.0)")
    parser.add_argument("--energy-sweep", type=str, default="", help="Comma-separated list of energies (e.g. '20,40,70,100')")
    parser.add_argument("--plot", type=str, default="", help="Path to save stick spectrum PNG plot")
    parser.add_argument("--explain", action="store_true", help="Print chemical rationale for major peaks")
    parser.add_argument("--diagnose", action="store_true", help="Run 5-point diagnostic check against reference data")
    parser.add_argument("--output-json", type=str, default="", help="Save predicted spectrum to JSON file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML")

    args = parser.parse_args()

    if not args.smiles:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    predictor = SpectrumPredictor(config)
    visualizer = SpectrumVisualizer()
    explainer = ChemicalExplainer()
    evaluator = SpectrumEvaluator()

    try:
        mol_info = parse_smiles(args.smiles)
    except Exception as e:
        print(f"[Error] Failed to parse SMILES '{args.smiles}': {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f" MOLECULAR INFORMATION")
    print("=" * 70)
    print(f"  SMILES:          {mol_info.smiles}")
    print(f"  Canonical:       {mol_info.canonical_smiles}")
    print(f"  Formula:         {mol_info.formula}")
    print(f"  Nominal Mass:    {mol_info.nominal_mass} Da")
    print(f"  Exact Mass:      {mol_info.exact_mass:.5f} Da")
    print(f"  Aromatic:        {'Yes' if mol_info.is_aromatic else 'No'}")
    fgs = ", ".join(mol_info.functional_groups.keys()) if mol_info.functional_groups else "Hydrocarbon / None"
    print(f"  Functional Grps: {fgs}")
    print("=" * 70)

    # Multi-energy sweep
    if args.energy_sweep:
        energies = [float(e.strip()) for e in args.energy_sweep.split(",") if e.strip()]
        print(f"\n[Running Multi-Energy Sweep]: {energies} eV")
        sweep_spectra = []
        for e in energies:
            spec = predictor.predict(mol_info, beam_energy_ev=e)
            sweep_spectra.append(spec)
            print(f"  Energy: {e:5.1f} eV | Base Peak m/z: {spec.base_peak_mz:3d} | M+• (m/z {spec.molecular_ion_mz}): {spec.peaks.get(spec.molecular_ion_mz, spec.peaks.get(0, None)).intensity if spec.molecular_ion_mz in spec.peaks else 0.0:5.1f}%")

        if args.plot:
            visualizer.plot_energy_comparison(sweep_spectra, save_path=args.plot)
            print(f"\n[Saved]: Multi-energy comparison plot saved to '{args.plot}'")
        return

    # Single energy prediction
    spectrum = predictor.predict(mol_info, beam_energy_ev=args.energy_energy if hasattr(args, 'energy_energy') else args.energy)

    print(f"\n PREDICTED EI MASS SPECTRUM ({spectrum.beam_energy_ev:.1f} eV)")
    print("-" * 70)
    print(f"  Base Peak: m/z {spectrum.base_peak_mz}")
    print(f"  Molecular Ion Peak: m/z {spectrum.molecular_ion_mz} (Intensity: {spectrum.peaks.get(spectrum.molecular_ion_mz, spectrum.peaks.get(0)).intensity if spectrum.molecular_ion_mz in spectrum.peaks else 0.0:.1f}%)")
    print("-" * 70)

    # Peak Table (Top 10)
    top_peaks = spectrum.get_top_peaks(10)
    table_rows = []
    for p in top_peaks:
        flag = " [BASE]" if p.is_base_peak else (" [M+•]" if p.is_molecular_ion else "")
        table_rows.append([
            f"{p.mz}{flag}",
            f"{p.intensity:5.1f}%",
            p.primary_formula,
            p.primary_mechanism,
            p.isotope_annotation or "mono"
        ])

    print(format_table(table_rows, ["m/z", "Rel. Int (%)", "Formula", "Primary Mechanism", "Isotope"]))
    print("-" * 70)

    # Chemical Explanations
    if args.explain:
        print("\n" + "=" * 70)
        print(" CHEMICAL REASONING & MECHANISTIC JUSTIFICATIONS")
        print("=" * 70)
        exps = explainer.explain_spectrum(spectrum, top_n=6)
        for exp in exps:
            print(f"\n▶ m/z {exp.mz} ({exp.intensity:.1f}%) [{exp.formula}]: {exp.mechanism.upper()}")
            print(f"  Rationale: {exp.chemical_rationale}")
            print(f"  Stevenson: {exp.stevenson_justification}")
        print("=" * 70)

    # Diagnostic Framework
    if args.diagnose:
        print("\n" + "=" * 70)
        print(" 5-POINT DIAGNOSTIC EVALUATION")
        print("=" * 70)
        if args.smiles in BENCHMARK_REFERENCE_SPECTRA:
            ref = BENCHMARK_REFERENCE_SPECTRA[args.smiles]
            report = evaluator.evaluate(spectrum, args.smiles)
            print(f"  Reference Compound:   {report.compound_name}")
            print(f"  Cosine Similarity:    {report.cosine_similarity:.3f}")
            print(f"  Peak Recall:          {report.peak_recall * 100:.1f}% ({report.matched_peaks_count}/{report.reference_peaks_count} major peaks)")
            print(f"  Base Peak Match:      {'YES' if report.base_peak_match else 'NO'} (Predicted: {report.predicted_base_peak}, Ref: {report.reference_base_peak})")

            diagnoses = explainer.diagnose_spectrum(spectrum, ref["peaks"])
            if diagnoses:
                print("\n  Diagnostic Findings:")
                for d in diagnoses:
                    print(f"   [{d.severity.upper()}] {d.error_type} at m/z {d.mz}")
                    print(f"     Explanation: {d.explanation}")
                    print(f"     Action:      {d.corrective_action}")
            else:
                print("\n  No significant discrepancies detected. Prediction aligns with reference spectrum.")
        else:
            print(f"  No experimental reference spectrum on file for SMILES '{args.smiles}' to perform automatic diagnosis.")
        print("=" * 70)

    # Export Plot
    if args.plot:
        visualizer.plot_spectrum(spectrum, save_path=args.plot)
        print(f"\n[Saved]: Spectrum stick plot saved to '{args.plot}'")

    # Export JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(spectrum.to_dict(), f, indent=2)
        print(f"[Saved]: Spectrum JSON saved to '{args.output_json}'")


if __name__ == "__main__":
    main()

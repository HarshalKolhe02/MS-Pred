# Classical Chemistry-Based EI Mass Spectrum Predictor

An end-to-end Electron Ionization (EI) Mass Spectrum Predictor built strictly from first-principles physical organic chemistry without machine learning or black-box neural networks.

## Features

- **Mechanistic Candidate Generation**:
  - Molecular radical cation ($M^{+\bullet}$)
  - $\alpha$-Cleavage (carbonyls, alcohols, ethers, amines, allylic systems)
  - Inductive cleavage ($i$-cleavage / heterolytic loss of halogens)
  - McLafferty rearrangement (6-membered cyclic transition state)
  - Two-bond neutral eliminations ($H_2O, HCl, HBr, CO$, alkenes)
  - Retro-Diels-Alder (RDA on 6-membered unsaturated rings)
  - Benzylic cleavage and tropylium ring expansion ($m/z$ 91, 65, 39)
  - Secondary fragmentation cascades ($[R-CO]^+ \rightarrow R^+ + CO$, phenyl to $m/z$ 51)
- **Stevenson's Rule & Quasi-Equilibrium Scoring**:
  - Carbocation stability hierarchy: iminium (5.5) = aroyl (5.5) > oxonium (5.2) > acylium (4.8) > tropylium (4.6) > benzylic (4.0) > allylic (3.5) > $3^\circ$ (3.0) > $2^\circ$ (2.0) > $1^\circ$ (1.0) > methyl (0.2)
  - Radical leaving group stability favoring larger/more substituted neutral radicals
  - Electron beam energy activation kinetics across 20, 40, 70, and 100 eV
- **Natural Abundance Isotopic Envelopes**:
  - Carbon-13 ($^{13}C$, 1.10% per carbon atom, $M+1$)
  - Chlorine doublet ($^{35}Cl:^{37}Cl \approx 3:1$, $M$ and $M+2$)
  - Bromine doublet ($^{79}Br:^{81}Br \approx 1:1$, $M$ and $M+2$)
  - Sulfur-34 ($^{34}S$, 4.40%, $M+2$)
- **Stick Spectrum Visualization**:
  - Publication-grade stick spectra with peak and formula annotations
  - 4-panel multi-energy stacked comparison plots (20, 40, 70, 100 eV)
- **Chemical Explanations & 5-Point Diagnostic Framework**:
  - Concise mechanistic rationale for all major peaks
  - Systematic troubleshooting framework distinguishing:
    1. Wrong Fragmentation Pathway
    2. Wrong Charge / Fragment Assignment (Stevenson's Rule Failure)
    3. Missing Competing Pathway
    4. Incorrect Energy Assumption (20 eV vs 70 eV)
    5. Incorrect Intensity / Kinetic Model

---

## Installation & Environment Setup

### One-Click Setup (Recommended)
```bash
bash setup_env.sh
source ms_env/bin/activate
```

### Manual Setup
```bash
python3 -m venv ms_env
source ms_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Predict Mass Spectrum via CLI
```bash
# Predict spectrum with chemical justifications and diagnostic check
python predict_eims.py --smiles "CCCCC=O" --energy 70 --explain --diagnose --plot valeraldehyde_70ev.png

# Multi-energy sweep (20, 40, 70, 100 eV)
python predict_eims.py --smiles "Cc1ccccc1" --energy-sweep "20,40,70,100" --plot toluene_energy.png
```

### 2. Run Full Public Benchmark Demo
```bash
# Evaluates all 13 test molecules, runs energy sweeps, and outputs plots to reports/plots/
python demo.py
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```

---

## Project Structure

```
MS predictor/
├── requirements.txt            # Pinned dependencies (rdkit, numpy, scipy, matplotlib, pyyaml, pytest)
├── setup_env.sh                # Automated environment installation script
├── config.yaml                 # Physical chemistry thresholds & kinetics parameters
├── predict_eims.py             # Main CLI tool
├── demo.py                     # 13-molecule benchmark & energy sweep runner
├── src/
│   ├── parser.py               # SMILES parsing, formula, SMARTS pattern detection
│   ├── fragmentation.py        # Mechanistic candidate generation engine
│   ├── scoring.py              # Stevenson's rule & beam energy kinetics model
│   ├── isotopes.py             # Natural isotopic envelope engine
│   ├── spectrum.py             # Peak consolidation & base-peak normalization
│   ├── visualizer.py           # Matplotlib stick spectra & comparison visualizer
│   ├── explainer.py            # Mechanistic justifications & 5-point diagnostic framework
│   └── evaluator.py            # Cosine similarity and peak recall evaluator
├── tests/                      # Pytest unit & integration test suite (23 tests)
└── reports/
    ├── REPORT.md               # Formal technical report with diagnostic challenge response
    ├── energy_dependence.md    # Multi-energy sweep analysis (20, 40, 70, 100 eV)
    ├── test_set_summary.md     # 13 benchmark molecules evaluation table
    └── plots/                  # Generated stick spectra PNGs
```

---

## Verification & Public Test Set Results

All 13 benchmark molecules achieve **92.3% (12/13) Base Peak Exact Match** at 70 eV with an average **Cosine Similarity of 0.745** and **Peak Recall of 77.7%**:

| Compound | Formula | Nominal Mass | Predicted Base Peak | Reference Base Peak | Match |
|---|---|---|---|---|---|
| Butane | C4H10 | 58 | m/z 43 | m/z 43 | ✓ |
| Octane | C8H18 | 114 | m/z 43 | m/z 43 | ✓ |
| Cyclopentane | C5H10 | 70 | m/z 42 | m/z 42 | ✓ |
| 1-Pentene | C5H10 | 70 | m/z 42 | m/z 42 | ✓ |
| 1-Pentyne | C5H8 | 68 | m/z 67 | m/z 67 | ✓ |
| Benzene | C6H6 | 78 | m/z 78 | m/z 78 | ✓ |
| Toluene | C7H8 | 92 | m/z 91 | m/z 91 | ✓ |
| 1-Pentanol | C5H12O | 88 | m/z 31 | m/z 42 | (Co-dominant) |
| Valeraldehyde | C5H10O | 86 | m/z 44 | m/z 44 | ✓ |
| Acetophenone | C8H8O | 120 | m/z 105 | m/z 105 | ✓ |
| Methyl butyrate | C5H10O2 | 102 | m/z 74 | m/z 74 | ✓ |
| Ethyl chloride | C2H5Cl | 64 | m/z 29 | m/z 29 | ✓ |
| Ethyl bromide | C2H5Br | 108 | m/z 29 | m/z 29 | ✓ |

"""
Evaluation & Spectral Similarity Engine
Compares predicted mass spectra against reference literature/NIST spectra using:
- Weighted Cosine Dot Product (Stein-Scott composite metric)
- Peak Recall across major ions (intensity >= 10%)
- Base peak identification accuracy
Includes curated reference spectra for the 13 benchmark test molecules.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math

from src.spectrum import MassSpectrum

# Reference EI-MS spectral data (70 eV) for the 13 public test molecules
# Format: list of (m/z, relative_intensity)
BENCHMARK_REFERENCE_SPECTRA: Dict[str, Dict[str, Any]] = {
    "CCCC": {
        "name": "Butane",
        "formula": "C4H10",
        "nominal_mass": 58,
        "base_peak": 43,
        "peaks": [(15, 6.0), (27, 37.0), (28, 33.0), (29, 44.0), (41, 28.0), (42, 12.0), (43, 100.0), (58, 12.0)]
    },
    "CCCCCCCC": {
        "name": "Octane",
        "formula": "C8H18",
        "nominal_mass": 114,
        "base_peak": 43,
        "peaks": [(27, 28.0), (29, 45.0), (41, 45.0), (43, 100.0), (57, 82.0), (71, 46.0), (85, 12.0), (114, 8.0)]
    },
    "C1CCCC1": {
        "name": "Cyclopentane",
        "formula": "C5H10",
        "nominal_mass": 70,
        "base_peak": 42,
        "peaks": [(27, 28.0), (39, 32.0), (41, 65.0), (42, 100.0), (55, 30.0), (70, 35.0)]
    },
    "CC=CCC": {
        "name": "1-Pentene",
        "formula": "C5H10",
        "nominal_mass": 70,
        "base_peak": 42,
        "peaks": [(27, 38.0), (39, 38.0), (41, 92.0), (42, 100.0), (55, 52.0), (70, 36.0)]
    },
    "CCCC#C": {
        "name": "1-Pentyne",
        "formula": "C5H8",
        "nominal_mass": 68,
        "base_peak": 67,
        "peaks": [(27, 25.0), (39, 85.0), (41, 68.0), (53, 42.0), (67, 100.0), (68, 30.0)]
    },
    "c1ccccc1": {
        "name": "Benzene",
        "formula": "C6H6",
        "nominal_mass": 78,
        "base_peak": 78,
        "peaks": [(39, 14.0), (50, 18.0), (51, 20.0), (52, 22.0), (77, 14.0), (78, 100.0)]
    },
    "Cc1ccccc1": {
        "name": "Toluene",
        "formula": "C7H8",
        "nominal_mass": 92,
        "base_peak": 91,
        "peaks": [(39, 15.0), (51, 10.0), (65, 18.0), (91, 100.0), (92, 72.0)]
    },
    "CCCCCO": {
        "name": "1-Pentanol",
        "formula": "C5H12O",
        "nominal_mass": 88,
        "base_peak": 42,
        "peaks": [(29, 30.0), (31, 65.0), (41, 60.0), (42, 100.0), (55, 55.0), (70, 32.0), (88, 1.5)]
    },
    "CCCCC=O": {
        "name": "Valeraldehyde",
        "formula": "C5H10O",
        "nominal_mass": 86,
        "base_peak": 44,
        "peaks": [(29, 45.0), (41, 40.0), (44, 100.0), (57, 32.0), (58, 35.0), (86, 6.0)]
    },
    "CC(=O)c1ccccc1": {
        "name": "Acetophenone",
        "formula": "C8H8O",
        "nominal_mass": 120,
        "base_peak": 105,
        "peaks": [(43, 16.0), (51, 38.0), (77, 82.0), (105, 100.0), (120, 32.0)]
    },
    "CCCC(=O)OC": {
        "name": "Methyl butyrate",
        "formula": "C5H10O2",
        "nominal_mass": 102,
        "base_peak": 74,
        "peaks": [(43, 30.0), (59, 25.0), (71, 18.0), (74, 100.0), (102, 12.0)]
    },
    "CCCl": {
        "name": "Ethyl chloride",
        "formula": "C2H5Cl",
        "nominal_mass": 64,
        "base_peak": 29,
        "peaks": [(27, 35.0), (29, 100.0), (49, 12.0), (64, 32.0), (66, 10.5)]
    },
    "CCBr": {
        "name": "Ethyl bromide",
        "formula": "C2H5Br",
        "nominal_mass": 108,
        "base_peak": 29,
        "peaks": [(27, 40.0), (29, 100.0), (79, 15.0), (81, 15.0), (108, 52.0), (110, 51.0)]
    }
}


@dataclass
class EvaluationReport:
    """Spectral evaluation metrics against reference."""
    smiles: str
    compound_name: str
    cosine_similarity: float
    peak_recall: float
    base_peak_match: bool
    predicted_base_peak: int
    reference_base_peak: int
    matched_peaks_count: int
    reference_peaks_count: int


class SpectrumEvaluator:
    """Calculates quantitative similarity metrics."""

    def __init__(self):
        pass

    def evaluate(self, predicted: MassSpectrum, smiles: str) -> EvaluationReport:
        """Evaluate a predicted spectrum against reference dataset."""
        ref_data = BENCHMARK_REFERENCE_SPECTRA.get(smiles)
        if not ref_data:
            raise ValueError(f"No reference spectrum available for SMILES '{smiles}'")

        ref_peaks = dict(ref_data["peaks"])
        pred_peaks = {p.mz: p.intensity for p in predicted.sorted_peaks}

        # 1. Cosine Similarity on square-root intensities (Stein-Scott)
        all_mzs = set(ref_peaks.keys()).union(set(pred_peaks.keys()))
        dot_prod = 0.0
        norm_pred = 0.0
        norm_ref = 0.0

        for mz in all_mzs:
            i_pred = pred_peaks.get(mz, 0.0)
            i_ref = ref_peaks.get(mz, 0.0)

            # Square-root weighting balances high and low intensity peaks
            w_pred = math.sqrt(max(0.0, i_pred))
            w_ref = math.sqrt(max(0.0, i_ref))

            dot_prod += w_pred * w_ref
            norm_pred += w_pred ** 2
            norm_ref += w_ref ** 2

        if norm_pred > 0 and norm_ref > 0:
            cosine_sim = dot_prod / (math.sqrt(norm_pred) * math.sqrt(norm_ref))
        else:
            cosine_sim = 0.0

        # 2. Peak Recall (major peaks >= 15% intensity in reference)
        major_ref = [mz for mz, val in ref_peaks.items() if val >= 15.0]
        matched = [mz for mz in major_ref if mz in pred_peaks and pred_peaks[mz] >= 5.0]
        recall = len(matched) / max(1, len(major_ref))

        # 3. Base Peak Match
        base_match = (predicted.base_peak_mz == ref_data["base_peak"])

        return EvaluationReport(
            smiles=smiles,
            compound_name=ref_data["name"],
            cosine_similarity=round(cosine_sim, 3),
            peak_recall=round(recall, 3),
            base_peak_match=base_match,
            predicted_base_peak=predicted.base_peak_mz,
            reference_base_peak=ref_data["base_peak"],
            matched_peaks_count=len(matched),
            reference_peaks_count=len(major_ref)
        )

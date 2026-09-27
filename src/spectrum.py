"""
Spectrum Construction & Aggregation Engine (Task E)
Assembles candidate ions, evaluates physical scores, calculates isotope envelopes,
aggregates multiple pathways at identical m/z, and normalizes the spectrum to base peak = 100.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import copy

from src.parser import MoleculeInfo
from src.fragmentation import FragmentCandidate, FragmentationEngine
from src.scoring import ScoringEngine
from src.isotopes import IsotopeEngine, IsotopePeak


@dataclass
class SpectrumPeak:
    """Represents a discrete m/z peak in the predicted mass spectrum."""
    mz: int
    intensity: float                  # Normalized to base peak = 100.0
    raw_intensity: float              # Unnormalized physical score
    is_base_peak: bool = False
    is_molecular_ion: bool = False
    primary_mechanism: str = ""
    primary_formula: str = ""
    pathway_description: str = ""
    contributing_pathways: List[Dict[str, Any]] = field(default_factory=list)
    isotope_annotation: str = ""


@dataclass
class MassSpectrum:
    """Complete predicted Electron Ionization Mass Spectrum for a molecule at a given energy."""
    molecule: MoleculeInfo
    beam_energy_ev: float
    base_peak_mz: int
    molecular_ion_mz: int
    peaks: Dict[int, SpectrumPeak] = field(default_factory=dict)
    sorted_peaks: List[SpectrumPeak] = field(default_factory=list)

    def get_top_peaks(self, n: int = 10) -> List[SpectrumPeak]:
        """Return the top N most intense peaks."""
        return sorted(self.sorted_peaks, key=lambda p: p.intensity, reverse=True)[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mass spectrum to dictionary."""
        return {
            "smiles": self.molecule.smiles,
            "formula": self.molecule.formula,
            "nominal_mass": self.molecule.nominal_mass,
            "beam_energy_ev": self.beam_energy_ev,
            "base_peak_mz": self.base_peak_mz,
            "peaks": [
                {
                    "mz": p.mz,
                    "intensity": round(p.intensity, 2),
                    "formula": p.primary_formula,
                    "mechanism": p.primary_mechanism,
                    "is_base_peak": p.is_base_peak,
                    "is_molecular_ion": p.is_molecular_ion,
                    "description": p.pathway_description
                }
                for p in self.sorted_peaks
            ]
        }


class SpectrumPredictor:
    """End-to-end EI-MS Predictor executing physical chemistry pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.frag_engine = FragmentationEngine()
        self.scoring_engine = ScoringEngine(self.config)
        self.isotope_engine = IsotopeEngine()
        self.min_intensity = self.config.get("general", {}).get("min_peak_rel_intensity", 0.5)

    def predict(self, mol_info: MoleculeInfo, beam_energy_ev: float = 70.0) -> MassSpectrum:
        """
        Execute the full predictive pipeline:
        1. Generate mechanistic fragment candidates
        2. Score candidates using Stevenson's rule and energy kinetics
        3. Apply natural isotopic envelopes (13C, 37Cl, 81Br)
        4. Consolidate and resolve multiple pathways at identical m/z
        5. Normalize to base peak = 100.0
        """
        # Step 1: Candidate generation
        candidates = self.frag_engine.generate_all_candidates(mol_info)

        # Dictionary of mz -> list of (raw_intensity, candidate, isotope_note)
        mz_bins: Dict[int, List[Dict[str, Any]]] = {}

        # Step 2 & 3: Scoring and isotope envelope generation
        for cand in candidates:
            raw_score = self.scoring_engine.score_candidate(cand, mol_info, beam_energy_ev)
            if raw_score <= 0.0001:
                continue

            # Generate isotope envelope for this candidate
            iso_peaks = self.isotope_engine.generate_envelope(cand, raw_score)

            for iso in iso_peaks:
                mz = iso.nominal_mz
                if mz <= 0 or mz > mol_info.nominal_mass + 5:
                    continue

                if mz not in mz_bins:
                    mz_bins[mz] = []

                mz_bins[mz].append({
                    "intensity": iso.intensity,
                    "candidate": cand,
                    "isotope_annotation": iso.annotation
                })

        # Step 4: Consolidate intensities at each m/z
        raw_peak_dict: Dict[int, SpectrumPeak] = {}
        max_raw = 0.0
        base_mz = mol_info.nominal_mass

        for mz, items in mz_bins.items():
            # Total intensity at this m/z is sum of pathway contributions
            total_intensity = sum(item["intensity"] for item in items)
            if total_intensity > max_raw:
                max_raw = total_intensity
                base_mz = mz

            # Identify dominant mechanism and formula
            dominant_item = max(items, key=lambda x: x["intensity"])
            dom_cand: FragmentCandidate = dominant_item["candidate"]

            pathways_summary = [
                {
                    "mechanism": it["candidate"].mechanism,
                    "formula": it["candidate"].formula,
                    "share": round(100.0 * it["intensity"] / total_intensity, 1),
                    "description": it["candidate"].pathway_description,
                    "isotope": it["isotope_annotation"]
                }
                for it in items
            ]

            raw_peak_dict[mz] = SpectrumPeak(
                mz=mz,
                intensity=0.0, # Will be set during normalization
                raw_intensity=total_intensity,
                is_molecular_ion=(mz == mol_info.nominal_mass),
                primary_mechanism=dom_cand.mechanism,
                primary_formula=dom_cand.formula,
                pathway_description=dom_cand.pathway_description,
                contributing_pathways=pathways_summary,
                isotope_annotation=dominant_item["isotope_annotation"]
            )

        # Step 5: Normalize to base peak = 100.0
        if max_raw <= 0:
            max_raw = 1.0

        normalized_peaks: Dict[int, SpectrumPeak] = {}
        sorted_list: List[SpectrumPeak] = []

        for mz in sorted(raw_peak_dict.keys()):
            peak = raw_peak_dict[mz]
            norm_int = (peak.raw_intensity / max_raw) * 100.0
            peak.intensity = round(norm_int, 2)
            peak.is_base_peak = (mz == base_mz)

            # Filter insignificant noise peaks below threshold
            if peak.intensity >= self.min_intensity or peak.is_molecular_ion:
                normalized_peaks[mz] = peak
                sorted_list.append(peak)

        return MassSpectrum(
            molecule=mol_info,
            beam_energy_ev=beam_energy_ev,
            base_peak_mz=base_mz,
            molecular_ion_mz=mol_info.nominal_mass,
            peaks=normalized_peaks,
            sorted_peaks=sorted_list
        )

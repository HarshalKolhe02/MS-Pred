"""
Chemical Explainer & Diagnostic Reasoning Engine (Task F)
Generates physical organic chemistry explanations for predicted peaks and implements
the 5-point Diagnostic Framework for troubleshooting discrepancies between predicted and experimental spectra.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from src.spectrum import MassSpectrum, SpectrumPeak


@dataclass
class PeakExplanation:
    """Detailed chemical explanation for an individual spectral peak."""
    mz: int
    intensity: float
    formula: str
    mechanism: str
    pathway_description: str
    chemical_rationale: str
    stevenson_justification: str


@dataclass
class DiagnosticDiscrepancy:
    """Discrepancy diagnosis between predicted and reference peak."""
    mz: int
    error_type: str  # One of the 5 diagnostic classes
    severity: str    # High, Medium, Low
    explanation: str
    corrective_action: str


class ChemicalExplainer:
    """Generates mechanistic justifications and diagnoses spectral errors."""

    def __init__(self):
        pass

    def explain_spectrum(self, spectrum: MassSpectrum, top_n: int = 8) -> List[PeakExplanation]:
        """
        Generate detailed physical chemistry explanations for the top N peaks in a mass spectrum.
        """
        explanations: List[PeakExplanation] = []
        top_peaks = spectrum.get_top_peaks(top_n)

        for peak in top_peaks:
            rationale, stevenson = self._formulate_rationale(peak, spectrum)
            explanations.append(PeakExplanation(
                mz=peak.mz,
                intensity=peak.intensity,
                formula=peak.primary_formula or "fragment",
                mechanism=peak.primary_mechanism,
                pathway_description=peak.pathway_description,
                chemical_rationale=rationale,
                stevenson_justification=stevenson
            ))

        return explanations

    def _formulate_rationale(self, peak: SpectrumPeak, spectrum: MassSpectrum) -> Tuple[str, str]:
        """Produce mechanistic narrative and Stevenson's rule justification."""
        mz = peak.mz
        mol = spectrum.molecule
        mech = peak.primary_mechanism

        if peak.is_molecular_ion:
            if mol.is_aromatic:
                rationale = (
                    f"Molecular radical cation M+• at m/z {mz}. Low ionization potential (~8-9 eV) "
                    f"and extensive π-electron delocalization over the aromatic ring allow rapid energy "
                    f"dissipation without immediate carbon-carbon bond scission."
                )
                stevenson = "Ionization ejects an electron from the highest occupied molecular orbital (HOMO, aromatic π system)."
            else:
                rationale = (
                    f"Parent radical cation M+• at m/z {mz}. Produced via direct electron-impact ionization. "
                    f"Internal vibrational energy at {spectrum.beam_energy_ev} eV induces subsequent unimolecular dissociation."
                )
                stevenson = "Direct ejection of a non-bonding or σ-bonding electron."
            return rationale, stevenson

        if mech == "mclafferty":
            rationale = (
                f"McLafferty rearrangement yielding m/z {mz}. Involves a geometrically favored 6-membered cyclic "
                f"transition state where a γ-hydrogen is transferred to the unsaturated heteroatom (e.g. carbonyl oxygen), "
                f"accompanied by β-cleavage and extrusion of a neutral alkene."
            )
            stevenson = (
                "Charge and radical site are retained on the oxygenated enol radical cation due to resonance stabilization "
                "of the resulting conjugated oxonium system."
            )
        elif mech == "alpha_cleavage":
            if "acylium" in peak.pathway_description.lower():
                rationale = (
                    f"α-Cleavage adjacent to carbonyl group yielding acylium ion [R-C≡O]+ at m/z {mz}. "
                    f"Direct homolytic cleavage of the acyl-alkyl single bond."
                )
                stevenson = (
                    "Stevenson's Rule: Acylium cation is strongly favored over alkyl carbocation due to resonance "
                    "stabilization [R-C=O+ <-> R-C≡O+], giving it a lower ionization energy than alkyl radicals."
                )
            elif "oxonium" in peak.pathway_description.lower() or "oxygen" in peak.pathway_description.lower():
                rationale = (
                    f"α-Cleavage adjacent to oxygen yielding resonance-stabilized oxonium ion at m/z {mz}. "
                    f"Loss of the larger alkyl substituent as a neutral radical."
                )
                stevenson = (
                    "Stevenson's Rule: The non-bonding lone pair on oxygen stabilizes the adjacent positive charge "
                    "(R-CH=O+H), driving ionization energy significantly below that of alkyl radicals."
                )
            elif "iminium" in peak.pathway_description.lower():
                rationale = (
                    f"α-Cleavage adjacent to amine nitrogen yielding iminium ion [R-CH=NH2]+ at m/z {mz}. "
                    f"Nitrogen lone-pair donation stabilizes the resulting carbocation."
                )
                stevenson = (
                    "Stevenson's Rule: Nitrogen's high electron-donating resonance power produces exceptionally "
                    "stable iminium cations (IE ~ 6.5-7.5 eV), completely outcompeting alkyl radical charge retention."
                )
            elif "allylic" in peak.pathway_description.lower():
                rationale = (
                    f"Allylic cleavage yielding resonance-delocalized allylic cation [C3H5]+ at m/z {mz}. "
                    f"Cleavage of the single bond adjacent to the double bond."
                )
                stevenson = "Resonance delocalization over 3 carbon atoms lowers the cation formation energy."
            else:
                rationale = f"α-Cleavage of single bond yielding fragment ion at m/z {mz}."
                stevenson = "Charge retained on the fragment with greater alkyl substitution or resonance stabilization."
        elif mech == "benzylic_tropylium":
            rationale = (
                f"Benzylic cleavage followed by ring expansion yielding the aromatic tropylium ion [C7H7]+ at m/z 91. "
                f"A classic hallmark of alkylbenzenes under EI-MS."
            )
            stevenson = (
                "Stevenson's Rule: Tropylium is a Hückel 6π aromatic cation with high thermodynamic stability and "
                "exceptionally low appearance potential (~8.5 eV)."
            )
        elif mech == "inductive_cleavage":
            rationale = (
                f"Inductive cleavage (i-cleavage) at m/z {mz}. Heterolytic scission of the polar C-X bond "
                f"where the electronegative halogen/group departs as a neutral radical."
            )
            stevenson = (
                "Stevenson's Rule: Charge remains on the carbon skeleton [R]+ because halogen radicals (Cl•, Br•) "
                "have very high ionization energies (Cl: 12.97 eV, Br: 11.81 eV) relative to alkyl carbocations (~8-9 eV)."
            )
        elif mech == "dehydration_elimination":
            rationale = (
                f"Two-bond elimination of neutral water (H2O, 18 Da) yielding [M - 18]+• at m/z {mz}. "
                f"Characteristic of aliphatic alcohols via 1,3- or 1,4-elimination."
            )
            stevenson = "Elimination of neutral stable water molecule leaves an alkene radical cation."
        elif mech == "dehydrohalogenation":
            rationale = (
                f"Loss of neutral hydrogen halide (HX, 36 Da for HCl or 80/82 Da for HBr) yielding [M - HX]+• at m/z {mz}."
            )
            stevenson = "Extrusion of neutral HX leaves an alkene-like radical cation."
        elif mech == "retro_diels_alder":
            rationale = (
                f"Retro-Diels-Alder (RDA) fragmentation at m/z {mz}. Concerted cleavage of two C-C single bonds "
                f"in a cyclohexene core yielding a conjugated diene radical cation and neutral alkene."
            )
            stevenson = "Diene radical cation retains charge due to conjugated π-system stabilization."
        elif mech == "secondary_cascade":
            rationale = (
                f"High-energy secondary fragmentation cascade at m/z {mz}. Further decomposition of primary "
                f"fragment ions via extrusion of stable neutral molecules (CO, C2H2, C2H4) or loss of H•."
            )
            stevenson = "Successive unimolecular decomposition driven by high internal vibrational energy at 70 eV."
        else:
            rationale = f"Homolytic cleavage producing fragment ion at m/z {mz}."
            stevenson = "Thermodynamically favored charge retention."

        return rationale, stevenson

    def diagnose_spectrum(
        self,
        predicted: MassSpectrum,
        experimental_peaks: List[Tuple[int, float]]
    ) -> List[DiagnosticDiscrepancy]:
        """
        Implements the 5-point Diagnostic Framework:
        1. Wrong Fragmentation Pathway (chemically invalid route)
        2. Wrong Charge / Fragment Assignment (Stevenson's rule failure)
        3. Missing Competing Pathway (parallel high-rate process omitted)
        4. Incorrect Energy Assumption (energy regime mismatch: 20 vs 70 eV)
        5. Incorrect Intensity / Kinetic Model (peak positions correct, ratios skewed)
        """
        diagnoses: List[DiagnosticDiscrepancy] = []
        exp_dict = {mz: intensity for mz, intensity in experimental_peaks}
        pred_dict = {p.mz: p.intensity for p in predicted.sorted_peaks}

        exp_mzs = set(exp_dict.keys())
        pred_mzs = set(pred_dict.keys())

        # 1. Missing Major Experimental Peaks (Missing Competing Pathway or Wrong Pathway)
        missing_in_pred = exp_mzs - pred_mzs
        for mz in missing_in_pred:
            exp_int = exp_dict[mz]
            if exp_int >= 20.0:
                # Check if it's the complementary Stevenson fragment
                comp_mz = predicted.molecule.nominal_mass - mz
                if comp_mz in pred_mzs:
                    diagnoses.append(DiagnosticDiscrepancy(
                        mz=mz,
                        error_type="Wrong Charge / Fragment Assignment (Stevenson's Rule Failure)",
                        severity="High",
                        explanation=(
                            f"Experimental base/major peak at m/z {mz} (intensity {exp_int}%) is missing in prediction, "
                            f"while complementary fragment at m/z {comp_mz} was predicted. The cleavage was identified, "
                            f"but charge was assigned to the wrong fragment partner."
                        ),
                        corrective_action="Recalculate ionization energies and carbocation stability rankings for this fragment pair."
                    ))
                elif mz == predicted.molecule.nominal_mass - 18 and predicted.molecule.num_oxygens > 0:
                    diagnoses.append(DiagnosticDiscrepancy(
                        mz=mz,
                        error_type="Missing Competing Pathway",
                        severity="High",
                        explanation=f"Missing dehydration peak [M - 18]+• at m/z {mz}. Aliphatic alcohol elimination pathway was omitted.",
                        corrective_action="Activate two-bond dehydration mechanism for oxygenated species."
                    ))
                else:
                    diagnoses.append(DiagnosticDiscrepancy(
                        mz=mz,
                        error_type="Missing Competing Pathway",
                        severity="Medium",
                        explanation=f"Major experimental peak at m/z {mz} (intensity {exp_int}%) was not generated by any active mechanism.",
                        corrective_action="Expand candidate generation engine with missing rearrangement or cascade pathway."
                    ))

        # 2. Spurious Predicted Peaks (Wrong Fragmentation Pathway)
        spurious_in_pred = pred_mzs - exp_mzs
        for mz in spurious_in_pred:
            pred_int = pred_dict[mz]
            if pred_int >= 25.0:
                diagnoses.append(DiagnosticDiscrepancy(
                    mz=mz,
                    error_type="Wrong Fragmentation Pathway",
                    severity="High",
                    explanation=(
                        f"Predicted strong peak at m/z {mz} (intensity {pred_int}%) does not exist in experimental spectrum. "
                        f"The proposed mechanism ({predicted.peaks[mz].primary_mechanism}) is chemically forbidden or has an insurmountable barrier."
                    ),
                    corrective_action="Constrain SMARTS topological prerequisites (e.g. ensure 6-membered cyclic geometry for McLafferty)."
                ))

        # 3. Energy Regime Anomalies
        # Molecular Ion discrepancy check
        mol_mz = predicted.molecular_ion_mz
        if mol_mz in exp_dict and mol_mz in pred_dict:
            exp_mol_int = exp_dict[mol_mz]
            pred_mol_int = pred_dict[mol_mz]
            if exp_mol_int > 50.0 and pred_mol_int < 10.0:
                diagnoses.append(DiagnosticDiscrepancy(
                    mz=mol_mz,
                    error_type="Incorrect Energy Assumption",
                    severity="Medium",
                    explanation=(
                        f"Experimental molecular ion M+• at m/z {mol_mz} is very intense ({exp_mol_int}%), "
                        f"whereas prediction at {predicted.beam_energy_ev} eV heavily dissociated it ({pred_mol_int}%). "
                        f"Experimental spectrum was likely acquired at low electron energy (15-20 eV)."
                    ),
                    corrective_action="Adjust beam energy parameter to low-eV regime (e.g. 20 eV) to match experimental conditions."
                ))

        # 4. Intensity / Kinetic Model Skew
        common_mzs = exp_mzs.intersection(pred_mzs)
        for mz in common_mzs:
            diff = abs(exp_dict[mz] - pred_dict[mz])
            if diff > 40.0:
                diagnoses.append(DiagnosticDiscrepancy(
                    mz=mz,
                    error_type="Incorrect Intensity / Kinetic Model",
                    severity="Low",
                    explanation=(
                        f"Peak m/z {mz} exists in both spectra, but relative intensity deviates significantly "
                        f"(Experimental: {exp_dict[mz]}%, Predicted: {pred_dict[mz]}%, Δ = {diff:.1f}%). "
                        f"Rate constant k(E) or transition state entropy is miscalibrated."
                    ),
                    corrective_action="Tune mechanism prefactor or activation threshold E0 in config.yaml."
                ))

        return diagnoses

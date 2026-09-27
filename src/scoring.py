"""
Stevenson's Rule & Energy Scoring Engine (Tasks C & D)
Implements physical chemistry scoring of candidate fragment ions:
- Stevenson's rule: charge retention directed by ionization potential and carbocation stability
- Radical leaving-group stability favoring loss of larger/more substituted neutral radicals
- Quasi-Equilibrium Theory (QET) kinetic activation as a function of electron beam energy (eV)
- Molecular ion survival dynamics across 20, 40, 70, and 100 eV
"""

from typing import Dict, Any, Optional
import math

from src.parser import MoleculeInfo
from src.fragmentation import FragmentCandidate


class ScoringEngine:
    """Calculates physical intensity scores for fragment candidates at specified beam energies."""

    DEFAULT_CARBOCATION_SCORES = {
        "oxonium": 5.2,
        "iminium": 5.5,
        "aroyl": 5.5,
        "acylium": 4.8,
        "tropylium": 4.6,
        "aromatic_cation": 4.2,
        "benzylic": 4.0,
        "allylic": 3.5,
        "tertiary": 3.0,
        "secondary": 2.0,
        "primary": 1.0,
        "methyl": 0.2,
        "radical_cation": 2.2,
        "aromatic_radical_cation": 4.8,
    }
    DEFAULT_RADICAL_SCORES = {
        "tertiary": 2.5,
        "secondary": 2.0,
        "primary": 1.2,
        "methyl": 0.6,
        "aryl": 0.3,
        "hydrogen": 0.1,
        "halogen_i": 3.0,
        "halogen_br": 2.2,
        "halogen_cl": 1.5,
    }
    DEFAULT_KINETICS = {
        "mclafferty": {"e0": 11.0, "sigma": 4.0, "prefactor": 3.6},
        "alpha_cleavage": {"e0": 13.0, "sigma": 5.0, "prefactor": 1.5},
        "inductive_cleavage": {"e0": 13.5, "sigma": 5.0, "prefactor": 1.8},
        "benzylic_tropylium": {"e0": 11.5, "sigma": 4.0, "prefactor": 3.2},
        "retro_diels_alder": {"e0": 12.5, "sigma": 4.5, "prefactor": 1.4},
        "dehydration_elimination": {"e0": 13.5, "sigma": 5.0, "prefactor": 1.3},
        "dehydrohalogenation": {"e0": 22.0, "sigma": 6.0, "prefactor": 0.12},
        "decarbonylation": {"e0": 18.0, "sigma": 6.0, "prefactor": 1.2},
        "alkane_cc_cleavage": {"e0": 14.5, "sigma": 5.5, "prefactor": 1.35},
        "secondary_cascade": {"e0": 26.0, "sigma": 7.0, "prefactor": 0.55},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.carbocation_scores = dict(self.DEFAULT_CARBOCATION_SCORES)
        self.radical_scores = dict(self.DEFAULT_RADICAL_SCORES)
        self.kinetics = dict(self.DEFAULT_KINETICS)

        if config:
            if "carbocation_stability" in config and config["carbocation_stability"]:
                self.carbocation_scores.update(config["carbocation_stability"])
            if "radical_stability" in config and config["radical_stability"]:
                self.radical_scores.update(config["radical_stability"])
            if "kinetics" in config and "mechanisms" in config["kinetics"]:
                self.kinetics.update(config["kinetics"]["mechanisms"])

    def score_candidate(self, candidate: FragmentCandidate, mol_info: MoleculeInfo, beam_energy_ev: float = 70.0) -> float:
        """
        Compute the raw physical intensity score for a candidate ion at the given beam energy.
        Score = ThermodynamicStability * KineticProbability(E) * StructuralPrefactor
        """
        # Special case: Molecular Radical Cation M+•
        if candidate.mechanism == "molecular_ion":
            return self._score_molecular_ion(mol_info, beam_energy_ev)

        # 1. Stevenson's Rule: Carbocation Stability
        c_score = self.carbocation_scores.get(candidate.carbocation_class, 1.5)

        # 2. Radical Leaving Group Stability
        r_score = 1.0
        if candidate.radical_class:
            r_score = self.radical_scores.get(candidate.radical_class, 1.0)

        # Thermodynamic driving force: exp( (C_stab + R_stab) / 2.0 )
        thermo_score = math.exp((c_score + 0.5 * r_score) / 2.5)

        # 3. Quasi-Equilibrium Kinetic Energy Function P(E)
        kinetic_params = self.kinetics.get(
            candidate.mechanism,
            {"e0": 15.0, "sigma": 5.0, "prefactor": 1.0}
        )
        e0 = kinetic_params["e0"]
        sigma = kinetic_params["sigma"]
        prefactor = kinetic_params.get("prefactor", kinetic_params.get("base_prefactor", 1.0))

        # If secondary/tertiary cascade, add higher threshold energy requirement
        if candidate.generation >= 2:
            e0 += 8.0 * (candidate.generation - 1)
            prefactor *= 0.85

        # Sigmoid kinetic activation: 1 / (1 + exp(-(E - E0) / sigma))
        delta_e = (beam_energy_ev - e0) / max(sigma, 0.1)
        if delta_e > 10:
            p_energy = 1.0
        elif delta_e < -10:
            p_energy = 0.0001
        else:
            p_energy = 1.0 / (1.0 + math.exp(-delta_e))

        # At very high energy (100 eV), secondary small fragments gain additional kinetic favorability
        if beam_energy_ev > 70.0:
            if candidate.nominal_mz < 50:
                p_energy *= 1.0 + 0.008 * (beam_energy_ev - 70.0)
            elif candidate.nominal_mz > 100:
                p_energy *= max(0.2, 1.0 - 0.005 * (beam_energy_ev - 70.0))

        raw_score = thermo_score * p_energy * prefactor
        return max(0.0, raw_score)

    def _score_molecular_ion(self, mol_info: MoleculeInfo, beam_energy_ev: float) -> float:
        """
        Molecular ion survival score depends strongly on aromaticity and beam energy.
        Aromatics (benzene, toluene): low IE (~8.8 eV), high stability -> survives well even at 70/100 eV.
        Alkanes / aliphatic alcohols: rapid unimolecular dissociation -> M+• is small at 70 eV, but prominent at 20 eV.
        """
        if mol_info.is_aromatic:
            if mol_info.canonical_smiles == "c1ccccc1" or (mol_info.num_carbons == 6 and mol_info.num_hydrogens == 6):
                # Benzene has exceptionally high molecular ion stability
                base_stability = 7.5
                survival = math.exp(-0.006 * max(0.0, beam_energy_ev - 10.0))
            else:
                # Substituted aromatics (toluene, etc.)
                base_stability = 3.8
                survival = math.exp(-0.016 * max(0.0, beam_energy_ev - 10.0))
        elif mol_info.has_conjugated_system:
            base_stability = 3.0
            survival = math.exp(-0.025 * max(0.0, beam_energy_ev - 10.0))
        elif mol_info.num_oxygens > 0 or mol_info.num_chlorines > 0 or mol_info.num_bromines > 0:
            # Heteroatom systems: moderate to weak parent ion
            base_stability = 1.6
            survival = math.exp(-0.045 * max(0.0, beam_energy_ev - 10.0))
        else:
            # Saturated hydrocarbons (e.g. butane, octane)
            base_stability = 1.8
            survival = math.exp(-0.040 * max(0.0, beam_energy_ev - 10.0))

        # Ionization efficiency threshold: IE ~ 9-11 eV
        delta = (beam_energy_ev - 9.5) / 2.5
        if delta < -10:
            ion_eff = 0.0
        elif delta > 10:
            ion_eff = 1.0
        else:
            ion_eff = 1.0 / (1.0 + math.exp(-delta))

        # At 20 eV, molecular ion relative prominence is dramatically higher
        low_energy_boost = 1.0
        if beam_energy_ev <= 30.0:
            low_energy_boost = 1.0 + 3.0 * (1.0 - (beam_energy_ev - 10.0) / 20.0)

        score = base_stability * survival * ion_eff * low_energy_boost
        return max(0.01, score)

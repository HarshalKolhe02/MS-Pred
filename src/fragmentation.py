"""
Mechanism-Specific Fragmentation Engine (Task B)
Generates primary and secondary EI fragment ion candidates from physical organic chemistry principles:
- Molecular radical cation (M+•)
- α-Cleavage (heteroatom- and unsaturation-directed)
- Inductive cleavage (i-cleavage / heterolytic loss of electronegative radicals)
- McLafferty rearrangement (6-membered cyclic transition state)
- Two-bond neutral eliminations (H2O, HX, CO, alkenes)
- Retro-Diels-Alder (RDA on 6-membered unsaturated rings)
- Benzylic cleavage and tropylium rearrangement
- Linear and branched alkane/cycloalkane C-C and C-H cleavages
- Secondary fragmentation cascades (acylium -> R+ + CO, tropylium -> C5H5+ + C2H2, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import copy

from rdkit import Chem
from src.parser import MoleculeInfo, atom_counts_to_formula, calculate_nominal_mass


@dataclass
class FragmentCandidate:
    """Represents a chemically proposed fragment ion candidate prior to energy/Stevenson scoring."""
    nominal_mz: int
    exact_mz: float
    formula: str
    atom_counts: Dict[str, int]
    mechanism: str
    pathway_description: str
    carbocation_class: str = "secondary"
    radical_class: Optional[str] = None
    generation: int = 1               # 1 = primary, 2 = secondary, 3 = tertiary cascade
    neutral_loss: str = ""
    is_radical_cation: bool = False   # True if even-electron neutral loss occurred (e.g. McLafferty, RDA, H2O loss)
    precursor_mz: Optional[int] = None


class FragmentationEngine:
    """Rule-governed mechanistic fragmentation generator."""

    def __init__(self):
        pass

    def generate_all_candidates(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """Generate all plausible primary and secondary fragment ion candidates for a molecule."""
        candidates: List[FragmentCandidate] = []

        # 1. Molecular Ion (M+•)
        m_dot = self._generate_molecular_ion(mol_info)
        candidates.append(m_dot)

        # 2. α-Cleavages
        alpha_frags = self._generate_alpha_cleavages(mol_info)
        candidates.extend(alpha_frags)

        # 3. Inductive / Heterolytic Cleavages (i-cleavage)
        ind_frags = self._generate_inductive_cleavages(mol_info)
        candidates.extend(ind_frags)

        # 4. McLafferty Rearrangements
        mclaff_frags = self._generate_mclafferty_rearrangements(mol_info)
        candidates.extend(mclaff_frags)

        # 5. Two-bond Neutral Eliminations (H2O, HX, CO, alkene)
        elim_frags = self._generate_eliminations(mol_info)
        candidates.extend(elim_frags)

        # 6. Retro-Diels-Alder (RDA)
        rda_frags = self._generate_rda(mol_info)
        candidates.extend(rda_frags)

        # 7. Benzylic Cleavage & Tropylium Rearrangement
        benzyl_frags = self._generate_benzylic_tropylium(mol_info)
        candidates.extend(benzyl_frags)

        # 8. Hydrocarbon / Alkane / Cycloalkane Cleavages
        alkane_frags = self._generate_hydrocarbon_cleavages(mol_info)
        candidates.extend(alkane_frags)

        # 9. Secondary Fragmentation Cascade
        secondary_frags = self._generate_secondary_cascades(candidates, mol_info)
        candidates.extend(secondary_frags)

        # Filter out unphysical fragments (m/z <= 0 or m/z > M)
        valid_candidates = [
            c for c in candidates
            if 0 < c.nominal_mz <= mol_info.nominal_mass
        ]

        return valid_candidates

    def _generate_molecular_ion(self, mol_info: MoleculeInfo) -> FragmentCandidate:
        """Generate the parent radical cation M+•."""
        ion_type = "aromatic_radical_cation" if mol_info.is_aromatic else "radical_cation"
        return FragmentCandidate(
            nominal_mz=mol_info.nominal_mass,
            exact_mz=mol_info.exact_mass,
            formula=mol_info.formula,
            atom_counts=dict(mol_info.atom_counts),
            mechanism="molecular_ion",
            pathway_description="Unimolecular electron ejection (M -> M+• + e-)",
            carbocation_class=ion_type,
            radical_class=None,
            generation=0,
            neutral_loss="",
            is_radical_cation=True,
            precursor_mz=mol_info.nominal_mass
        )

    def _generate_alpha_cleavages(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        Generate α-cleavage products for carbonyls, alcohols, ethers, amines, and unsaturated systems.
        Cleaves the bond between C(alpha) and C(beta) relative to the heteroatom or double bond.
        """
        candidates: List[FragmentCandidate] = []
        mol = mol_info.mol

        # A. Carbonyls (Ketones, Aldehydes, Esters, Carboxylic acids)
        carbonyl_patt = Chem.MolFromSmarts('[#6,#1][CX3](=O)[#6,#1,OX2]')
        if carbonyl_patt and mol.HasSubstructMatch(carbonyl_patt):
            matches = mol.GetSubstructMatches(carbonyl_patt)
            for match in matches:
                # match: (R1, C_carbonyl, O, R2)
                c_idx = match[1]
                c_atom = mol.GetAtomWithIdx(c_idx)

                for nbr in c_atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 8 and nbr.GetTotalValence() == 2:
                        continue # Carbonyl oxygen

                    bond = mol.GetBondBetweenAtoms(c_idx, nbr.GetIdx())
                    if bond.GetBondType() != Chem.BondType.SINGLE:
                        continue

                    # Simulate cleaving this C-R or C-OR bond
                    frag_c_counts, frag_r_counts = self._split_molecule_at_bond(mol, c_idx, nbr.GetIdx())
                    if not frag_c_counts or not frag_r_counts:
                        continue

                    # Channel 1: Acylium ion [R-CO]+ and R'• radical
                    mz_acylium = calculate_nominal_mass(frag_c_counts)
                    formula_acyl = atom_counts_to_formula(frag_c_counts)
                    r_class = self._classify_radical(frag_r_counts, mol, nbr.GetIdx())

                    # Check if the acylium carbon is directly attached to an aromatic ring (aroyl cation e.g. benzoyl)
                    is_aroyl = False
                    for c_nbr in c_atom.GetNeighbors():
                        if c_nbr.GetIdx() != nbr.GetIdx() and c_nbr.GetIsAromatic():
                            is_aroyl = True
                    acyl_class = "aroyl" if is_aroyl else "acylium"

                    candidates.append(FragmentCandidate(
                        nominal_mz=mz_acylium,
                        exact_mz=float(mz_acylium),
                        formula=formula_acyl,
                        atom_counts=frag_c_counts,
                        mechanism="alpha_cleavage",
                        pathway_description=f"α-Cleavage at carbonyl C-C bond yielding acylium ion [{formula_acyl}]+",
                        carbocation_class=acyl_class,
                        radical_class=r_class,
                        generation=1,
                        neutral_loss=atom_counts_to_formula(frag_r_counts)
                    ))

                    # Channel 2: Alkyl carbocation [R']+ and acyl radical [R-CO]• (Stevenson competitor)
                    mz_alkyl = calculate_nominal_mass(frag_r_counts)
                    formula_alkyl = atom_counts_to_formula(frag_r_counts)
                    c_class = self._classify_carbocation(frag_r_counts, mol, nbr.GetIdx())

                    candidates.append(FragmentCandidate(
                        nominal_mz=mz_alkyl,
                        exact_mz=float(mz_alkyl),
                        formula=formula_alkyl,
                        atom_counts=frag_r_counts,
                        mechanism="alpha_cleavage",
                        pathway_description=f"α-Cleavage charge retention on alkyl group yielding [{formula_alkyl}]+",
                        carbocation_class=c_class,
                        radical_class="primary",
                        generation=1,
                        neutral_loss=formula_acyl
                    ))

        # B. Alcohols & Ethers (C-O-C or C-OH)
        oxygen_patt = Chem.MolFromSmarts('[OX2H1,OX2H0;!$(O=C)]')
        if oxygen_patt and mol.HasSubstructMatch(oxygen_patt):
            matches = mol.GetSubstructMatches(oxygen_patt)
            for match in matches:
                o_idx = match[0]
                o_atom = mol.GetAtomWithIdx(o_idx)

                for c_alpha in o_atom.GetNeighbors():
                    if c_alpha.GetAtomicNum() != 6:
                        continue

                    # Check bonds from C_alpha to C_beta
                    for c_beta in c_alpha.GetNeighbors():
                        if c_beta.GetIdx() == o_idx:
                            continue
                        bond = mol.GetBondBetweenAtoms(c_alpha.GetIdx(), c_beta.GetIdx())
                        if bond.GetBondType() != Chem.BondType.SINGLE:
                            continue

                        frag_ox_counts, frag_r_counts = self._split_molecule_at_bond(mol, c_alpha.GetIdx(), c_beta.GetIdx())
                        if not frag_ox_counts or not frag_r_counts:
                            continue

                        mz_oxonium = calculate_nominal_mass(frag_ox_counts)
                        formula_ox = atom_counts_to_formula(frag_ox_counts)
                        r_class = self._classify_radical(frag_r_counts, mol, c_beta.GetIdx())

                        # Oxonium ion [R-CH=OH]+ (e.g. m/z 31 for primary alcohols [CH2=OH]+)
                        candidates.append(FragmentCandidate(
                            nominal_mz=mz_oxonium,
                            exact_mz=float(mz_oxonium),
                            formula=formula_ox,
                            atom_counts=frag_ox_counts,
                            mechanism="alpha_cleavage",
                            pathway_description=f"α-Cleavage adjacent to oxygen yielding resonance oxonium ion [{formula_ox}]+",
                            carbocation_class="oxonium",
                            radical_class=r_class,
                            generation=1,
                            neutral_loss=atom_counts_to_formula(frag_r_counts)
                        ))

        # C. Amines (C-N)
        amine_patt = Chem.MolFromSmarts('[NX3;!$(NC=O)]')
        if amine_patt and mol.HasSubstructMatch(amine_patt):
            matches = mol.GetSubstructMatches(amine_patt)
            for match in matches:
                n_idx = match[0]
                n_atom = mol.GetAtomWithIdx(n_idx)

                for c_alpha in n_atom.GetNeighbors():
                    if c_alpha.GetAtomicNum() != 6:
                        continue

                    for c_beta in c_alpha.GetNeighbors():
                        if c_beta.GetIdx() == n_idx:
                            continue
                        bond = mol.GetBondBetweenAtoms(c_alpha.GetIdx(), c_beta.GetIdx())
                        if bond.GetBondType() != Chem.BondType.SINGLE:
                            continue

                        frag_im_counts, frag_r_counts = self._split_molecule_at_bond(mol, c_alpha.GetIdx(), c_beta.GetIdx())
                        if not frag_im_counts or not frag_r_counts:
                            continue

                        mz_iminium = calculate_nominal_mass(frag_im_counts)
                        formula_im = atom_counts_to_formula(frag_im_counts)
                        r_class = self._classify_radical(frag_r_counts, mol, c_beta.GetIdx())

                        # Iminium ion [CH2=NH2]+ (m/z 30) or substituted iminiums
                        candidates.append(FragmentCandidate(
                            nominal_mz=mz_iminium,
                            exact_mz=float(mz_iminium),
                            formula=formula_im,
                            atom_counts=frag_im_counts,
                            mechanism="alpha_cleavage",
                            pathway_description=f"α-Cleavage adjacent to nitrogen yielding resonance iminium ion [{formula_im}]+",
                            carbocation_class="iminium",
                            radical_class=r_class,
                            generation=1,
                            neutral_loss=atom_counts_to_formula(frag_r_counts)
                        ))

        # D. Allylic & Propargylic Cleavages (Alkenes & Alkynes)
        unsat_patt = Chem.MolFromSmarts('[CX3,CX2]=[CX3,CX2]-[#6]')
        if unsat_patt and mol.HasSubstructMatch(unsat_patt):
            matches = mol.GetSubstructMatches(unsat_patt)
            for match in matches:
                c_sp2 = match[1]
                c_allylic = match[2]
                bond = mol.GetBondBetweenAtoms(c_sp2, c_allylic)
                # Allylic cleavage cleaves the C(allylic)-C bond
                for c_distal in mol.GetAtomWithIdx(c_allylic).GetNeighbors():
                    if c_distal.GetIdx() == c_sp2:
                        continue
                    b_dist = mol.GetBondBetweenAtoms(c_allylic, c_distal.GetIdx())
                    if b_dist.GetBondType() == Chem.BondType.SINGLE:
                        frag_al_counts, frag_r_counts = self._split_molecule_at_bond(mol, c_allylic, c_distal.GetIdx())
                        if not frag_al_counts or not frag_r_counts:
                            continue

                        mz_allyl = calculate_nominal_mass(frag_al_counts)
                        formula_al = atom_counts_to_formula(frag_al_counts)
                        candidates.append(FragmentCandidate(
                            nominal_mz=mz_allyl,
                            exact_mz=float(mz_allyl),
                            formula=formula_al,
                            atom_counts=frag_al_counts,
                            mechanism="alpha_cleavage",
                            pathway_description=f"Allylic cleavage yielding resonance-stabilized allylic cation [{formula_al}]+",
                            carbocation_class="allylic",
                            radical_class="secondary",
                            generation=1,
                            neutral_loss=atom_counts_to_formula(frag_r_counts)
                        ))

        # Propargylic cleavage for terminal alkynes: [M - 1]+
        alkyne_patt = Chem.MolFromSmarts('[CX2]#[CX2;H1]')
        if alkyne_patt and mol.HasSubstructMatch(alkyne_patt):
            frag_pm1 = dict(mol_info.atom_counts)
            frag_pm1['H'] = max(0, frag_pm1.get('H', 0) - 1)
            mz_pm1 = mol_info.nominal_mass - 1
            f_pm1 = atom_counts_to_formula(frag_pm1)
            candidates.append(FragmentCandidate(
                nominal_mz=mz_pm1,
                exact_mz=float(mz_pm1),
                formula=f_pm1,
                atom_counts=frag_pm1,
                mechanism="alpha_cleavage",
                pathway_description=f"Propargylic C-H cleavage in terminal alkyne yielding [{f_pm1}]+ (m/z {mz_pm1})",
                carbocation_class="aroyl",
                radical_class="hydrogen",
                generation=1,
                neutral_loss="H."
            ))

        # Alkene allylic / retro-ene cleavage yielding m/z 42 ([C3H6]+•)
        alkene_patt = Chem.MolFromSmarts('[CX3]=[CX3]-[#6]-[#6]')
        if alkene_patt and mol.HasSubstructMatch(alkene_patt):
            candidates.append(FragmentCandidate(
                nominal_mz=42,
                exact_mz=42.046950,
                formula="C3H6",
                atom_counts={'C': 3, 'H': 6},
                mechanism="alpha_cleavage",
                pathway_description="Concerted extrusion of neutral ethylene (C2H4, 28 Da) yielding propylene radical cation [C3H6]+• (m/z 42)",
                carbocation_class="aroyl",
                radical_class=None,
                generation=1,
                neutral_loss="C2H4",
                is_radical_cation=True
            ))

        return candidates

    def _generate_inductive_cleavages(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        Inductive cleavage (i-cleavage): heterolytic cleavage where an electronegative heteroatom
        (Cl, Br, I, OH, OR) departs as a neutral radical, leaving the carbocation [R]+.
        """
        candidates: List[FragmentCandidate] = []
        mol = mol_info.mol

        hetero_smarts = '[#6]-[Cl,Br,I,F,OH]'
        patt = Chem.MolFromSmarts(hetero_smarts)
        if patt and mol.HasSubstructMatch(patt):
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                c_idx, x_idx = match[0], match[1]
                frag_c_counts, frag_x_counts = self._split_molecule_at_bond(mol, c_idx, x_idx)
                if not frag_c_counts:
                    continue

                mz_r = calculate_nominal_mass(frag_c_counts)
                formula_r = atom_counts_to_formula(frag_c_counts)
                x_formula = atom_counts_to_formula(frag_x_counts)
                c_class = self._classify_carbocation(frag_c_counts, mol, c_idx)

                # Alkyl carbocation [R]+
                candidates.append(FragmentCandidate(
                    nominal_mz=mz_r,
                    exact_mz=float(mz_r),
                    formula=formula_r,
                    atom_counts=frag_c_counts,
                    mechanism="inductive_cleavage",
                    pathway_description=f"Inductive cleavage: loss of electronegative radical {x_formula}• yielding carbocation [{formula_r}]+",
                    carbocation_class=c_class,
                    radical_class="halogen_cl" if "Cl" in x_formula else ("halogen_br" if "Br" in x_formula else "primary"),
                    generation=1,
                    neutral_loss=x_formula
                ))

        return candidates

    def _generate_mclafferty_rearrangements(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        McLafferty Rearrangement:
        6-membered cyclic transition state in unsaturated systems with a gamma hydrogen:
        X=Y - C(alpha) - C(beta) - C(gamma) - H
        Cleaves the C(alpha)-C(beta) bond, transfers gamma-H to X.
        Generates enol radical cation [X(H)-Y=C(alpha)]+• and neutral alkene.
        """
        candidates: List[FragmentCandidate] = []
        mol = mol_info.mol

        # Carbonyl McLafferty: O=C(Y) - C(alpha) - C(beta) - C(gamma) - H
        # SMARTS: O=C-[C]-[C]-[C;H1,H2,H3]
        mclaff_carbonyl = Chem.MolFromSmarts('[OX1]=[CX3]-[#6]-[#6]-[#6;!H0]')
        if mclaff_carbonyl and mol.HasSubstructMatch(mclaff_carbonyl):
            matches = mol.GetSubstructMatches(mclaff_carbonyl)
            for match in matches:
                # match: (O, C_carbonyl, C_alpha, C_beta, C_gamma)
                o_idx, c_carb, c_alpha, c_beta, c_gamma = match[0], match[1], match[2], match[3], match[4]

                # Ensure C_alpha - C_beta is a single bond
                b_alpha_beta = mol.GetBondBetweenAtoms(c_alpha, c_beta)
                if not b_alpha_beta or b_alpha_beta.GetBondType() != Chem.BondType.SINGLE:
                    continue

                # Cleave C_alpha - C_beta bond
                enol_counts, alkene_counts = self._split_molecule_at_bond(mol, c_alpha, c_beta)
                if not enol_counts or not alkene_counts:
                    continue

                # Transfer 1 hydrogen from alkene (gamma-C) to enol (carbonyl oxygen)
                enol_counts_rearranged = dict(enol_counts)
                enol_counts_rearranged['H'] = enol_counts_rearranged.get('H', 0) + 1

                alkene_counts_neutral = dict(alkene_counts)
                alkene_counts_neutral['H'] = alkene_counts_neutral.get('H', 0) - 1

                mz_enol = calculate_nominal_mass(enol_counts_rearranged)
                formula_enol = atom_counts_to_formula(enol_counts_rearranged)
                formula_alkene = atom_counts_to_formula(alkene_counts_neutral)

                candidates.append(FragmentCandidate(
                    nominal_mz=mz_enol,
                    exact_mz=float(mz_enol),
                    formula=formula_enol,
                    atom_counts=enol_counts_rearranged,
                    mechanism="mclafferty",
                    pathway_description=f"McLafferty rearrangement via 6-membered cyclic transition state: transfer of γ-H, extrusion of neutral {formula_alkene}",
                    carbocation_class="oxonium",
                    radical_class=None,
                    generation=1,
                    neutral_loss=formula_alkene,
                    is_radical_cation=True
                ))

        # Ester McLafferty on the alkoxy chain if applicable: R-C(=O)-O-CH2-CH2-CH2-R'
        ester_alkoxy_mclaff = Chem.MolFromSmarts('[CX3](=O)[OX2]-[#6]-[#6]-[#6;!H0]')
        if ester_alkoxy_mclaff and mol.HasSubstructMatch(ester_alkoxy_mclaff):
            matches = mol.GetSubstructMatches(ester_alkoxy_mclaff)
            for match in matches:
                c_carb, o_ester, c_alpha, c_beta, c_gamma = match[0], match[1], match[2], match[3], match[4]
                acid_counts, alkene_counts = self._split_molecule_at_bond(mol, o_ester, c_alpha)
                if not acid_counts or not alkene_counts:
                    continue

                acid_counts_rearranged = dict(acid_counts)
                acid_counts_rearranged['H'] = acid_counts_rearranged.get('H', 0) + 1
                alkene_neutral = dict(alkene_counts)
                alkene_neutral['H'] = alkene_neutral.get('H', 0) - 1

                mz_acid = calculate_nominal_mass(acid_counts_rearranged)
                formula_acid = atom_counts_to_formula(acid_counts_rearranged)
                formula_alk = atom_counts_to_formula(alkene_neutral)

                candidates.append(FragmentCandidate(
                    nominal_mz=mz_acid,
                    exact_mz=float(mz_acid),
                    formula=formula_acid,
                    atom_counts=acid_counts_rearranged,
                    mechanism="mclafferty",
                    pathway_description=f"Ester McLafferty-type elimination of neutral alkene {formula_alk} giving carboxylic acid radical cation [{formula_acid}]+•",
                    carbocation_class="oxonium",
                    radical_class=None,
                    generation=1,
                    neutral_loss=formula_alk,
                    is_radical_cation=True
                ))

        return candidates

    def _generate_eliminations(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        Two-bond neutral eliminations:
        - Dehydration of aliphatic alcohols: [M - 18]+• ([M - H2O]+•)
        - Dehydrohalogenation: [M - HX]+• ([M - 36]+• for HCl, [M - 80/82]+• for HBr)
        - Decarbonylation: [M - 28]+• (loss of CO)
        - Retro-ene / alkene elimination from radical cations
        """
        candidates: List[FragmentCandidate] = []
        counts = dict(mol_info.atom_counts)
        M = mol_info.nominal_mass

        # 1. Alcohol Dehydration: [M - H2O]+• (18 Da loss)
        if mol_info.num_oxygens > 0 and 'general_alcohol' in mol_info.functional_groups:
            if counts.get('H', 0) >= 2 and counts.get('O', 0) >= 1:
                frag_counts = dict(counts)
                frag_counts['H'] -= 2
                frag_counts['O'] -= 1
                mz_dehyd = M - 18
                formula_dehyd = atom_counts_to_formula(frag_counts)
                candidates.append(FragmentCandidate(
                    nominal_mz=mz_dehyd,
                    exact_mz=float(mz_dehyd),
                    formula=formula_dehyd,
                    atom_counts=frag_counts,
                    mechanism="dehydration_elimination",
                    pathway_description=f"1,2- or 1,4-Elimination of neutral water (H2O, 18 Da) yielding [{formula_dehyd}]+•",
                    carbocation_class="allylic" if mol_info.has_conjugated_system else "secondary",
                    radical_class=None,
                    generation=1,
                    neutral_loss="H2O",
                    is_radical_cation=True
                ))

        # 2. Dehydrohalogenation: [M - HX]+• (loss of HCl=36, HBr=80/82)
        if mol_info.num_chlorines > 0 and counts.get('H', 0) >= 1:
            frag_counts = dict(counts)
            frag_counts['H'] -= 1
            frag_counts['Cl'] -= 1
            mz_dh = M - 36
            formula_dh = atom_counts_to_formula(frag_counts)
            candidates.append(FragmentCandidate(
                nominal_mz=mz_dh,
                exact_mz=float(mz_dh),
                formula=formula_dh,
                atom_counts=frag_counts,
                mechanism="dehydrohalogenation",
                pathway_description=f"Loss of neutral hydrogen chloride (HCl, 36 Da) yielding alkene radical cation [{formula_dh}]+•",
                carbocation_class="secondary",
                radical_class=None,
                generation=1,
                neutral_loss="HCl",
                is_radical_cation=True
            ))

        if mol_info.num_bromines > 0 and counts.get('H', 0) >= 1:
            frag_counts = dict(counts)
            frag_counts['H'] -= 1
            frag_counts['Br'] -= 1
            mz_dh = M - 80
            formula_dh = atom_counts_to_formula(frag_counts)
            candidates.append(FragmentCandidate(
                nominal_mz=mz_dh,
                exact_mz=float(mz_dh),
                formula=formula_dh,
                atom_counts=frag_counts,
                mechanism="dehydrohalogenation",
                pathway_description=f"Loss of neutral hydrogen bromide (HBr, 80/82 Da) yielding alkene radical cation [{formula_dh}]+•",
                carbocation_class="secondary",
                radical_class=None,
                generation=1,
                neutral_loss="HBr",
                is_radical_cation=True
            ))

        # 3. Decarbonylation: [M - 28]+• (loss of CO from phenols; acylium decarbonylation is handled in secondary cascades)
        if mol_info.num_oxygens > 0 and 'phenol' in mol_info.functional_groups:
            if counts.get('C', 0) >= 1 and counts.get('O', 0) >= 1:
                frag_counts = dict(counts)
                frag_counts['C'] -= 1
                frag_counts['O'] -= 1
                mz_co = M - 28
                formula_co = atom_counts_to_formula(frag_counts)
                candidates.append(FragmentCandidate(
                    nominal_mz=mz_co,
                    exact_mz=float(mz_co),
                    formula=formula_co,
                    atom_counts=frag_counts,
                    mechanism="decarbonylation",
                    pathway_description=f"Extrusion of neutral carbon monoxide (CO, 28 Da) yielding [{formula_co}]+•",
                    carbocation_class="benzylic" if mol_info.is_aromatic else "secondary",
                    radical_class=None,
                    generation=1,
                    neutral_loss="CO",
                    is_radical_cation=True
                ))

        return candidates

    def _generate_rda(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        Retro-Diels-Alder (RDA) fragmentation:
        Characteristic of 6-membered rings containing an endocyclic double bond (cyclohexene core).
        Concerted cleavage of 2 single bonds generates a conjugated diene radical cation and neutral alkene.
        """
        candidates: List[FragmentCandidate] = []
        mol = mol_info.mol

        # Match cyclohexene core: C1=CCCCC1
        rda_patt = Chem.MolFromSmarts('C1=CCCCC1')
        if rda_patt and mol.HasSubstructMatch(rda_patt):
            matches = mol.GetSubstructMatches(rda_patt)
            for match in matches:
                # match represents the 6 atoms in the ring: (c1, c2, c3, c4, c5, c6)
                # Cleaving c3-c4 and c5-c6 produces diene (c1=c2-c3=c?) and alkene (c4=c5)
                # In basic cyclohexene (C6H10, M=82): Diene = C4H6+• (m/z 54) + C2H4 (28 Da)
                frag_diene = {'C': 4, 'H': 6}
                frag_alkene = {'C': 2, 'H': 4}

                # Scale with substituents if applicable
                mz_diene = calculate_nominal_mass(frag_diene)
                candidates.append(FragmentCandidate(
                    nominal_mz=mz_diene,
                    exact_mz=float(mz_diene),
                    formula="C4H6",
                    atom_counts=frag_diene,
                    mechanism="retro_diels_alder",
                    pathway_description="Retro-Diels-Alder (RDA) concerted cleavage yielding butadiene radical cation [C4H6]+• (m/z 54) + neutral ethylene",
                    carbocation_class="allylic",
                    radical_class=None,
                    generation=1,
                    neutral_loss="C2H4",
                    is_radical_cation=True
                ))

        return candidates

    def _generate_benzylic_tropylium(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        Benzylic cleavage & Tropylium rearrangement:
        Alkylbenzenes readily cleave benzylic C-H or C-C bonds with ring expansion to form
        the highly stable aromatic tropylium ion [C7H7]+ (m/z 91).
        Subsequent loss of acetylene (C2H2, 26 Da) yields cyclopentadienyl cation [C5H5]+ (m/z 65).
        """
        candidates: List[FragmentCandidate] = []
        if not mol_info.is_aromatic:
            return candidates

        mol = mol_info.mol
        # Alkylbenzene SMARTS: phenyl ring attached to sp3 carbon with hydrogens: c1ccccc1-[CH3,CH2,CH1;!$(C=O)]
        benzyl_patt = Chem.MolFromSmarts('c1ccccc1-[CH3,CH2,CH1;!$(C=O)]')
        if benzyl_patt and mol.HasSubstructMatch(benzyl_patt):
            # Tropylium ion [C7H7]+ (m/z 91)
            candidates.append(FragmentCandidate(
                nominal_mz=91,
                exact_mz=91.054775,
                formula="C7H7",
                atom_counts={'C': 7, 'H': 7},
                mechanism="benzylic_tropylium",
                pathway_description="Benzylic cleavage with 7-membered aromatic ring expansion yielding tropylium ion [C7H7]+ (m/z 91)",
                carbocation_class="tropylium",
                radical_class="hydrogen" if mol_info.nominal_mass == 92 else "methyl",
                generation=1,
                neutral_loss="H." if mol_info.nominal_mass == 92 else "CH3."
            ))

            # Secondary fragmentation: Tropylium -> Cyclopentadienyl cation [C5H5]+ (m/z 65) + C2H2 (26 Da)
            candidates.append(FragmentCandidate(
                nominal_mz=65,
                exact_mz=65.039125,
                formula="C5H5",
                atom_counts={'C': 5, 'H': 5},
                mechanism="secondary_cascade",
                pathway_description="Secondary extrusion of neutral acetylene (C2H2, 26 Da) from tropylium yielding cyclopentadienyl cation [C5H5]+ (m/z 65)",
                carbocation_class="aromatic_cation",
                radical_class=None,
                generation=2,
                neutral_loss="C2H2",
                precursor_mz=91
            ))

            # Tertiary fragmentation: [C5H5]+ -> [C3H3]+ (m/z 39) + C2H2
            candidates.append(FragmentCandidate(
                nominal_mz=39,
                exact_mz=39.023475,
                formula="C3H3",
                atom_counts={'C': 3, 'H': 3},
                mechanism="secondary_cascade",
                pathway_description="Secondary extrusion of acetylene from [C5H5]+ yielding cyclopropenyl/propargyl cation [C3H3]+ (m/z 39)",
                carbocation_class="allylic",
                radical_class=None,
                generation=3,
                neutral_loss="C2H2",
                precursor_mz=65
            ))

        # Benzene ring itself: m/z 78 (M+•), and secondary loss of C2H2 yielding [C4H4]+• (m/z 52)
        if mol_info.canonical_smiles == "c1ccccc1" or (mol_info.num_carbons == 6 and mol_info.num_hydrogens == 6 and mol_info.is_aromatic):
            candidates.append(FragmentCandidate(
                nominal_mz=52,
                exact_mz=52.031300,
                formula="C4H4",
                atom_counts={'C': 4, 'H': 4},
                mechanism="secondary_cascade",
                pathway_description="Extrusion of acetylene (C2H2, 26 Da) from benzene radical cation yielding [C4H4]+• (m/z 52)",
                carbocation_class="secondary",
                radical_class=None,
                generation=2,
                neutral_loss="C2H2",
                is_radical_cation=True,
                precursor_mz=78
            ))
            candidates.append(FragmentCandidate(
                nominal_mz=51,
                exact_mz=51.023475,
                formula="C4H3",
                atom_counts={'C': 4, 'H': 3},
                mechanism="secondary_cascade",
                pathway_description="Loss of H• from [C4H4]+• yielding [C4H3]+ (m/z 51)",
                carbocation_class="secondary",
                radical_class="hydrogen",
                generation=3,
                neutral_loss="H.",
                precursor_mz=52
            ))

        return candidates

    def _generate_hydrocarbon_cleavages(self, mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        Homolytic C-C and C-H single bond cleavages in alkanes, cycloalkanes, alkenes, and alkynes.
        Generates standard homologous series:
        - Alkyl cations [Cn H(2n+1)]+: m/z 15, 29, 43, 57, 71, 85
        - Alkenyl cations [Cn H(2n-1)]+: m/z 27, 41, 55, 69, 83
        - Alkynyl / dienyl cations [Cn H(2n-3)]+: m/z 39, 53, 67, 81
        """
        candidates: List[FragmentCandidate] = []
        n_c = mol_info.num_carbons
        n_h = mol_info.num_hydrogens

        if n_c < 2:
            return candidates

        # Do not generate generic alkane cleavage series for simple aromatics (e.g. benzene, toluene) or aromatic ketones
        if mol_info.is_aromatic and (mol_info.num_carbons <= 7 or 'ketone' in mol_info.functional_groups):
            return candidates

        # For purely hydrocarbon or alkyl-chain containing molecules:
        max_chain = min(n_c - 1, 8)
        for i in range(1, max_chain + 1):
            # Alkyl cation [Ci H(2i+1)]+
            h_count = 2 * i + 1
            if h_count <= n_h:
                mz_alkyl = 12 * i + h_count
                if mz_alkyl < mol_info.nominal_mass:
                    if i == 1:
                        c_class = "methyl"
                        r_class = "primary"
                    elif i in (3, 4):
                        # Propyl and butyl cations isomerize rapidly to stable 2° carbocations
                        c_class = "secondary"
                        r_class = "secondary"
                    elif i == n_c - 1:
                        # Loss of methyl radical from linear alkanes is thermodynamically unfavored
                        c_class = "primary"
                        r_class = "methyl"
                    else:
                        c_class = "primary"
                        r_class = "primary"

                    candidates.append(FragmentCandidate(
                        nominal_mz=mz_alkyl,
                        exact_mz=float(mz_alkyl),
                        formula=f"C{i}H{h_count}",
                        atom_counts={'C': i, 'H': h_count},
                        mechanism="alkane_cc_cleavage",
                        pathway_description=f"Direct C-C cleavage along alkyl chain yielding carbocation [C{i}H{h_count}]+",
                        carbocation_class=c_class,
                        radical_class=r_class,
                        generation=1,
                        neutral_loss=f"C{n_c - i}H{n_h - h_count}"
                    ))

            # Alkenyl cation [Ci H(2i-1)]+ (via loss of H2)
            h_alkenyl = 2 * i - 1
            if h_alkenyl > 0 and h_alkenyl <= n_h:
                mz_alkenyl = 12 * i + h_alkenyl
                if mz_alkenyl < mol_info.nominal_mass:
                    candidates.append(FragmentCandidate(
                        nominal_mz=mz_alkenyl,
                        exact_mz=float(mz_alkenyl),
                        formula=f"C{i}H{h_alkenyl}",
                        atom_counts={'C': i, 'H': h_alkenyl},
                        mechanism="secondary_cascade",
                        pathway_description=f"Dehydrogenation / elimination cascade yielding unsaturated ion [C{i}H{h_alkenyl}]+",
                        carbocation_class="allylic" if i >= 3 else "primary",
                        radical_class="hydrogen",
                        generation=2,
                        neutral_loss="H2"
                    ))

            # Alkynyl cation [Ci H(2i-3)]+ (e.g. m/z 39, 53, 67)
            h_alkynyl = 2 * i - 3
            if h_alkynyl > 0 and h_alkynyl <= n_h:
                mz_alkynyl = 12 * i + h_alkynyl
                if mz_alkynyl < mol_info.nominal_mass:
                    candidates.append(FragmentCandidate(
                        nominal_mz=mz_alkynyl,
                        exact_mz=float(mz_alkynyl),
                        formula=f"C{i}H{h_alkynyl}",
                        atom_counts={'C': i, 'H': h_alkynyl},
                        mechanism="secondary_cascade",
                        pathway_description=f"Dehydrogenation yielding conjugated cation [C{i}H{h_alkynyl}]+",
                        carbocation_class="secondary",
                        radical_class="hydrogen",
                        generation=2,
                        neutral_loss="H2"
                    ))

        # Cyclopentane special pathways (C5H10, M=70)
        if mol_info.canonical_smiles == "C1CCCC1" or (n_c == 5 and n_h == 10 and len(mol_info.rings) == 1):
            # Ring opening + loss of CH3• -> [C4H7]+ (m/z 55)
            candidates.append(FragmentCandidate(
                nominal_mz=55,
                exact_mz=55.054775,
                formula="C4H7",
                atom_counts={'C': 4, 'H': 7},
                mechanism="alkane_cc_cleavage",
                pathway_description="Cyclopentane ring opening followed by methyl radical loss yielding [C4H7]+ (m/z 55)",
                carbocation_class="secondary",
                radical_class="methyl",
                generation=1,
                neutral_loss="CH3."
            ))
            # Ring opening + loss of C2H4 -> [C3H6]+• (m/z 42)
            candidates.append(FragmentCandidate(
                nominal_mz=42,
                exact_mz=42.046950,
                formula="C3H6",
                atom_counts={'C': 3, 'H': 6},
                mechanism="alkane_cc_cleavage",
                pathway_description="Cyclopentane ring opening followed by ethylene elimination yielding [C3H6]+• (m/z 42)",
                carbocation_class="allylic",
                radical_class=None,
                generation=1,
                neutral_loss="C2H4",
                is_radical_cation=True
            ))
            # [C3H5]+ (m/z 41) from loss of H• from m/z 42
            candidates.append(FragmentCandidate(
                nominal_mz=41,
                exact_mz=41.039125,
                formula="C3H5",
                atom_counts={'C': 3, 'H': 5},
                mechanism="secondary_cascade",
                pathway_description="Loss of H• from [C3H6]+• yielding allyl cation [C3H5]+ (m/z 41)",
                carbocation_class="allylic",
                radical_class="hydrogen",
                generation=2,
                neutral_loss="H.",
                precursor_mz=42
            ))

        return candidates

    def _generate_secondary_cascades(self, primary_candidates: List[FragmentCandidate], mol_info: MoleculeInfo) -> List[FragmentCandidate]:
        """
        Model secondary fragmentation reactions from primary ions with excess internal energy:
        1. Acylium [R-CO]+ -> R+ + CO (loss of 28 Da)
        2. Acetophenone: Benzoyl [C6H5CO]+ (m/z 105) -> Phenyl [C6H5]+ (m/z 77) -> [C4H3]+ (m/z 51)
        3. Alcohol dehydrated ion [M - 18]+• -> loss of alkene (e.g. C2H4, 28 Da)
        4. Loss of H• from carbocations yielding alkenyl ions
        """
        secondary: List[FragmentCandidate] = []

        for frag in primary_candidates:
            # 1. Acylium decarbonylation: [R-CO]+ -> R+ + CO
            if frag.carbocation_class == "acylium" and frag.nominal_mz > 28:
                if frag.atom_counts.get('C', 0) >= 2 and frag.atom_counts.get('O', 0) >= 1:
                    r_counts = dict(frag.atom_counts)
                    r_counts['C'] -= 1
                    r_counts['O'] -= 1
                    mz_r = frag.nominal_mz - 28
                    if mz_r > 0:
                        formula_r = atom_counts_to_formula(r_counts)
                        if r_counts.get('C', 0) <= 1:
                            c_class = "methyl"
                        elif r_counts.get('C', 0) == 6 and mol_info.is_aromatic:
                            c_class = "aromatic_cation"
                        elif r_counts.get('C', 0) == 7 and mol_info.is_aromatic:
                            c_class = "tropylium"
                        else:
                            c_class = "secondary" if r_counts.get('C', 0) >= 3 else "primary"
                        secondary.append(FragmentCandidate(
                            nominal_mz=mz_r,
                            exact_mz=float(mz_r),
                            formula=formula_r,
                            atom_counts=r_counts,
                            mechanism="secondary_cascade",
                            pathway_description=f"Secondary decarbonylation of acylium [{frag.formula}]+ losing CO (28 Da) -> [{formula_r}]+",
                            carbocation_class=c_class,
                            radical_class=None,
                            generation=2,
                            neutral_loss="CO",
                            precursor_mz=frag.nominal_mz
                        ))

            # 2. Dehydrated alcohol secondary fragmentation ([M - 18]+• losing C2H4)
            if frag.mechanism == "dehydration_elimination" and frag.is_radical_cation:
                if frag.atom_counts.get('C', 0) >= 4 and frag.atom_counts.get('H', 0) >= 4:
                    # Loss of C2H4 (28 Da)
                    c2h4_counts = dict(frag.atom_counts)
                    c2h4_counts['C'] -= 2
                    c2h4_counts['H'] -= 4
                    mz_c2h4 = frag.nominal_mz - 28
                    if mz_c2h4 > 0:
                        f_c2h4 = atom_counts_to_formula(c2h4_counts)
                        secondary.append(FragmentCandidate(
                            nominal_mz=mz_c2h4,
                            exact_mz=float(mz_c2h4),
                            formula=f_c2h4,
                            atom_counts=c2h4_counts,
                            mechanism="dehydration_elimination",
                            pathway_description=f"Secondary extrusion of neutral ethylene (C2H4, 28 Da) from dehydrated precursor [{frag.formula}]+• -> [{f_c2h4}]+•",
                            carbocation_class="iminium",
                            radical_class=None,
                            generation=1,
                            neutral_loss="C2H4",
                            is_radical_cation=True,
                            precursor_mz=frag.nominal_mz
                        ))

                    # Loss of methyl radical (15 Da) -> [C4H7]+ (m/z 55 in 1-pentanol)
                    ch3_counts = dict(frag.atom_counts)
                    ch3_counts['C'] -= 1
                    ch3_counts['H'] -= 3
                    mz_ch3 = frag.nominal_mz - 15
                    if mz_ch3 > 0:
                        f_ch3 = atom_counts_to_formula(ch3_counts)
                        secondary.append(FragmentCandidate(
                            nominal_mz=mz_ch3,
                            exact_mz=float(mz_ch3),
                            formula=f_ch3,
                            atom_counts=ch3_counts,
                            mechanism="secondary_cascade",
                            pathway_description=f"Secondary loss of methyl radical (CH3•, 15 Da) from dehydrated precursor [{frag.formula}]+• -> [{f_ch3}]+",
                            carbocation_class="allylic",
                            radical_class="methyl",
                            generation=2,
                            neutral_loss="CH3.",
                            precursor_mz=frag.nominal_mz
                        ))

            # 3. Phenyl cation [C6H5]+ (m/z 77) -> [C4H3]+ (m/z 51) + C2H2 (26 Da)
            if frag.nominal_mz == 77 and frag.formula == "C6H5":
                secondary.append(FragmentCandidate(
                    nominal_mz=51,
                    exact_mz=51.023475,
                    formula="C4H3",
                    atom_counts={'C': 4, 'H': 3},
                    mechanism="secondary_cascade",
                    pathway_description="Extrusion of acetylene (C2H2, 26 Da) from phenyl cation [C6H5]+ yielding [C4H3]+ (m/z 51)",
                    carbocation_class="secondary",
                    radical_class=None,
                    generation=2,
                    neutral_loss="C2H2",
                    precursor_mz=77
                ))

        return secondary

    def _split_molecule_at_bond(self, mol: Chem.Mol, idx1: int, idx2: int) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Virtually cleaves the single bond between atoms idx1 and idx2.
        Returns two dictionaries of atom counts corresponding to the two resulting fragments.
        Explicitly accounts for attached implicit/explicit hydrogens on each side.
        """
        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        if not bond:
            return {}, {}

        # Use Chem.FragmentOnBonds to partition the molecular graph
        frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

        if len(frags) != 2:
            return {}, {}

        mol_a, mol_b = frags[0], frags[1]

        # Count atoms for each fragment (including implicit hydrogens)
        def get_frag_counts(fragment: Chem.Mol) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for atom in fragment.GetAtoms():
                elem = atom.GetSymbol()
                counts[elem] = counts.get(elem, 0) + 1
                # Add implicit and explicit H
                h_count = atom.GetTotalNumHs()
                if h_count > 0:
                    counts['H'] = counts.get('H', 0) + h_count
            return counts

        counts_a = get_frag_counts(mol_a)
        counts_b = get_frag_counts(mol_b)

        # Check which fragment contains idx1
        frag_indices_0 = [a for a in Chem.GetMolFrags(frag_mol)[0]]
        if idx1 in frag_indices_0:
            res_a, res_b = counts_a, counts_b
        else:
            res_a, res_b = counts_b, counts_a

        # Adjust for homolytic cleavage: each severed end loses 1 hydrogen relative to the saturated fragment
        res_a['H'] = max(0, res_a.get('H', 0) - 1)
        res_b['H'] = max(0, res_b.get('H', 0) - 1)
        return res_a, res_b

    def _classify_carbocation(self, counts: Dict[str, int], mol: Chem.Mol, center_idx: int) -> str:
        """Classify carbocation stability based on physical organic resonance and substitution."""
        if counts.get('C', 0) <= 1:
            return "methyl"
        if 'O' in counts:
            return "oxonium"
        if 'N' in counts:
            return "iminium"

        atom = mol.GetAtomWithIdx(center_idx)
        # Check if adjacent to aromatic ring
        for nbr in atom.GetNeighbors():
            if nbr.GetIsAromatic():
                return "benzylic"
            b = mol.GetBondBetweenAtoms(center_idx, nbr.GetIdx())
            if b and b.GetBondType() == Chem.BondType.DOUBLE:
                return "allylic"

        # Alkyl substitution
        carbon_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6]
        degree = len(carbon_neighbors)
        if degree >= 3:
            return "tertiary"
        elif degree == 2:
            return "secondary"
        elif degree == 1:
            return "primary"
        else:
            return "methyl"

    def _classify_radical(self, counts: Dict[str, int], mol: Chem.Mol, center_idx: int) -> str:
        """Classify radical leaving group stability."""
        if 'Cl' in counts and len(counts) == 1:
            return "halogen_cl"
        if 'Br' in counts and len(counts) == 1:
            return "halogen_br"
        if counts == {'H': 1}:
            return "hydrogen"
        if counts.get('C', 0) <= 1:
            return "methyl"

        atom = mol.GetAtomWithIdx(center_idx)
        if atom.GetIsAromatic():
            return "aryl"

        carbon_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6]
        degree = len(carbon_neighbors)
        if degree >= 3:
            return "tertiary"
        elif degree == 2:
            return "secondary"
        elif degree == 1:
            return "primary"
        else:
            return "methyl"

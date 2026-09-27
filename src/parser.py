"""
Molecular Graph & Feature Parser (Task A)
Parses SMILES representations into RDKit molecular graphs, extracts molecular properties,
computes monoisotopic/nominal masses, and identifies functional groups and rearrangement sites
using curated SMARTS patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional
import re

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Element nominal masses for fast mass calculations
NOMINAL_MASSES: Dict[str, int] = {
    'H': 1, 'C': 12, 'N': 14, 'O': 16, 'F': 19,
    'P': 31, 'S': 32, 'Cl': 35, 'Br': 79, 'I': 127
}

EXACT_MASSES: Dict[str, float] = {
    'H': 1.007825, 'C': 12.000000, 'N': 14.003074, 'O': 15.994915, 'F': 18.998403,
    'P': 30.973762, 'S': 31.972071, 'Cl': 34.968853, 'Br': 78.918338, 'I': 126.904473
}

# Curated SMARTS patterns for functional groups relevant to EI-MS fragmentation
SMARTS_PATTERNS: Dict[str, str] = {
    # Carbonyls
    'aldehyde': '[CX3H1](=O)[#6]',
    'ketone': '[#6][CX3](=O)[#6]',
    'carboxylic_acid': '[CX3](=O)[OX2H]',
    'ester': '[#6][CX3](=O)[OX2][#6]',
    'acyl_halide': '[CX3](=O)[F,Cl,Br,I]',
    'amide': '[CX3](=O)[NX3]',

    # Alcohols & Phenols
    'primary_alcohol': '[CH2][OX2H]',
    'secondary_alcohol': '[CH1]([#6])[OX2H]',
    'tertiary_alcohol': '[CH0]([#6])([#6])[OX2H]',
    'phenol': '[c][OX2H]',
    'general_alcohol': '[#6][OX2H]',

    # Amines & Ethers
    'primary_amine': '[#6][NX3;H2]',
    'secondary_amine': '[#6][NX3;H1][#6]',
    'tertiary_amine': '[#6][NX3;H0]([#6])[#6]',
    'ether': '[#6][OX2][#6;!$(C=O)]',

    # Halides
    'alkyl_fluoride': '[#6][F]',
    'alkyl_chloride': '[#6][Cl]',
    'alkyl_bromide': '[#6][Br]',
    'alkyl_iodide': '[#6][I]',

    # Unsaturations & Aromatics
    'alkene': '[CX3]=[CX3]',
    'alkyne': '[CX2]#[CX2]',
    'aromatic_ring': 'c1ccccc1',
    'alkylbenzene': 'c1ccccc1[CH3,CH2,CH1,CH0]',
    'benzylic_ch': '[c][CH3,CH2,CH1]',

    # RDA Substrates: Cyclohexene core (6-membered ring with 1 double bond)
    'cyclohexene_rda': 'C1=CCCCC1',

    # McLafferty Rearrangement motif: X=Y-C(alpha)-C(beta)-C(gamma)
    'mclafferty_carbonyl': '[OX1]=[CX3]-[#6]-[#6]-[#6]'
}


@dataclass
class MoleculeInfo:
    """Encapsulates chemical properties and graph representations of an input molecule."""
    smiles: str
    canonical_smiles: str
    mol: Chem.Mol
    mol_with_h: Chem.Mol
    formula: str
    atom_counts: Dict[str, int]
    exact_mass: float
    nominal_mass: int
    functional_groups: Dict[str, List[Tuple[int, ...]]] = field(default_factory=dict)
    is_aromatic: bool = False
    has_conjugated_system: bool = False
    rings: List[Tuple[int, ...]] = field(default_factory=list)
    num_carbons: int = 0
    num_hydrogens: int = 0
    num_chlorines: int = 0
    num_bromines: int = 0
    num_oxygens: int = 0
    num_nitrogens: int = 0
    num_sulfurs: int = 0


def parse_formula(formula_str: str) -> Dict[str, int]:
    """Parse a molecular formula string (e.g. 'C5H12O', 'C6H5Cl') into atom counts."""
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula_str)
    counts: Dict[str, int] = {}
    for elem, num_str in matches:
        if not elem:
            continue
        count = int(num_str) if num_str else 1
        counts[elem] = counts.get(elem, 0) + count
    return counts


def calculate_nominal_mass(atom_counts: Dict[str, int]) -> int:
    """Compute nominal mass from atom counts using standard integer isotopic masses."""
    mass = 0
    for elem, count in atom_counts.items():
        elem_mass = NOMINAL_MASSES.get(elem, round(EXACT_MASSES.get(elem, 0.0)))
        mass += elem_mass * count
    return mass


def parse_smiles(smiles: str) -> MoleculeInfo:
    """
    Parse SMILES string, validate chemistry, add explicit hydrogens,
    detect functional groups via SMARTS, and return a MoleculeInfo object.
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES string provided: '{smiles}'")

    # Sanitize molecule to ensure correct valences and aromaticity
    Chem.SanitizeMol(mol)

    canonical_smiles = Chem.MolToSmiles(mol)
    mol_with_h = Chem.AddHs(mol)

    # Compute formula and exact mass
    formula = rdMolDescriptors.CalcMolFormula(mol)
    atom_counts = parse_formula(formula)
    exact_mass = float(Descriptors.ExactMolWt(mol))
    nominal_mass = calculate_nominal_mass(atom_counts)

    # Ring information
    ring_info = mol.GetRingInfo()
    rings = [tuple(ring) for ring in ring_info.AtomRings()]

    # Aromaticity
    is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    # Detect functional groups via SMARTS
    functional_groups: Dict[str, List[Tuple[int, ...]]] = {}
    for fg_name, smarts in SMARTS_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            matches = mol.GetSubstructMatches(patt)
            if matches:
                functional_groups[fg_name] = [tuple(m) for m in matches]

    # Conjugation detection (alternating double/single bonds or aromatic + alkene)
    has_conjugated_system = is_aromatic or ('alkene' in functional_groups and len(functional_groups.get('alkene', [])) > 1)

    return MoleculeInfo(
        smiles=smiles,
        canonical_smiles=canonical_smiles,
        mol=mol,
        mol_with_h=mol_with_h,
        formula=formula,
        atom_counts=atom_counts,
        exact_mass=exact_mass,
        nominal_mass=nominal_mass,
        functional_groups=functional_groups,
        is_aromatic=is_aromatic,
        has_conjugated_system=has_conjugated_system,
        rings=rings,
        num_carbons=atom_counts.get('C', 0),
        num_hydrogens=atom_counts.get('H', 0),
        num_chlorines=atom_counts.get('Cl', 0),
        num_bromines=atom_counts.get('Br', 0),
        num_oxygens=atom_counts.get('O', 0),
        num_nitrogens=atom_counts.get('N', 0),
        num_sulfurs=atom_counts.get('S', 0),
    )


def atom_counts_to_formula(counts: Dict[str, int]) -> str:
    """Format atom counts as standard Hill system formula (C first, then H, then alphabetical)."""
    parts = []
    if 'C' in counts and counts['C'] > 0:
        c_val = counts['C']
        parts.append(f"C{c_val if c_val > 1 else ''}")
    if 'H' in counts and counts['H'] > 0:
        h_val = counts['H']
        parts.append(f"H{h_val if h_val > 1 else ''}")

    for elem in sorted(counts.keys()):
        if elem in ('C', 'H'):
            continue
        val = counts[elem]
        if val > 0:
            parts.append(f"{elem}{val if val > 1 else ''}")

    return "".join(parts) if parts else "H"

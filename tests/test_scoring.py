"""
Unit tests for Stevenson's Rule & Energy Scoring Engine (Tasks C & D).
"""

import pytest
from src.parser import parse_smiles
from src.fragmentation import FragmentCandidate
from src.scoring import ScoringEngine


@pytest.fixture
def scoring():
    return ScoringEngine()


def test_stevensons_rule_hierarchy(scoring):
    mol = parse_smiles("CC(=O)CC")

    cand_oxonium = FragmentCandidate(
        nominal_mz=43, exact_mz=43.0, formula="C2H3O", atom_counts={'C': 2, 'H': 3, 'O': 1},
        mechanism="alpha_cleavage", pathway_description="", carbocation_class="acylium", radical_class="primary"
    )

    cand_primary_alkyl = FragmentCandidate(
        nominal_mz=29, exact_mz=29.0, formula="C2H5", atom_counts={'C': 2, 'H': 5},
        mechanism="alpha_cleavage", pathway_description="", carbocation_class="primary", radical_class="primary"
    )

    score_ox = scoring.score_candidate(cand_oxonium, mol, beam_energy_ev=70.0)
    score_alk = scoring.score_candidate(cand_primary_alkyl, mol, beam_energy_ev=70.0)

    # Stevenson's rule: Acylium is significantly favored over primary alkyl carbocation
    assert score_ox > score_alk


def test_molecular_ion_energy_dependence(scoring):
    mol = parse_smiles("CCCC")
    cand_m = FragmentCandidate(
        nominal_mz=58, exact_mz=58.0, formula="C4H10", atom_counts={'C': 4, 'H': 10},
        mechanism="molecular_ion", pathway_description="", carbocation_class="radical_cation"
    )

    score_20 = scoring.score_candidate(cand_m, mol, beam_energy_ev=20.0)
    score_70 = scoring.score_candidate(cand_m, mol, beam_energy_ev=70.0)
    score_100 = scoring.score_candidate(cand_m, mol, beam_energy_ev=100.0)

    # Lower beam energy leaves higher fraction of intact molecular ions
    assert score_20 > score_70 > score_100


def test_secondary_cascade_energy_threshold(scoring):
    mol = parse_smiles("CC(=O)c1ccccc1")
    cand_secondary = FragmentCandidate(
        nominal_mz=51, exact_mz=51.0, formula="C4H3", atom_counts={'C': 4, 'H': 3},
        mechanism="secondary_cascade", pathway_description="", carbocation_class="secondary",
        generation=2
    )

    score_20 = scoring.score_candidate(cand_secondary, mol, beam_energy_ev=20.0)
    score_70 = scoring.score_candidate(cand_secondary, mol, beam_energy_ev=70.0)

    # Secondary cascades requiring high internal energy are heavily suppressed at 20 eV
    assert score_70 > score_20 * 5.0

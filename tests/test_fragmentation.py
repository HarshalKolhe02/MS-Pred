"""
Unit tests for Fragmentation Engine (Task B).
"""

import pytest
from src.parser import parse_smiles
from src.fragmentation import FragmentationEngine


@pytest.fixture
def engine():
    return FragmentationEngine()


def test_molecular_ion_generation(engine):
    mol = parse_smiles("CCCC")
    candidates = engine.generate_all_candidates(mol)
    mol_ion = [c for c in candidates if c.mechanism == "molecular_ion"]
    assert len(mol_ion) == 1
    assert mol_ion[0].nominal_mz == 58


def test_mclafferty_valeraldehyde(engine):
    mol = parse_smiles("CCCCC=O")
    candidates = engine.generate_all_candidates(mol)
    mclaff = [c for c in candidates if c.mechanism == "mclafferty"]
    assert len(mclaff) >= 1
    # Characteristic enol radical cation at m/z 44
    assert any(c.nominal_mz == 44 for c in mclaff)


def test_mclafferty_methyl_butyrate(engine):
    mol = parse_smiles("CCCC(=O)OC")
    candidates = engine.generate_all_candidates(mol)
    mclaff = [c for c in candidates if c.mechanism == "mclafferty"]
    assert len(mclaff) >= 1
    # Characteristic enol ester radical cation at m/z 74
    assert any(c.nominal_mz == 74 for c in mclaff)


def test_benzylic_tropylium_toluene(engine):
    mol = parse_smiles("Cc1ccccc1")
    candidates = engine.generate_all_candidates(mol)
    tropylium = [c for c in candidates if c.mechanism == "benzylic_tropylium" and c.nominal_mz == 91]
    assert len(tropylium) >= 1
    assert tropylium[0].carbocation_class == "tropylium"


def test_alcohol_alpha_cleavage_and_dehydration(engine):
    mol = parse_smiles("CCCCCO")
    candidates = engine.generate_all_candidates(mol)
    # Oxonium ion at m/z 31: [CH2=OH]+
    oxonium = [c for c in candidates if c.nominal_mz == 31]
    assert len(oxonium) >= 1

    # Dehydration [M - 18]+• at m/z 70
    dehyd = [c for c in candidates if c.mechanism == "dehydration_elimination" and c.nominal_mz == 70]
    assert len(dehyd) >= 1


def test_acetophenone_alpha_cleavage(engine):
    mol = parse_smiles("CC(=O)c1ccccc1")
    candidates = engine.generate_all_candidates(mol)
    # Acylium/Aroyl ion [C6H5-CO]+ at m/z 105
    acylium = [c for c in candidates if c.nominal_mz == 105 and c.carbocation_class in ["acylium", "aroyl"]]
    assert len(acylium) >= 1

    # Secondary decarbonylation [C6H5]+ at m/z 77
    phenyl = [c for c in candidates if c.nominal_mz == 77]
    assert len(phenyl) >= 1


def test_inductive_cleavage_ethyl_chloride(engine):
    mol = parse_smiles("CCCl")
    candidates = engine.generate_all_candidates(mol)
    ethyl = [c for c in candidates if c.mechanism == "inductive_cleavage" and c.nominal_mz == 29]
    assert len(ethyl) >= 1

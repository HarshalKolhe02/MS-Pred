"""
Unit tests for Molecular Graph & Feature Parser (Task A).
"""

import pytest
from src.parser import parse_smiles, parse_formula, calculate_nominal_mass, atom_counts_to_formula


def test_parse_smiles_alkane():
    mol = parse_smiles("CCCC")
    assert mol.formula == "C4H10"
    assert mol.nominal_mass == 58
    assert mol.num_carbons == 4
    assert mol.num_hydrogens == 10
    assert not mol.is_aromatic


def test_parse_smiles_aromatic():
    mol = parse_smiles("Cc1ccccc1")
    assert mol.formula == "C7H8"
    assert mol.nominal_mass == 92
    assert mol.is_aromatic
    assert "alkylbenzene" in mol.functional_groups


def test_parse_smiles_alcohol():
    mol = parse_smiles("CCCCCO")
    assert mol.formula == "C5H12O"
    assert mol.nominal_mass == 88
    assert mol.num_oxygens == 1
    assert "primary_alcohol" in mol.functional_groups or "general_alcohol" in mol.functional_groups


def test_parse_smiles_carbonyl_mclafferty():
    mol = parse_smiles("CCCCC=O")
    assert mol.formula == "C5H10O"
    assert mol.nominal_mass == 86
    assert "aldehyde" in mol.functional_groups
    assert "mclafferty_carbonyl" in mol.functional_groups


def test_parse_smiles_halides():
    mol_cl = parse_smiles("CCCl")
    assert mol_cl.formula == "C2H5Cl"
    assert mol_cl.nominal_mass == 64
    assert mol_cl.num_chlorines == 1

    mol_br = parse_smiles("CCBr")
    assert mol_br.formula == "C2H5Br"
    assert mol_br.nominal_mass == 108
    assert mol_br.num_bromines == 1


def test_invalid_smiles():
    with pytest.raises(ValueError):
        parse_smiles("NotAValidSMILESString123")

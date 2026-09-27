"""
Unit tests for Natural Abundance Isotope Patterns (Task E).
"""

import pytest
from src.fragmentation import FragmentCandidate
from src.isotopes import IsotopeEngine


@pytest.fixture
def iso_engine():
    return IsotopeEngine()


def test_carbon_13_satellite(iso_engine):
    # Decane fragment C10H21+ (m/z 141)
    cand = FragmentCandidate(
        nominal_mz=141, exact_mz=141.0, formula="C10H21", atom_counts={'C': 10, 'H': 21},
        mechanism="alkane_cc_cleavage", pathway_description=""
    )

    envelope = iso_engine.generate_envelope(cand, base_intensity=100.0)
    peak_m = next(p for p in envelope if p.nominal_mz == 141)
    peak_m1 = next(p for p in envelope if p.nominal_mz == 142)

    assert peak_m.intensity == 100.0
    # 10 carbons * 1.1% = ~11%
    assert 10.0 <= peak_m1.intensity <= 12.0


def test_chlorine_isotope_ratio(iso_engine):
    # Ethyl chloride molecular ion C2H5Cl+• (m/z 64)
    cand = FragmentCandidate(
        nominal_mz=64, exact_mz=64.0, formula="C2H5Cl", atom_counts={'C': 2, 'H': 5, 'Cl': 1},
        mechanism="molecular_ion", pathway_description="", is_radical_cation=True
    )

    envelope = iso_engine.generate_envelope(cand, base_intensity=100.0)
    peak_35 = next(p for p in envelope if p.nominal_mz == 64)
    peak_37 = next(p for p in envelope if p.nominal_mz == 66)

    # Ratio of 35Cl to 37Cl is ~3:1 (100 to ~32)
    ratio = peak_35.intensity / peak_37.intensity
    assert 2.8 <= ratio <= 3.4


def test_bromine_isotope_ratio(iso_engine):
    # Ethyl bromide molecular ion C2H5Br+• (m/z 108)
    cand = FragmentCandidate(
        nominal_mz=108, exact_mz=108.0, formula="C2H5Br", atom_counts={'C': 2, 'H': 5, 'Br': 1},
        mechanism="molecular_ion", pathway_description="", is_radical_cation=True
    )

    envelope = iso_engine.generate_envelope(cand, base_intensity=100.0)
    peak_79 = next(p for p in envelope if p.nominal_mz == 108)
    peak_81 = next(p for p in envelope if p.nominal_mz == 110)

    # Ratio of 79Br to 81Br is ~1:1 (100 to ~97)
    ratio = peak_79.intensity / peak_81.intensity
    assert 0.95 <= ratio <= 1.06

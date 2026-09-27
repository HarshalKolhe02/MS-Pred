"""
End-to-End Integration Tests for EI Mass Spectrum Predictor.
"""

import pytest
from src.parser import parse_smiles
from src.spectrum import SpectrumPredictor
from src.explainer import ChemicalExplainer
from src.evaluator import SpectrumEvaluator


@pytest.fixture
def predictor():
    return SpectrumPredictor()


@pytest.fixture
def explainer():
    return ChemicalExplainer()


@pytest.fixture
def evaluator():
    return SpectrumEvaluator()


def test_valeraldehyde_prediction(predictor, explainer, evaluator):
    mol = parse_smiles("CCCCC=O")
    spectrum = predictor.predict(mol, beam_energy_ev=70.0)

    # Base peak must be m/z 44 (McLafferty rearrangement)
    assert spectrum.base_peak_mz == 44
    assert 44 in spectrum.peaks
    assert spectrum.peaks[44].intensity == 100.0
    assert spectrum.peaks[44].primary_mechanism == "mclafferty"

    # Explanations check
    exps = explainer.explain_spectrum(spectrum, top_n=3)
    assert len(exps) >= 1
    assert any("mclafferty" in e.mechanism.lower() for e in exps)

    # Evaluation check
    report = evaluator.evaluate(spectrum, "CCCCC=O")
    assert report.base_peak_match is True
    assert report.cosine_similarity > 0.70


def test_toluene_prediction(predictor, evaluator):
    mol = parse_smiles("Cc1ccccc1")
    spectrum = predictor.predict(mol, beam_energy_ev=70.0)

    # Base peak must be m/z 91 (tropylium ion)
    assert spectrum.base_peak_mz == 91
    assert 91 in spectrum.peaks
    assert spectrum.peaks[91].intensity == 100.0

    # Secondary fragment m/z 65 must be present
    assert 65 in spectrum.peaks

    report = evaluator.evaluate(spectrum, "Cc1ccccc1")
    assert report.base_peak_match is True
    assert report.cosine_similarity > 0.85


def test_acetophenone_prediction(predictor, evaluator):
    mol = parse_smiles("CC(=O)c1ccccc1")
    spectrum = predictor.predict(mol, beam_energy_ev=70.0)

    # Base peak must be m/z 105 (benzoyl acylium ion)
    assert spectrum.base_peak_mz == 105
    # Secondary fragment m/z 77 (phenyl cation) must be present
    assert 77 in spectrum.peaks
    assert spectrum.peaks[77].intensity > 20.0

    report = evaluator.evaluate(spectrum, "CC(=O)c1ccccc1")
    assert report.base_peak_match is True
    assert report.cosine_similarity > 0.80


def test_ethyl_halide_isotopes(predictor):
    mol = parse_smiles("CCCl")
    spectrum = predictor.predict(mol, beam_energy_ev=70.0)

    # Both 35Cl (64) and 37Cl (66) must be present
    assert 64 in spectrum.peaks
    assert 66 in spectrum.peaks
    # 35Cl peak should be ~3x 37Cl peak
    int_64 = spectrum.peaks[64].intensity
    int_66 = spectrum.peaks[66].intensity
    assert int_64 > int_66
    assert 2.5 <= (int_64 / int_66) <= 3.5

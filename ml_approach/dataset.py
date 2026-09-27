"""
MassBank EI-MS Dataset Parser, Featurizer, and PyTorch Dataset
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def extract_molecular_features(smiles: str) -> Optional[Tuple[np.ndarray, float]]:
    """
    Extract multi-scale molecular features:
    1. Morgan Fingerprint (ECFP6, radius 3, 2048 bits)
    2. MACCS structural keys (166 bits)
    3. 24 Physicochemical and elemental constitutional descriptors
    Total feature length: 2048 + 166 + 24 = 2238 dimensions.
    Returns: (feature_vector, exact_mw) or None if invalid molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        # 1. Morgan Fingerprint (ECFP6)
        if hasattr(rdMolDescriptors, "GetMorganFingerprintAsBitVect"):
            try:
                from rdkit.Chem import rdFingerprintGenerator
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
                fpgen = gen.GetFingerprint(mol)
            except Exception:
                fpgen = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        else:
            fpgen = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        fp_arr = np.zeros((2048,), dtype=np.float32)
        for bit in fpgen.GetOnBits():
            fp_arr[bit] = 1.0

        # 2. MACCS keys (166 bits)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.array([int(b) for b in maccs.ToBitString()[1:]], dtype=np.float32)

        # 3. Physicochemical & Constitutional Descriptors
        exact_mw = Descriptors.ExactMolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        h_acc = Descriptors.NumHAcceptors(mol)
        h_don = Descriptors.NumHDonors(mol)
        rings = Descriptors.RingCount(mol)
        arom_rings = Descriptors.NumAromaticRings(mol)
        sat_rings = Descriptors.NumSaturatedRings(mol)
        frac_csp3 = Descriptors.FractionCSP3(mol)
        val_elec = Descriptors.NumValenceElectrons(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        # Specific atom counts
        atom_counts = {el: 0 for el in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in atom_counts:
                atom_counts[sym] += 1

        elem_vec = [atom_counts[el] for el in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']]

        # Normalized descriptor vector
        desc_vec = np.array([
            exact_mw / 100.0,
            tpsa / 100.0,
            logp / 5.0,
            rot_bonds / 10.0,
            h_acc / 10.0,
            h_don / 10.0,
            rings / 5.0,
            arom_rings / 5.0,
            sat_rings / 5.0,
            frac_csp3,
            val_elec / 100.0,
            heavy_atoms / 30.0,
            *elem_vec
        ], dtype=np.float32)

        # Concatenate into unified vector
        full_feat = np.concatenate([fp_arr, maccs_arr, desc_vec])
        return full_feat, exact_mw

    except Exception:
        return None


def parse_massbank_msp(msp_file_path: str, max_mz: int = 500, max_records: Optional[int] = None) -> List[Dict]:
    """
    Parses a MassBank NIST-format MSP file, specifically extracting authentic EI-MS records.
    """
    if not os.path.exists(msp_file_path):
        raise FileNotFoundError(f"MSP file not found at: {msp_file_path}")

    records = []
    current_entry = {}
    peaks = []
    reading_peaks = False

    with open(msp_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()

            if not line_str:
                if current_entry and peaks:
                    current_entry["peaks"] = peaks
                    if _is_valid_ei_record(current_entry):
                        records.append(current_entry)
                        if max_records and len(records) >= max_records:
                            break
                current_entry = {}
                peaks = []
                reading_peaks = False
                continue

            if not reading_peaks:
                if line_str.startswith("Name:"):
                    current_entry["name"] = line_str[5:].strip()
                elif line_str.startswith("SMILES:"):
                    current_entry["smiles"] = line_str[7:].strip()
                elif line_str.startswith("Formula:"):
                    current_entry["formula"] = line_str[8:].strip()
                elif line_str.startswith("MW:"):
                    try:
                        current_entry["mw"] = float(line_str[3:].strip())
                    except ValueError:
                        pass
                elif line_str.startswith("ExactMass:"):
                    try:
                        current_entry["exact_mass"] = float(line_str[10:].strip())
                    except ValueError:
                        pass
                elif line_str.startswith("Instrument_type:"):
                    current_entry["instrument_type"] = line_str[16:].strip()
                elif line_str.startswith("Ion_mode:"):
                    current_entry["ion_mode"] = line_str[9:].strip()
                elif line_str.startswith("Num Peaks:") or line_str.startswith("Num peaks:"):
                    reading_peaks = True
            else:
                parts = line_str.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        peaks.append((mz, intensity))
                    except ValueError:
                        pass

        # Handle last record
        if current_entry and peaks and _is_valid_ei_record(current_entry):
            current_entry["peaks"] = peaks
            records.append(current_entry)

    return records


def _is_valid_ei_record(entry: Dict) -> bool:
    """Filter for valid EI-MS records with proper SMILES and reasonable MW."""
    inst = entry.get("instrument_type", "").upper()
    if "EI" not in inst:
        return False
    smiles = entry.get("smiles", "")
    if not smiles or smiles == "N/A":
        return False
    if len(entry.get("peaks", [])) < 3:
        return False
    return True


def peaks_to_vector(peaks: List[Tuple[float, float]], max_mz: int = 500) -> np.ndarray:
    """
    Bins continuous (m/z, intensity) pairs into an integer m/z spectrum vector (1 to max_mz).
    Normalized such that base peak = 100.0.
    """
    vec = np.zeros(max_mz, dtype=np.float32)
    for mz, intensity in peaks:
        idx = int(round(mz)) - 1  # 0-indexed: mz 1 -> index 0, mz 500 -> index 499
        if 0 <= idx < max_mz:
            vec[idx] = max(vec[idx], float(intensity))

    max_val = np.max(vec)
    if max_val > 0:
        vec = (vec / max_val) * 100.0
    return vec


class EIMassSpectraDataset(Dataset):
    """
    PyTorch Dataset for EI-MS spectra prediction.
    Features: Multi-scale molecular descriptors (X in R^2238)
    Targets: Normalized binned spectrum vector (Y in R^max_mz)
    Auxiliary: Exact MW for physical masking
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, mws: np.ndarray, smiles_list: List[str]):
        if torch is not None:
            self.X = torch.tensor(features, dtype=torch.float32)
            self.Y = torch.tensor(targets, dtype=torch.float32)
            self.mws = torch.tensor(mws, dtype=torch.float32)
        else:
            self.X = features
            self.Y = targets
            self.mws = mws
        self.smiles = smiles_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mws[idx], self.smiles[idx]


def build_or_load_dataset(msp_path: str, cache_path: str = "data/processed_eims_dataset.npz",
                          max_mz: int = 500, max_records: Optional[int] = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extracts features and builds the vectorized dataset, caching the result in .npz format.
    """
    if os.path.exists(cache_path):
        print(f"[Dataset] Loading cached processed dataset from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["targets"], data["mws"], list(data["smiles"])

    print(f"[Dataset] Parsing authentic MassBank MSP records from: {msp_path}")
    raw_records = parse_massbank_msp(msp_path, max_mz=max_mz, max_records=max_records)
    print(f"[Dataset] Found {len(raw_records)} valid EI-MS experimental records.")

    features = []
    targets = []
    mws = []
    smiles_list = []

    for rec in raw_records:
        smi = rec["smiles"]
        feat_info = extract_molecular_features(smi)
        if feat_info is None:
            continue
        feat_vec, exact_mw = feat_info
        if exact_mw > (max_mz - 5):
            continue

        spec_vec = peaks_to_vector(rec["peaks"], max_mz=max_mz)
        if np.max(spec_vec) <= 0:
            continue

        features.append(feat_vec)
        targets.append(spec_vec)
        mws.append(exact_mw)
        smiles_list.append(smi)

    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    mws = np.array(mws, dtype=np.float32)

    print(f"[Dataset] Successfully processed {len(features)} valid molecular spectra pairs.")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, features=features, targets=targets, mws=mws, smiles=np.array(smiles_list))
    print(f"[Dataset] Cached processed dataset to: {cache_path}")

    return features, targets, mws, smiles_list

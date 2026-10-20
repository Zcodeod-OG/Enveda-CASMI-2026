"""Library-search baseline for Enveda CASMI 2026.

For each test molecule: compute its neutral mass from precursor m/z + adduct,
pull training structures within a ppm window, score each candidate by spectral
entropy similarity against its training spectra, and write the top 25
(deduplicated on InChIKey first block) to submission.csv.

Usage (Kaggle notebook or local):
    python library_search.py --data /kaggle/input/enveda-CASMI26-molecule-id-mass-spectra \
                             --out /kaggle/working/submission.csv
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

PROTON = 1.007276
ELECTRON = 0.000549
TOP_K = 25

# Exact masses of groups that appear in adduct strings.
GROUP_MASS = {
    "H": 1.007825, "Na": 22.989770, "K": 38.963707, "NH4": 18.034374, "NH3": 17.026549,
    "H2O": 18.010565, "Cl": 34.968853, "Br": 78.918338, "Li": 7.016004,
    "FA": 46.005479, "HCOO": 44.997654, "HCOOH": 46.005479, "CH3COO": 59.013304,
    "Hac": 60.021129, "ACN": 41.026549, "CH3OH": 32.026215, "MeOH": 32.026215,
    "CO2": 43.989829, "CH2O2": 46.005479, "C2H4O2": 60.021129, "OH": 17.002740,
}
ADDUCT_RE = re.compile(r"\[(\d*)M(.*?)\](\d*)([+-])")
TERM_RE = re.compile(r"([+-])(\d*)([A-Za-z0-9]+)")


def parse_adduct(adduct):
    """Return (n_M, delta, charge) such that mz = (n*M + delta) / charge, or None."""
    if not isinstance(adduct, str):
        return None
    m = ADDUCT_RE.match(adduct.replace(" ", ""))
    if not m:
        return None
    n = int(m.group(1) or 1)
    charge = int(m.group(3) or 1)
    sign = 1 if m.group(4) == "+" else -1
    delta = 0.0
    for s, cnt, grp in TERM_RE.findall(m.group(2)):
        if grp not in GROUP_MASS:
            return None
        delta += (1 if s == "+" else -1) * int(cnt or 1) * GROUP_MASS[grp]
    delta -= sign * charge * ELECTRON
    return n, delta, charge


def neutral_mass(precursor_mz, adduct):
    p = parse_adduct(adduct)
    if p is None:  # fall back to protonated / deprotonated
        pos = isinstance(adduct, str) and adduct.strip().endswith("+")
        return precursor_mz - PROTON if pos or not isinstance(adduct, str) else precursor_mz + PROTON
    n, delta, charge = p
    return (precursor_mz * charge - delta) / n


_tautomer = rdMolStandardize.TautomerEnumerator()


def ik14(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _tautomer.Canonicalize(mol)
    except Exception:
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def mol_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return ExactMolWt(mol) if mol is not None else np.nan


# ---------------------------------------------------------------- spectra
def clean(mzs, ints, precursor_mz, top_n=64, min_rel=0.01):
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    keep = (mzs < precursor_mz - 1.5) & (ints > 0)
    mzs, ints = mzs[keep], ints[keep]
    if len(ints) == 0:
        return mzs, ints
    keep = ints >= ints.max() * min_rel
    mzs, ints = mzs[keep], ints[keep]
    if len(ints) > top_n:
        idx = np.argsort(ints)[-top_n:]
        mzs, ints = mzs[idx], ints[idx]
    order = np.argsort(mzs)
    mzs, ints = mzs[order], ints[order]
    return mzs, ints / ints.sum()


def _entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def _weight(ints):
    s = _entropy(ints)
    if s < 3:
        ints = ints ** (0.25 + s * 0.25)
        ints = ints / ints.sum()
    return ints


def entropy_similarity(a, b, tol=0.01):
    """Unweighted-then-weighted spectral entropy similarity (Li et al. 2021)."""
    (ma, ia), (mb, ib) = a, b
    if len(ma) == 0 or len(mb) == 0:
        return 0.0
    ia, ib = _weight(ia), _weight(ib)
    # Greedy one-to-one matching on sorted m/z.
    j = np.searchsorted(mb, ma - tol)
    ma_out, mb_out = ia.copy(), ib.copy()
    pa, pb = [], []
    used = set()
    for i, start in enumerate(j):
        k = start
        while k < len(mb) and mb[k] <= ma[i] + tol:
            if k not in used:
                used.add(k)
                pa.append(i)
                pb.append(k)
                break
            k += 1
    if not pa:
        return 0.0
    pa, pb = np.array(pa), np.array(pb)
    ma_out[pa] = 0
    mb_out[pb] = 0
    mix = np.concatenate([(ia[pa] + ib[pb]) / 2, ma_out / 2, mb_out / 2])
    s_ab = _entropy(mix)
    return float(1 - (2 * s_ab - _entropy(ia) - _entropy(ib)) / np.log(4))


# ---------------------------------------------------------------- pipeline
def build_library(train, max_spectra_per_struct=12):
    print(f"train spectra: {len(train):,}")
    structs = pd.DataFrame({"smiles": train["normalized_smiles"].unique()})
    structs["mass"] = [mol_mass(s) for s in structs["smiles"]]
    structs = structs.dropna().sort_values("mass").reset_index(drop=True)
    print(f"train structures: {len(structs):,}")
    sid = dict(zip(structs["smiles"], structs.index))
    spectra = defaultdict(list)
    for smi, pmz, mzs, ints in zip(train["normalized_smiles"], train["precursor_mz"],
                                   train["ms2_mzs"], train["ms2_normalized_intensities"]):
        k = sid.get(smi)
        if k is None or len(spectra[k]) >= max_spectra_per_struct:
            continue
        spec = clean(mzs, ints, pmz)
        if len(spec[0]):
            spectra[k].append(spec)
    return structs, spectra


def rank_molecule(test_specs, target, structs, spectra, ppm=10.0):
    masses = structs["mass"].values
    lo = np.searchsorted(masses, target * (1 - ppm * 1e-6))
    hi = np.searchsorted(masses, target * (1 + ppm * 1e-6))
    scored = []
    for k in range(lo, hi):
        lib = spectra.get(k, [])
        if lib and test_specs:
            per_test = [max(entropy_similarity(t, l) for l in lib) for t in test_specs]
            score = float(np.mean(per_test))
        else:
            score = 0.0
        scored.append((score, -abs(masses[k] - target), k))
    scored.sort(reverse=True)
    ranked = [structs.at[k, "smiles"] for _, _, k in scored]
    if len(ranked) < TOP_K * 2:  # pad with mass-nearest structures
        c = np.searchsorted(masses, target)
        near = np.argsort(np.abs(masses[max(0, c - 200):c + 200] - target)) + max(0, c - 200)
        ranked += [structs.at[k, "smiles"] for k in near]
    out, seen = [], set()
    for smi in ranked:
        key = ik14(smi)
        if key and key not in seen:
            seen.add(key)
            out.append(smi)
        if len(out) == TOP_K:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--ppm", type=float, default=10.0)
    args = ap.parse_args()
    data = Path(args.data)

    cols = ["normalized_smiles", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    train = pd.read_parquet(data / "train.parquet", columns=cols)
    structs, spectra = build_library(train)
    del train

    test = pd.read_parquet(data / "test.parquet")
    rows = []
    for mid, g in test.groupby("molecule_id", sort=False):
        target = float(np.median([neutral_mass(p, a) for p, a in zip(g["precursor_mz"], g["adduct"])]))
        specs = [clean(m, i, p) for m, i, p in
                 zip(g["ms2_mzs"], g["ms2_normalized_intensities"], g["precursor_mz"])]
        specs = [s for s in specs if len(s[0])]
        cands = rank_molecule(specs, target, structs, spectra, args.ppm)
        rows.append((mid, ";".join(cands)))
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    sub.to_csv(args.out, index=False)
    print(f"wrote {len(sub)} rows to {args.out}")


if __name__ == "__main__":
    main()

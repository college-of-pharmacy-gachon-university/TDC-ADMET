# %% [markdown]
# # 06 - Construction of the CYP2C19 cross-isoform probe set
#
# This script documents, exactly, how the CYP2C19 substrate evaluation set reported in
# Table 7 and Table S32 was assembled. It is published for transparency, not as a recommended
# procedure.
#
# What it does:
#
# * **positives** - CYP2C19 substrates from the external source, after the Tanimoto 0.8
#   filter of script 01
# * **negatives** - non-substrates taken from the held-out **TDC CYP2C9 test partition**,
#   further filtered to a Tanimoto similarity below 0.3 from any positive
#
# Three things follow, and the manuscript states all three (Section 2.8.4):
#
# 1. No CYP2C19 model exists. These compounds are scored with the TDC-trained **CYP2C9**
#    substrate model, on the grounds that the two isoforms are closely related members of the
#    CYP2C subfamily. The result bears on cross-isoform transfer, not on CYP2C19 prediction.
# 2. The negative class derives from the benchmark rather than from an independent source. It
#    comes from the held-out test partition, so there is no training leakage, but the set is
#    not external in the sense the rest of Section 3.5 intends.
# 3. Filtering the negative class away from the positive class in chemical space widens the
#    separation between the classes and inflates any discrimination metric.
#
# Report the outcome as a cross-isoform transfer probe, with the construction described.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "admet_config.py").exists())
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

import admet_config as cfg

RDLogger.DisableLog("rdApp.*")

# ------------------------------------------------------------------ settings
POSITIVES_CSV = cfg.EXTERNAL_FILTERED_DIR / "Admetica" / "cyp2c19-substrate.csv"
TDC_REFERENCE_ENDPOINT = "CYP2C9_Substrate_CarbonMangels"
NEGATIVE_DISSIMILARITY = 0.3
OUT_CSV = cfg.EXTERNAL_FILTERED_DIR / "Admetica" / "cyp2c19_cross_isoform_probe.csv"

_fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=cfg.MORGAN_RADIUS, fpSize=cfg.MORGAN_NBITS_SIMILARITY)

# %%
pos_all = pd.read_csv(POSITIVES_CSV, encoding="utf-8-sig")
positives = pos_all[pos_all["Label"] == 1][["Canonical_SMILES", "Label"]].copy()
print(f"positives (external CYP2C19 substrates, post 0.8 filter): {len(positives)}")

ref = pd.read_csv(cfg.CANONICAL_DIR / f"canonical_{TDC_REFERENCE_ENDPOINT}.csv", encoding="utf-8-sig")
tdc_test_negatives = ref[(ref["Split"] == "test") & (ref["Label"] == 0)].copy()
print(f"candidate negatives (TDC {TDC_REFERENCE_ENDPOINT} test partition, label 0): "
      f"{len(tdc_test_negatives)}")


# %%
def fps(smiles_iter):
    out = []
    for s in smiles_iter:
        mol = MolFromSmiles(s)
        if mol is not None:
            out.append((s, _fpgen.GetFingerprint(mol)))
    return out


pos_fps = [fp for _, fp in fps(positives["Canonical_SMILES"])]

kept = []
for smi, fp in fps(tdc_test_negatives["Canonical_SMILES"]):
    sims = DataStructs.BulkTanimotoSimilarity(fp, pos_fps)
    if (max(sims) if sims else 1.0) < NEGATIVE_DISSIMILARITY:
        kept.append(smi)

negatives = pd.DataFrame({"Canonical_SMILES": kept, "Label": 0})
print(f"negatives retained at Tanimoto < {NEGATIVE_DISSIMILARITY}: {len(negatives)} "
      f"(of {len(tdc_test_negatives)})")

# %%
probe = pd.concat([positives, negatives], ignore_index=True).drop_duplicates(
    subset=["Canonical_SMILES"])
probe["Source"] = ["external_positive"] * len(positives) + ["TDC_test_negative"] * len(negatives)
probe = probe.drop_duplicates(subset=["Canonical_SMILES"])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
probe.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nprobe set: {len(probe)} compounds "
      f"({int((probe['Label'] == 1).sum())} active, {int((probe['Label'] == 0).sum())} inactive)")
print("written to:", OUT_CSV)
print("\nReport any result from this file as a cross-isoform transfer probe, naming the "
      "construction, and not as external validation of CYP2C19 prediction.")

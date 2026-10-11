# Part of the ADMET-Stack revision analysis set.
# Section 2.8.4: overlap between the in-house AMES collection and TDC AMES.
# Inputs: run artifacts; set the path constant at the top
# -*- coding: utf-8 -*-
"""The overlap check Section 2.8.5 asks for, on the in-house AMES set.

Section 2.8.2 states that for each external dataset the nearest TDC neighbour was found by
Morgan Tanimoto similarity and that external compounds above a threshold were removed.
Section 2.8.5 then asks for the result of that check on the in-house collections. This
computes it for in-house AMES against the TDC AMES benchmark:

  * how many in-house structures are exactly the same canonical structure as a TDC compound,
    split by which TDC partition the match falls in, because the training and validation
    partitions were pooled for fitting and so a match there is a compound the model saw;
  * where the labels agree on those matches;
  * the distribution of each in-house compound's nearest TDC neighbour, and how many sit at
    or above the 0.8 threshold of Section 2.8.2;
  * whether the n = 25,536 reported in Table S32 is the count before or after that filter.

Only the in-house AMES set is examined. The in-house P-glycoprotein collection is not
reported anywhere in the manuscript, so there is nothing in the paper for a check on it to
support.
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
sys.stdout.reconfigure(encoding="utf8")
HERE = os.path.dirname(os.path.abspath(__file__))

IH = (r"G:\K-MELLODDY-Project\data\02_in-house_CSV_files\Ames_inhouse"
      r"\Ames_mutagenicity_canonical.csv")
TDC_CSV = r"G:\K-MELLODDY-Project\data\04_TDC_data\AMES.csv"
THRESHOLD = 0.8
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


# --------------------------------------------------------------------- the two sides
ih = pd.read_csv(IH)
ih["cs"] = ih["Canonical_SMILES"].map(canon)
ih = ih.dropna(subset=["cs"])
print(f"in-house AMES          : {len(ih):,} rows, "
      f"{ih.cs.nunique():,} unique canonical structures, "
      f"prevalence {ih.Label.mean():.3f}")

tdc = pd.read_csv(TDC_CSV)
tdc["cs"] = tdc["Drug"].map(canon)
tdc = tdc.dropna(subset=["cs"])
print(f"TDC AMES benchmark     : {len(tdc):,} rows, "
      f"{tdc.cs.nunique():,} unique canonical structures, "
      f"prevalence {tdc.Y.mean():.3f}")

# --------------------------------------------------------------------- TDC partitions
# The scaffold split is the official one; the pipeline pools train and valid before fitting,
# so "seen by the model" means train or valid.
try:
    from tdc.single_pred import Tox
    split = Tox(name="AMES").get_split(method="scaffold", seed=42)
    part = {}
    for name in ("train", "valid", "test"):
        for s in split[name]["Drug"]:
            c = canon(s)
            if c:
                part.setdefault(c, set()).add(name)
    print(f"official scaffold split: "
          + ", ".join(f"{k} {len(split[k]):,}" for k in ("train", "valid", "test")))
    have_split = True
except Exception as exc:
    print(f"\n  could not load the official split ({type(exc).__name__}: "
          f"{str(exc)[:70]}); the partition breakdown is omitted and the overlap is "
          f"reported against the whole benchmark")
    part, have_split = {}, False

# --------------------------------------------------------------------- exact overlap
tdc_label = tdc.groupby("cs").Y.agg(lambda s: s.mode().iloc[0])
shared = sorted(set(ih.cs) & set(tdc_label.index))
print("\n" + "=" * 92)
print("exact canonical-structure overlap with the TDC AMES benchmark")
print("=" * 92)
print(f"  shared structures        : {len(shared):,}")
print(f"  share of the in-house set: {len(shared) / ih.cs.nunique():.1%}")
print(f"  share of the benchmark   : {len(shared) / tdc.cs.nunique():.1%}")

if shared:
    ih_label = ih.groupby("cs").Label.agg(lambda s: s.mode().iloc[0])
    agree = int(sum(ih_label[c] == tdc_label[c] for c in shared))
    print(f"  labels agree on          : {agree:,} of {len(shared):,} "
          f"({agree / len(shared):.1%})")
    if have_split:
        counts = {"train": 0, "valid": 0, "test": 0, "not in the split file": 0}
        for c in shared:
            names = part.get(c)
            if not names:
                counts["not in the split file"] += 1
            else:
                for n in names:
                    counts[n] += 1
        print("\n  which TDC partition each shared structure falls in:")
        for k, v in counts.items():
            print(f"    {k:24s} {v:,}")
        seen = counts["train"] + counts["valid"]
        print(f"\n  structures the model was fitted on (train + valid pooled): {seen:,}"
              f"  = {seen / ih.cs.nunique():.2%} of the in-house set")

# --------------------------------------------------------------------- nearest neighbour
print("\n" + "=" * 92)
print(f"nearest TDC neighbour of each in-house structure (Morgan r = 2, 2048 bits)")
print("=" * 92)
tdc_fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in tdc_label.index]
ih_unique = sorted(set(ih.cs))
nn = np.empty(len(ih_unique), dtype=np.float32)
for i, s in enumerate(ih_unique):
    fp = GEN.GetFingerprint(Chem.MolFromSmiles(s))
    nn[i] = max(DataStructs.BulkTanimotoSimilarity(fp, tdc_fps))
    if (i + 1) % 5000 == 0:
        print(f"    {i + 1:,} / {len(ih_unique):,}")

for lo in (1.0, 0.9, 0.8, 0.7, 0.5):
    k = int((nn >= lo).sum())
    print(f"  nearest neighbour >= {lo:.1f} : {k:6,}  ({k / len(nn):6.2%})")
print(f"  median nearest neighbour  : {np.median(nn):.3f}")
print(f"  mean   nearest neighbour  : {nn.mean():.3f}")

above = int((nn >= THRESHOLD).sum())
would_remain = len(nn) - above
print("\n" + "=" * 92)
print("was the Section 2.8.2 filter applied to this set?")
print("=" * 92)
print(f"  reported in Table S32                      : 25,536")
print(f"  rows in the source file                    : {len(ih):,}")
print(f"  unique structures in the source file       : {len(ih_unique):,}")
print(f"  structures with a TDC neighbour >= {THRESHOLD}     : {above:,}")
print(f"  structures that would remain after filtering: {would_remain:,}")
if len(ih) == 25536 and above > 0:
    print("\n  -> the reported count equals the unfiltered row count, so the "
          f"similarity filter\n     of Section 2.8.2 was NOT applied to the in-house AMES "
          f"evaluation.")
elif above == 0:
    print("\n  -> no structure reaches the threshold, so filtering would remove nothing "
          "and the\n     reported count is consistent with the filter having been applied.")

pd.DataFrame({"canonical_smiles": ih_unique, "nearest_tdc_tanimoto": nn}).to_csv(
    os.path.join(HERE, "data_inhouse_ames_nn.csv"), index=False)
print("\n-> data_inhouse_ames_nn.csv")

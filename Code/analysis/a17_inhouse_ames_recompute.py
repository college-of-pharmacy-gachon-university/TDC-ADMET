# Part of the ADMET-Stack revision analysis set.
# Section 2.8.4: metrics on the deduplicated, overlap-filtered AMES subset.
# Inputs: run artifacts; set the path constant at the top
# -*- coding: utf-8 -*-
"""Recompute the in-house AMES evaluation on the set the Methods section describes.

Section 2.8.1 states that exact duplicate structures were removed from each external set and
Section 2.8.2 that external compounds within Tanimoto 0.8 of a benchmark compound were
removed. Neither held for this set: the 25,536 rows reported in Table S32 and Section 3.5
contain 4,497 distinct structures, and 2,981 of those sit at or above 0.8 of a TDC AMES
compound.

This recomputes the evaluation three ways from the same stored predictions:

  as reported        all 25,536 rows, structures repeating up to 29 times
  deduplicated       one row per distinct structure, 4,497 compounds
  independent        deduplicated and then filtered at Tanimoto 0.8, 1,516 compounds

The bootstrap resamples compounds, not rows, in the second and third, so the interval
reflects the number of distinct compounds actually evidenced.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef, roc_auc_score)

sys.stdout.reconfigure(encoding="utf8")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = (r"G:\K-MELLODDY-Project\data\02_in-house_CSV_files\Ames_inhouse"
       r"\Ames_mutagenicity_canonical.csv")
PRED = (r"G:\K-MELLODDY-Project\data\02_in-house_CSV_files\in-house_data"
        r"\in-house_canonical_SMILES\ames_in_house\5Folds\ames_Outputs")
NBOOT = 2000
RNG = np.random.default_rng(7)

src = pd.read_csv(SRC)
nn = pd.read_csv(os.path.join(HERE, "data_inhouse_ames_nn.csv"))
near = dict(zip(nn.canonical_smiles, nn.nearest_tdc_tanimoto))

files = sorted(glob.glob(os.path.join(PRED, "Model_*_actual_vs_predicted.csv")))
print(f"{len(files)} per-seed prediction files")
probs, y = [], None
for f in files:
    d = pd.read_csv(f)
    assert len(d) == len(src), f"{os.path.basename(f)}: {len(d)} rows against {len(src)}"
    probs.append(d["Predicted_Prob"].to_numpy(float))
    if y is None:
        y = d["Actual"].to_numpy(int)

# the prediction rows must line up with the source rows, or nothing below means anything
assert np.array_equal(y, src["Label"].to_numpy(int)), \
    "the Actual column does not match the source Label column row for row"
print("row alignment with the source file: verified")
P = np.column_stack(probs)

src = src.assign(_i=np.arange(len(src)))
src["near"] = src["Canonical_SMILES"].map(near)


def evaluate(idx, label, unit):
    """Metrics over the rows in `idx`, bootstrapping whatever `unit` says is independent."""
    yy, pp = y[idx], P[idx]
    per_seed = []
    for k in range(pp.shape[1]):
        d = (pp[:, k] >= 0.5).astype(int)
        per_seed.append(dict(ACC=accuracy_score(yy, d),
                             BAL=balanced_accuracy_score(yy, d),
                             MCC=matthews_corrcoef(yy, d),
                             AUROC=roc_auc_score(yy, pp[:, k]),
                             AUPRC=average_precision_score(yy, pp[:, k])))
    ps = pd.DataFrame(per_seed)

    n = len(idx)
    boot = []
    for _ in range(NBOOT):
        b = RNG.integers(0, n, n)
        if len(np.unique(yy[b])) < 2:
            continue
        boot.append(np.mean([roc_auc_score(yy[b], pp[b, k])
                             for k in range(pp.shape[1])]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n{label}")
    print(f"  {unit:28s} {n:,}")
    print(f"  prevalence                   {yy.mean():.3f}")
    print(f"  ACC                          {ps.ACC.mean():.3f} \u00b1 {ps.ACC.std(ddof=1):.3f}")
    print(f"  balanced accuracy            {ps.BAL.mean():.3f} \u00b1 {ps.BAL.std(ddof=1):.3f}")
    print(f"  MCC                          {ps.MCC.mean():.3f} \u00b1 {ps.MCC.std(ddof=1):.3f}")
    print(f"  AUROC                        {ps.AUROC.mean():.3f} \u00b1 "
          f"{ps.AUROC.std(ddof=1):.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  AUPRC                        {ps.AUPRC.mean():.3f} \u00b1 "
          f"{ps.AUPRC.std(ddof=1):.3f}")
    return dict(set=label, n=n, prevalence=round(float(yy.mean()), 4),
                ACC=f"{ps.ACC.mean():.3f} \u00b1 {ps.ACC.std(ddof=1):.3f}",
                BAL_ACC=f"{ps.BAL.mean():.3f} \u00b1 {ps.BAL.std(ddof=1):.3f}",
                MCC=f"{ps.MCC.mean():.3f} \u00b1 {ps.MCC.std(ddof=1):.3f}",
                AUROC=f"{ps.AUROC.mean():.3f} \u00b1 {ps.AUROC.std(ddof=1):.3f}",
                AUROC_CI=f"[{lo:.3f}, {hi:.3f}]",
                AUPRC=f"{ps.AUPRC.mean():.3f} \u00b1 {ps.AUPRC.std(ddof=1):.3f}")


rows = [evaluate(src._i.to_numpy(), "as reported in Table S32", "rows")]

# one row per structure: the first occurrence, after dropping structures whose own rows
# disagree on the label, since for those no single label is defensible
lab = src.groupby("Canonical_SMILES").Label.nunique()
conflicting = set(lab[lab > 1].index)
print(f"\nstructures whose own rows carry conflicting labels: {len(conflicting):,} "
      f"(excluded from the deduplicated sets)")
dedup = src[~src.Canonical_SMILES.isin(conflicting)].drop_duplicates("Canonical_SMILES")
rows.append(evaluate(dedup._i.to_numpy(), "deduplicated, one row per structure",
                     "distinct compounds"))

indep = dedup[dedup.near < 0.8]
rows.append(evaluate(indep._i.to_numpy(),
                     "deduplicated and filtered at Tanimoto 0.8 against TDC AMES",
                     "independent compounds"))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "data_inhouse_ames_recomputed.csv"), index=False)
print("\n-> data_inhouse_ames_recomputed.csv")

print("\n" + "=" * 92)
print("what changes between the reported set and the independent one")
print("=" * 92)
a, c = rows[0], rows[-1]
print(f"  n                 {a['n']:,} rows  ->  {c['n']:,} independent compounds")
print(f"  AUROC             {a['AUROC']}  ->  {c['AUROC']}")
print(f"  AUROC 95% CI      {a['AUROC_CI']}  ->  {c['AUROC_CI']}")
print(f"  prevalence        {a['prevalence']}  ->  {c['prevalence']}")

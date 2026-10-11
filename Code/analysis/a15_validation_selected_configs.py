# Part of the ADMET-Stack revision analysis set.
# Table S44: configurations chosen on a held-out validation partition.
# Inputs: run artifacts; set the path constant at the top
# -*- coding: utf-8 -*-
"""Harvest the complete validation/test grid and redo the per-endpoint selection honestly.

The reviewers' most serious objection is that the per-endpoint configuration reported in
Table 6 was identified on the official test partition. The revision so far answered that with
two surrogates: a selection-free fixed configuration (Table S39) and a split-sample estimate
of the optimism (Table S40).

Both become unnecessary. The results tree

    G:\\K-MELLODDY-Project\\data\\ADMET_PK_Public_Dataset\\00_00_TDC_data
        <legacy configuration>\\<endpoint>\\<endpoint>_Outpus\\
            <endpoint>_validation_test_metrics_sf_scaled.csv

holds, for every one of the 54 x 22 = 1,188 cells, a held-out VALIDATION score beside the
test score, in the endpoint's official metric. The configuration can therefore be selected on
validation and scored on test - the selection the paper should have made.

This script
  1. harvests all 1,188 cells and checks the grid is complete,
  2. selects per endpoint on validation and reports the corresponding test score,
  3. compares that with the published test-selected value, giving the selection optimism
     directly rather than through a resampling surrogate,
  4. checks whether the published RMSE column is in fact a mean squared error.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"E:\ADMET_Stack_project_for_review261003\Repo_Ready_20261007\src")
from admet_config import ENDPOINTS, LEGACY_FOLDER  # noqa: E402

GRID = (r"G:\K-MELLODDY-Project\data\ADMET_PK_Public_Dataset\00_00_TDC_data")
CFG_OF_FOLDER = {v: k for k, v in LEGACY_FOLDER.items()}
# the column each endpoint's official metric lives in, and whether larger is better
METRIC_COL = {"AUROC": ("AUROC", True), "AUPRC": ("AUPRC", True),
              "MAE": ("MAE", False), "Spearman": ("SpearmanR", True)}

rows = []
for folder in sorted(os.listdir(GRID)):
    cfg = CFG_OF_FOLDER.get(folder)
    if cfg is None:
        print(f"  unmapped results folder: {folder}")
        continue
    for ep, meta in ENDPOINTS.items():
        path = os.path.join(GRID, folder, ep, f"{ep}_Outpus",
                            f"{ep}_validation_test_metrics_sf_scaled.csv")
        if not os.path.exists(path):
            rows.append(dict(configuration=cfg, endpoint=ep, valid=np.nan, test=np.nan,
                             missing=True))
            continue
        d = pd.read_csv(path).set_index("Set")
        col, higher = METRIC_COL[meta["metric"]]
        if col not in d.columns:
            print(f"  {ep} / {folder}: no {col} column, has {list(d.columns)}")
            continue
        rec = dict(configuration=cfg, endpoint=ep, metric=meta["metric"],
                   higher_is_better=higher, missing=False,
                   valid=float(d.loc["Validation", col]),
                   test=float(d.loc["Test", col]))
        for extra in ("RMSE", "R2", "MAE", "Accuracy"):
            if extra in d.columns:
                rec[f"test_{extra}"] = float(d.loc["Test", extra])
        rows.append(rec)

g = pd.DataFrame(rows)
g.to_csv(os.path.join(HERE, "data_validation_test_grid.csv"), index=False)
print(f"grid harvested: {len(g)} cells, {g.configuration.nunique()} configurations x "
      f"{g.endpoint.nunique()} endpoints")
print(f"cells with no file: {int(g.missing.sum())}")
g = g[~g.missing]

# ------------------------------------------------------- selection on validation vs on test
print("\n" + "=" * 104)
print("per-endpoint selection: on the held-out validation partition against on the test "
      "partition")
print("=" * 104)
print(f"{'endpoint':32s}{'metric':10s}{'selected on validation':>34s}"
      f"{'selected on test':>26s}")
sel = []
for ep, grp in g.groupby("endpoint", sort=False):
    higher = bool(grp.higher_is_better.iloc[0])
    metric = grp.metric.iloc[0]
    iv = grp.valid.idxmax() if higher else grp.valid.idxmin()
    it = grp.test.idxmax() if higher else grp.test.idxmin()
    v_cfg, v_test = grp.loc[iv, "configuration"], grp.loc[iv, "test"]
    t_cfg, t_test = grp.loc[it, "configuration"], grp.loc[it, "test"]
    opt = (t_test - v_test) if higher else (v_test - t_test)
    sel.append(dict(endpoint=ep, metric=metric,
                    config_by_validation=v_cfg, test_of_validation_pick=round(v_test, 4),
                    config_by_test=t_cfg, best_test=round(t_test, 4),
                    optimism=round(opt, 4), same_configuration=v_cfg == t_cfg))
    print(f"{ep:32s}{metric:10s}{v_cfg:>22s} {v_test:>10.4f}"
          f"{t_cfg:>16s} {t_test:>8.4f}")

s = pd.DataFrame(sel)
s.to_csv(os.path.join(HERE, "data_honest_selection.csv"), index=False)

print("\n--- how often the validation pick is also the test pick ---")
print(f"  {int(s.same_configuration.sum())} of {len(s)} endpoints")

print("\n--- selection optimism (best test minus the test score of the validation pick) ---")
for metric, grp in s.groupby("metric"):
    print(f"  {metric:9s} n={len(grp):2d}  mean {grp.optimism.mean():+.4f}   "
          f"median {grp.optimism.median():+.4f}   max {grp.optimism.max():+.4f}")
print("\n  largest optimism, by endpoint:")
for _, r in s.sort_values("optimism", ascending=False).head(8).iterrows():
    print(f"    {r['endpoint']:32s} {r['metric']:9s} "
          f"validation pick {r['config_by_validation']:>20s} -> test "
          f"{r['test_of_validation_pick']:.4f}   best test {r['best_test']:.4f}   "
          f"{r['optimism']:+.4f}")

# ------------------------------------------------------- is the published RMSE really an MSE?
print("\n" + "=" * 104)
print("the RMSE column of the stored results, against its own MAE")
print("=" * 104)
reg = g[g.metric.isin(["MAE", "Spearman"])].dropna(subset=["test_RMSE"])
if len(reg):
    bad = reg[reg.test_RMSE < reg.test_MAE]
    print(f"  {len(reg)} regression cells carry an RMSE column")
    print(f"  cells where the stored RMSE is smaller than the MAE (impossible for a true "
          f"RMSE): {len(bad)}")
    if len(bad):
        for _, r in bad.head(10).iterrows():
            print(f"    {r['endpoint']:28s} {r['configuration']:22s} "
                  f"MAE {r['test_MAE']:.4f}  stored RMSE {r['test_RMSE']:.4f}  "
                  f"sqrt(stored) {np.sqrt(r['test_RMSE']):.4f}")

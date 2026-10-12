# Part of the ADMET-Stack revision analysis set.
# Table S31: class prevalence, majority-class rate and threshold-corrected metrics.
# Inputs: data/external_evaluation.csv
# -*- coding: utf-8 -*-
"""Recomputes the class-balance columns of Table S31 (continued) (a) from the reported
accuracy, AUROC and AUPRC of each external evaluation set.

The input file is the Table S31 grid exported verbatim from the Supplementary Information,
so the numbers here cannot drift away from the published table. AUROC and AUPRC are the
means over five seeds, except for the AutoML rows, which come from a single stored model
and were recomputed from the retained predictions.
"""
import csv
import os
from pathlib import Path

DATA = Path(os.environ.get("ADMET_DATA",
                           Path(__file__).resolve().parent.parent / "data"))

with open(DATA / "external_evaluation.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

hdr = (f'{"property":22s}{"source":13s}{"n":>7s}{"prev":>7s}{"maj":>7s}'
       f'{"ACC":>8s}{"AUROC":>8s}{"AUPRC":>8s}{"AP-prev":>9s}  flags')
print(hdr)
print("-" * len(hdr))

n_below_maj = n_tiny = n_below_chance = 0
for r in rows:
    n = int(r["n_total"]); a = int(r["n_active"]); i = int(r["n_inactive"])
    acc = float(r["acc"]); roc = float(r["auroc"]); prc = float(r["auprc"])
    prev = a / n
    maj = max(a, i) / n
    lift = prc - prev
    flags = []
    if roc < 0.5:  flags.append("AUROC<0.5");  n_below_chance += 1
    if lift <= 0:  flags.append("AUPRC<=prev")
    if acc < maj:  flags.append("ACC<majority"); n_below_maj += 1
    if n < 100:    flags.append("n<100"); n_tiny += 1
    print(f'{r["property"]:22s}{r["source"]:13s}{n:7d}{prev:7.3f}{maj:7.3f}'
          f'{acc:8.3f}{roc:8.3f}{prc:8.3f}{lift:+9.3f}  {", ".join(flags)}')

print()
print(f"evaluation sets: {len(rows)}")
print(f"accuracy below the majority-class rate: {n_below_maj}")
print(f"fewer than 100 compounds:               {n_tiny}")
print(f"below-chance AUROC:                     {n_below_chance}")

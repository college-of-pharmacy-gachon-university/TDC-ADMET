# Part of the ADMET-Stack revision analysis set.
# Diagnostics for the two external sets with below-chance AUROC.
# Inputs: run artifacts; set the path constant at the top
# -*- coding: utf-8 -*-
"""Diagnose the two below-chance external results.

Admetica BBB gives AUROC 0.267 on n = 4,924 and Admetica oral bioavailability 0.384 on
n = 462. Performance that far below chance at these sample sizes is systematic, not noise.
The decisive test is label agreement: take the compounds that appear in BOTH the external
set and the corresponding TDC benchmark (before any similarity filtering) and compare their
labels. If the conventions agree, the labels should match; if the external convention is
reversed, they should be anti-correlated.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf8")

ADM = (r"G:\K-MELLODDY-Project\data\06_ADMET_Admetica"
       r"\datagrok-ai admetica main ADMET\Admetica_canonical_SMILES_all")
TDC = r"G:\K-MELLODDY-Project\data\04_TDC_data"

CASES = [
    ("BBB", os.path.join(ADM, "canonical_bbb", "canonical_bbb.csv"),
     os.path.join(TDC, "BBB_Martins.csv")),
    ("Bioavailability", os.path.join(ADM, "canonical_bioavailability",
                                     "canonical_bioavailability.csv"),
     os.path.join(TDC, "Bioavailability_Ma.csv")),
    ("hERG (control)", os.path.join(ADM, "canonical_herg", "canonical_herg.csv"),
     os.path.join(TDC, "hERG.csv")),
]


def label_col(df):
    for c in ("Label", "Y", "label", "y", "Activity", "Class", "class", "Bioavailability", "hERG", "pIC50"):
        if c in df.columns:
            return c
    raise KeyError(f"no label column in {list(df.columns)[:8]}")


def smiles_col(df):
    for c in ("Canonical_SMILES", "SMILES", "Smiles", "Drug", "smiles"):
        if c in df.columns:
            return c
    raise KeyError(f"no SMILES column in {list(df.columns)[:8]}")


def canon_tdc(df):
    """The TDC reference files may not carry a canonical column; add one if needed."""
    if "Canonical_SMILES" in df.columns:
        return df
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    sc = smiles_col(df)
    df = df.copy()
    df["Canonical_SMILES"] = [
        (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(str(s)))
        for s in df[sc]
    ]
    return df.dropna(subset=["Canonical_SMILES"])


for name, ext_path, tdc_path in CASES:
    print("=" * 92)
    print(name)
    if not (os.path.exists(ext_path) and os.path.exists(tdc_path)):
        print("  file missing:", ext_path if not os.path.exists(ext_path) else tdc_path)
        continue

    ext = pd.read_csv(ext_path, encoding="utf-8-sig")
    tdc = pd.read_csv(tdc_path, encoding="utf-8-sig")
    print(f"  external columns : {list(ext.columns)[:7]}")
    print(f"  TDC columns      : {list(tdc.columns)[:7]}")

    el, tl = label_col(ext), label_col(tdc)
    try:
        tdc = canon_tdc(tdc)
    except ImportError:
        print("  rdkit unavailable; matching on the raw SMILES column instead")
        tdc["Canonical_SMILES"] = tdc[smiles_col(tdc)]

    e = ext[["Canonical_SMILES", el]].dropna().drop_duplicates("Canonical_SMILES")
    t = tdc[["Canonical_SMILES", tl]].dropna().drop_duplicates("Canonical_SMILES")
    e.columns = ["smi", "ext"]
    t.columns = ["smi", "tdc"]

    print(f"  external n = {len(e):6d}   positives {int((e.ext == 1).sum()):6d} "
          f"({(e.ext == 1).mean():.3f})")
    print(f"  TDC      n = {len(t):6d}   positives {int((t.tdc == 1).sum()):6d} "
          f"({(t.tdc == 1).mean():.3f})")

    m = e.merge(t, on="smi", how="inner")
    if len(m) < 10:
        print(f"  only {len(m)} shared structures - cannot diagnose from overlap")
        continue

    agree = int((m.ext == m.tdc).sum())
    print(f"\n  shared structures: {len(m)}")
    print(f"    labels agree   : {agree:5d}  ({agree / len(m):.1%})")
    print(f"    labels disagree: {len(m) - agree:5d}  ({1 - agree / len(m):.1%})")
    ct = pd.crosstab(m.tdc, m.ext)
    ct.index.name, ct.columns.name = "TDC", "external"
    print("\n" + "\n".join("    " + ln for ln in ct.to_string().split("\n")))

    if agree / len(m) < 0.35:
        print("\n  >> The labels are ANTI-correlated: the external convention is reversed "
              "relative to TDC.")
    elif agree / len(m) > 0.65:
        print("\n  >> The labels agree: polarity is NOT the explanation; look at the "
              "feature-column alignment instead.")
    else:
        print("\n  >> Inconclusive from overlap alone.")

# Part of the ADMET-Stack revision analysis set.
# Table S38: bootstrap CIs and equivalence tests for the baseline differences.
# Inputs: none (self-contained)
# -*- coding: utf-8 -*-
"""Confidence intervals on the mean ADMET-Stack minus baseline difference.

`02_Analysis_Report.md` §3.3 recommends replacing "classification performance remained
statistically indistinguishable" with "we did not detect a difference" plus a confidence
interval, and quotes a CI of "roughly -4 % to +2 %". That figure was an illustration; it was
never computed. This script computes it.

Three quantities are produced for every baseline, separately for the classification and the
regression endpoints:

  * the mean difference with a percentile bootstrap CI over endpoints, 20,000 replicates,
    endpoints being the independent sampling units (the same resampling unit as the
    endpoint-cluster bootstrap used for the block analysis);
  * the Hodges-Lehmann median difference with its distribution-free CI, which is the
    estimate and interval that belong beside the Wilcoxon p-value already reported;
  * a two-one-sided-test equivalence result against a pre-specified margin, which is what
    licences a claim of parity rather than a failure to detect a difference.

Both the absolute difference in the metric and the relative difference in per cent are
reported, because the manuscript states gains in per cent while the metrics are bounded.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf8")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ADMET_DATA", os.path.join(os.path.dirname(HERE), "data"))
NBOOT = 20_000
RNG = np.random.default_rng(20261008)
TOST_MARGIN_PCT = 5.0        # the +/-5 % relative margin suggested in the report
FIXED_CONFIG = "M_A_E_R_MC_TI"   # the single configuration used for the selection-free row

T = json.load(open(os.path.join(DATA, "si_tables.json"), encoding="utf-8"))
CONFIGS = sorted({n for e in T for n, _, _ in T[e]["rows"]})
SCORE = {(e, n): v for e in T for n, v, _ in T[e]["rows"]}
LOWER_IS_BETTER = {e: T[e]["metric"] == "MAE" for e in T}

# Baselines as printed in Table S26. `None` where that baseline reports no value.
B = {
    "Caco2_Wang":      dict(task="R", ai=0.285, trix=0.326, mini=0.350, maplight=0.287),
    "HIA_Hou":         dict(task="C", ai=0.988, trix=0.985, mini=0.993, maplight=0.989),
    "Pgp_Broccatelli": dict(task="C", ai=0.935, trix=0.926, mini=None,  maplight=0.938),
    "Bioavailability": dict(task="C", ai=0.748, trix=0.749, mini=0.942, maplight=0.938),
    "Lipophilicity":   dict(task="R", ai=0.456, trix=0.512, mini=0.456, maplight=0.525),
    "Solubility":      dict(task="R", ai=0.761, trix=0.789, mini=0.741, maplight=0.789),
    "BBB_Martins":     dict(task="C", ai=0.915, trix=0.906, mini=0.924, maplight=0.913),
    "PPBR":            dict(task="R", ai=7.788, trix=8.200, mini=7.696, maplight=7.526),
    "VDss_Lombardo":   dict(task="R", ai=0.627, trix=0.475, mini=0.535, maplight=0.713),
    "CYP3A4_Sub.":     dict(task="C", ai=0.662, trix=0.642, mini=0.663, maplight=0.647),
    "CYP2D6_Sub.":     dict(task="C", ai=0.736, trix=0.699, mini=0.695, maplight=0.720),
    "CYP2C9_Sub.":     dict(task="C", ai=0.441, trix=0.377, mini=0.474, maplight=0.434),
    "CYP3A4_In.":      dict(task="C", ai=0.904, trix=0.884, mini=0.877, maplight=0.916),
    "CYP2D6_In.":      dict(task="C", ai=0.739, trix=0.718, mini=0.719, maplight=0.790),
    "CYP2C9_In.":      dict(task="C", ai=0.839, trix=0.789, mini=0.823, maplight=0.859),
    "Half_Life":       dict(task="R", ai=0.547, trix=0.372, mini=0.495, maplight=0.557),
    "Clearance H.":    dict(task="R", ai=0.440, trix=0.447, mini=0.446, maplight=0.498),
    "Clearance M.":    dict(task="R", ai=0.625, trix=0.552, mini=0.628, maplight=0.630),
    "LD50_Zhu":        dict(task="R", ai=0.552, trix=0.573, mini=0.585, maplight=0.633),
    "hERG":            dict(task="C", ai=0.874, trix=0.836, mini=0.846, maplight=0.880),
    "AMES":            dict(task="C", ai=0.871, trix=0.870, mini=0.849, maplight=0.869),
    "DILI":            dict(task="C", ai=0.925, trix=0.906, mini=0.956, maplight=0.917),
}
assert set(B) == set(T), set(B) ^ set(T)
BASELINES = [("ADMET-AI", "ai"), ("ADMETrix", "trix"),
             ("MiniMol", "mini"), ("MapLight+GNN", "maplight")]


def stack_value(endpoint: str, mode: str) -> float:
    if mode == "best":
        vals = [SCORE[(endpoint, c)] for c in CONFIGS]
        return min(vals) if LOWER_IS_BETTER[endpoint] else max(vals)
    return SCORE[(endpoint, mode)]


def signed(endpoint: str, ours: float, base: float, relative: bool) -> float:
    """Difference in the direction where positive always means ADMET-Stack is better."""
    diff = (base - ours) if LOWER_IS_BETTER[endpoint] else (ours - base)
    return diff / base * 100.0 if relative else diff


def boot_mean_ci(d: np.ndarray):
    """Percentile CI on the mean, resampling endpoints."""
    n = len(d)
    means = d[RNG.integers(0, n, (NBOOT, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi, means


def hodges_lehmann(d: np.ndarray):
    """The HL estimate and its distribution-free CI, matched to the signed-rank test."""
    walsh = np.array([(d[i] + d[j]) / 2
                      for i in range(len(d)) for j in range(i, len(d))])
    walsh.sort()
    est = float(np.median(walsh))
    n, m = len(d), len(walsh)
    # The normal approximation to the signed-rank statistic gives the index offset into the
    # ordered Walsh averages that bounds the interval.
    k = max(int(stats.norm.ppf(0.025) * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                + n * (n + 1) / 4), 0)
    lo = walsh[k] if k < m else walsh[0]
    hi = walsh[m - 1 - k] if k < m else walsh[-1]
    return est, float(lo), float(hi)


def tost(d: np.ndarray, margin: float):
    """Two one-sided t-tests for equivalence within +/- margin."""
    n, mean, se = len(d), d.mean(), stats.sem(d)
    if se == 0:
        return (0.0, 0.0, True) if abs(mean) < margin else (1.0, 1.0, False)
    t_lo = (mean - (-margin)) / se          # H0: mean <= -margin
    t_hi = (mean - margin) / se             # H0: mean >= +margin
    p_lo = stats.t.sf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return float(p_lo), float(p_hi), bool(max(p_lo, p_hi) < 0.05)


def analyse(mode: str, label: str, relative: bool):
    unit = "%" if relative else "metric units"
    print("\n" + "=" * 108)
    print(f"ADMET-Stack value = {label}   |   difference in {unit}   "
          f"|   positive favours ADMET-Stack")
    print("=" * 108)
    print(f"{'baseline':15s}{'task':6s}{'n':>3s}{'mean':>9s}"
          f"{'bootstrap 95% CI':>22s}{'HL median':>11s}{'HL 95% CI':>20s}"
          f"{'Wilcoxon p':>12s}{'TOST':>7s}")
    out = []
    for name, key in BASELINES:
        for task, tag in (("classification", "C"), ("regression", "R")):
            d = np.array([signed(e, stack_value(e, mode), B[e][key], relative)
                          for e in B
                          if B[e][key] is not None and B[e]["task"] == tag])
            lo, hi, _ = boot_mean_ci(d)
            hl, hl_lo, hl_hi = hodges_lehmann(d)
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            margin = TOST_MARGIN_PCT if relative else 0.02
            p_lo, p_hi, equiv = tost(d, margin)
            print(f"{name:15s}{tag:6s}{len(d):3d}{d.mean():+9.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>22s}{hl:+11.3f}"
                  f"{f'[{hl_lo:+.3f}, {hl_hi:+.3f}]':>20s}{p:12.3f}"
                  f"{('yes' if equiv else 'no'):>7s}")
            out.append(dict(selection=label, scale=unit, baseline=name, task=task,
                            n_endpoints=len(d), mean=round(d.mean(), 4),
                            ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                            hl_median=round(hl, 4), hl_lo=round(hl_lo, 4),
                            hl_hi=round(hl_hi, 4), wilcoxon_p=round(p, 4),
                            tost_margin=margin, tost_p=round(max(p_lo, p_hi), 4),
                            equivalent=equiv,
                            ci_excludes_zero=bool(lo > 0 or hi < 0)))
    return out


rows = []
for mode, label in (("best", "per-endpoint best (as published)"),
                    (FIXED_CONFIG, f"fixed {FIXED_CONFIG} (selection-free)")):
    for relative in (True, False):
        rows += analyse(mode, label, relative)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "out_classification_difference_CIs.csv"), index=False)

print("\n" + "=" * 108)
print("the sentence Section 3.3 needs")
print("=" * 108)
for mode_label in res.selection.unique():
    for scale in res.scale.unique():
        g = res[(res.selection == mode_label) & (res.scale == scale)
                & (res.task == "classification")]
        print(f"\n{mode_label}, classification, {scale}:")
        for _, r in g.iterrows():
            verdict = ("a difference IS detected" if r.ci_excludes_zero
                       else "no difference detected")
            eq = (f"and equivalence within {r.tost_margin:g} {scale} IS established"
                  if r.equivalent else
                  f"and equivalence within {r.tost_margin:g} {scale} is NOT established")
            print(f"  vs {r['baseline']:14s} mean {r['mean']:+.2f} "
                  f"95% CI [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  -> {verdict}, {eq}")

print("\n-> data_difference_CIs.csv")

# %% [markdown]
# # 06 - Evaluation on external datasets
#
# Scores the stored ADMET-Stack ensembles on an external dataset that has been canonicalized and
# similarity-filtered by script 01 and featurized by script 03.
#
# ## Reporting basis
#
# Table S31 of the Supplementary Information reports, for each external evaluation set, the **mean
# over the retained runs** rather than the result of any single run. This script therefore scores
# **every** `model_seed*.pkl` found for the endpoint and configuration and reports mean +/- SD
# across those runs. Where only one run was retained — which is the case for the AutoML rows, whose
# original artifacts could not be reproduced — the SD is undefined and the row is a single stored
# model. The script states which case applies.
#
# Bootstrap intervals follow the same definition as the paper: compounds are resampled, the metric
# is computed for each run on the resampled compounds, and those per-run values are averaged inside
# each replicate before the percentiles are taken.
#
# Scoring a single seed instead would not reproduce the published table, so `MODEL_SEEDS` defaults
# to every seed present.
#
# The original implementation selected feature columns by hard-coded index arithmetic, which is
# correct only if the external matrix was built with exactly the same blocks in the same order, and
# which fails silently otherwise because the subsequent reindex zero-fills rather than raising. Here
# the matrices are required to agree, and a mismatch is an error.
#
# Reported alongside each metric:
#
# * the positive-class **prevalence**, which is the AUPRC of a constant predictor and the correct
#   reference point for reading an AUPRC
# * the **majority-class rate**, which is the accuracy of a constant predictor
# * **balanced accuracy** and the **Matthews correlation coefficient**, which do not depend on the
#   class balance
# * **bootstrap 95% confidence intervals**, because several external sets are small
#
# Manuscript: Sections 2.8.1, 2.8.3 and 3.5. Supplementary: Table S31 and Table S31 (continued).

# %%
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "admet_config.py").exists())
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score)

import admet_config as cfg

# ------------------------------------------------------------------ settings
DATASET_NAME = "pgp_AutoML"
CONFIGURATION = "M_A_E_R_MC"
TDC_ENDPOINT = "Pgp_Broccatelli"
MODEL_SEEDS = None        # None = every model_seed*.pkl present; or e.g. [1, 2, 3, 4, 5]
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(0)

FEAT_DIR = cfg.EXTERNAL_DIR / "features" / DATASET_NAME / CONFIGURATION
MODEL_DIR = cfg.model_dir(TDC_ENDPOINT, CONFIGURATION)
OUT_DIR = cfg.EXTERNAL_DIR / "results" / DATASET_NAME / CONFIGURATION
OUT_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_SEEDS is None:
    model_paths = sorted(MODEL_DIR.glob("model_seed*.pkl"),
                         key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
else:
    model_paths = [MODEL_DIR / f"model_seed{s}.pkl" for s in MODEL_SEEDS]

if not model_paths:
    raise FileNotFoundError(f"No model_seed*.pkl under {MODEL_DIR}")

print(f"dataset       : {DATASET_NAME}")
print(f"configuration : {CONFIGURATION}")
print(f"runs found    : {len(model_paths)} ({', '.join(p.stem for p in model_paths)})")
if len(model_paths) == 1:
    print("               only one run is retained, so this row is a single stored model and the "
          "SD is undefined")

blocks = CONFIGURATION.split("_")
if "EM32" in blocks or "EM64" in blocks:
    print("\nNOTE: this configuration contains a random SELFIES projection. Unless script 03 "
          "loaded the projection saved by the benchmark builder, those columns were produced "
          "by a different random map than the model was trained with and carry no usable "
          "information. Prefer a projection-free configuration for external evaluation.")

# %%
X_raw = pd.read_csv(FEAT_DIR / f"{DATASET_NAME}_fp_features_scaled.csv")
meta = pd.read_csv(FEAT_DIR / f"{DATASET_NAME}_fp_features_raw.csv",
                   usecols=["Canonical_SMILES", "Label"])
y = meta["Label"].values
X_raw.columns = X_raw.columns.astype(str)


def aligned(model):
    """Return the external matrix in the column order the model was fitted with."""
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        return X_raw
    if len(expected) != X_raw.shape[1]:
        raise ValueError(
            f"Feature-count mismatch: the model expects {len(expected)} columns but the "
            f"external matrix has {X_raw.shape[1]}. Rebuild the external features with "
            f"CONFIGURATION = '{CONFIGURATION}' in script 03."
        )
    missing = [c for c in expected if c not in X_raw.columns]
    if missing:
        raise ValueError(f"{len(missing)} expected columns are absent, e.g. {missing[:5]}. "
                         "Silently zero-filling them would invalidate the evaluation.")
    return X_raw[expected]


proba, pred = [], []
for p in model_paths:
    m = joblib.load(p)
    Xm = aligned(m)
    proba.append(m.predict_proba(Xm)[:, 1])
    pred.append(m.predict(Xm))
proba = np.vstack(proba)          # runs x compounds
pred = np.vstack(pred)

print(f"{len(y)} compounds, {X_raw.shape[1]} features, {proba.shape[0]} run(s)")

prevalence = float(np.mean(y))
majority = float(max(prevalence, 1 - prevalence))

METRICS = ("ACC", "Balanced_ACC", "MCC", "AUROC", "AUPRC")


def run_scores(idx, run):
    """Metrics for one run on one set of compound indices, or None if a class is absent."""
    yy = y[idx]
    if len(np.unique(yy)) < 2:
        return None
    pp, dd = proba[run][idx], pred[run][idx]
    return dict(ACC=accuracy_score(yy, dd),
                Balanced_ACC=balanced_accuracy_score(yy, dd),
                MCC=matthews_corrcoef(yy, dd),
                AUROC=roc_auc_score(yy, pp),
                AUPRC=average_precision_score(yy, pp))


def averaged(idx):
    """Mean of each metric across runs, as the paper reports it."""
    per_run = [run_scores(idx, r) for r in range(proba.shape[0])]
    per_run = [s for s in per_run if s]
    if not per_run:
        return None, None
    return ({k: float(np.mean([s[k] for s in per_run])) for k in METRICS},
            {k: float(np.std([s[k] for s in per_run], ddof=0)) for k in METRICS})


full = np.arange(len(y))
scores, sds = averaged(full)
boot = [s for s in (averaged(RNG.integers(0, len(y), len(y)))[0] for _ in range(N_BOOTSTRAP)) if s]
ci = {k: np.percentile([b[k] for b in boot], [2.5, 97.5]) for k in METRICS}

row = {
    "Dataset": DATASET_NAME,
    "Configuration": CONFIGURATION,
    "Contains_projection_block": ("EM32" in blocks or "EM64" in blocks),
    "N_runs": int(proba.shape[0]),
    "Basis": "mean of %d runs" % proba.shape[0] if proba.shape[0] > 1 else "single stored model",
    "N_total": len(y),
    "N_active": int(np.sum(y == 1)),
    "N_inactive": int(np.sum(y == 0)),
    "Prevalence": round(prevalence, 4),
    "Majority_class_rate": round(majority, 4),
}
for k in METRICS:
    row[k] = round(scores[k], 4)
    row[f"{k}_SD"] = round(sds[k], 4) if proba.shape[0] > 1 else ""
    row[f"{k}_CI95"] = f"[{ci[k][0]:.4f}, {ci[k][1]:.4f}]"
row["AUPRC_lift_over_prevalence"] = round(scores["AUPRC"] - prevalence, 4)

pd.DataFrame([row]).to_csv(OUT_DIR / "evaluation_metrics.csv", index=False)
pd.DataFrame({"Canonical_SMILES": meta["Canonical_SMILES"],
              "y_true": y,
              "y_pred_mean_vote": pred.mean(axis=0),
              "y_proba_mean": proba.mean(axis=0)}).to_csv(OUT_DIR / "predictions.csv", index=False)

for k, v in row.items():
    print(f"  {k:30s} {v}")

print()
if scores["ACC"] < majority:
    print("Accuracy is below the majority-class rate while the ranking metrics may still be "
          "informative. That is a decision-threshold effect inherited from the class balance "
          "of the training data, not a failure of discrimination - report balanced accuracy "
          "and MCC alongside it.")
if scores["AUROC"] < 0.5:
    print(f"AUROC is below chance ({scores['AUROC']:.3f}). At this sample size that indicates "
          f"a systematic cause rather than noise. Check the label polarity of the source file "
          f"(inverted would give {1 - scores['AUROC']:.3f}) and the feature-column alignment "
          f"before reporting it.")
print("\nWritten to:", OUT_DIR)

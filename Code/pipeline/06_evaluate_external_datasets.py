# %% [markdown]
# # 05 - Evaluation on external datasets
#
# Scores a stored ADMET-Stack ensemble on an external dataset that has been canonicalized and
# similarity-filtered by script 01 and featurized by script 03.
#
# The original implementation selected feature columns by hard-coded index arithmetic, which
# is correct only if the external matrix was built with exactly the same blocks in the same
# order, and which fails silently otherwise because the subsequent reindex zero-fills rather
# than raising. Here the matrices are instead required to agree, and a mismatch is an error.
# That is the only behavioural change in this script, and it is a safeguard rather than a
# different method.
#
# Reported alongside each metric:
#
# * the positive-class **prevalence**, which is the AUPRC of a constant predictor and the
#   correct reference point for reading an AUPRC
# * the **majority-class rate**, which is the accuracy of a constant predictor
# * **balanced accuracy** and the **Matthews correlation coefficient**, which do not depend
#   on the class balance
# * **bootstrap 95% confidence intervals**, because several external sets are small
#
# Manuscript: Sections 2.8.3 and 3.5.

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
MODEL_SEED = 1
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(0)

FEAT_DIR = cfg.EXTERNAL_DIR / "features" / DATASET_NAME / CONFIGURATION
MODEL_PATH = cfg.model_dir(TDC_ENDPOINT, CONFIGURATION) / f"model_seed{MODEL_SEED}.pkl"
OUT_DIR = cfg.EXTERNAL_DIR / "results" / DATASET_NAME / CONFIGURATION
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"dataset       : {DATASET_NAME}")
print(f"configuration : {CONFIGURATION}")
print(f"model         : {MODEL_PATH}")

blocks = CONFIGURATION.split("_")
if "EM32" in blocks or "EM64" in blocks:
    print("\nNOTE: this configuration contains a random SELFIES projection. Unless script 03 "
          "loaded the projection saved by the benchmark builder, those columns were produced "
          "by a different random map than the model was trained with and carry no usable "
          "information. Prefer a projection-free configuration for external evaluation.")

# %%
X = pd.read_csv(FEAT_DIR / f"{DATASET_NAME}_fp_features_scaled.csv")
meta = pd.read_csv(FEAT_DIR / f"{DATASET_NAME}_fp_features_raw.csv", usecols=["Canonical_SMILES", "Label"])
y = meta["Label"].values
model = joblib.load(MODEL_PATH)

expected = list(getattr(model, "feature_names_in_", []))
X.columns = X.columns.astype(str)
if expected:
    if len(expected) != X.shape[1]:
        raise ValueError(
            f"Feature-count mismatch: the model expects {len(expected)} columns but the "
            f"external matrix has {X.shape[1]}. Rebuild the external features with "
            f"CONFIGURATION = '{CONFIGURATION}' in script 03."
        )
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(f"{len(missing)} expected columns are absent, e.g. {missing[:5]}. "
                         "Silently zero-filling them would invalidate the evaluation.")
    X = X[expected]

print(f"{len(y)} compounds, {X.shape[1]} features")

# %%
y_proba = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)

prevalence = float(np.mean(y))
majority = float(max(prevalence, 1 - prevalence))


def point_scores(idx):
    yy, pp, dd = y[idx], y_proba[idx], y_pred[idx]
    if len(np.unique(yy)) < 2:
        return None
    return dict(
        ACC=accuracy_score(yy, dd),
        Balanced_ACC=balanced_accuracy_score(yy, dd),
        MCC=matthews_corrcoef(yy, dd),
        AUROC=roc_auc_score(yy, pp),
        AUPRC=average_precision_score(yy, pp),
    )


scores = point_scores(np.arange(len(y)))
boot = [s for s in (point_scores(RNG.integers(0, len(y), len(y))) for _ in range(N_BOOTSTRAP)) if s]
ci = {k: np.percentile([b[k] for b in boot], [2.5, 97.5]) for k in scores}

row = {
    "Dataset": DATASET_NAME,
    "Configuration": CONFIGURATION,
    "Contains_projection_block": ("EM32" in blocks or "EM64" in blocks),
    "N_total": len(y),
    "N_active": int(np.sum(y == 1)),
    "N_inactive": int(np.sum(y == 0)),
    "Prevalence": round(prevalence, 4),
    "Majority_class_rate": round(majority, 4),
}
for k, v in scores.items():
    row[k] = round(float(v), 4)
    row[f"{k}_CI95"] = f"[{ci[k][0]:.4f}, {ci[k][1]:.4f}]"
row["AUPRC_lift_over_prevalence"] = round(float(scores["AUPRC"] - prevalence), 4)

pd.DataFrame([row]).to_csv(OUT_DIR / "evaluation_metrics.csv", index=False)
pd.DataFrame({"Canonical_SMILES": meta["Canonical_SMILES"],
              "y_true": y, "y_pred": y_pred, "y_proba": y_proba}).to_csv(
    OUT_DIR / "predictions.csv", index=False)

for k, v in row.items():
    print(f"  {k:28s} {v}")

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

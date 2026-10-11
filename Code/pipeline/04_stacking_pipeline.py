# %% [markdown]
# # 03 - Stacked-ensemble training and evaluation
#
# Trains the ADMET-Stack ensemble for a given endpoint and configuration, over five learner
# seeds, and writes the per-compound test predictions and the per-seed metrics.
#
# The architecture is identical for every endpoint and every configuration; only the input
# feature matrix changes. That is what makes the 54-configuration comparison a comparison of
# representations rather than of models.
#
# * base learners - ExtraTrees, RandomForest, LightGBM, XGBoost
# * meta-learner  - LogisticRegression (classification) or Ridge (regression)
# * stacking      - scikit-learn `Stacking*` with 10-fold out-of-fold meta-features
# * fitting data  - the pooled training and validation partitions; the official TDC test
#                   partition is never used for fitting
# * repeats       - learner `random_state` 1..5 on the same partition. These are **model
#                   seeds, not split seeds**; the resulting standard deviation describes
#                   run-to-run reproducibility and is not comparable with the TDC
#                   leaderboard's split-seed standard deviations.
#
# **Hyperparameters are left at their library defaults, deliberately.** Tuning each
# configuration separately would confound representation quality with tuning effort, so a
# more heavily tuned configuration would look like a better representation. The optional
# Optuna study is in `s05_hyperparameter_search.py` and was not used for the benchmark.
#
# Manuscript: Section 2.4.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "admet_config.py").exists())
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor

import admet_config as cfg

# ------------------------------------------------------------------ settings
ENDPOINTS_TO_RUN = list(cfg.ENDPOINTS)
CONFIGS_TO_RUN = list(cfg.CONFIGURATIONS)
DEVICE = "cpu"          # "cuda" to use a GPU for the gradient-boosting learners
SAVE_MODELS = False     # the fitted ensembles are large; predictions are what matter
print(f"DATA_ROOT: {cfg.DATA_ROOT}")
print(f"runs to perform: {len(ENDPOINTS_TO_RUN) * len(CONFIGS_TO_RUN)} x {len(cfg.SEEDS)} seeds")


# %%
def make_stack(task: str, seed: int):
    if task == "classification":
        base = [
            ("et",   ExtraTreesClassifier(random_state=seed, n_jobs=-1)),
            ("rf",   RandomForestClassifier(random_state=seed, n_jobs=-1)),
            ("lgbm", LGBMClassifier(device="cpu", random_state=seed, force_col_wise=True, verbose=-1)),
            ("xgb",  XGBClassifier(tree_method="hist", device=DEVICE,
                                   eval_metric="logloss", random_state=seed)),
        ]
        return StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(solver="liblinear", random_state=seed),
            cv=cfg.STACKING_CV, n_jobs=-1)
    base = [
        ("et",   ExtraTreesRegressor(random_state=seed, n_jobs=-1)),
        ("rf",   RandomForestRegressor(random_state=seed, n_jobs=-1)),
        ("lgbm", LGBMRegressor(device="cpu", random_state=seed, force_col_wise=True, verbose=-1)),
        ("xgb",  XGBRegressor(tree_method="hist", device=DEVICE,
                              random_state=seed, eval_metric="rmse")),
    ]
    return StackingRegressor(estimators=base, final_estimator=Ridge(random_state=seed),
                             cv=cfg.STACKING_CV, n_jobs=-1)


def load_split(endpoint: str, configuration: str):
    d = cfg.feature_dir(endpoint, configuration)
    out = {}
    for part in ("train", "valid", "test"):
        X = pd.read_csv(d / f"{endpoint}_{part}_fp_features_scaled.csv")
        y = pd.read_csv(d / f"{endpoint}_{part}_fp_features_raw.csv")["Label"]
        out[part] = (X, y)
    return out


def score_classification(y, y_pred, y_proba) -> dict:
    return dict(
        AUROC=roc_auc_score(y, y_proba),
        AUPRC=average_precision_score(y, y_proba),
        Accuracy=accuracy_score(y, y_pred),
        Precision=precision_score(y, y_pred, zero_division=0),
        Recall=recall_score(y, y_pred, zero_division=0),
    )


def score_regression(y, y_pred) -> dict:
    mse = mean_squared_error(y, y_pred)
    return dict(
        MAE=mean_absolute_error(y, y_pred),
        MSE=mse,                      # NOTE: the published tables label this column RMSE
        RMSE=float(np.sqrt(mse)),     #       the true RMSE is this one
        R2=r2_score(y, y_pred),
        Spearman=spearmanr(y, y_pred).statistic,
    )


def run_one(endpoint: str, configuration: str) -> pd.DataFrame | None:
    meta = cfg.ENDPOINTS[endpoint]
    task = meta["task"]
    try:
        data = load_split(endpoint, configuration)
    except FileNotFoundError:
        print(f"  missing features: {endpoint} / {configuration}")
        return None

    X_full = pd.concat([data["train"][0], data["valid"][0]], axis=0, ignore_index=True)
    y_full = np.concatenate([data["train"][1], data["valid"][1]])
    X_test, y_test = data["test"]

    out_dir = cfg.result_dir(endpoint, configuration)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in cfg.SEEDS:
        stack = make_stack(task, seed)
        stack.fit(X_full, y_full)

        if task == "classification":
            y_pred = stack.predict(X_test)
            y_proba = stack.predict_proba(X_test)[:, 1]
            scores = score_classification(y_test, y_pred, y_proba)
            preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba})
        else:
            y_pred = stack.predict(X_test)
            scores = score_regression(y_test, y_pred)
            preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

        preds.to_csv(out_dir / f"predictions_seed{seed}.csv", index=False)
        if SAVE_MODELS:
            joblib.dump(stack, cfg.model_dir(endpoint, configuration) / f"model_seed{seed}.pkl")

        rows.append(dict(Seed=seed, **scores))
        print("   seed %d  " % seed + "  ".join(f"{k} {v:.4f}" for k, v in scores.items()))

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "all_results.csv", index=False)
    summary = res.drop(columns=["Seed"]).agg(["mean", "std"]).T
    summary["Mean ± SD"] = summary.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    summary.to_csv(out_dir / "summary.csv")

    pd.DataFrame([{
        "Split": s, "X_shape": str(data[p][0].shape), "y_shape": str(data[p][1].shape)
    } for s, p in [("Train", "train"), ("Valid", "valid"), ("Test", "test")]]).to_csv(
        out_dir / "data_shapes.csv", index=False)
    return res


# %%
for endpoint in ENDPOINTS_TO_RUN:
    for configuration in CONFIGS_TO_RUN:
        done = cfg.result_dir(endpoint, configuration) / "summary.csv"
        if done.exists():
            print(f"skip (done)  {endpoint:34s} {configuration}")
            continue
        print(f"\n=== {endpoint}  /  {configuration}  ({cfg.ENDPOINTS[endpoint]['task']}) ===")
        run_one(endpoint, configuration)

print("\nDone. The per-compound predictions under DATA_ROOT/results/ are what make every "
      "number in the paper independently reproducible without retraining.")

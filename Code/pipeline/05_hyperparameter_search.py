# %% [markdown]
# # 04 - Optional hyperparameter search (auxiliary study)
#
# **This was not used for the 54-configuration benchmark.** The benchmark deliberately uses
# untuned learner hyperparameters, so that performance differences between configurations
# reflect the representation rather than how much tuning effort each one received. This
# script reproduces the auxiliary Optuna study reported in Section 2.4 and Tables 3, 4 and
# S23 of the manuscript.
#
# Protocol, unchanged from the original study:
#
# * 30 trials per base learner, scored by 5-fold cross-validation
# * 10 trials for the meta-learner, scored by 10-fold cross-validation
# * **every search runs on the training partition only** - no validation or test data enters
#   the search, so the procedure is leakage-free
#
# Manuscript: Section 2.4.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "admet_config.py").exists())
sys.path.insert(0, str(ROOT / "src"))

import json

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

import admet_config as cfg

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------ settings
ENDPOINT = "CYP3A4_Veith"
CONFIGURATION = "M_A_E_R_TI"
N_TRIALS_BASE = 30
N_TRIALS_META = 10
CV_BASE = 5
CV_META = 10
SEED = 42

TASK = cfg.ENDPOINTS[ENDPOINT]["task"]
SCORING = "roc_auc" if TASK == "classification" else "neg_mean_absolute_error"
OUT_DIR = cfg.model_dir(ENDPOINT, CONFIGURATION) / "hyperparameter_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

d = cfg.feature_dir(ENDPOINT, CONFIGURATION)
X_train = pd.read_csv(d / f"{ENDPOINT}_train_fp_features_scaled.csv")
y_train = pd.read_csv(d / f"{ENDPOINT}_train_fp_features_raw.csv")["Label"]
print(f"{ENDPOINT} / {CONFIGURATION}  ({TASK})   training set: {X_train.shape}")
print("All searches use the training partition only.")


# %% [markdown]
# ## Base learners - 30 trials each, 5-fold cross-validation on the training partition

# %%
def suggest_tree(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_categorical("max_depth", [None, 20, 40]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        random_state=SEED, n_jobs=1,
    )


def suggest_boost(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 30, 150),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=SEED,
    )


def suggest_xgb(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        random_state=SEED,
    )


CLS = {
    "ExtraTrees":   (ExtraTreesClassifier,   suggest_tree,  {}),
    "RandomForest": (RandomForestClassifier, suggest_tree,  {}),
    "LightGBM":     (LGBMClassifier,         suggest_boost, dict(verbose=-1, force_col_wise=True)),
    "XGBoost":      (XGBClassifier,          suggest_xgb,   dict(tree_method="hist", eval_metric="logloss")),
}
REG = {
    "ExtraTrees":   (ExtraTreesRegressor,   suggest_tree,  {}),
    "RandomForest": (RandomForestRegressor, suggest_tree,  {}),
    "LightGBM":     (LGBMRegressor,         suggest_boost, dict(verbose=-1, force_col_wise=True)),
    "XGBoost":      (XGBRegressor,          suggest_xgb,   dict(tree_method="hist", eval_metric="rmse")),
}

best_params, best_models = {}, []
for name, (klass, suggest, fixed) in (CLS if TASK == "classification" else REG).items():
    def objective(trial, klass=klass, suggest=suggest, fixed=fixed):
        model = klass(**{**suggest(trial), **fixed})
        return cross_val_score(model, X_train, y_train, cv=CV_BASE,
                               scoring=SCORING, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(objective, n_trials=N_TRIALS_BASE, n_jobs=1, show_progress_bar=False)

    best_params[name] = study.best_params
    model = klass(**{**study.best_params, **fixed, "random_state": SEED})
    model.fit(X_train, y_train)
    joblib.dump(model, OUT_DIR / f"best_{name}.pkl")
    best_models.append((name.lower()[:4], model))
    print(f"{name:14s} best {SCORING} = {study.best_value:.4f}")

(OUT_DIR / "best_base_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")


# %% [markdown]
# ## Meta-learner - 10 trials, 10-fold cross-validation on the training partition

# %%
cv = (StratifiedKFold(n_splits=CV_META, shuffle=True, random_state=SEED)
      if TASK == "classification" else KFold(n_splits=CV_META, shuffle=True, random_state=SEED))


def objective_meta(trial):
    if TASK == "classification":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear" if penalty == "l1" else trial.suggest_categorical("solver", ["lbfgs", "saga"])
        meta = LogisticRegression(C=trial.suggest_float("C", 1e-3, 5, log=True),
                                  penalty=penalty, solver=solver,
                                  random_state=SEED, max_iter=1000)
        stack = StackingClassifier(estimators=best_models, final_estimator=meta, cv=cv, n_jobs=-1)
    else:
        meta = Ridge(alpha=trial.suggest_float("alpha", 1e-4, 100, log=True),
                     solver=trial.suggest_categorical("solver", ["auto", "svd", "lsqr", "saga"]),
                     random_state=SEED)
        stack = StackingRegressor(estimators=best_models, final_estimator=meta, cv=cv, n_jobs=-1)
    return cross_val_score(stack, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1).mean()


study_meta = optuna.create_study(direction="maximize", study_name="meta")
study_meta.optimize(objective_meta, n_trials=N_TRIALS_META, n_jobs=1, show_progress_bar=False)

print("best meta parameters:", study_meta.best_params)
(OUT_DIR / "best_meta_params.json").write_text(json.dumps(study_meta.best_params, indent=2),
                                               encoding="utf-8")

pd.DataFrame([{"component": k, **v} for k, v in best_params.items()]).to_csv(
    OUT_DIR / "best_base_params.csv", index=False)
print("\nWritten to:", OUT_DIR)
print("These results belong to the auxiliary tuning study; the benchmark in script 04 uses "
      "untuned defaults by design.")

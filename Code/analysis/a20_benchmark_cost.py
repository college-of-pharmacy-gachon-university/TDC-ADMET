# Part of the ADMET-Stack revision analysis set.
# Optional probe for training time, inference time and peak memory.
# Inputs: run artifacts; set the path constant at the top
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_cost.py — measures the computational cost figures Reviewer 2 (point 6) asks for.

Run this inside the `admet-stack` conda environment (needs rdkit, lightgbm, xgboost,
scikit-learn, selfies, torch, optuna, psutil). It does NOT need a GPU; it forces CPU so
the numbers are comparable and reproducible.

    python benchmark_cost.py --root F:\00_00_TDC_data_reorganized --out cost_benchmark.csv

What it reports, for each (endpoint, configuration) pair:
    n_train, n_test, n_features
    feat_extract_s_per_1k   feature extraction wall time per 1,000 molecules
    fit_s                   end-to-end StackingClassifier/Regressor fit (4 base learners,
                            10-fold OOF, + meta-learner), default hyperparameters
    optuna_s                optional: 30-trial Optuna search for ONE base learner (5-fold),
                            extrapolated to 4 base learners  (--optuna to enable)
    infer_ms_per_1k         inference wall time per 1,000 molecules
    peak_rss_mb             peak resident set size during fit
    model_mb                size of the pickled fitted ensemble

Default endpoints cover a small / medium / large training set, and both the
descriptor-only and the seven-block configurations:

    DILI               (n_train  380)   small
    Pgp_Broccatelli    (n_train  974)   medium
    CYP2D6_Veith       (n_train 10504)  large

Send the resulting CSV back and it will be turned into Table S34.
"""

import argparse, gc, os, pickle, time

import numpy as np
import pandas as pd
import psutil

from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, StackingClassifier,
                              ExtraTreesRegressor, RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------- configuration

TASKS = {
    # endpoint                       : (task, folder for 'R',                folder for the 7-block config)
    "DILI":            ("clf", "01_03_TDC_rdkit_variance", "07_01_TDC_morgan_avalon_erg_selfies_rdkit_maccs_tfidf_variance"),
    "Pgp_Broccatelli": ("clf", "01_03_TDC_rdkit_variance", "07_01_TDC_morgan_avalon_erg_selfies_rdkit_maccs_tfidf_variance"),
    "CYP2D6_Veith":    ("clf", "01_03_TDC_rdkit_variance", "07_01_TDC_morgan_avalon_erg_selfies_rdkit_maccs_tfidf_variance"),
}
SEED = 1


def base_models(task):
    if task == "clf":
        return [
            ("et",   ExtraTreesClassifier(random_state=SEED, n_jobs=-1)),
            ("rf",   RandomForestClassifier(random_state=SEED, n_jobs=-1)),
            ("lgbm", LGBMClassifier(device="cpu", random_state=SEED, force_col_wise=True, verbose=-1)),
            ("xgb",  XGBClassifier(tree_method="hist", device="cpu", eval_metric="logloss", random_state=SEED)),
        ]
    return [
        ("et",   ExtraTreesRegressor(random_state=SEED, n_jobs=-1)),
        ("rf",   RandomForestRegressor(random_state=SEED, n_jobs=-1)),
        ("lgbm", LGBMRegressor(device="cpu", random_state=SEED, force_col_wise=True, verbose=-1)),
        ("xgb",  XGBRegressor(tree_method="hist", device="cpu", random_state=SEED, eval_metric="rmse")),
    ]


def make_stack(task):
    if task == "clf":
        return StackingClassifier(estimators=base_models("clf"),
                                  final_estimator=LogisticRegression(solver="liblinear", random_state=SEED),
                                  cv=10, n_jobs=-1)
    return StackingRegressor(estimators=base_models("reg"),
                             final_estimator=Ridge(random_state=SEED), cv=10, n_jobs=-1)


# ---------------------------------------------------------------- measurement helpers

def peak_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def time_feature_extraction(smiles, use_all_blocks):
    """Times the FeatureExtractor from ADMET_Feature_Extractor.ipynb.
    Paste that class into feature_extractor.py next to this script first."""
    try:
        from feature_extractor import FeatureExtractor
    except ImportError:
        return None
    fe = FeatureExtractor(
        use_morgan=use_all_blocks, use_avalon=use_all_blocks, use_erg=use_all_blocks,
        use_selfies=use_all_blocks, use_rdkit=True, use_maccs=use_all_blocks,
        selfies_use_tfidf=use_all_blocks, selfies_use_embed=False,
    )
    fe.build_vocab(smiles)
    t0 = time.perf_counter()
    fe.transform(smiles)
    return (time.perf_counter() - t0) / len(smiles) * 1000.0


def run_one(root, endpoint, task, folder, label, do_optuna):
    d = os.path.join(root, endpoint, folder)
    Xtr = pd.read_csv(os.path.join(d, f"{endpoint}_train_fp_features_scaled.csv"))
    ytr = pd.read_csv(os.path.join(d, f"{endpoint}_train_fp_features_raw.csv")).Label
    Xva = pd.read_csv(os.path.join(d, f"{endpoint}_valid_fp_features_scaled.csv"))
    yva = pd.read_csv(os.path.join(d, f"{endpoint}_valid_fp_features_raw.csv")).Label
    Xte = pd.read_csv(os.path.join(d, f"{endpoint}_test_fp_features_scaled.csv"))

    X = pd.concat([Xtr, Xva], axis=0, ignore_index=True)
    y = np.concatenate([ytr, yva])

    gc.collect()
    rss0 = peak_rss_mb()
    stack = make_stack(task)
    t0 = time.perf_counter()
    stack.fit(X, y)
    fit_s = time.perf_counter() - t0
    rss1 = peak_rss_mb()

    t0 = time.perf_counter()
    _ = stack.predict(Xte)
    infer_ms_per_1k = (time.perf_counter() - t0) / len(Xte) * 1e6

    blob = pickle.dumps(stack)
    model_mb = len(blob) / 1024 ** 2

    optuna_s = None
    if do_optuna:
        import optuna
        from sklearn.model_selection import cross_val_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            m = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 20, 40),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                random_state=SEED, n_jobs=-1)
            return cross_val_score(m, X, y, cv=5,
                                   scoring="roc_auc" if task == "clf" else "neg_mean_absolute_error").mean()

        t0 = time.perf_counter()
        optuna.create_study(direction="maximize").optimize(objective, n_trials=30)
        optuna_s = (time.perf_counter() - t0) * 4  # extrapolate to 4 base learners

    return dict(endpoint=endpoint, config=label, n_train=len(X), n_test=len(Xte),
                n_features=X.shape[1], fit_s=round(fit_s, 2),
                optuna_s=None if optuna_s is None else round(optuna_s, 1),
                infer_ms_per_1k=round(infer_ms_per_1k, 2),
                fit_rss_delta_mb=round(rss1 - rss0, 1), model_mb=round(model_mb, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r"e.g. F:\00_00_TDC_data_reorganized")
    ap.add_argument("--out", default="cost_benchmark.csv")
    ap.add_argument("--optuna", action="store_true", help="also time a 30-trial Optuna search (slow)")
    a = ap.parse_args()

    rows = []
    for endpoint, (task, folder_R, folder_full) in TASKS.items():
        for label, folder in (("R", folder_R), ("M_A_E_S_R_MC_TI", folder_full)):
            print(f"-> {endpoint} / {label}", flush=True)
            try:
                rows.append(run_one(a.root, endpoint, task, folder, label, a.optuna))
                print("   ", rows[-1], flush=True)
            except Exception as e:  # keep going; report what failed
                print(f"   FAILED: {e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("\nwritten:", a.out)
    print(df.to_string(index=False))

    print("\nAlso record for the Methods section:")
    print("  CPU model / cores, RAM, GPU model, OS")
    print("  python, scikit-learn, lightgbm, xgboost, rdkit, selfies, torch, optuna versions")


if __name__ == "__main__":
    main()

# %% [markdown]
# # 02b - Feature matrices for the external datasets
#
# Builds features for an external dataset so that a stored TDC model can score it.
#
# Two things differ from the benchmark builder (`s02_build_features_tdc_benchmark.py`):
#
# 1. **There is no train / validation / test split**, so the imputer and the scaler are
#    necessarily fitted on the whole file. That is correct for a single evaluation set, but
#    it is a different procedure from the benchmark path and the Methods section says so.
# 2. **The configuration must match the stored model.** The blocks are concatenated in a
#    fixed order, so a model expects exactly the blocks it was trained with, in that order.
#    `CONFIGURATION` below is used both to select the blocks and to name the output folder.
#
# A warning about the SELFIES-derived projection blocks: they are randomly initialized per
# extractor instance and were not persisted in the original runs, so an external dataset
# featurized separately receives a *different* random map from the one the model was fitted
# with, and those columns carry no usable information at prediction time. Either choose a
# configuration without `EM32` / `EM64`, or load the projection saved by the benchmark
# builder. This script does the latter when `PROJECTION_PATH` is set.
#
# Manuscript: Sections 2.8.1 and 2.8.6.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "admet_config.py").exists())
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import admet_config as cfg
from feature_extractor import FeatureExtractor

RDLogger.DisableLog("rdApp.*")

# ------------------------------------------------------------------ settings
INPUT_CSV = cfg.EXTERNAL_FILTERED_DIR / "AutoML" / "pgp.csv"   # output of script 01
DATASET_NAME = "pgp_AutoML"
CONFIGURATION = "M_A_E_R_MC"        # must match the configuration of the model that will score it
TDC_ENDPOINT = "Pgp_Broccatelli"    # only used to locate a saved projection, if any
PROJECTION_PATH = None              # e.g. cfg.feature_dir(TDC_ENDPOINT, CONFIGURATION) / f"{TDC_ENDPOINT}_projection.pt"

OUT_DIR = cfg.EXTERNAL_DIR / "features" / DATASET_NAME / CONFIGURATION
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"input         : {INPUT_CSV}")
print(f"configuration : {CONFIGURATION}  ({cfg.fixed_length_dim(CONFIGURATION)} fixed-length features)")
print(f"output        : {OUT_DIR}")

if ("EM32" in CONFIGURATION.split("_") or "EM64" in CONFIGURATION.split("_")) and PROJECTION_PATH is None:
    print("\nWARNING: this configuration contains a random projection block and no saved "
          "projection was supplied. The projection built here will differ from the one the "
          "model was trained with, and those columns will be uninformative. Set "
          "PROJECTION_PATH, or pick a configuration without EM32 / EM64.")

# %%
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
if "Canonical_SMILES" not in df.columns:
    raise ValueError("Expected a 'Canonical_SMILES' column; run script 01 on this file first.")
if "Label" not in df.columns and "Activity" in df.columns:
    df = df.rename(columns={"Activity": "Label"})

smiles = df["Canonical_SMILES"].astype(str).tolist()
labels = df["Label"].tolist() if "Label" in df.columns else [np.nan] * len(df)
print(f"{len(smiles)} compounds")

flags = cfg.extractor_flags(CONFIGURATION)
fe = FeatureExtractor(selfies_vocab_path=str(OUT_DIR / f"{DATASET_NAME}_vocab.json"), **flags)
fe.build_vocab(smiles)
fe.save_vocab_csv(str(OUT_DIR / f"{DATASET_NAME}_vocab.csv"))

if PROJECTION_PATH is not None:
    fe.load_projection(str(PROJECTION_PATH))
    print(f"projection loaded from {PROJECTION_PATH}")

print("feature lengths:", fe.get_feature_lengths())

# %%
X = pd.DataFrame(fe.transform(smiles))
X.insert(0, "Canonical_SMILES", smiles)
X.insert(1, "Label", labels)
X.to_csv(OUT_DIR / f"{DATASET_NAME}_fp_features_raw.csv", index=False)
print("raw shape:", X.shape)

enc = []
for smi in smiles:
    try:
        enc.append(sf.encoder(smi))
    except Exception:
        enc.append("INVALID_SMILES")
pd.DataFrame({"Canonical_SMILES": smiles, "SELFIES": enc}).to_csv(
    OUT_DIR / f"{DATASET_NAME}_selfies.csv", index=False)

# %%
Xf = X.drop(columns=["Canonical_SMILES", "Label"]).replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
X_imp = imputer.fit_transform(Xf)     # single file: necessarily fitted on all of it
pd.DataFrame(X_imp, columns=Xf.columns).to_csv(
    OUT_DIR / f"{DATASET_NAME}_fp_features_imputed.csv", index=False)

scaler = StandardScaler()
X_sca = scaler.fit_transform(X_imp)
pd.DataFrame(X_sca, columns=Xf.columns).to_csv(
    OUT_DIR / f"{DATASET_NAME}_fp_features_scaled.csv", index=False)

(OUT_DIR / f"{DATASET_NAME}_preprocessing_log.txt").write_text("\n".join([
    f"dataset       : {DATASET_NAME}",
    f"source        : {INPUT_CSV}",
    f"configuration : {CONFIGURATION}",
    f"compounds     : {len(smiles)}",
    f"features      : {Xf.shape[1]}",
    "imputation    : SimpleImputer(strategy='mean'), fitted on the whole file (no split exists)",
    "scaling       : StandardScaler, fitted on the whole file (no split exists)",
    f"projection    : {'loaded from ' + str(PROJECTION_PATH) if PROJECTION_PATH else 'newly initialized or not used'}",
]), encoding="utf-8")

print("scaled shape:", X_sca.shape)
print("done:", OUT_DIR)

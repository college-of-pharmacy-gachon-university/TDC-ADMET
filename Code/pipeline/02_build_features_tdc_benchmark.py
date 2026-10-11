# %% [markdown]
# # 02a - Feature matrices for the TDC benchmark
#
# Builds the feature matrices on which the 54-configuration study is computed:
# **22 endpoints x 54 configurations**, each as train / validation / test matrices.
#
# For every endpoint and configuration the script writes
#
#     DATA_ROOT/features/<endpoint>/<configuration>/
#         <endpoint>_{train,valid,test}_fp_features_raw.csv       features + SMILES + Label
#         <endpoint>_{train,valid,test}_fp_features_imputed.csv   after mean imputation
#         <endpoint>_{train,valid,test}_fp_features_scaled.csv    after standardization
#         <endpoint>_vocab.csv                                    SELFIES vocabulary
#         <endpoint>_preprocessing_log.txt
#
# Fitting scope, which matters for the Methods section:
#
# * the SELFIES vocabulary, the 99th-percentile padded length and the TF-IDF vectorizer are
#   fitted on the pooled train + validation + test structures (`VOCAB_ON_TRAIN_ONLY = False`,
#   which reproduces the published matrices). No labels are involved.
# * the imputer and the scaler are fitted on the **training partition only** and applied
#   unchanged to validation and test.
#
# Manuscript: Sections 2.2 and 2.3.
#
# **Cost.** 22 x 54 = 1188 feature matrices. Use `ENDPOINTS_TO_RUN` and `CONFIGS_TO_RUN` to
# work through it in batches.

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
from tdc.single_pred import ADME, Tox

import admet_config as cfg
from feature_extractor import FeatureExtractor

RDLogger.DisableLog("rdApp.*")
cfg.ensure_dirs()

LOADERS = {"ADME": ADME, "Tox": Tox}

# ------------------------------------------------------------------ what to build
ENDPOINTS_TO_RUN = list(cfg.ENDPOINTS)        # all 22; narrow this to run in batches
CONFIGS_TO_RUN = list(cfg.CONFIGURATIONS)     # all 54; narrow this to run in batches

VOCAB_ON_TRAIN_ONLY = False   # False reproduces the published matrices
SAVE_SELFIES = True           # also write the SELFIES strings per split
SAVE_PROJECTION = True        # persist the random projection so external sets can reuse it
PROJECTION_SEED = None        # set an integer to make EM32 / EM64 reproducible

print(f"DATA_ROOT        : {cfg.DATA_ROOT}")
print(f"endpoints to run : {len(ENDPOINTS_TO_RUN)}")
print(f"configurations   : {len(CONFIGS_TO_RUN)}")
print(f"matrices to build: {len(ENDPOINTS_TO_RUN) * len(CONFIGS_TO_RUN)}")


# %%
def build_one(endpoint: str, configuration: str, split) -> dict:
    out_dir = cfg.feature_dir(endpoint, configuration)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = [f"endpoint      : {endpoint}",
           f"configuration : {configuration}",
           f"legacy folder : {cfg.LEGACY_FOLDER.get(configuration, '-')}"]

    flags = cfg.extractor_flags(configuration)
    fe = FeatureExtractor(
        selfies_vocab_path=str(out_dir / f"{endpoint}_vocab.json"),
        projection_seed=PROJECTION_SEED,
        **flags,
    )

    # ---- vocabulary / padded length / TF-IDF
    if VOCAB_ON_TRAIN_ONLY:
        vocab_smiles = split["train"]["Drug"].tolist()
    else:
        vocab_smiles = [s for part in ("train", "valid", "test") for s in split[part]["Drug"]]
    fe.build_vocab(vocab_smiles)
    fe.save_vocab_csv(str(out_dir / f"{endpoint}_vocab.csv"))
    log.append(f"vocabulary fitted on: {'train only' if VOCAB_ON_TRAIN_ONLY else 'train + valid + test'}")
    log += [f"  block {k}: {v}" for k, v in fe.get_feature_lengths().items()]

    if SAVE_PROJECTION and fe.model is not None:
        fe.save_projection(str(out_dir / f"{endpoint}_projection.pt"))
        log.append("projection weights saved (reuse these for external datasets)")

    # ---- raw features per split
    frames = {}
    for part in ("train", "valid", "test"):
        smiles = split[part]["Drug"].tolist()
        labels = split[part]["Y"].tolist()
        X = pd.DataFrame(fe.transform(smiles))
        X.insert(0, "SMILES", smiles)
        X.insert(1, "Label", labels)
        X.to_csv(out_dir / f"{endpoint}_{part}_fp_features_raw.csv", index=False)
        frames[part] = X
        log.append(f"{part:5s} raw shape: {X.shape}")

        if SAVE_SELFIES:
            enc = []
            for smi in smiles:
                try:
                    enc.append(sf.encoder(smi))
                except Exception:
                    enc.append("INVALID_SMILES")
            pd.DataFrame({"SMILES": smiles, "SELFIES": enc}).to_csv(
                out_dir / f"{endpoint}_{part}_selfies.csv", index=False)

    # ---- imputation and scaling, fitted on train only
    Xs = {p: frames[p].drop(columns=["SMILES", "Label"]).replace([np.inf, -np.inf], np.nan)
          for p in ("train", "valid", "test")}
    cols = Xs["train"].columns

    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    imp = {"train": imputer.fit_transform(Xs["train"])}
    imp["valid"] = imputer.transform(Xs["valid"])
    imp["test"] = imputer.transform(Xs["test"])
    for part in ("train", "valid", "test"):
        pd.DataFrame(imp[part], columns=cols).to_csv(
            out_dir / f"{endpoint}_{part}_fp_features_imputed.csv", index=False)
    log.append("imputation: SimpleImputer(strategy='mean'), fitted on train only")

    scaler = StandardScaler()
    sca = {"train": scaler.fit_transform(imp["train"])}
    sca["valid"] = scaler.transform(imp["valid"])
    sca["test"] = scaler.transform(imp["test"])
    for part in ("train", "valid", "test"):
        pd.DataFrame(sca[part], columns=cols).to_csv(
            out_dir / f"{endpoint}_{part}_fp_features_scaled.csv", index=False)
    log.append("scaling: StandardScaler, fitted on train only")

    (out_dir / f"{endpoint}_preprocessing_log.txt").write_text("\n".join(log), encoding="utf-8")
    return dict(endpoint=endpoint, configuration=configuration,
                n_train=len(frames["train"]), n_valid=len(frames["valid"]),
                n_test=len(frames["test"]), n_features=len(cols))


# %%
records = []
for endpoint in ENDPOINTS_TO_RUN:
    meta = cfg.ENDPOINTS[endpoint]
    split = LOADERS[meta["loader"]](name=endpoint).get_split()
    for configuration in CONFIGS_TO_RUN:
        out_dir = cfg.feature_dir(endpoint, configuration)
        marker = out_dir / f"{endpoint}_test_fp_features_scaled.csv"
        if marker.exists():
            print(f"skip (exists)  {endpoint:34s} {configuration}")
            continue
        rec = build_one(endpoint, configuration, split)
        records.append(rec)
        print(f"built          {endpoint:34s} {configuration:24s} "
              f"{rec['n_features']:5d} features")

if records:
    df = pd.DataFrame(records)
    df.to_csv(cfg.FEATURES_DIR / "feature_matrix_index.csv", index=False)
    print(f"\n{len(df)} matrices built; index written to feature_matrix_index.csv")
    print("This index supplies the per-endpoint dimensionality column of Supplementary Table S33.")

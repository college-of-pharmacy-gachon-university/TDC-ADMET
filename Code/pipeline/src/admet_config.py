"""
admet_config.py — single source of truth for paths, endpoints and the 54 representation
configurations used in the ADMET-Stack study.

Every notebook and script in this repository imports from here. Nothing else contains a
hard-coded path.

Setting DATA_ROOT
-----------------
Resolution order, first match wins:

1. the environment variable ``ADMET_DATA_ROOT``
2. a file named ``data_root.txt`` at the repository root, containing one line with the path
3. ``<repository root>/data``

So either

    export ADMET_DATA_ROOT=/mnt/storage/ADMET          # Linux / macOS
    set    ADMET_DATA_ROOT=E:\\ADMET_data              # Windows cmd
    $env:ADMET_DATA_ROOT = 'E:\\ADMET_data'            # PowerShell

or simply write the path into ``data_root.txt`` next to this repository's README.

Expected layout under DATA_ROOT
-------------------------------
    DATA_ROOT/
    ├── canonical/                 # canonicalized TDC reference files (one CSV per endpoint)
    ├── external/                  # external datasets, one folder per source
    │   ├── raw/                   #   as downloaded
    │   ├── canonical/             #   after canonicalization
    │   └── filtered/              #   after the Tanimoto 0.8 filter
    ├── features/<endpoint>/<configuration>/
    │                              # *_{train,valid,test}_fp_features_{raw,imputed,scaled}.csv
    ├── models/<endpoint>/<configuration>/
    └── results/<endpoint>/<configuration>/
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- repository
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_root() -> Path:
    env = os.environ.get("ADMET_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    marker = REPO_ROOT / "data_root.txt"
    if marker.exists():
        line = marker.read_text(encoding="utf-8").strip().splitlines()
        if line and line[0].strip():
            return Path(line[0].strip()).expanduser().resolve()
    return (REPO_ROOT / "data").resolve()


DATA_ROOT = _resolve_data_root()

CANONICAL_DIR = DATA_ROOT / "canonical"
EXTERNAL_DIR = DATA_ROOT / "external"
EXTERNAL_RAW_DIR = EXTERNAL_DIR / "raw"
EXTERNAL_CANONICAL_DIR = EXTERNAL_DIR / "canonical"
EXTERNAL_FILTERED_DIR = EXTERNAL_DIR / "filtered"
FEATURES_DIR = DATA_ROOT / "features"
MODELS_DIR = DATA_ROOT / "models"
RESULTS_DIR = DATA_ROOT / "results"


def ensure_dirs() -> None:
    """Create the expected directory tree under DATA_ROOT."""
    for d in (CANONICAL_DIR, EXTERNAL_RAW_DIR, EXTERNAL_CANONICAL_DIR,
              EXTERNAL_FILTERED_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def feature_dir(endpoint: str, configuration: str) -> Path:
    return FEATURES_DIR / endpoint / configuration


def model_dir(endpoint: str, configuration: str) -> Path:
    return MODELS_DIR / endpoint / configuration


def result_dir(endpoint: str, configuration: str) -> Path:
    return RESULTS_DIR / endpoint / configuration


# --------------------------------------------------------------------------- endpoints
# The 22 official TDC ADMET Leaderboard benchmarks.
# task: "classification" | "regression";  loader: the TDC class that serves the dataset.
ENDPOINTS: dict[str, dict] = {
    "Caco2_Wang":                     dict(loader="ADME", task="regression",     metric="MAE"),
    "HIA_Hou":                        dict(loader="ADME", task="classification", metric="AUROC"),
    "Pgp_Broccatelli":                dict(loader="ADME", task="classification", metric="AUROC"),
    "Bioavailability_Ma":             dict(loader="ADME", task="classification", metric="AUROC"),
    "Lipophilicity_AstraZeneca":      dict(loader="ADME", task="regression",     metric="MAE"),
    "Solubility_AqSolDB":             dict(loader="ADME", task="regression",     metric="MAE"),
    "BBB_Martins":                    dict(loader="ADME", task="classification", metric="AUROC"),
    "PPBR_AZ":                        dict(loader="ADME", task="regression",     metric="MAE"),
    "VDss_Lombardo":                  dict(loader="ADME", task="regression",     metric="Spearman"),
    "CYP3A4_Substrate_CarbonMangels": dict(loader="ADME", task="classification", metric="AUROC"),
    "CYP2D6_Substrate_CarbonMangels": dict(loader="ADME", task="classification", metric="AUPRC"),
    "CYP2C9_Substrate_CarbonMangels": dict(loader="ADME", task="classification", metric="AUPRC"),
    "CYP3A4_Veith":                   dict(loader="ADME", task="classification", metric="AUPRC"),
    "CYP2D6_Veith":                   dict(loader="ADME", task="classification", metric="AUPRC"),
    "CYP2C9_Veith":                   dict(loader="ADME", task="classification", metric="AUPRC"),
    "Half_Life_Obach":                dict(loader="ADME", task="regression",     metric="Spearman"),
    "Clearance_Hepatocyte_AZ":        dict(loader="ADME", task="regression",     metric="Spearman"),
    "Clearance_Microsome_AZ":         dict(loader="ADME", task="regression",     metric="Spearman"),
    "LD50_Zhu":                       dict(loader="Tox",  task="regression",     metric="MAE"),
    "hERG":                           dict(loader="Tox",  task="classification", metric="AUROC"),
    "AMES":                           dict(loader="Tox",  task="classification", metric="AUROC"),
    "DILI":                           dict(loader="Tox",  task="classification", metric="AUROC"),
}

CLASSIFICATION_ENDPOINTS = [k for k, v in ENDPOINTS.items() if v["task"] == "classification"]
REGRESSION_ENDPOINTS = [k for k, v in ENDPOINTS.items() if v["task"] == "regression"]

# --------------------------------------------------------------------------- blocks
# Feature blocks and the FeatureExtractor flags that switch each one on.
BLOCKS: dict[str, dict] = {
    "M":    dict(flag="use_morgan",        dim=1024, label="Morgan fingerprint (ECFP, r = 2)"),
    "A":    dict(flag="use_avalon",        dim=1024, label="Avalon fingerprint"),
    "E":    dict(flag="use_erg",           dim=315,  label="Extended Reduced Graph"),
    "S":    dict(flag="use_selfies",       dim=None, label="Tokenized SELFIES (endpoint-specific length)"),
    "R":    dict(flag="use_rdkit",         dim=217,  label="RDKit molecular descriptors"),
    "MC":   dict(flag="use_maccs",         dim=167,  label="MACCS keys"),
    "TI":   dict(flag="selfies_use_tfidf", dim=None, label="SELFIES TF-IDF (endpoint-specific length)"),
    "EM32": dict(flag="selfies_use_embed", dim=32,   label="32-dim random SELFIES projection (untrained)"),
    "EM64": dict(flag="selfies_use_embed", dim=64,   label="64-dim random SELFIES projection (untrained)"),
}

BLOCK_ORDER = ["M", "A", "E", "S", "R", "MC", "TI", "EM32", "EM64"]

# Legacy folder names of the released results, so that a regenerated feature matrix can be
# matched to the corresponding published result directory.
LEGACY_FOLDER: dict[str, str] = {
    "S":                      "01_02_TDC_selfies_variance",
    "R":                      "01_03_TDC_rdkit_variance",
    "MC":                     "01_04_TDC_maccs_variance",
    "TI":                     "01_05_TDC_tfidf_variance",
    "EM32":                   "01_06_TDC_embed_variance",
    "EM64":                   "01_07_TDC_embed64_variance",
    "M_A_E":                  "03_01_TDC_morgan_avalon_erg_variance",
    "M_A_E_S":                "04_01_TDC_morgan_avalon_erg_selfies_variance",
    "M_A_E_R":                "04_02_TDC_morgan_avalon_erg_rdkit_variance",
    "M_A_E_MC":               "04_03_TDC_morgan_avalon_erg_maccs_variance",
    "M_A_E_TI":               "04_04_TDC_morgan_avalon_erg_tfidf_variance",
    "M_A_E_EM32":             "04_05_TDC_morgan_avalon_erg_embed_variance",
    "M_A_E_EM64":             "04_06_TDC_morgan_avalon_erg_embed64_variance",
    "M_A_E_S_R":              "05_01_TDC_morgan_avalon_erg_selfies_rdkit_variance",
    "M_A_E_S_MC":             "05_02_TDC_morgan_avalon_erg_selfies_maccs_variance",
    "M_A_E_S_TI":             "05_03_TDC_morgan_avalon_erg_selfies_tfidf_variance",
    "M_A_E_S_EM32":           "05_04_TDC_morgan_avalon_erg_selfies_embed_variance",
    "M_A_E_S_EM64":           "05_05_TDC_morgan_avalon_erg_selfies_embed64_variance",
    "M_A_E_R_MC":             "05_06_TDC_morgan_avalon_erg_rdkit_maccs_variance",
    "M_A_E_R_TI":             "05_07_TDC_morgan_avalon_erg_rdkit_tfidf_variance",
    "M_A_E_R_EM32":           "05_08_TDC_morgan_avalon_erg_rdkit_embed_variance",
    "M_A_E_R_EM64":           "05_09_TDC_morgan_avalon_erg_rdkit_embed64_variance",
    "M_A_E_MC_TI":            "05_10_TDC_morgan_avalon_erg_maccs_tfidf_variance",
    "M_A_E_MC_EM32":          "05_11_TDC_morgan_avalon_erg_maccs_embed_variance",
    "M_A_E_MC_EM64":          "05_12_TDC_morgan_avalon_erg_maccs_embed64_variance",
    "M_A_E_TI_EM32":          "05_13_TDC_morgan_avalon_erg_tfidf_embed_variance",
    "M_A_E_TI_EM64":          "05_14_TDC_morgan_avalon_erg_tfidf_embed64_variance",
    "M_A_E_S_R_MC":           "06_01_TDC_morgan_avalon_erg_selfies_rdkit_maccs_variance",
    "M_A_E_S_R_TI":           "06_02_TDC_morgan_avalon_erg_selfies_rdkit_tfidf_variance",
    "M_A_E_S_R_EM32":         "06_03_TDC_morgan_avalon_erg_selfies_rdkit_embed_variance",
    "M_A_E_S_R_EM64":         "06_04_TDC_morgan_avalon_erg_selfies_rdkit_embed64_variance",
    "M_A_E_S_MC_TI":          "06_05_TDC_morgan_avalon_erg_selfies_maccs_tfidf_variance",
    "M_A_E_S_MC_EM32":        "06_06_TDC_morgan_avalon_erg_selfies_maccs_embed_variance",
    "M_A_E_S_MC_EM64":        "06_07_TDC_morgan_avalon_erg_selfies_maccs_embed64_variance",
    "M_A_E_S_TI_EM32":        "06_08_TDC_morgan_avalon_erg_selfies_tfidf_embed_variance",
    "M_A_E_S_TI_EM64":        "06_09_TDC_morgan_avalon_erg_selfies_tfidf_embed64_variance",
    "M_A_E_R_MC_TI":          "06_10_TDC_morgan_avalon_erg_rdkit_maccs_tfidf_variance",
    "M_A_E_R_MC_EM32":        "06_11_TDC_morgan_avalon_erg_rdkit_maccs_embed_variance",
    "M_A_E_R_MC_EM64":        "06_12_TDC_morgan_avalon_erg_rdkit_maccs_embed64_variance",
    "M_A_E_R_TI_EM32":        "06_13_TDC_morgan_avalon_erg_rdkit_tfidf_embed_variance",
    "M_A_E_R_TI_EM64":        "06_14_TDC_morgan_avalon_erg_rdkit_tfidf_embed64_variance",
    "M_A_E_MC_TI_EM32":       "06_15_TDC_morgan_avalon_erg_maccs_tfidf_embed_variance",
    "M_A_E_MC_TI_EM64":       "06_16_TDC_morgan_avalon_erg_maccs_tfidf_embed64_variance",
    "M_A_E_S_R_MC_TI":        "07_01_TDC_morgan_avalon_erg_selfies_rdkit_maccs_tfidf_variance",
    "M_A_E_S_R_MC_EM32":      "07_02_TDC_morgan_avalon_erg_selfies_rdkit_maccs_embed_variance",
    "M_A_E_S_R_MC_EM64":      "07_03_TDC_morgan_avalon_erg_selfies_rdkit_maccs_embed64_variance",
    "M_A_E_S_R_TI_EM32":      "07_04_TDC_morgan_avalon_erg_selfies_rdkit_tfidf_embed_variance",
    "M_A_E_S_R_TI_EM64":      "07_05_TDC_morgan_avalon_erg_selfies_rdkit_tfidf_embed64_variance",
    "M_A_E_S_MC_TI_EM32":     "07_06_TDC_morgan_avalon_erg_selfies_maccs_tfidf_embed_variance",
    "M_A_E_S_MC_TI_EM64":     "07_07_TDC_morgan_avalon_erg_selfies_maccs_tfidf_embed64_variance",
    "M_A_E_R_MC_TI_EM32":     "07_08_TDC_morgan_avalon_erg_rdkit_maccs_tfidf_embed_variance",
    "M_A_E_R_MC_TI_EM64":     "07_09_TDC_morgan_avalon_erg_rdkit_maccs_tfidf_embed64_variance",
    "M_A_E_S_R_MC_TI_EM32":   "08_01_TDC_morgan_avalon_erg_selfies_rdkit_maccs_tfidf_embed_variance",
    "M_A_E_S_R_MC_TI_EM64":   "08_02_TDC_morgan_avalon_erg_selfies_rdkit_maccs_tfidf_embed64_variance",
}


def build_configurations() -> list[str]:
    """Return the 54 configuration codes of the study.

    48 core configurations: the shared Morgan+Avalon+ErG backbone, crossed with the four
    present/absent switches S, R, MC, TI (2**4 = 16) and the three-level projection factor
    (none, EM32, EM64), giving 16 * 3 = 48.
    6 single-block baselines: R, MC, S, TI, EM32, EM64 each used alone.
    """
    core = []
    for s in (False, True):
        for r in (False, True):
            for mc in (False, True):
                for ti in (False, True):
                    for emb in (None, "EM32", "EM64"):
                        toks = ["M", "A", "E"]
                        if s:
                            toks.append("S")
                        if r:
                            toks.append("R")
                        if mc:
                            toks.append("MC")
                        if ti:
                            toks.append("TI")
                        if emb:
                            toks.append(emb)
                        core.append("_".join(toks))
    singles = ["R", "MC", "S", "TI", "EM32", "EM64"]
    ordered = sorted(set(core), key=lambda c: (len(c.split("_")), c)) + singles
    assert len(core) == 48 and len(set(core)) == 48, "core design is not complete"
    assert len(ordered) == 54, f"expected 54 configurations, built {len(ordered)}"
    return ordered


CONFIGURATIONS = build_configurations()


def extractor_flags(configuration: str) -> dict:
    """Translate a configuration code into FeatureExtractor keyword arguments."""
    toks = configuration.split("_")
    flags = dict(
        use_morgan="M" in toks,
        use_avalon="A" in toks,
        use_erg="E" in toks,
        use_selfies="S" in toks,
        use_rdkit="R" in toks,
        use_maccs="MC" in toks,
        selfies_use_tfidf="TI" in toks,
        selfies_use_embed=("EM32" in toks or "EM64" in toks),
    )
    dim = 64 if "EM64" in toks else 32
    flags["selfies_emb_dim"] = dim
    flags["selfies_emb_out"] = dim
    return flags


def fixed_length_dim(configuration: str) -> int:
    """Dimensionality of the blocks whose width does not depend on the endpoint."""
    toks = configuration.split("_")
    return sum(BLOCKS[b]["dim"] for b in toks if BLOCKS[b]["dim"] is not None)


# --------------------------------------------------------------------------- misc
SEEDS = [1, 2, 3, 4, 5]          # learner random_state values used for the five repeats
STACKING_CV = 10                 # folds for the out-of-fold meta-feature construction
SIMILARITY_THRESHOLD = 0.8       # external compounds above this Tanimoto to TDC are removed
MORGAN_RADIUS = 2
MORGAN_NBITS_SIMILARITY = 2048   # fingerprint used for the similarity filter only


if __name__ == "__main__":
    print(f"REPO_ROOT  = {REPO_ROOT}")
    print(f"DATA_ROOT  = {DATA_ROOT}")
    print(f"endpoints  = {len(ENDPOINTS)} "
          f"({len(CLASSIFICATION_ENDPOINTS)} classification, {len(REGRESSION_ENDPOINTS)} regression)")
    print(f"configurations = {len(CONFIGURATIONS)}")
    missing = [c for c in CONFIGURATIONS if c not in LEGACY_FOLDER]
    print(f"legacy folder names mapped: {len(CONFIGURATIONS) - len(missing)}/{len(CONFIGURATIONS)}")
    if missing:
        print("  unmapped:", missing)
    print()
    print("example:", CONFIGURATIONS[0], "->", extractor_flags(CONFIGURATIONS[0]))

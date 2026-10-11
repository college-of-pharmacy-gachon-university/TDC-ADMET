# %% [markdown]
# # 01 - Canonicalization and external-set independence filter
#
# Produces, for every external dataset, a canonicalized and similarity-filtered copy that is
# independent of the corresponding TDC benchmark.
#
# Steps
# 1. canonicalize the TDC reference structures for each endpoint
# 2. canonicalize each external dataset
# 3. for every external compound, compute the Tanimoto similarity to its nearest TDC
#    neighbour (Morgan, radius 2, 2048 bits)
# 4. drop every external compound whose nearest-neighbour similarity exceeds 0.8
#
# Because exact duplicates have a similarity of 1.0, step 4 removes structural overlap as
# well as close analogues. The retained similarity value is written into the output as a
# `Tanimoto_Similarity` column, so the filter is auditable after the fact.
#
# Manuscript: Sections 2.8.1 and 2.8.2.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "admet_config.py").exists())
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MolFromSmiles

import admet_config as cfg
from feature_extractor import canonicalize

RDLogger.DisableLog("rdApp.*")
cfg.ensure_dirs()
print("DATA_ROOT:", cfg.DATA_ROOT)

# %% [markdown]
# ## Step 1 - canonicalize the TDC reference structures
#
# One CSV per endpoint, containing every structure of that benchmark. These files are the
# reference against which external compounds are compared.

# %%
from tdc.single_pred import ADME, Tox  # noqa: E402

LOADERS = {"ADME": ADME, "Tox": Tox}


def write_tdc_reference(endpoint: str) -> Path:
    meta = cfg.ENDPOINTS[endpoint]
    split = LOADERS[meta["loader"]](name=endpoint).get_split()
    rows = []
    for part in ("train", "valid", "test"):
        df = split[part][["Drug", "Y"]].copy()
        df.columns = ["SMILES", "Label"]
        df["Split"] = part
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["Canonical_SMILES"] = out["SMILES"].astype(str).map(canonicalize)
    out = out.dropna(subset=["Canonical_SMILES"])
    path = cfg.CANONICAL_DIR / f"canonical_{endpoint}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


for endpoint in cfg.ENDPOINTS:
    p = write_tdc_reference(endpoint)
    print(f"{endpoint:34s} -> {p.name}")

# %% [markdown]
# ## Step 2 - canonicalize the external datasets
#
# Place the external datasets under `DATA_ROOT/external/raw/<source>/<name>.csv`, each with a
# `SMILES` column and a `Label` column.

# %%
def canonicalize_external(src_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(src_csv, encoding="utf-8-sig")
    smiles_col = "SMILES" if "SMILES" in df.columns else "Drug"
    df["Canonical_SMILES"] = df[smiles_col].astype(str).map(canonicalize)
    before = len(df)
    df = df.dropna(subset=["Canonical_SMILES"]).copy()
    if "Y" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"Y": "Label"})
    df = df.drop_duplicates(subset=["Canonical_SMILES"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  {src_csv.name}: {before} -> {len(df)} after parsing and deduplication")
    return df


for src in sorted(cfg.EXTERNAL_RAW_DIR.rglob("*.csv")):
    rel = src.relative_to(cfg.EXTERNAL_RAW_DIR)
    canonicalize_external(src, cfg.EXTERNAL_CANONICAL_DIR / rel)

# %% [markdown]
# ## Step 3 and 4 - nearest-neighbour similarity and the 0.8 filter
#
# `ENDPOINT_FOR_EXTERNAL` maps each external file to the TDC endpoint it should be compared
# against. Edit it to match the files you placed under `external/raw/`.

# %%
ENDPOINT_FOR_EXTERNAL = {
    # "AutoML/pgp.csv":            "Pgp_Broccatelli",
    # "AutoML/cyp3a4_inhibition.csv": "CYP3A4_Veith",
    # "PharmaBench/bbb.csv":       "BBB_Martins",
    # "Admetica/herg.csv":         "hERG",
    # "in-house/ames.csv":         "AMES",
}

_fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=cfg.MORGAN_RADIUS, fpSize=cfg.MORGAN_NBITS_SIMILARITY
)


def fingerprints(smiles_iter):
    fps, kept = [], []
    for s in smiles_iter:
        mol = MolFromSmiles(s)
        if mol is not None:
            fps.append(_fpgen.GetFingerprint(mol))
            kept.append(s)
    return fps, kept


def nearest_neighbour_similarity(external_smiles, reference_smiles) -> pd.DataFrame:
    """Highest Tanimoto similarity of each external compound to any reference compound."""
    ext_fps, ext_kept = fingerprints(external_smiles)
    ref_fps, _ = fingerprints(reference_smiles)
    out = []
    for smi, fp in zip(ext_kept, ext_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        out.append({"Canonical_SMILES": smi, "Tanimoto_Similarity": max(sims) if sims else 0.0})
    return pd.DataFrame(out)


def filter_external(rel_path: str, endpoint: str) -> pd.DataFrame:
    ext = pd.read_csv(cfg.EXTERNAL_CANONICAL_DIR / rel_path, encoding="utf-8-sig")
    ref = pd.read_csv(cfg.CANONICAL_DIR / f"canonical_{endpoint}.csv", encoding="utf-8-sig")

    sim = nearest_neighbour_similarity(
        ext["Canonical_SMILES"].dropna().unique(),
        ref["Canonical_SMILES"].dropna().unique(),
    )
    merged = ext.merge(sim, on="Canonical_SMILES", how="inner")
    kept = merged[merged["Tanimoto_Similarity"] <= cfg.SIMILARITY_THRESHOLD].copy()

    out = cfg.EXTERNAL_FILTERED_DIR / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"{rel_path:40s} vs {endpoint:28s} "
          f"{len(ext):6d} -> {len(kept):6d} kept "
          f"({len(ext) - len(kept)} removed at Tanimoto > {cfg.SIMILARITY_THRESHOLD})")
    return kept


summary = []
for rel_path, endpoint in ENDPOINT_FOR_EXTERNAL.items():
    kept = filter_external(rel_path, endpoint)
    n_before = len(pd.read_csv(cfg.EXTERNAL_CANONICAL_DIR / rel_path, encoding="utf-8-sig"))
    summary.append(dict(dataset=rel_path, tdc_endpoint=endpoint,
                        n_before=n_before, n_after=len(kept),
                        prevalence=round(float(np.mean(kept["Label"])), 4) if "Label" in kept else None))

if summary:
    df = pd.DataFrame(summary)
    df.to_csv(cfg.EXTERNAL_FILTERED_DIR / "filtering_summary.csv", index=False)
    print()
    print(df.to_string(index=False))
    print("\nThis summary supplies the before/after columns of Supplementary Table S32.")
else:
    print("ENDPOINT_FOR_EXTERNAL is empty - add your external files to it and re-run.")

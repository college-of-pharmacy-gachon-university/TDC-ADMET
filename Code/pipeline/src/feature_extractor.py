"""
feature_extractor.py — the molecular feature extractor used throughout the ADMET-Stack study.

This is a cleaned, single copy of the class that previously existed in several notebooks.
**Its behaviour is unchanged**: it reproduces the feature matrices on which the published
results were computed. Where the implementation has a quirk that affects interpretation, the
quirk is kept and documented rather than silently fixed, so that the repository matches the
paper. Every such point is marked with a `NOTE:` comment and is also stated in the manuscript.

Nine feature blocks are available:

    M     Morgan fingerprint (ECFP, radius 2)                           1024 bits
    A     Avalon fingerprint                                            1024 bits
    E     Extended Reduced Graph (ErG), zero-padded / truncated          315
    R     RDKit molecular descriptors (Descriptors._descList)            217
    MC    MACCS keys                                                     167
    S     Tokenized SELFIES, integer indices                            endpoint-specific
    TI    SELFIES TF-IDF                                                endpoint-specific
    EM32  32-dimensional SELFIES-derived random projection               32
    EM64  64-dimensional SELFIES-derived random projection               64

The blocks are concatenated in the order listed above.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import selfies as sf
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class SelfiesProjection(nn.Module):
    """Token embedding -> mean pooling -> linear projection.

    NOTE: this module is **never trained**. There is no objective function, optimizer or
    training loop anywhere in the study. The weights remain at PyTorch's default random
    initialization and the module is used in ``eval()`` mode under ``torch.no_grad()``.
    What it computes is therefore a fixed random linear projection of the length-normalized
    SELFIES token-count vector — a random sketch of a bag-of-tokens representation. The
    blocks are named EM32 / EM64 in the published results and RP32 / RP64 in the revised
    manuscript, where their untrained character is relevant.

    NOTE: no seed is set, so two instances produce different projections. A projection built
    for an external dataset is therefore NOT the one a stored model was trained with; see
    ``make_reproducible`` below and Section 2.8.6 of the manuscript.
    """

    def __init__(self, vocab_size: int, emb_dim: int = 32, output_dim: int = 32, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, output_dim)

    def forward(self, x):
        emb = self.embedding(x)       # (batch, seq, emb_dim)
        pooled = emb.mean(dim=1)      # NOTE: mean over the full padded length, not the true
                                      # token count, so the vector is scaled by 1 / pad_len
        return self.fc(pooled)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert SMILES strings into the concatenated feature blocks of the study."""

    def __init__(
        self,
        selfies_vocab_path: str = "selfies_vocab.json",
        selfies_pad_len: int = 300,
        selfies_use_tfidf: bool = True,
        selfies_use_embed: bool = True,
        selfies_emb_dim: int = 32,
        selfies_emb_out: int = 32,
        morgan_bits: int = 1024,
        avalon_bits: int = 1024,
        erg_len: int = 315,
        maccs_bits: int = 167,
        rdkit_descriptors: list[str] | None = None,
        use_morgan: bool = True,
        use_avalon: bool = True,
        use_erg: bool = True,
        use_rdkit: bool = True,
        use_selfies: bool = True,
        use_maccs: bool = True,
        projection_seed: int | None = None,
    ):
        self.selfies_vocab_path = selfies_vocab_path
        self.selfies_pad_len = selfies_pad_len
        self.selfies_use_tfidf = selfies_use_tfidf
        self.selfies_use_embed = selfies_use_embed
        self.selfies_emb_dim = selfies_emb_dim
        self.selfies_emb_out = selfies_emb_out
        self.morgan_bits = morgan_bits
        self.avalon_bits = avalon_bits
        self.erg_len = erg_len
        self.maccs_bits = maccs_bits
        self.use_morgan = use_morgan
        self.use_avalon = use_avalon
        self.use_erg = use_erg
        self.use_rdkit = use_rdkit
        self.use_selfies = use_selfies
        self.use_maccs = use_maccs
        self.projection_seed = projection_seed

        # NOTE: the descriptor count is RDKit-version dependent — len(Descriptors._descList)
        # returns 217 from RDKit 2024.03 onward and 210 in earlier releases. The pinned
        # environment fixes this; do not change the RDKit pin without re-checking.
        self.rdkit_descriptors = (
            [name for name, _ in Descriptors._descList] if rdkit_descriptors is None else rdkit_descriptors
        )

        self.selfies_vocab_stoi = self._load_vocab() if os.path.exists(self.selfies_vocab_path) else None
        self.vectorizer = None
        self.token2idx = None
        self.model = None
        self._morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.morgan_bits)

    # ------------------------------------------------------------------ fitting
    def build_vocab(self, smiles_list):
        """Fit the SELFIES vocabulary, the padded length and the TF-IDF vectorizer.

        NOTE: in the published study this was called on the pooled training, validation and
        test structures of each endpoint. No labels are involved, so this is transductive
        feature construction rather than label leakage, and it affects only the S, TI, EM32
        and EM64 blocks. To fit on the training partition alone instead, pass only the
        training SMILES here; the rest of the pipeline is unchanged.
        """
        selfies_set, selfies_texts, token_lengths, valid = set(), [], [], []
        for smi in smiles_list:
            if not smi or not isinstance(smi, str):
                continue
            try:
                tokens = list(sf.split_selfies(sf.encoder(smi)))
            except Exception:
                continue
            selfies_set.update(tokens)
            token_lengths.append(len(tokens))
            if self.selfies_use_tfidf:
                selfies_texts.append(" ".join(tokens))
            valid.append(smi)

        if not valid:
            raise ValueError("No valid SMILES found; cannot build the SELFIES vocabulary.")

        max_len_99 = int(np.percentile(token_lengths, 99))
        self.max_selfies_len = max_len_99
        self.selfies_pad_len = max_len_99

        # NOTE: the vocabulary is indexed from 0, so a token unseen at fitting time takes the
        # index of the first token in sorted order rather than a reserved unknown symbol.
        vocab = {tok: idx for idx, tok in enumerate(sorted(selfies_set))}
        if self.selfies_vocab_path:
            with open(self.selfies_vocab_path, "w", encoding="utf-8") as fh:
                json.dump(vocab, fh, indent=4)
        self.selfies_vocab_stoi = vocab

        if self.selfies_use_tfidf and selfies_texts:
            self.vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"\[.*?\]",
                lowercase=False,
                max_features=max_len_99,
            )
            self.vectorizer.fit(selfies_texts)
        else:
            self.vectorizer = None

        if self.selfies_use_embed and selfies_set:
            # index 0 is reserved for padding, and an unknown token also maps to 0
            self.token2idx = {tok: i + 1 for i, tok in enumerate(sorted(selfies_set))}
            self.model = SelfiesProjection(
                vocab_size=len(self.token2idx) + 1,
                emb_dim=self.selfies_emb_dim,
                output_dim=self.selfies_emb_out,
                seed=self.projection_seed,
            )
            self.model.eval()

        return vocab

    def fit(self, smiles_list, y=None):
        self.build_vocab(smiles_list)
        return self

    def _load_vocab(self):
        with open(self.selfies_vocab_path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------ metadata
    def get_feature_lengths(self) -> dict:
        lengths = {}
        if self.use_morgan:
            lengths["Morgan"] = self.morgan_bits
        if self.use_avalon:
            lengths["Avalon"] = self.avalon_bits
        if self.use_erg:
            lengths["ErG"] = self.erg_len
        if self.use_rdkit:
            lengths["RDKit"] = len(self.rdkit_descriptors)
        if self.use_selfies:
            lengths["SELFIES_int"] = self.selfies_pad_len
        if self.selfies_use_tfidf and self.vectorizer is not None:
            lengths["SELFIES_TFIDF"] = len(self.vectorizer.get_feature_names_out())
        if self.selfies_use_embed and self.model is not None:
            lengths["SELFIES_Projection"] = self.selfies_emb_out
        if self.use_maccs:
            lengths["MACCS"] = self.maccs_bits
        return lengths

    def save_vocab_csv(self, csv_path: str = "selfies_vocab.csv") -> None:
        if self.selfies_vocab_stoi is None:
            raise ValueError("Vocabulary has not been built yet.")
        pd.DataFrame(list(self.selfies_vocab_stoi.items()), columns=["Token", "Index"]).to_csv(
            csv_path, index=False
        )

    def save_projection(self, path: str) -> None:
        """Persist the random projection so that external datasets can use the same map."""
        if self.model is None:
            raise ValueError("No projection module was built for this configuration.")
        torch.save({"state_dict": self.model.state_dict(), "token2idx": self.token2idx,
                    "pad_len": self.selfies_pad_len, "emb_dim": self.selfies_emb_dim,
                    "out_dim": self.selfies_emb_out}, path)

    def load_projection(self, path: str) -> None:
        blob = torch.load(path, map_location="cpu")
        self.token2idx = blob["token2idx"]
        self.selfies_pad_len = blob["pad_len"]
        self.selfies_emb_dim = blob["emb_dim"]
        self.selfies_emb_out = blob["out_dim"]
        self.model = SelfiesProjection(len(self.token2idx) + 1, blob["emb_dim"], blob["out_dim"])
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    # ------------------------------------------------------------------ transform
    def transform(self, smiles_list):
        rows = []
        for smi in smiles_list:
            mol = self._smiles_to_mol(smi)
            parts = []

            if self.use_morgan:
                parts.append(self._bitvec(lambda m: self._morgan_gen.GetFingerprint(m), mol, self.morgan_bits))

            if self.use_avalon:
                parts.append(
                    self._bitvec(lambda m: pyAvalonTools.GetAvalonFP(m, nBits=self.avalon_bits),
                                 mol, self.avalon_bits)
                )

            if self.use_erg:
                parts.append(self._erg(mol))

            if self.use_rdkit:
                parts.append(
                    self._compute_descriptors(mol) if mol is not None
                    else np.zeros(len(self.rdkit_descriptors), dtype=np.float32)
                )

            if self.use_selfies and self.selfies_vocab_stoi is not None:
                parts.append(self._selfies_indices(smi))

            if self.selfies_use_tfidf and self.vectorizer is not None:
                parts.append(self._selfies_tfidf(smi))

            if self.selfies_use_embed and self.model is not None and self.token2idx is not None:
                parts.append(self._selfies_projection(smi))

            if self.use_maccs:
                parts.append(self._bitvec(MACCSkeys.GenMACCSKeys, mol, self.maccs_bits))

            rows.append(np.concatenate(parts))
        return np.vstack(rows)

    # ------------------------------------------------------------------ block helpers
    @staticmethod
    def _smiles_to_mol(smi):
        try:
            return Chem.MolFromSmiles(smi)
        except Exception:
            return None

    @staticmethod
    def _bitvec(fn, mol, n_bits):
        arr = np.zeros((n_bits,), dtype=np.int8)
        if mol is not None:
            try:
                ConvertToNumpyArray(fn(mol), arr)
            except Exception:
                arr = np.zeros((n_bits,), dtype=np.int8)
        return arr.astype(np.float32)

    def _erg(self, mol):
        if mol is None:
            return np.zeros((self.erg_len,), dtype=np.float32)
        try:
            arr = np.asarray(list(GetErGFingerprint(mol)), dtype=np.float32)
        except Exception:
            return np.zeros((self.erg_len,), dtype=np.float32)
        if len(arr) < self.erg_len:
            arr = np.pad(arr, (0, self.erg_len - len(arr)))
        return arr[: self.erg_len].astype(np.float32)

    def _compute_descriptors(self, mol):
        # NOTE: a descriptor that returns a non-finite value, or whose calculation raises, is
        # replaced by 0.0 here — not by a column mean. For descriptors such as
        # MaxPartialCharge, 0.0 is a meaningful value rather than a missing marker. The
        # SimpleImputer applied later therefore has little left to do.
        values = []
        for name in self.rdkit_descriptors:
            try:
                val = getattr(Descriptors, name)(mol)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                values.append(float(val))
            except Exception:
                values.append(0.0)
        return np.array(values, dtype=np.float32)

    def _pad_ids(self, ids):
        if len(ids) < self.selfies_pad_len:
            return ids + [0] * (self.selfies_pad_len - len(ids))
        return ids[: self.selfies_pad_len]

    def _selfies_indices(self, smi):
        try:
            ids = [self.selfies_vocab_stoi.get(t, 0) for t in sf.split_selfies(sf.encoder(smi))]
        except Exception:
            return np.zeros(self.selfies_pad_len, dtype=np.float32)
        return np.array(self._pad_ids(ids), dtype=np.int16).astype(np.float32)

    def _selfies_tfidf(self, smi):
        n = len(self.vectorizer.get_feature_names_out())
        try:
            text = " ".join(sf.split_selfies(sf.encoder(smi)))
            return self.vectorizer.transform([text]).toarray().flatten().astype(np.float32)
        except Exception:
            return np.zeros(n, dtype=np.float32)

    def _selfies_projection(self, smi):
        try:
            ids = [self.token2idx.get(t, 0) for t in sf.split_selfies(sf.encoder(smi))]
            with torch.no_grad():
                emb = self.model(torch.tensor([self._pad_ids(ids)])).numpy().flatten()
            return emb.astype(np.float32)
        except Exception:
            return np.zeros(self.selfies_emb_out, dtype=np.float32)


def canonicalize(smiles: str) -> str | None:
    """RDKit canonical SMILES, or None if the string cannot be parsed."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None

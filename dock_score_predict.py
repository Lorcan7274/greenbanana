import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
from transformers import RobertaTokenizer, RobertaModel

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import spearmanr


def canonicalize_smiles(smiles: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def load_csv(
    csv_path: str,
    smiles_col: str = "SMILES",
    y_col: str = "best_score",
    drop_unnamed: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if drop_unnamed and "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df[[smiles_col, y_col]].copy()
    df.columns = ["smiles", "y"]

    df["smiles"] = df["smiles"].astype(str)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["smiles", "y"]).reset_index(drop=True)

    df["smiles"] = df["smiles"].map(canonicalize_smiles)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    return df


def rdkit_descriptors(smiles_list: List[str]) -> np.ndarray:
    feats = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            feats.append([np.nan] * 18)
            continue

        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        tpsa = Descriptors.TPSA(m)

        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        rot = Lipinski.NumRotatableBonds(m)
        rings = Lipinski.RingCount(m)

        arom_rings = Lipinski.NumAromaticRings(m)
        aliph_rings = Lipinski.NumAliphaticRings(m)

        heavy = m.GetNumHeavyAtoms()
        fr_csp3 = Lipinski.FractionCSP3(m)

        atoms = m.GetNumAtoms()
        hetero = Lipinski.NumHeteroatoms(m)

        formal_charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
        n_chiral = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        mr = Crippen.MolMR(m)

        n_n = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N")
        n_o = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "O")

        feats.append(
            [
                mw,
                logp,
                tpsa,
                hbd,
                hba,
                rot,
                rings,
                arom_rings,
                aliph_rings,
                heavy,
                fr_csp3,
                atoms,
                hetero,
                formal_charge,
                n_chiral,
                mr,
                n_n,
                n_o,
            ]
        )

    X = np.asarray(feats, dtype=float)
    med = np.nanmedian(X, axis=0)
    bad = np.isnan(X)
    X[bad] = np.take(med, np.where(bad)[1])
    return X


def murcko_scaffold(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "INVALID"
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    if scaf is None:
        return "NONE"
    return Chem.MolToSmiles(scaf, canonical=True)


def scaffold_groups(smiles_list: List[str]) -> np.ndarray:
    gs = [murcko_scaffold(s) for s in smiles_list]
    uniq = {g: i for i, g in enumerate(sorted(set(gs)))}
    return np.array([uniq[g] for g in gs], dtype=int)


class ChemBERTaFeaturizer:
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 16,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = RobertaModel.from_pretrained(model_name)

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        outs = []
        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            outs.append(pooled.detach().cpu().numpy())
        return np.vstack(outs)


def get_regressor(random_state: int = 42):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        depth=4,
        learning_rate=0.05,
        n_estimators=2000,
        l2_leaf_reg=6.0,
        subsample=0.8,
        random_seed=random_state,
        verbose=False,
    ), "catboost"


class DockingSurrogate:
    def __init__(
        self,
        pca_components: Optional[int] = 100,
        use_descriptors: bool = True,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.pca_components = pca_components
        self.use_descriptors = use_descriptors
        self.model_name = model_name
        self.random_state = random_state
        self.device = device

        self.featurizer = ChemBERTaFeaturizer(model_name=model_name, device=device)
        self.pca = None
        self.model = None
        self.model_kind = None

    def _features(self, smiles: List[str]) -> np.ndarray:
        X_bert = self.featurizer.transform(smiles)
        if not self.use_descriptors:
            return X_bert
        X_desc = rdkit_descriptors(smiles)
        return np.hstack([X_bert, X_desc])

    def cross_validate(self, smiles: List[str], y: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
        groups = scaffold_groups(smiles)
        n_groups = len(np.unique(groups))
        n_splits = min(n_splits, n_groups)

        X_raw = self._features(smiles)
        gkf = GroupKFold(n_splits=n_splits)

        rows = []
        for fold, (tr, te) in enumerate(gkf.split(X_raw, y, groups=groups), start=1):
            Xtr, Xte = X_raw[tr], X_raw[te]

            if self.pca_components is not None:
                pca = PCA(n_components=self.pca_components, random_state=self.random_state + fold)
                Xtr = pca.fit_transform(Xtr)
                Xte = pca.transform(Xte)

            model, kind = get_regressor(self.random_state + fold)
            model.fit(Xtr, y[tr])
            pred = model.predict(Xte)

            mae = mean_absolute_error(y[te], pred)
            rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
            spr = spearmanr(y[te], pred).correlation
            rows.append({"fold": fold, "model": kind, "MAE": mae, "RMSE": rmse, "Spearman": spr})

        return pd.DataFrame(rows)

    def fit(self, smiles: List[str], y: np.ndarray):
        X_raw = self._features(smiles)

        if self.pca_components is not None:
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X = self.pca.fit_transform(X_raw)
        else:
            X = X_raw

        self.model, self.model_kind = get_regressor(self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict().")

        X_raw = self._features(smiles)
        X = self.pca.transform(X_raw) if self.pca is not None else X_raw
        return np.asarray(self.model.predict(X), dtype=float)


def rank_candidates(model: DockingSurrogate, smiles_list: List[str]) -> pd.DataFrame:
    can = []
    for s in smiles_list:
        cs = canonicalize_smiles(str(s))
        if cs is not None:
            can.append(cs)

    preds = model.predict(can)
    out = pd.DataFrame({"smiles": can, "pred_score": preds})
    out = out.sort_values("pred_score", ascending=True).reset_index(drop=True)  # more negative = better
    return out


if __name__ == "__main__":
    df = load_csv("output.csv", smiles_col="SMILES", y_col="best_score")
    smiles = df["smiles"].tolist()
    y = df["y"].to_numpy(dtype=float)

    m = DockingSurrogate(pca_components=100, use_descriptors=True)
    print(m.cross_validate(smiles, y))
    m.fit(smiles, y)

    demo = ["CCO", "c1ccccc1"]
    print("Demo preds:", m.predict(demo))

    ranked = rank_candidates(m, smiles[:200])
    print(ranked.head(20))

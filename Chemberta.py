import math
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
import rdkit
# -----------------------------
# 0) Load + clean
# -----------------------------
def canonicalize_smiles(smiles: str) -> Optional[str]:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)

def load_csv(csv_path: str, smiles_col="SMILES", y_col="best_score") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop common junk index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df[[smiles_col, y_col]].copy()
    df.rename(columns={smiles_col: "smiles", y_col: "y"}, inplace=True)

    df["smiles"] = df["smiles"].astype(str)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["smiles", "y"]).reset_index(drop=True)

    # Canonicalize + drop invalid
    can = []
    for s in df["smiles"].tolist():
        cs = canonicalize_smiles(s)
        can.append(cs)
    df["smiles"] = can
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    # Deduplicate identical molecules
    df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    return df

# -----------------------------
# 1) RDKit descriptor features
# -----------------------------
def rdkit_descriptors(smiles_list: List[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen

    feats = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            feats.append([np.nan] * 18)
            continue

        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        rot = Lipinski.NumRotatableBonds(m)
        rings = Lipinski.RingCount(m)
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(m)
        aliph_rings = rdMolDescriptors.CalcNumAliphaticRings(m)
        heavy = m.GetNumHeavyAtoms()
        fr_csp3 = rdMolDescriptors.CalcFractionCSP3(m)
        atoms = m.GetNumAtoms()
        hetero = rdMolDescriptors.CalcNumHeteroatoms(m)
        formal_charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
        n_chiral = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        mr = Crippen.MolMR(m)
        n_n = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N")
        n_o = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "O")

        feats.append([
            mw, logp, tpsa, hbd, hba, rot, rings, arom_rings, aliph_rings,
            heavy, fr_csp3, atoms, hetero, formal_charge, n_chiral, mr, n_n, n_o
        ])

    X = np.asarray(feats, dtype=float)
    # Median impute
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X

# -----------------------------
# 2) Scaffold groups (for honest CV)
# -----------------------------
def murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
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

# -----------------------------
# 3) ChemBERTa embedding (frozen)
# -----------------------------
class ChemBERTaFeaturizer:
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
        summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
        counts = mask.sum(dim=1).clamp(min=1.0)                         # (B,1)
        return summed / counts

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        import torch
        outs = []
        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i:i + self.batch_size]
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

# -----------------------------
# 4) Choose GBM regressor
# -----------------------------
def get_regressor(random_state: int = 42):
    # CatBoost
    try:
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
    except Exception:
        pass

    # LightGBM
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
        ), "lightgbm"
    except Exception:
        pass

    # sklearn fallback
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=4,
        max_iter=600,
        l2_regularization=1.0,
        random_state=random_state,
    ), "sklearn_histgb"

# -----------------------------
# 5) Train + CV + final model
# -----------------------------
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

        self.featurizer = None
        self.pca = None
        self.model = None
        self.model_kind = None

    def _build_X(self, smiles: List[str], fit_pca: bool) -> np.ndarray:
        from sklearn.decomposition import PCA

        if self.featurizer is None:
            self.featurizer = ChemBERTaFeaturizer(
                model_name=self.model_name, device=self.device, max_length=128, batch_size=16
            )

        X_bert = self.featurizer.transform(smiles)
        if self.use_descriptors:
            X_desc = rdkit_descriptors(smiles)
            X = np.hstack([X_bert, X_desc])
        else:
            X = X_bert

        if self.pca_components is not None:
            if fit_pca:
                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
                X = self.pca.fit_transform(X)
            else:
                X = self.pca.transform(X)
        return X

    def cross_validate(self, smiles: List[str], y: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from scipy.stats import spearmanr

        groups = scaffold_groups(smiles)
        gkf = GroupKFold(n_splits=n_splits)

        # Build features once (fit PCA on full set for CV convenience)
        X = self._build_X(smiles, fit_pca=True)

        rows = []
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
            model, kind = get_regressor(self.random_state + fold)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])

            mae = mean_absolute_error(y[te], pred)
            rmse = math.sqrt(mean_squared_error(y[te], pred))
            spr = spearmanr(y[te], pred).correlation
            rows.append({"fold": fold, "model": kind, "MAE": mae, "RMSE": rmse, "Spearman": spr})

        return pd.DataFrame(rows)

    def fit(self, smiles: List[str], y: np.ndarray):
        X = self._build_X(smiles, fit_pca=True)
        self.model, self.model_kind = get_regressor(self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X = self._build_X(smiles, fit_pca=False)
        return np.asarray(self.model.predict(X), dtype=float)

# -----------------------------
# Example usage
# -----------------------------
# df = load_csv("your_file.csv", smiles_col="SMILES", y_col="best_score")
# smiles = df["smiles"].tolist()
# y = df["y"].values.astype(float)
#
# surrogate = DockingSurrogate(pca_components=100, use_descriptors=True)
# cv = surrogate.cross_validate(smiles, y, n_splits=5)
# print(cv)
# print("CV mean:", cv[["MAE","RMSE","Spearman"]].mean().to_dict())
#
# surrogate.fit(smiles, y)
# print(surrogate.predict(["CCO", "c1ccccc1"]))

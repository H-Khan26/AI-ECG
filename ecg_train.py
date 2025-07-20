# ai_ecg/train.py

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data    import Dataset, DataLoader, Subset
from sklearn.ensemble    import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics     import roc_auc_score
from sklearn.utils       import resample

# from ecg.augmentations import ECGAugmentations
# from ecg.models        import TwoTower

# ─── USER CONFIG ──────────────────────────────────────────────────────────────

# Your master CSV from xml_to_csv + append_ecg_features
CSV_PATH   = "data/ecg_master.csv"

# Lead order in the CSV
LEAD_ORDER = ["I","II","V1","V2","V3","V4","V5","V6"]

# Tabular features to pull from CSV
TAB_FEATURES = [
    # demographics & intervals
    "PatientAge", "Gender",
    "AtrialRate","VentricularRate","HeartRate",
    "QTInterval","QRSDuration","TAxis", "Abnormal",
    # clinical covariates
    "SMOKE","acs_BEFORE_ICI","arrhythmia_BEFORE_ICI",
    "cad_BEFORE_ICI","hf_BEFORE_ICI","cardiacarrest_BEFORE_ICI",
    "stroke_BEFORE_ICI","AGE_AT_FIRST_ICI","Lifetime_ICI_COUNTS",
    "ICI_GROUP","htn","hld","dm","CR","ef_prior"
]

# RF / bootstrap params
RF_TREES   = 200
BOOTSTRAPS = 200

# CNN params
BATCH_SIZE = 16
N_EPOCHS   = 30
LR         = 1e-3
WD         = 1e-5  # weight decay
N_SPLITS   = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── UTIL: BOOTSTRAP VALIDATION ────────────────────────────────────────────────

def bootstrap_val(model, X, y, n_boot=200, random_state=0):
    """
    Fit `model` once on (X,y), then bootstrap the evaluation to get a 95% CI on ROC-AUC.
    """
    rng  = np.random.RandomState(random_state)
    p0   = model.fit(X, y).predict_proba(X)[:,1]
    aucs = []
    n    = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, pb = y[idx], p0[idx]
        # skip if only one class
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
    mean = np.mean(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"Bootstrapped RF ROC-AUC: {mean:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])")
    return mean, (lo, hi)


# ─── LOADERS: TABULAR & WAVEFORM ───────────────────────────────────────────────

def load_tabular(df: pd.DataFrame):
    """
    Extract X_tab (numpy) and y (0/1) from df[TAB_FEATURES + 'Group'].
    """
    df = df.copy()
    df["Group"] = df["Group"].str.strip()
    y = (df["Group"] == "CTRCD").astype(int).values

    tab = df[TAB_FEATURES].copy()
    # numeric vs. categorical
    num_cols = tab.select_dtypes(include=["int64","float64"]).columns
    cat_cols = tab.select_dtypes(include=["object","bool","category"]).columns

    X_num = tab[num_cols].fillna(0).values
    X_cat = pd.get_dummies(tab[cat_cols].fillna("__MISSING__")).values

    X_tab = np.hstack([X_num, X_cat])
    return X_tab.astype(np.float32), y


def load_waveforms(df: pd.DataFrame):
    """
    Parse JSON-encoded Lead_<lead> columns into an array [N, T, L].
    """
    N = len(df)
    L = len(LEAD_ORDER)
    # infer T from first row
    first = np.array(json.loads(df.iloc[0][f"Lead_{LEAD_ORDER[0]}"]), dtype=float)
    T = first.shape[0]

    X = np.zeros((N, T, L), dtype=np.float32)
    for i, row in df.iterrows():
        for j, lead in enumerate(LEAD_ORDER):
            sig = np.array(json.loads(row[f"Lead_{lead}"]), dtype=float)
            if sig.shape[0] != T:
                # fallback linear interp
                sig = np.interp(
                    np.linspace(0, sig.shape[0], T, endpoint=False),
                    np.arange(sig.shape[0]),
                    sig
                )
            X[i, :, j] = sig
    return X


# ─── DATASET & TRAIN/EVAL LOOP ────────────────────────────────────────────────

class ECGDataset(Dataset):
    def __init__(self, X_ecg, X_tab, y, augment=None):
        self.X_ecg = X_ecg
        self.X_tab = X_tab
        self.y     = y
        self.aug   = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        ecg = self.X_ecg[idx]               # [T, L]
        if self.aug:
            ecg = self.aug(ecg)            # still [T, L]
        # to [L, T] for Conv1d
        ecg_t = torch.from_numpy(ecg.T).float()
        tab   = torch.from_numpy(self.X_tab[idx]).float()
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return ecg_t, tab, label


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for ecg, tab, y in loader:
        ecg, tab, y = ecg.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(ecg, tab)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_p, all_y = [], []
    for ecg, tab, y in loader:
        ecg, tab = ecg.to(DEVICE), tab.to(DEVICE)
        p = model(ecg, tab).cpu().numpy()
        all_p.append(p); all_y.append(y.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # 1) load & split
    df = pd.read_csv(CSV_PATH)
    X_tab, y   = load_tabular(df)
    X_ecg      = load_waveforms(df)

    # 2) RF baseline + bootstrap CI
    print("\n=== RANDOM FOREST BASELINE ===")
    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        class_weight="balanced",
        random_state=42
    )
    bootstrap_val(rf, X_tab, y, n_boot=BOOTSTRAPS, random_state=42)

    # 3) two-tower CNN + 5-fold CV
    print("\n=== TWO-TOWER CNN + MLP (5-fold CV) ===")
    aug = ECGAugmentations(
        noise_std=0.01,
        max_shift_s=0.2,
        crop_len_s=2.0,
        fs=250,
        warp_pct=0.05,
        drop_lead_p=0.1
    )
    dataset = ECGDataset(X_ecg, X_tab, y, augment=aug)
    skf     = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_ecg, y), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")
        tr_ds = Subset(dataset, tr_idx)
        va_ds = Subset(dataset, va_idx)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE)

        model   = TwoTower(tab_in_dim=X_tab.shape[1]).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        for ep in range(1, N_EPOCHS+1):
            loss = train_epoch(model, tr_dl, optimizer, loss_fn)
            p_val, y_val = eval_epoch(model, va_dl)
            auc = roc_auc_score(y_val, p_val)
            print(f" Epoch {ep:02d}: loss={loss:.3f}, AUC={auc:.3f}", end="\r")
            best_auc = max(best_auc, auc)

        print(f"\n → Best AUC fold {fold}: {best_auc:.3f}")
        fold_scores.append(best_auc)

    mean_auc, std_auc = np.mean(fold_scores), np.std(fold_scores)
    print(f"\nCV ROC-AUC: {mean_auc:.3f} ± {std_auc:.3f}")


if __name__ == "__main__":
    main()

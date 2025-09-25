#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, random, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser("Mamba RUL training (with auto by-file validation split)")
parser.add_argument('--data-dir', type=str, required=True, help='Training folder (CSV files)')
parser.add_argument('--outdir', type=str, default='rul_model_mamba', help='Where to save checkpoints/plots')

parser.add_argument('--use-cuda', action='store_true', help='Use CUDA')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=1e-5)

parser.add_argument('--hidden-dim', type=int, default=16)
parser.add_argument('--layer-num', type=int, default=2)

args = parser.parse_args()
device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')


# -------------------------
# RUL Normalization utilities
# -------------------------
def normalize_rul_by_max(rul_values: np.ndarray, verbose: bool = False):
    """
    Normalize RUL values by maximum value to [0,1] range.    
    Args:
        rul_values: RUL array 
        verbose: Print normalization info
    
    Returns:
        normalized_rul: RUL values in [0,1] range
        max_rul: The maximum RUL value used for normalization
    """
    rul_clean = rul_values[np.isfinite(rul_values)]
    if len(rul_clean) == 0:
        if verbose: print("[RUL] Warning: No valid RUL values found")
        return rul_values, 1.0
    
    max_rul = np.max(rul_clean)
    min_rul = np.min(rul_clean)
    normalized_rul = np.clip(rul_values / max_rul, 0.0, 1.0)
    
    if verbose: 
        print(f"[RUL] Normalizing by max_rul={max_rul}")
        print(f"[RUL] Range: [{min_rul:.1f}, {max_rul:.1f}] -> [0, 1]")
    
    return normalized_rul, max_rul


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)


def find_csvs(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))
    if not files:
        raise ValueError(f"No CSV files under {folder}")
    return files


def read_one_csv(path: str):
    df = pd.read_csv(path)

    if 'RUL' not in df.columns:
        raise ValueError(f"{path} has no 'RUL' column")
    
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].interpolate('linear', limit_direction='both')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    rul_raw = df['RUL'].astype(float).values
    y, max_rul = normalize_rul_by_max(rul_raw, verbose=True)
    
    drop_cols = ['RUL']
    for col in ['SOH', 'Profile']:
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=drop_cols).values.astype(float)

    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[finite], y[finite], os.path.basename(path), max_rul


def load_folder(folder: str):
    data = []
    for f in find_csvs(folder):
        try:
            X, y, name, max_rul = read_one_csv(f)
            if len(y) == 0:
                print(f"[warn] skip empty after cleaning: {name}")
                continue
            data.append({'file': name, 'X': X, 'y': y, 'max_rul': max_rul})
            print(f"[load] {name:20s} -> X{X.shape} y{y.shape} max_rul={max_rul}")
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
    if not data:
        raise ValueError(f"No valid CSVs in {folder}")
    return data


def split_train_val_by_file(data, ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * ratio))
    val_idx = set(idx[:n_val])
    tr, va = [], []
    for i, d in enumerate(data):
        (va if i in val_idx else tr).append(d)
    print(f"[split] by-file ratio={ratio} -> train={len(tr)} files, val={len(va)} files")
    return tr, va


# -------------------------
# Model
# -------------------------
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers):
        super().__init__()
        self.config = MambaConfig(d_model=hidden, n_layers=layers)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Mamba(self.config),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, L, F)
        y = self.net(x)    # (B, L, 1)
        return y[..., 0]   # (B, L)


# -------------------------
# Training
# -------------------------
def train_model(train_list, val_list):
    Scaler = RobustScaler
    scaler = Scaler()
    Xcat = np.concatenate([d['X'] for d in train_list], axis=0)
    scaler.fit(Xcat)

    for d in train_list:
        d['X'] = scaler.transform(d['X'])
        d['X'][~np.isfinite(d['X'])] = 0.0
    for d in val_list:
        d['X'] = scaler.transform(d['X'])
        d['X'][~np.isfinite(d['X'])] = 0.0

    in_dim = train_list[0]['X'].shape[1]
    model = Net(in_dim=in_dim, out_dim=1, hidden=args.hidden_dim, layers=args.layer_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf')
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "best_model.pth")

    max_rul_info = {d['file']: d['max_rul'] for d in train_list + val_list}

    for e in range(args.epochs):
        model.train()
        train_losses = []
        for d in train_list:
            xt = torch.from_numpy(d['X']).float().unsqueeze(0).to(device)
            yt = torch.from_numpy(d['y']).float().unsqueeze(0).to(device)
            pred = model(xt)
            loss = F.l1_loss(pred, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for d in val_list:
                xt = torch.from_numpy(d['X']).float().unsqueeze(0).to(device)
                yt = torch.from_numpy(d['y']).float().unsqueeze(0).to(device)
                pred = model(xt)
                val_losses.append(F.l1_loss(pred, yt).item())
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'model_state_dict': model.state_dict(),
                        'in_dim': in_dim,
                        'hidden': args.hidden_dim,
                        'layers': args.layer_num,
                        'task': 'RUL',
                        'max_rul_info': max_rul_info}, ckpt_path)
            print(f"[save] epoch {e:03d} | val_loss={avg_val:.6f} (new best)")

        if e % 10 == 0 or e == args.epochs - 1:
            print(f"Epoch {e:03d} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

    print(f"[done] best val_loss={best_val:.6f}, model saved -> {ckpt_path}")
    
    pickle.dump(scaler, open(os.path.join(args.outdir, "best_model_scaler.pkl"), "wb"))
    return model, scaler


# -------------------------
# Main
# -------------------------
def main():
    print("[load] data ...")
    all_list = load_folder(args.data_dir)
    train_list, val_list = split_train_val_by_file(all_list, ratio=0.2, seed=args.seed)
    train_model(train_list, val_list)
    


if __name__ == "__main__":
    main()

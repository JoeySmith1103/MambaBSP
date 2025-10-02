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
parser = argparse.ArgumentParser("Mamba SOC training with data augmentation")
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--outdir', type=str, default='soc_model_mamba_augmented')

parser.add_argument('--use-cuda', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=1e-5)

parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--layer-num', type=int, default=2)

parser.add_argument('--min-seq-len', type=int, default=300)
parser.add_argument('--num-augment', type=int, default=6)

args = parser.parse_args()
device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')


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


def detect_and_normalize_soc(soc_values: np.ndarray, verbose: bool = False):
    soc_clean = soc_values[np.isfinite(soc_values)]
    if len(soc_clean) == 0:
        if verbose: print("[SOC] Warning: No valid SOC values found")
        return soc_values, 1.0
    
    soc_min, soc_max = np.min(soc_clean), np.max(soc_clean)
    
    if soc_max <= 1.1 and soc_min >= -0.1:
        if verbose: print(f"[SOC] Detected [0,1] format: range [{soc_min:.3f}, {soc_max:.3f}]")
        return np.clip(soc_values, 0.0, 1.0), 1.0
    elif soc_max <= 110 and soc_min >= -10:
        if verbose: print(f"[SOC] Detected [0,100] format: range [{soc_min:.1f}, {soc_max:.1f}] -> converting to [0,1]")
        return np.clip(soc_values / 100.0, 0.0, 1.0), 0.01
    elif soc_max > 10:
        if verbose: print(f"[SOC] Unknown format with max={soc_max:.1f}, assuming [0,100] -> converting to [0,1]")
        return np.clip(soc_values / 100.0, 0.0, 1.0), 0.01
    else:
        if verbose: print(f"[SOC] Assuming [0,1] format: range [{soc_min:.3f}, {soc_max:.3f}]")
        return np.clip(soc_values, 0.0, 1.0), 1.0


def find_csvs(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))
    if not files:
        raise ValueError(f"No CSV files under {folder}")
    return files


def read_one_csv(path: str):
    df = pd.read_csv(path)

    if 'SOC' not in df.columns:
        raise ValueError(f"{path} has no 'SOC' column")
    if not {'Current', 'Voltage'}.issubset(df.columns):
        raise ValueError(f"{path} must contain 'Current' and 'Voltage' columns")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].interpolate('linear', limit_direction='both')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    df['Power'] = df['Current'] * df['Voltage']
    df['Power_Squared'] = df['Power'] ** 2
    
    soc_raw = df['SOC'].astype(float).values
    y, _ = detect_and_normalize_soc(soc_raw, verbose=True)
    
    drop_cols = ['SOC']
    if 'Profile' in df.columns:
        drop_cols.append('Profile')
    if 'Time' in df.columns:
        drop_cols.append('Time')
    
    X = df.drop(columns=drop_cols, errors='ignore').values.astype(float)

    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[finite], y[finite], os.path.basename(path)


def augment_data(X, y, num_augment, min_seq_len):
    total_len = len(y)
    if total_len < min_seq_len:
        return [(X, y)] 
    
    augmented = [(X, y)]
    
    for _ in range(num_augment):
        max_start = total_len - min_seq_len
        start = np.random.randint(0, max_start + 1)
        
        max_len = total_len - start
        seq_len = np.random.randint(min_seq_len, max_len + 1)
        
        end = start + seq_len
        
        X_aug = X[start:end]
        y_aug = y[start:end]
        
        augmented.append((X_aug, y_aug))
    
    return augmented


def load_folder_with_augmentation(folder: str, num_augment, min_seq_len):
    data = []
    for f in find_csvs(folder):
        try:
            X, y, name = read_one_csv(f)
            if len(y) == 0:
                print(f"[warn] skip empty: {name}")
                continue
            
            augmented_sequences = augment_data(X, y, num_augment, min_seq_len)
            
            for idx, (X_aug, y_aug) in enumerate(augmented_sequences):
                soc_range = f"[{y_aug[0]:.2f}, {y_aug[-1]:.2f}]"
                data.append({
                    'file': f"{name}_aug{idx}" if idx > 0 else name,
                    'X': X_aug,
                    'y': y_aug
                })
                if idx == 0:
                    print(f"[load] {name:20s} -> X{X_aug.shape}, SOC{soc_range}")
                else:
                    print(f"  [aug{idx}] {name:20s} -> X{X_aug.shape}, SOC{soc_range}")
                    
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
    
    if not data:
        raise ValueError(f"No valid CSVs in {folder}")
    
    print(f"\n[augment] origin file num: {len(find_csvs(folder))}, after augmentation: {len(data)}")
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
    print(f"[split] train={len(tr)} sequences, val={len(va)} sequences")
    return tr, va


# -------------------------
# Model
# -------------------------
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers):
        super().__init__()
        self.config = MambaConfig(d_model=hidden, n_layers=layers)
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )
        
        self.mamba = Mamba(self.config)
        
        self.feature_refine = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input_proj(x)
        h_mamba = self.mamba(h)
        h = h + h_mamba
        h = self.feature_refine(h)
        y = self.output_proj(h)
        return y[..., 0]


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

    for e in range(args.epochs):
        model.train()
        random.shuffle(train_list)
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'in_dim': in_dim,
                'hidden': args.hidden_dim,
                'layers': args.layer_num
            }, ckpt_path)
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
    print("[Data Augmentation Mode]")
    print(f"[config] every file generate {args.num_augment} subsequences（different initial SOC）")
    print(f"[config] subseq min len: {args.min_seq_len}")
    print(f"[config] Hidden: {args.hidden_dim}, Layers: {args.layer_num}")
    print()
    
    all_list = load_folder_with_augmentation(
        args.data_dir, 
        num_augment=args.num_augment,
        min_seq_len=args.min_seq_len
    )
    
    train_list, val_list = split_train_val_by_file(all_list, ratio=0.2, seed=args.seed)
    train_model(train_list, val_list)


if __name__ == "__main__":
    main()

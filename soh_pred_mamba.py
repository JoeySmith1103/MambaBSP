#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler

import torch
import torch.nn as nn
from mamba import Mamba, MambaConfig
from scipy.signal import savgol_filter


# -------------------------
# Model (mirror training)
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

    def forward(self, x):                 # x: (B, L, F)
        y = self.net(x)                   # (B, L, 1)
        return y[..., 0]                  # (B, L)


# -------------------------
# SOH Normalization utilities (mirror training)
# -------------------------
def detect_and_normalize_soh(soh_values: np.ndarray, verbose: bool = False):
    """
    Automatically detect SOH format and normalize to [0,1] range.
    """
    soh_clean = soh_values[np.isfinite(soh_values)]
    if len(soh_clean) == 0:
        if verbose: print("[SOH] Warning: No valid SOH values found")
        return soh_values, 1.0
    
    soh_min, soh_max = np.min(soh_clean), np.max(soh_clean)
    
    # Check if already in [0,1] range (with some tolerance)
    if soh_max <= 1.1 and soh_min >= -0.1:
        if verbose: print(f"[SOH] Detected [0,1] format: range [{soh_min:.3f}, {soh_max:.3f}]")
        return np.clip(soh_values, 0.0, 1.0), 1.0
    
    # Check if in [0,100] range
    elif soh_max <= 110 and soh_min >= -10:
        if verbose: print(f"[SOH] Detected [0,100] format: range [{soh_min:.1f}, {soh_max:.1f}] -> converting to [0,1]")
        return np.clip(soh_values / 100.0, 0.0, 1.0), 0.01
    
    # Unknown format - try to guess based on max value
    elif soh_max > 10:
        if verbose: print(f"[SOH] Unknown format with max={soh_max:.1f}, assuming [0,100] -> converting to [0,1]")
        return np.clip(soh_values / 100.0, 0.0, 1.0), 0.01
    
    else:
        if verbose: print(f"[SOH] Assuming [0,1] format: range [{soh_min:.3f}, {soh_max:.3f}]")
        return np.clip(soh_values, 0.0, 1.0), 1.0


# -------------------------
# IO helpers (mirror training)
# -------------------------
def find_csvs(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))
    if not files:
        raise ValueError(f"No CSV files under {folder}")
    return files


def read_one_csv(path: str, has_true_label: bool = True):
    df = pd.read_csv(path)
    if has_true_label and 'SOH' not in df.columns:
        raise ValueError(f"{path} has no 'SOH' column")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].interpolate('linear', limit_direction='both')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    if has_true_label or 'SOH' in df.columns:
        soh_raw = df['SOH'].astype(float).values
        y, soh_scale = detect_and_normalize_soh(soh_raw, verbose=True)
        drop_cols = ['SOH']
    else:
        y = None
        drop_cols = []
    
    for col in ['RUL', 'Profile']:
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=drop_cols).values.astype(float)

    if has_true_label:
        finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[finite], y[finite], os.path.basename(path)
    else:
        finite = np.all(np.isfinite(X), axis=1)
        return X[finite], None, os.path.basename(path)


def load_folder(folder: str, has_true_label: bool = True):
    data = []
    for f in find_csvs(folder):
        try:
            X, y, name = read_one_csv(f, has_true_label)
            if has_true_label and len(y) == 0:
                print(f"[warn] skip empty after cleaning: {name}")
                continue
            elif not has_true_label and len(X) == 0:
                print(f"[warn] skip empty after cleaning: {name}")
                continue
            data.append({'file': name, 'X': X, 'y': y})
            if has_true_label:
                print(f"[load] {name:20s} -> X{X.shape} y{y.shape}")
            else:
                print(f"[load] {name:20s} -> X{X.shape}")
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
    if not data:
        raise ValueError(f"No valid CSVs in {folder}")
    return data


# -------------------------
# Load model & scaler
# -------------------------
def load_model_and_scaler(model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    for k in ['model_state_dict', 'in_dim', 'hidden', 'layers']:
        if k not in ckpt:
            raise ValueError(f"Checkpoint missing key: {k}")

    if ckpt.get('task') != 'SOH':
        print(f"[warn] Model task is {ckpt.get('task')}, expected SOH")

    scaler_path_guess = os.path.join(os.path.dirname(model_path), "best_model_scaler.pkl")
    if not os.path.exists(scaler_path_guess):
        raise FileNotFoundError(f"Missing scaler: {scaler_path_guess}")

    with open(scaler_path_guess, 'rb') as f:
        scaler = pickle.load(f)

    model = Net(ckpt['in_dim'], 1, ckpt['hidden'], ckpt['layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, scaler


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def predict_one(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(X).float().unsqueeze(0).to(device)   # (1, L, F)
    out = model(xt).squeeze(0).detach().cpu().numpy()          # (L,)
    return np.clip(out, -1e-6, 1.0 + 1e-6)


# -------------------------
# Evaluate / Save
# -------------------------
def evaluate_and_save(test_list, preds, outdir: str, has_true_label: bool = True, enable_plot: bool = True):
    os.makedirs(outdir, exist_ok=True)
    
    if has_true_label:
        rows = []
        for d, p in zip(test_list, preds):
            y = d['y']; L = min(len(y), len(p))
            y = y[:L]; p = p[:L]

            mse = mean_squared_error(y, p)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, p)
            r2 = r2_score(y, p)

            rows.append(dict(file=d['file'], MSE=mse, RMSE=rmse, MAE=mae, R2=r2))
            
            if enable_plot:
                pred_sg = savgol_filter(p, window_length=min(51, len(p)//2*2+1), polyorder=3)
                # per-file curve with true and pred
                plt.figure(figsize=(12,5))
                plt.plot(y, label='True', lw=2)
                plt.plot(pred_sg, label='Pred', lw=2)
                plt.title(f"SOH Estimation - {d['file']}")
                plt.xlabel("Timestep"); plt.ylabel("SOH (0~1)")
                plt.grid(alpha=0.3); plt.legend()
                plt.savefig(os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.png"),
                            dpi=200, bbox_inches='tight')
                plt.close()

            # Save CSV with both true and pred
            pd.DataFrame({'SOH_true': y, 'SOH_pred': p}).to_csv(
                os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.csv"), index=False
            )

        df = pd.DataFrame(rows).sort_values('file')
        df.to_csv(os.path.join(outdir, 'metrics.csv'), index=False)
        print(df[['file', 'MSE', 'RMSE', 'MAE', 'R2']])
        print("\nAvg:", df[['MSE', 'RMSE', 'MAE', 'R2']].mean().to_dict())
    else:
        # No true labels - just save predictions
        for d, p in zip(test_list, preds):
            if enable_plot:
                pred_sg = savgol_filter(p, window_length=min(51, len(p)//2*2+1), polyorder=3)
                # per-file curve with pred only
                plt.figure(figsize=(12,5))
                plt.plot(pred_sg, label='Pred', lw=2)
                plt.title(f"SOH Estimation - {d['file']}")
                plt.xlabel("Timestep"); plt.ylabel("SOH (0~1)")
                plt.grid(alpha=0.3); plt.legend()
                plt.savefig(os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.png"),
                            dpi=200, bbox_inches='tight')
                plt.close()

            # Save CSV with pred only
            pd.DataFrame({'SOH_pred': p}).to_csv(
                os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.csv"), index=False
            )
        
        print("Predictions saved (no metrics calculated - no true labels)")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Mamba SOH testing (aligned with training)")
    ap.add_argument('--model-path', type=str, required=True)
    ap.add_argument('--test-dir', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='soh_pred_results_mamba')
    ap.add_argument('--use-cuda', action='store_true')
    ap.add_argument('--true-label', action='store_true',
                    help='Test data has true SOH labels for evaluation')
    ap.add_argument('--plot', action='store_true',
                    help='Generate visualization plots')
    args = ap.parse_args()

    device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')
    print("device =", device)
    print(f"true_label = {args.true_label}, plot = {args.plot}")

    print("[load] model & scaler")
    model, scaler = load_model_and_scaler(args.model_path, device)

    print("[load] testing data ...")
    test_list = load_folder(args.test_dir, args.true_label)

    print("[scale] apply train-time scaler")
    for d in test_list:
        Xs = scaler.transform(d['X'])
        Xs[~np.isfinite(Xs)] = 0.0
        d['X'] = Xs

    print("[infer] ...")
    preds = [predict_one(model, d['X'], device) for d in test_list]

    print("[eval] ...")
    os.makedirs(args.outdir, exist_ok=True)
    evaluate_and_save(test_list, preds, args.outdir, args.true_label, args.plot)
    print("Done.")

if __name__ == "__main__":
    main()

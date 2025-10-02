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
# RUL Normalization utilities (mirror training)
# -------------------------
def normalize_rul_by_max(rul_values: np.ndarray, verbose: bool = False):
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
# IO helpers (mirror training)
# -------------------------
def find_csvs(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))
    if not files:
        raise ValueError(f"No CSV files under {folder}")
    return files


def read_one_csv(path: str, has_true_label: bool = True):
    df = pd.read_csv(path)
    if has_true_label and 'RUL' not in df.columns:
        raise ValueError(f"{path} has no 'RUL' column")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].interpolate('linear', limit_direction='both')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    if has_true_label or 'RUL' in df.columns:
        rul_raw = df['RUL'].astype(float).values
        y, max_rul = normalize_rul_by_max(rul_raw, verbose=True)
        drop_cols = ['RUL']
    else:
        y = None
        max_rul = 1.0  # Default value when no true labels
        drop_cols = []
    
    for col in ['SOH', 'Profile']:
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=drop_cols).values.astype(float)

    if has_true_label:
        finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[finite], y[finite], os.path.basename(path), max_rul
    else:
        finite = np.all(np.isfinite(X), axis=1)
        return X[finite], None, os.path.basename(path), max_rul


def load_folder(folder: str, has_true_label: bool = True):
    data = []
    for f in find_csvs(folder):
        try:
            X, y, name, max_rul = read_one_csv(f, has_true_label)
            if has_true_label and len(y) == 0:
                print(f"[warn] skip empty after cleaning: {name}")
                continue
            elif not has_true_label and len(X) == 0:
                print(f"[warn] skip empty after cleaning: {name}")
                continue
            data.append({'file': name, 'X': X, 'y': y, 'max_rul': max_rul})
            if has_true_label:
                print(f"[load] {name:20s} -> X{X.shape} y{y.shape} max_rul={max_rul}")
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
def load_model_and_scaler(model_dir: str, device: torch.device):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")
    
    model_path = None
    for pth_file in pth_files:
        if 'best' in os.path.basename(pth_file).lower():
            model_path = pth_file
            break
    
    if model_path is None:
        model_path = pth_files[0]
    
    print(f"[load] Using model file: {os.path.basename(model_path)}")
    
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {model_dir}")
    
    scaler_path = None
    for pkl_file in pkl_files:
        if 'scaler' in os.path.basename(pkl_file).lower():
            scaler_path = pkl_file
            break
    
    if scaler_path is None:
        scaler_path = pkl_files[0]
    
    print(f"[load] Using scaler file: {os.path.basename(scaler_path)}")
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    for k in ['model_state_dict', 'in_dim', 'hidden', 'layers']:
        if k not in ckpt:
            raise ValueError(f"Checkpoint missing key: {k}")

    if ckpt.get('task') != 'RUL':
        print(f"[warn] Model task is {ckpt.get('task')}, expected RUL")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model = Net(ckpt['in_dim'], 1, ckpt['hidden'], ckpt['layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    max_rul_info = ckpt.get('max_rul_info', {})
    
    return model, scaler, max_rul_info


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
def evaluate_and_save(test_list, preds, outdir: str, max_rul_info: dict = None, has_true_label: bool = True, enable_plot: bool = True):
    os.makedirs(outdir, exist_ok=True)
    
    if has_true_label:
        rows = []
        for d, p in zip(test_list, preds):
            y_norm = d['y']; L = min(len(y_norm), len(p))
            y_norm = y_norm[:L]; p_norm = p[:L]
            
            max_rul = d['max_rul']
            y_orig = y_norm * max_rul
            p_orig = p_norm * max_rul

            mse = mean_squared_error(y_orig, p_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_orig, p_orig)
            r2 = r2_score(y_orig, p_orig)

            rows.append(dict(file=d['file'], MSE=mse, RMSE=rmse, MAE=mae, R2=r2))
            
            if enable_plot:
                try:
                    pred_sg = savgol_filter(p_orig, window_length=min(51, len(p_orig)//2*2+1), polyorder=3)
                except:
                    pred_sg = p_orig
                    
                # per-file curve with true and pred
                plt.figure(figsize=(12,5))
                plt.plot(y_orig, label='True', lw=2)
                plt.plot(pred_sg, label='Pred', lw=2)
                plt.title(f"RUL Estimation - {d['file']}")
                plt.xlabel("Timestep"); plt.ylabel("RUL (cycles)")
                plt.grid(alpha=0.3); plt.legend()
                plt.savefig(os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.png"),
                            dpi=200, bbox_inches='tight')
                plt.close()

            # Save CSV with both true and pred
            pd.DataFrame({'RUL_true': y_orig, 'RUL_pred': p_orig}).to_csv(
                os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.csv"), index=False
            )

        df = pd.DataFrame(rows).sort_values('file')
        df.to_csv(os.path.join(outdir, 'metrics.csv'), index=False)
        print(df[['file', 'MSE', 'RMSE', 'MAE', 'R2']])
        print("\nAvg:", df[['MSE', 'RMSE', 'MAE', 'R2']].mean().to_dict())
    else:
        # No true labels - just save predictions
        for d, p in zip(test_list, preds):
            max_rul = d['max_rul']  # This will be 1.0 by default
            p_orig = p * max_rul
            
            if enable_plot:
                try:
                    pred_sg = savgol_filter(p_orig, window_length=min(51, len(p_orig)//2*2+1), polyorder=3)
                except:
                    pred_sg = p_orig
                    
                # per-file curve with pred only
                plt.figure(figsize=(12,5))
                plt.plot(pred_sg, label='Pred', lw=2)
                plt.title(f"RUL Estimation - {d['file']}")
                plt.xlabel("Timestep"); plt.ylabel("RUL (normalized)")
                plt.grid(alpha=0.3); plt.legend()
                plt.savefig(os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.png"),
                            dpi=200, bbox_inches='tight')
                plt.close()

            # Save CSV with pred only
            pd.DataFrame({'RUL_pred': p_orig}).to_csv(
                os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.csv"), index=False
            )
        
        print("Predictions saved (no metrics calculated - no true labels)")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Mamba RUL testing (aligned with training)")
    ap.add_argument('--model-dir', type=str, required=True,
                    help='Directory containing model .pth and scaler .pkl files')
    ap.add_argument('--test-dir', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='rul_pred_results')
    ap.add_argument('--use-cuda', action='store_true')
    ap.add_argument('--true-label', action='store_true',
                    help='Test data has true RUL labels for evaluation')
    ap.add_argument('--plot', action='store_true',
                    help='Generate visualization plots')
    args = ap.parse_args()

    device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')
    print("device =", device)
    print(f"true_label = {args.true_label}, plot = {args.plot}")

    print("[load] model & scaler")
    model, scaler, max_rul_info = load_model_and_scaler(args.model_dir, device)

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
    evaluate_and_save(test_list, preds, args.outdir, max_rul_info, args.true_label, args.plot)
    print("Done.")

if __name__ == "__main__":
    main()

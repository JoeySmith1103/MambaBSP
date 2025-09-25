import os, glob, argparse, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler 

import torch
import torch.nn as nn
from scipy.signal import savgol_filter


# -------------------------
# LSTM Model (mirror training)
# -------------------------
class LSTMNet(nn.Module):
    def __init__(self, in_dim, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            output: (batch_size, sequence_length)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Apply fully connected layers to each time step
        output = self.fc(lstm_out)  # (batch_size, seq_len, 1)
        
        return output.squeeze(-1)  # (batch_size, seq_len)


# -------------------------
# IO helpers (mirror training)
# -------------------------
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

    y = df['SOC'].astype(float).values
    drop_cols = ['SOC']
    if 'Profile' in df.columns:
        drop_cols.append('Profile')
    X = df.drop(columns=drop_cols).values.astype(float)

    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[finite], y[finite], os.path.basename(path)


def load_folder(folder: str):
    data = []
    for f in find_csvs(folder):
        try:
            X, y, name = read_one_csv(f)
            if len(y) == 0:
                print(f"[warn] skip empty after cleaning: {name}")
                continue
            data.append({'file': name, 'X': X, 'y': y})
            print(f"[load] {name:20s} -> X{X.shape} y{y.shape}")
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
    
    required_keys = ['model_state_dict', 'in_dim', 'hidden_size', 'num_layers']
    for k in required_keys:
        if k not in ckpt:
            raise ValueError(f"Checkpoint missing key: {k}")

    scaler_path_guess = os.path.join(os.path.dirname(model_path), "best_model_scaler.pkl")
    if not os.path.exists(scaler_path_guess):
        raise FileNotFoundError(f"Missing scaler: {scaler_path_guess}")

    with open(scaler_path_guess, 'rb') as f:
        scaler = pickle.load(f)

    model = LSTMNet(
        in_dim=ckpt['in_dim'],
        hidden_size=ckpt['hidden_size'],
        num_layers=ckpt['num_layers'],
        dropout=ckpt.get('dropout', 0.1)
    ).to(device)
    
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
def evaluate_and_save(test_list, preds, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for d, p in zip(test_list, preds):
        y = d['y']; L = min(len(y), len(p))
        y = y[:L]; p = p[:L]

        mse = mean_squared_error(y, p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, p)
        r2 = r2_score(y, p)

        rows.append(dict(file=d['file'], MSE=mse, RMSE=rmse, MAE=mae, R2=r2))
        
        pred_sg = savgol_filter(p, window_length=min(51, len(p)//2*2+1), polyorder=3)
        
        plt.figure(figsize=(12,5))
        plt.plot(y, label='True SOC', lw=2, alpha=0.8)
        plt.plot(pred_sg, label='Pred SOC', lw=2, alpha=0.8)
        plt.title(f"LSTM SOC Estimation - {d['file']}")
        plt.xlabel("Timestep"); plt.ylabel("SOC (0~1)")
        plt.grid(alpha=0.3); plt.legend()
        plt.ylim(0, 1)
        
        plt.savefig(os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()

        pd.DataFrame({
            'SOC_true': y, 
            'SOC_pred_raw': p,
            'SOC_pred_smoothed': pred_sg
        }).to_csv(
            os.path.join(outdir, f"{os.path.splitext(d['file'])[0]}.csv"), index=False
        )

    df = pd.DataFrame(rows).sort_values('file')
    df.to_csv(os.path.join(outdir, 'metrics.csv'), index=False)
    
    print("\n" + "="*80)
    print("LSTM Results")
    print("="*80)
    print(df[['file', 'MSE', 'RMSE', 'MAE', 'R2']].to_string(index=False))
    print("\nAverage:")
    avg_metrics = df[['MSE', 'RMSE', 'MAE', 'R2']].mean()
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.6f}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("LSTM SOC testing (aligned with training)")
    ap.add_argument('--model-path', type=str, required=True, 
                    help='Path to trained LSTM model (.pth file)')
    ap.add_argument('--test-dir', type=str, required=True,
                    help='Directory containing test CSV files')
    ap.add_argument('--outdir', type=str, default='soc_pred_results_lstm',
                    help='Output directory for results')
    ap.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA for inference')
    args = ap.parse_args()

    device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')

    try:
        model, scaler = load_model_and_scaler(args.model_path, device)
        print(f"LSTM model: {args.model_path}")
    except Exception as e:
        print(f"Load model failed: {e}")
        return

    print("Testing data...")
    try:
        test_list = load_folder(args.test_dir)
    except Exception as e:
        print(f"load testing data failed: {e}")
        return

    for d in test_list:
        Xs = scaler.transform(d['X'])
        Xs[~np.isfinite(Xs)] = 0.0
        d['X'] = Xs

    print("Test inference...")
    preds = []
    for i, d in enumerate(test_list):
        pred = predict_one(model, d['X'], device)
        preds.append(pred)

    os.makedirs(args.outdir, exist_ok=True)
    evaluate_and_save(test_list, preds, args.outdir)


if __name__ == "__main__":
    main()

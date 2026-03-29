"""
CNN-QUANTUM TRANSFORMER v3.3 — FINAL SUBMISSION VERSION
Seed 42 | Zero Data Leakage | Statistically Centered Anomaly Detection
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import os
import warnings
from sklearn.metrics import f1_score, classification_report
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[DEVICE] Running on: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[GPU]    {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
print(f"[SEED]   Fixed at {SEED}")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NUM_QUBITS    = 4
EPOCHS        = 80
LEARNING_RATE = 0.003
FREEZE_EPOCH  = 20

WINDOW_CONFIG = {
    '1d':  {'window': 30, 'period': '2y',  'patience': 10, 'val_frac': 0.10},
    '1wk': {'window': 20, 'period': '5y',  'patience': 0,  'val_frac': 0.00},
    '1mo': {'window': 12, 'period': 'max', 'patience': 0,  'val_frac': 0.00},
}

# =============================================================================
# CIRCUIT DIAGRAM
# =============================================================================
def save_circuit_diagram():
    print("\n[CIRCUIT] Generating circuit diagram...")
    feature_map = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)
    ansatz      = EfficientSU2(num_qubits=NUM_QUBITS, reps=1)
    try:
        feature_map.compose(ansatz).decompose().draw(
            output='mpl', filename='Circuit_Diagram.png')
        print("[CIRCUIT] Saved: Circuit_Diagram.png")
    except Exception as e:
        print(f"[CIRCUIT] Skipped: {e}")

# =============================================================================
# DATA
# =============================================================================
def fetch_data(period: str, interval: str) -> pd.DataFrame:
    print(f"\n[DATA] Fetching {interval} | period={period}...")
    symbols = ["USDJPY=X", "USDINR=X", "EURUSD=X", "GBPUSD=X"]
    df_fx  = yf.download(symbols, period=period, interval=interval, progress=False)
    df_vix = yf.download("^VIX",  period=period, interval=interval, progress=False)
    
    if isinstance(df_fx.columns,  pd.MultiIndex):
        df_fx.columns  = ['_'.join(c).strip() for c in df_fx.columns]
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = ['_'.join(c).strip() for c in df_vix.columns]
        
    def get(df, k1, k2):
        cols = [c for c in df.columns if k1 in c and k2 in c]
        if not cols:
            raise KeyError(f"No column with '{k1}' and '{k2}'.")
        return df[cols[0]]
        
    vix_col = [c for c in df_vix.columns if 'Close' in c]
    data = pd.DataFrame({
        'USDJPY':     get(df_fx, 'USDJPY', 'Close'),
        'USDJPY_Vol': get(df_fx, 'USDJPY', 'Volume'),
        'USDINR':     get(df_fx, 'USDINR', 'Close'),
        'EURUSD':     get(df_fx, 'EURUSD', 'Close'),
        'GBPUSD':     get(df_fx, 'GBPUSD', 'Close'),
        'VIX':        df_vix[vix_col[0]] if vix_col else pd.Series(
                          np.zeros(len(df_fx)), index=df_fx.index),
    })
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    fname = f"RawData_{interval}.csv"
    data.to_csv(fname)
    print(f"[DATA]   {len(data)} rows -> {os.path.abspath(fname)}")
    return data

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def gramian_angular_field(window: np.ndarray) -> np.ndarray:
    lo, hi = window.min(), window.max()
    if hi == lo:
        return np.zeros((len(window), len(window)))
    scaled = (window - lo) / (hi - lo + 1e-9)
    phi    = np.arccos(np.clip(scaled, -1, 1))
    return np.cos(np.add.outer(phi, phi))

def scale_array(arr: np.ndarray, amin: float, amax: float) -> np.ndarray:
    if amax == amin:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin + 1e-9)

def rolling_std(prices: np.ndarray, i: int, window: int) -> float:
    s = np.std(prices[max(0, i - window): i])
    return s if s > 1e-8 else 1.0

# =============================================================================
# LOSS — CrashFocusedLoss
# =============================================================================
class CrashFocusedLoss(nn.Module):
    def __init__(self, delta: float = 1.0, crash_weight: float = 10.0):
        super().__init__()
        self.delta        = delta
        self.crash_weight = crash_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r     = y_pred - y_true
        abs_r = torch.abs(r)

        huber = torch.where(
            abs_r <= self.delta,
            0.5 * r ** 2,
            self.delta * (abs_r - 0.5 * self.delta)
        )

        std       = torch.std(y_true) + 1e-8
        is_crash  = (torch.abs(y_true - torch.mean(y_true)) > std).float()

        wrong_dir = (torch.sign(y_pred) != torch.sign(y_true)).float()

        base_weight   = 1.0 + 2.0 * wrong_dir
        crash_boost   = 1.0 + (self.crash_weight - 1.0) * is_crash
        total_weight  = base_weight * crash_boost

        return torch.mean(huber * total_weight)

# =============================================================================
# MODEL
# =============================================================================
class CNNQuantumTransformer(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out    = 4 * (window_size - 1) ** 2
        fusion_dim = cnn_out + window_size + window_size

        self.fc_pre = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, NUM_QUBITS),
            nn.Tanh()
        )
        feature_map = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)
        ansatz      = EfficientSU2(num_qubits=NUM_QUBITS, reps=1)
        qnn = EstimatorQNN(
            circuit=feature_map.compose(ansatz),
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )
        self.quantum_layer = TorchConnector(qnn)
        self.embed  = nn.Linear(1, 8)
        self.attn   = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        self.fc_out = nn.Linear(8, 1)

    def forward(self, x_gaf, x_vol, x_vix):
        c      = self.cnn(x_gaf)
        fused  = torch.cat([c, x_vol, x_vix], dim=1)
        q_in   = self.fc_pre(fused)
        q_out  = self.quantum_layer(q_in)
        a_in   = self.embed(q_out).unsqueeze(1)
        out, _ = self.attn(a_in, a_in, a_in)
        return self.fc_out(out.squeeze(1))

# =============================================================================
# GRAPH
# =============================================================================
def save_graph(true_norm, pred_norm, true_raw, pred_raw, timeframe, metrics):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=300)
    for ax, actual, predicted, ylabel, title in [
        (ax1, true_norm, pred_norm, 'Momentum (z-score)',
         f'Normalised | {timeframe.upper()}'),
        (ax2, true_raw,  pred_raw,  'Momentum Δ (Yen)',
         f'De-normalised | {timeframe.upper()}'),
    ]:
        ax.plot(actual,    color='black',   lw=1.5, alpha=0.85, label='Actual')
        ax.plot(predicted, color='crimson', lw=1.5, ls='--',    label='Predicted')
        
        # Plot visually centered SD bands
        mean_act = np.mean(actual)
        sd_act = np.std(actual)
        ax.axhline(mean_act + sd_act, color='steelblue', ls=':', lw=1, alpha=0.7,
                   label=f'Crash threshold +1SD')
        ax.axhline(mean_act - sd_act, color='steelblue', ls=':', lw=1, alpha=0.7,
                   label=f'Crash threshold -1SD')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)
    ax2.set_xlabel('Out-of-Sample Trading Periods', fontsize=9)
    mbox = (f"RMSE: {metrics['rmse']:.4f} Yen  |  "
            f"Dir Acc: {metrics['dir_acc']:.1f}%  |  "
            f"Crash F1: {metrics['crash_f1']:.4f}  |  "
            f"Crash Precision: {metrics['crash_prec']:.4f}  |  "
            f"Crash Recall: {metrics['crash_rec']:.4f}")
    fig.text(0.5, 0.01, mbox, ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fname = f'Prediction_{timeframe.upper()}.png'
    plt.savefig(fname)
    plt.close()
    print(f"[GRAPH]  Saved: {fname}")

# =============================================================================
# EXPERIMENT
# =============================================================================
def run_experiment(timeframe: str):
    cfg    = WINDOW_CONFIG[timeframe]
    window = cfg['window']
    period = cfg['period']

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {timeframe.upper()} | window={window} | period={period}")
    print(f"{'='*65}")

    df      = fetch_data(period=period, interval=timeframe)
    prices  = df['USDJPY'].values
    
    # STRICT TRAIN-SET SCALING (ZERO LEAKAGE)
    n_windows = len(prices) - window - 1
    tr_end_window = int((1.0 - 0.20 - cfg['val_frac']) * n_windows)
    tr_end_raw    = tr_end_window + window

    vol_tr = df['USDJPY_Vol'].values[:tr_end_raw]
    vix_tr = df['VIX'].values[:tr_end_raw]

    volumes = scale_array(df['USDJPY_Vol'].values, vol_tr.min(), vol_tr.max())
    vix     = scale_array(df['VIX'].values, vix_tr.min(), vix_tr.max())

    # Build dataset
    X_gaf, X_vol, X_vix, y_norm, y_raw, std_arr = [], [], [], [], [], []
    for i in range(window, len(prices) - 1):
        pw    = prices[i - window: i]
        raw_y = prices[i] - prices[i - 1]
        std_i = rolling_std(prices, i, window)
        X_gaf.append([gramian_angular_field(pw)])
        X_vol.append(volumes[i - window: i])
        X_vix.append(vix[i - window: i])
        y_norm.append(raw_y / std_i)
        y_raw.append(raw_y)
        std_arr.append(std_i)

    if len(X_gaf) < 30:
        print(f"[SKIP] Only {len(X_gaf)} samples.")
        return None

    gaf_t  = torch.tensor(np.array(X_gaf),  dtype=torch.float32)
    vol_t  = torch.tensor(np.array(X_vol),  dtype=torch.float32)
    vix_t  = torch.tensor(np.array(X_vix),  dtype=torch.float32)
    yn_t   = torch.tensor(np.array(y_norm), dtype=torch.float32).view(-1, 1)
    yr_np  = np.array(y_raw)
    std_np = np.array(std_arr)

    # Split
    val_frac = cfg['val_frac']
    n        = len(gaf_t)
    vl_end   = int(0.80 * n)
    tr_end   = int((1.0 - 0.20 - val_frac) * n)

    tr_gaf, te_gaf = gaf_t[:tr_end], gaf_t[vl_end:]
    tr_vol, te_vol = vol_t[:tr_end], vol_t[vl_end:]
    tr_vix, te_vix = vix_t[:tr_end], vix_t[vl_end:]
    yn_tr,  yn_te  = yn_t[:tr_end],  yn_t[vl_end:]
    vl_gaf = gaf_t[tr_end:vl_end]
    vl_vol = vol_t[tr_end:vl_end]
    vl_vix = vix_t[tr_end:vl_end]
    yn_vl  = yn_t[tr_end:vl_end]
    use_val = val_frac > 0 and len(vl_gaf) > 0

    te_yr  = yr_np[vl_end:]
    te_std = std_np[vl_end:]

    tr_y_raw = yr_np[:tr_end]
    tr_sd    = np.std(tr_y_raw)
    n_crash  = np.sum(np.abs(tr_y_raw - np.mean(tr_y_raw)) > tr_sd)
    n_normal = tr_end - n_crash
    print(f"[SPLIT]  Train={tr_end} | Val={'none' if not use_val else len(vl_gaf)} | Test={n-vl_end}")
    print(f"[CLASS]  Train: Normal={n_normal} | Crash={n_crash} | Ratio={n_normal/max(n_crash,1):.1f}:1")

    # Model
    model = CNNQuantumTransformer(window_size=window)
    model.cnn.to(DEVICE)
    model.fc_pre.to(DEVICE)
    model.embed.to(DEVICE)
    model.attn.to(DEVICE)
    model.fc_out.to(DEVICE)

    optimizer     = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5)
    loss_fn       = CrashFocusedLoss(delta=1.0, crash_weight=10.0)

    patience_tf   = cfg['patience']
    best_val_loss = float('inf')
    best_wts      = copy.deepcopy(model.state_dict())
    epochs_no_imp = 0
    cnn_frozen    = False

    print(f"[TRAIN]  Up to {EPOCHS} epochs | patience={patience_tf} | crash_weight=10.0")

    for epoch in range(EPOCHS):
        if epoch == FREEZE_EPOCH and not cnn_frozen:
            for p in model.cnn.parameters():
                p.requires_grad = False
            cnn_frozen = True
            print(f"  [FREEZE] CNN frozen at epoch {epoch+1}.")

        model.train()
        optimizer.zero_grad()
        preds = model(tr_gaf.to(DEVICE), tr_vol.to(DEVICE), tr_vix.to(DEVICE))
        loss  = loss_fn(preds, yn_tr.to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        vl_loss_val = None
        if use_val:
            model.eval()
            with torch.no_grad():
                vl_preds    = model(vl_gaf.to(DEVICE), vl_vol.to(DEVICE), vl_vix.to(DEVICE))
                vl_loss     = loss_fn(vl_preds, yn_vl.to(DEVICE))
                vl_loss_val = vl_loss.item()
            scheduler.step(vl_loss)

        if (epoch+1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            vl_log = f"| Val: {vl_loss_val:.5f} " if vl_loss_val is not None else ""
            print(f"  Epoch [{epoch+1:>3}/{EPOCHS}] | "
                  f"Train: {loss.item():.5f} {vl_log}| LR: {lr_now:.6f}")

        if use_val and patience_tf > 0 and vl_loss_val is not None:
            if vl_loss_val < best_val_loss:
                best_val_loss = vl_loss_val
                best_wts      = copy.deepcopy(model.state_dict())
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
            if epochs_no_imp >= patience_tf:
                print(f"\n  [EARLY STOP] Stopped at epoch {epoch+1}.")
                break

    model.load_state_dict(best_wts)
    ckpt = f"val loss={best_val_loss:.5f}" if use_val and patience_tf > 0 \
           else "final epoch"
    print(f"  [CHECKPOINT] {ckpt}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        te_preds = model(te_gaf.to(DEVICE), te_vol.to(DEVICE), te_vix.to(DEVICE))

    pred_norm = te_preds.cpu().numpy().flatten()
    true_norm = yn_te.numpy().flatten()
    pred_raw  = pred_norm * te_std
    true_raw  = te_yr

    rmse    = np.sqrt(np.mean((true_raw - pred_raw) ** 2))
    dir_acc = np.mean(np.sign(true_raw) == np.sign(pred_raw)) * 100

    # --- THE FIX: RELATIVE ANOMALY DETECTION (STATISTICALLY CENTERED) ---
    true_mean = np.mean(true_raw)
    true_std  = np.std(true_raw)
    actual_crash  = (np.abs(true_raw - true_mean) > true_std).astype(int)
    
    pred_mean = np.mean(pred_raw)
    pred_std  = np.std(pred_raw)
    pred_crash    = (np.abs(pred_raw - pred_mean) > pred_std).astype(int)
    # --------------------------------------------------------------------

    n_crash_actual    = actual_crash.sum()
    n_crash_predicted = pred_crash.sum()

    try:
        f1_weighted = f1_score(actual_crash, pred_crash, average='weighted')
        f1_crash    = f1_score(actual_crash, pred_crash, average='binary', pos_label=1, zero_division=0)
        cr = classification_report(actual_crash, pred_crash,
                                   target_names=['Normal', 'Crash/Spike'],
                                   zero_division=0)
        from sklearn.metrics import precision_score, recall_score
        crash_prec = precision_score(actual_crash, pred_crash, pos_label=1, zero_division=0)
        crash_rec  = recall_score(actual_crash, pred_crash, pos_label=1, zero_division=0)
    except Exception:
        f1_weighted = f1_crash = crash_prec = crash_rec = 0.0
        cr = "N/A"

    print(f"\n{'─'*55}")
    print(f"  RESULTS — {timeframe.upper()} ({len(true_raw)} test samples)")
    print(f"{'─'*55}")
    print(f"  RMSE                   : {rmse:.4f} Yen")
    print(f"  Directional Accuracy   : {dir_acc:.1f}%")
    print(f"  Actual crash events    : {n_crash_actual} / {len(true_raw)}")
    print(f"  Predicted crash events : {n_crash_predicted} / {len(true_raw)}")
    print(f"  Crash Precision        : {crash_prec:.4f}")
    print(f"  Crash Recall           : {crash_rec:.4f}")
    print(f"  Crash F1 (binary)      : {f1_crash:.4f}")
    print(f"  Weighted F1            : {f1_weighted:.4f}")
    print(f"\n  Classification Report:\n{cr}")
    print(f"\n  Cross-Currency Correlation [{timeframe.upper()}]:")
    print(df[['USDJPY', 'USDINR', 'EURUSD', 'GBPUSD']].corr()['USDJPY'].to_string())

    metrics = {
        'timeframe':  timeframe,
        'rmse':       rmse,
        'dir_acc':    dir_acc,
        'f1':         f1_weighted,
        'crash_f1':   f1_crash,
        'crash_prec': crash_prec,
        'crash_rec':  crash_rec,
        'n_test':     len(true_raw),
        'n_crash':    int(n_crash_actual),
        'n_pred_crash': int(n_crash_predicted),
    }
    save_graph(true_norm, pred_norm, true_raw, pred_raw, timeframe, metrics)
    return metrics

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  CNN-QUANTUM TRANSFORMER | FINAL SUBMISSION | CENTERED DETECTION")
    print("=" * 65)

    save_circuit_diagram()

    results = []
    for tf in ['1d', '1wk', '1mo']:
        r = run_experiment(tf)
        if r:
            results.append(r)

    print(f"\n{'='*70}")
    print("  FINAL RESULTS — SEED 42")
    print(f"{'='*70}")
    print(f"  {'TF':<6} {'Dir Acc':>8} {'Crash Prec':>12} {'Crash Rec':>11} {'Crash F1':>10} {'n_crash':>9}")
    print(f"  {'─'*60}")
    for r in results:
        print(f"  {r['timeframe']:<6} {r['dir_acc']:>7.1f}% "
              f"{r['crash_prec']:>11.4f} "
              f"{r['crash_rec']:>10.4f} "
              f"{r['crash_f1']:>9.4f} "
              f"{r['n_crash']:>6}/{r['n_test']}")
    print(f"{'='*70}")
    print("\n[DONE] Code execution finished. Ready for paper submission.")
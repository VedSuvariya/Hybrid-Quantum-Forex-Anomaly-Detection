# Multi-Scale Volatility Anomaly Detection in USD/JPY Markets via a Hybrid CNN–Variational Quantum Circuit

> **Official implementation of the research paper submitted for peer review.**

---

## Abstract

We present a hybrid quantum-classical architecture for multi-scale volatility anomaly (crash) detection in the USD/JPY foreign exchange market. The system combines a 2D Convolutional Neural Network (CNN) for spatial feature extraction, a 4-qubit Variational Quantum Circuit (VQC) with an EfficientSU2 ansatz for non-linear Hilbert-space interactions, and a Multi-Head Attention decoder. The architecture is evaluated simultaneously across three horizons — daily (1D), weekly (1WK), and monthly (1MO) — under a strictly enforced zero-leakage data pipeline.

---

## Key Results

| Timeframe | Test Samples | RMSE (Yen) | Dir Acc | Crash Precision | Crash Recall | Crash F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D** | 98 | 0.8212 | 59.2% | 0.2188 | 0.2500 | 0.2333 |
| **1WK** | 48 | 1.7570 | 62.5% | 0.2500 | 0.3571 | 0.2941 |
| **1MO** | 51 | 6.0916 | 39.2% | 0.5000 | 0.3529 | 0.4138 |

**Analysis:** Crash F1 improves monotonically across horizons (0.23 → 0.29 → 0.41), demonstrating strengthening anomaly detection as crash event geometry becomes structurally distinct from noise at coarser timescales. The 1MO Crash Precision of 0.50 represents a 1.5× lift over the naive base rate of 33.3%.

---

## Novel Contributions

### 1. CrashFocusedLoss
An asymmetric Huber loss function designed to eliminate MSE regression-to-mean degeneracy.
- **Huber Base Loss (δ = 1.0):** Robustness to heavy-tailed outliers.
- **Directional Penalty:** 3× multiplier for sign-incorrect predictions.
- **Crash Amplification:** 10× multiplier for minority-class crash events.
- **Result:** A **30:1 effective penalty ratio** for wrong-direction crash predictions.

### 2. Zero-Leakage Data Pipeline
All normalization statistics (Volume/VIX bounds, rolling standard deviations) are computed exclusively on training-set samples. Test-set transformations use frozen training parameters, eliminating future-data contamination common in financial ML literature.

### 3. Barren Plateau Mitigation via Staged CNN Freeze
CNN parameters are frozen at epoch 20. This prevents the ~4,000-parameter classical gradient from numerically dominating the ~30-parameter quantum gradient, ensuring VQC trainability grounded in Cerezo et al. (2021).

### 4. Statistically Centered Anomaly Threshold
Crash events are flagged based on deviations from the empirical mean of the prediction series — not from an absolute zero. This makes metrics robust to the "amplitude dampening" artifact typical of neural regression models.

---

## Architecture

```
Price Window
     |
[Gramian Angular Summation Field (GASF)]  -->  W×W image
     |
[CNN: Conv2d(1->4, k=2×2) + ReLU + Flatten]
     |
[Fusion: CNN features + Volume (W) + VIX (W)]
     |
[FC Bridge: Linear(->32) + ReLU + Dropout(0.2) + Linear(->4) + Tanh]
     |   z_q bounded in [-1, 1]^4
[Quantum VQC: ZZFeatureMap(reps=1) + EfficientSU2(reps=1)]
     |   gradients via parameter-shift rule through TorchConnector
[Multi-Head Attention: embed(1->8), 2 heads, d_k=4]
     |
[Linear(8->1)]  -->  Predicted z-score momentum
```

**Quantum Circuit:** 4 qubits | ZZ pairwise entanglement | ~30 trainable parameters

---

## Data Sources

All data is fetched live from Yahoo Finance via `yfinance` at runtime.

| Symbol | Description | Role |
| :--- | :--- | :--- |
| `USDJPY=X` | USD/JPY close + volume | Primary prediction target |
| `USDINR=X` | USD/INR close | Cross-currency feature |
| `EURUSD=X` | EUR/USD close | Cross-currency feature |
| `GBPUSD=X` | GBP/USD close | Cross-currency feature |
| `^VIX` | CBOE Volatility Index | Market fear indicator |

---

## Requirements

### Python Version
**Python 3.10.x is required.** `qiskit-machine-learning` is not currently compatible with Python 3.11 or higher.

### Dependencies
```bash
pip install yfinance pandas numpy torch scikit-learn matplotlib scipy pylatexenc
pip install qiskit==0.45.3 qiskit-machine-learning==0.7.2 qiskit-algorithms==0.3.0
```

---

## How to Run

### On Windows (VS Code / PowerShell)
```powershell
$env:PYTHONIOENCODING="utf-8"; python main.py
```

### On Linux / Mac
```bash
python main.py
```

### On Google Colab (Recommended — Free T4 GPU)
1. Upload `main.py` to your Colab environment.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU.**
3. Run the `pip install` commands above.
4. Run the script:
```bash
!python main.py
```

---

## Output Files

All files are saved in the same directory as the script after execution:

| File | Description |
| :--- | :--- |
| `Circuit_Diagram.png` | Quantum circuit visualization |
| `RawData_1d.csv` | Raw fetched forex data — daily |
| `RawData_1wk.csv` | Raw fetched forex data — weekly |
| `RawData_1mo.csv` | Raw fetched forex data — monthly |
| `Prediction_1D.png` | Actual vs. predicted plots — daily |
| `Prediction_1WK.png` | Actual vs. predicted plots — weekly |
| `Prediction_1MO.png` | Actual vs. predicted plots — monthly |

---

## Reproducibility

Experiments use a fixed global seed of **42** across Python, NumPy, and PyTorch to ensure identical results across machines.

---

## Quantum Simulation Note

All variational quantum circuit computations are performed using Qiskit's statevector simulator executed on classical hardware. The circuit is interfaced with PyTorch via `TorchConnector`. No real quantum hardware was used in this study, consistent with standard practice in current QML research.

---


---

## License

This repository is released for academic and research purposes only.

import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-cfm-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── CFM model hyperparameters (Lipman et al. 2022) ───────────────────────────

# Feature window: number of lagged return bars fed as condition vector
FEATURE_WINDOW = 21        # ~1 month of lags as conditioning context

# Prediction horizon: forward log-return to predict
PRED_HORIZON = 21

# Flow network architecture
CFM_HIDDEN_DIM  = 64       # hidden layer width
CFM_N_LAYERS    = 3        # number of hidden layers in velocity field network
CFM_TIME_DIM    = 16       # sinusoidal time embedding dimension

# Training
CFM_N_EPOCHS    = 80
CFM_LR          = 3e-3
CFM_BATCH_SIZE  = 32
CFM_SIGMA_MIN   = 1e-4     # OT-CFM interpolant noise floor

# Inference: number of ODE integration steps (Euler method)
CFM_ODE_STEPS   = 20

# Number of samples drawn from learned distribution for score construction
CFM_N_SAMPLES   = 200

# Score construction from generated samples:
#   mean_score   : E[X_1] under learned distribution (direction signal)
#   sharpe_score : E[X_1] / std(X_1)  (risk-adjusted signal)
#   tail_score   : P(X_1 > 0) - 0.5   (probability of positive return)
WEIGHT_MEAN   = 0.50
WEIGHT_SHARPE = 0.30
WEIGHT_TAIL   = 0.20

TOP_N = 3

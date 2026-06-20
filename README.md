# 🌊 P2-ETF-CONDITIONAL-FLOW-MATCHING

**Conditional Flow Matching Engine — Lipman et al. (2022) OT-CFM**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine applies **Conditional Flow Matching (CFM)** to learn the
macro-conditioned distribution of ETF forward returns, then uses samples
from that distribution to rank ETFs by expected risk-adjusted performance.

CFM is the current state of the art in generative modelling, having
superseded both diffusion models and normalising flows in speed, sample
quality, and theoretical simplicity.

---

## Theory

### Flow Matching (Lipman et al. 2022)

Flow matching learns a time-dependent velocity field v_θ(x, t, c) such that
integrating the ODE transports a simple source p₀ = N(0,1) to the target
return distribution p₁:

```
dx/dt = v_θ(x, t, c)     x(0) ~ N(0,1)  →  x(1) ~ p₁(·|c)
```

### OT-CFM Interpolant (Tong et al. 2023)

The Optimal Transport CFM uses a straight-line interpolant between source
x₀ ~ N(0,1) and target x₁ (forward return):

```
x_t  = (1 − (1−σ_min)·t)·x₀  +  t·x₁        (interpolated point)
u_t  = x₁ − (1−σ_min)·x₀                     (target velocity, constant in t)
```

**Training objective:**
```
L(θ) = E_{t, x₁~p_data, x₀~N(0,1)} [ ‖v_θ(x_t, t, c) − u_t‖² ]
```

### Why CFM Supersedes Prior Approaches

| Method | Path geometry | ODE steps | Architecture constraint |
|--------|--------------|-----------|------------------------|
| DDPM / Score diffusion | Curved SDE | 100–1000 | Noise schedule tuning |
| Normalising Flow | Bijective exact | 1 | Invertible coupling layers |
| **OT-CFM (this engine)** | **Straight line** | **20** | **Unconstrained MLP** |

Straight-line paths → constant velocity fields → regression is trivial →
faster convergence, smaller networks, fewer integration steps.

### Conditioning

Each ETF model is conditioned on:
```
c = [lagged_returns(21d, z-scored),  ΔV IX,  ΔDXY,  ΔT10Y2Y,  ΔIG_SPREAD,  ΔHY_SPREAD]
```

The learned distribution p₁(·|c) captures how the forward-return distribution
changes with recent path and macro regime.

### Score Construction

After training, draw N=200 samples from p₁(·|c_today) via Euler integration:

| Component | Formula | Weight | Signal |
|-----------|---------|--------|--------|
| Mean | E[X₁ \| c] | 50% | Expected forward return |
| Sharpe | E/std | 30% | Risk-adjusted return |
| Tail prob | P(X₁>0)−0.5 | 20% | Probability of positive return |

Final score: **cross-sectional z-score** per universe per window.

---

## Network Architecture

**Velocity field v_θ(x, t, c):** MLP with sinusoidal time embedding

```
Input: [x(1d) ‖ sin/cos_embed(t, 16d) ‖ c(cond_dim)]
        ↓
    Linear → Tanh  (×3 layers, hidden_dim=64)
        ↓
    Linear → x_dot (1d)
```

**Training:** Adam optimiser, 80 epochs, lr=3e-3, batch_size=32

**Inference:** Euler integration, 20 steps, t: 0→1

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d` (run in parallel via matrix strategy)

---

## Repository Structure

```
P2-ETF-CONDITIONAL-FLOW-MATCHING/
├── config.py          # Universes, CFM hyperparameters, score weights
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── cfm_engine.py      # Core: OT-CFM interpolant, MLP velocity field, ODE solver
├── trainer.py         # Orchestrator: --window N (shard) or --merge
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Parallel matrix: 4 window jobs + merge job
```

---

## GitHub Actions Flow

```
23:30 UTC  ┌─ window=63  ─┐
            ├─ window=126 ─┤  (parallel, fail-fast: false)
            ├─ window=252 ─┤
            └─ window=504 ─┘
                   ↓ (all complete)
              merge job
           downloads shards →
           builds Tab1 + Tab2 JSON →
           uploads to HuggingFace
```

Each window job writes `shards/shard_N.json` and uploads as a GitHub artifact.
The merge job downloads all artifacts and combines them.

---

## Output JSON Schemas

### Tab 1 — `cfm_engine_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-20",
  "universes": {
    "FI_COMMODITIES": {
      "top_etfs": [
        {"ticker": "TLT", "cfm_score": 1.32, "best_window": 252}
      ],
      "full_scores": {
        "TLT": {"score": 1.32, "best_window": 252}
      }
    }
  }
}
```

### Tab 2 — `cfm_engine_windows_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-20",
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63":  {"top_etfs": [...], "full_ranking": [["TLT", 1.32], ...]},
        "252": {"top_etfs": [...], "full_ranking": [...]}
      }
    }
  }
}
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-CONDITIONAL-FLOW-MATCHING
cd P2-ETF-CONDITIONAL-FLOW-MATCHING
pip install -r requirements.txt

export HF_TOKEN=hf_...

# Run a single window locally
python trainer.py --window 252

# Or merge existing shards
python trainer.py --merge

streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-cfm-results`

---

## Relationship to Existing Generative Engines

| Engine | Method | CFM advantage |
|--------|--------|---------------|
| NORMALIZING-FLOW | Coupling layers (RealNVP) | No invertibility constraint |
| SCORE-DIFFUSION | Reverse SDE / DDPM | 20 vs 1000 steps |
| TEMPORAL-GAN | Adversarial training | Stable training, no mode collapse |
| VARIATIONAL-AUTOENCODER | ELBO lower bound | Exact likelihood, not a bound |
| **CFM (this engine)** | **OT straight-line ODE** | **Fastest, simplest, SOTA 2022** |

---

## References

- Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M. & Le, M. (2022).
  Flow Matching for Generative Modeling. *ICLR 2023*.
- Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K.,
  Wolf, G. & Bengio, Y. (2023). Improving and Generalizing Flow-Matching for
  Conditional Generation. *ICML 2023 Workshop on Structured Probabilistic Inference*.
- Albergo, M.S. & Vanden-Eijnden, E. (2023). Building Normalizing Flows with
  Stochastic Interpolants. *ICLR 2023*.
- Liu, X., Gong, C. & Liu, Q. (2022). Flow Straight and Fast: Learning to
  Generate and Transfer Data with Rectified Flow. *ICLR 2023*.

"""
cfm_engine.py — Conditional Flow Matching Engine
=================================================

Theory
------
**Flow Matching** (Lipman et al. 2022) learns a time-dependent velocity field
v_θ(x, t) such that integrating the ODE:

    dx/dt = v_θ(x, t),   x(0) ~ p_0  (simple source, e.g. N(0,1))

transports the source distribution p_0 to the target distribution p_1
(the empirical distribution of forward returns).

The key insight: instead of the intractable marginal flow matching objective,
we condition on individual data points x₁ and regress against the
*conditional* vector field, which has a simple closed form.

**Optimal Transport CFM (OT-CFM, Tong et al. 2023):**
The interpolant between source x₀ ~ N(0,1) and target x₁ is:

    x_t = (1 − (1−σ_min)·t) · x₀  +  t · x₁

The conditional velocity field is:

    u_t(x | x₁) = x₁ − (1 − σ_min) · x₀

Training objective (regression on velocity field):

    L(θ) = E_{t~U[0,1], x₁~p_data, x₀~N(0,1)} [ ‖v_θ(x_t, t, c) − u_t‖² ]

Where c is the conditioning context (recent return features + macro signals).

**Conditioning:**
Each ETF's model is conditioned on:
  c = [lagged_returns(FEATURE_WINDOW),  macro_changes(MACRO_COLS)]

This makes the learned distribution p_1(·|c) depend on the recent path and
macro state — the model learns how the forward-return distribution shifts
with context.

**Score construction:**
After training, we draw N samples from p_1(·|c_today) by integrating the
ODE from x₀ ~ N(0,1):

    x(0) = x₀,   dx/dt = v_θ(x, t, c_today),   t: 0 → 1

From the N samples {x̂¹,...,x̂ᴺ}:
  mean_score   = mean(samples)             — expected forward return
  sharpe_score = mean / std               — risk-adjusted return
  tail_score   = P(sample > 0) − 0.5     — probability of positive return

Composite = weighted blend, cross-sectionally z-scored per universe/window.

**Why CFM beats diffusion / normalising flows:**
  - Diffusion: requires T → ∞ noise schedule, slow reverse SDE/ODE
  - NF: requires invertible architecture (coupling layers, volume preservation)
  - CFM: straight-line ODE paths → fewer integration steps, simpler network,
         provably faster convergence (Lipman et al. 2022, Tong et al. 2023)

References
----------
- Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M. & Le, M. (2022).
  Flow Matching for Generative Modeling. ICLR 2023.
- Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J.,
  Fatras, K., Wolf, G. & Bengio, Y. (2023).
  Improving and Generalizing Flow-Matching for Conditional Generation.
  ICML 2023 Workshop.
- Albergo, M.S. & Vanden-Eijnden, E. (2023).
  Building Normalizing Flows with Stochastic Interpolants. ICLR 2023.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

import config


# ── Pure-numpy neural network (no torch dependency) ──────────────────────────
# Velocity field network: MLP with sinusoidal time embedding
# Architecture: [x_dim + time_dim + cond_dim] → hidden → ... → x_dim

class SinusoidalTimeEmbedding:
    """Sinusoidal positional embedding for scalar time t ∈ [0,1]."""
    def __init__(self, dim: int):
        self.dim = dim
        half = dim // 2
        self.freqs = np.array([10000 ** (-2 * i / dim) for i in range(half)])

    def __call__(self, t: float) -> np.ndarray:
        args = t * self.freqs
        emb  = np.concatenate([np.sin(args), np.cos(args)])
        return emb[:self.dim]


class MLP:
    """
    Simple MLP with tanh activations.
    Weights stored as list of (W, b) tuples.
    """
    def __init__(self, layer_sizes: List[int], rng: np.random.Generator):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialisation
            scale = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, scale, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.layers.append([W, b])

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def parameters(self):
        """Yield all (W, b) pairs as flat arrays for gradient updates."""
        return self.layers

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([np.concatenate([W.ravel(), b.ravel()])
                                for W, b in self.layers])

    def set_flat_params(self, flat: np.ndarray):
        idx = 0
        for W, b in self.layers:
            nW = W.size
            W[:] = flat[idx:idx+nW].reshape(W.shape); idx += nW
            nb = b.size
            b[:] = flat[idx:idx+nb];                  idx += nb


class VelocityField:
    """
    v_θ(x, t, c): maps (x_t, t, context) → velocity dx/dt

    Input  : [x(1d), time_emb(time_dim), context(cond_dim)]
    Output : x_dot (1d)
    """
    def __init__(self, cond_dim: int, rng: np.random.Generator):
        self.time_emb = SinusoidalTimeEmbedding(config.CFM_TIME_DIM)
        in_dim   = 1 + config.CFM_TIME_DIM + cond_dim
        sizes    = [in_dim] + [config.CFM_HIDDEN_DIM] * config.CFM_N_LAYERS + [1]
        self.net = MLP(sizes, rng)

    def __call__(self, x: float, t: float, c: np.ndarray) -> float:
        t_emb  = self.time_emb(t)
        inp    = np.concatenate([[x], t_emb, c])
        return float(self.net.forward(inp)[0])

    def forward_batch(self, X: np.ndarray, t: float,
                      C: np.ndarray) -> np.ndarray:
        """Vectorised forward pass for a batch of (x, c) pairs at fixed t."""
        t_emb = self.time_emb(t)
        T_emb = np.tile(t_emb, (len(X), 1))
        inp   = np.hstack([X.reshape(-1, 1), T_emb, C])
        return self.net.forward(inp).ravel()


# ── OT-CFM training ───────────────────────────────────────────────────────────

def _ot_interpolant(x0: np.ndarray, x1: np.ndarray,
                    t: float, sigma_min: float) -> np.ndarray:
    """x_t = (1 − (1−σ_min)·t)·x₀ + t·x₁"""
    return (1 - (1 - sigma_min) * t) * x0 + t * x1


def _ot_velocity(x0: np.ndarray, x1: np.ndarray,
                 sigma_min: float) -> np.ndarray:
    """u_t = x₁ − (1−σ_min)·x₀   (constant in t for OT-CFM)"""
    return x1 - (1 - sigma_min) * x0


def _compute_loss_and_grad(
    vf:         VelocityField,
    X1_batch:   np.ndarray,   # (B,) target samples
    C_batch:    np.ndarray,   # (B, cond_dim) conditions
    t_batch:    np.ndarray,   # (B,) times
    x0_batch:   np.ndarray,   # (B,) source samples
    sigma_min:  float,
    eps:        float = 1e-5,
) -> Tuple[float, list]:
    """
    Compute MSE loss and parameter gradients via finite differences.
    (Pure numpy — no autograd. Efficient enough for small MLP on daily data.)
    """
    B = len(X1_batch)

    # Forward pass for all batch elements
    losses = []
    for i in range(B):
        t  = t_batch[i]
        x0 = x0_batch[i]
        x1 = X1_batch[i]
        c  = C_batch[i]

        xt  = _ot_interpolant(x0, x1, t, sigma_min)
        ut  = _ot_velocity(x0, x1, sigma_min)
        vt  = vf(xt, t, c)
        losses.append((vt - ut) ** 2)

    loss = float(np.mean(losses))

    # Gradient via central finite differences on flat parameter vector
    flat  = vf.net.get_flat_params()
    grads = np.zeros_like(flat)

    for k in range(len(flat)):
        flat_p = flat.copy(); flat_p[k] += eps
        flat_m = flat.copy(); flat_m[k] -= eps

        vf.net.set_flat_params(flat_p)
        lp = np.mean([(vf(_ot_interpolant(x0_batch[i], X1_batch[i], t_batch[i], sigma_min),
                          t_batch[i], C_batch[i])
                       - _ot_velocity(x0_batch[i], X1_batch[i], sigma_min)) ** 2
                      for i in range(B)])

        vf.net.set_flat_params(flat_m)
        lm = np.mean([(vf(_ot_interpolant(x0_batch[i], X1_batch[i], t_batch[i], sigma_min),
                          t_batch[i], C_batch[i])
                       - _ot_velocity(x0_batch[i], X1_batch[i], sigma_min)) ** 2
                      for i in range(B)])

        grads[k] = (lp - lm) / (2 * eps)
        vf.net.set_flat_params(flat)   # restore

    return loss, grads


def _train_cfm(
    X1:       np.ndarray,   # (N,) target distribution samples (forward returns)
    C:        np.ndarray,   # (N, cond_dim) conditioning contexts
    rng:      np.random.Generator,
) -> VelocityField:
    """
    Train OT-CFM velocity field on (X1, C) pairs.
    Uses Adam optimiser with finite-difference gradients.
    """
    cond_dim = C.shape[1]
    vf       = VelocityField(cond_dim, rng)
    N        = len(X1)

    # Adam state
    flat   = vf.net.get_flat_params()
    m      = np.zeros_like(flat)
    v      = np.zeros_like(flat)
    lr     = config.CFM_LR
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    step   = 0

    B       = min(config.CFM_BATCH_SIZE, N)
    sigma   = config.CFM_SIGMA_MIN

    for epoch in range(config.CFM_N_EPOCHS):
        idx      = rng.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, B):
            batch_idx = idx[start:start + B]
            if len(batch_idx) < 2:
                continue

            X1_b = X1[batch_idx]
            C_b  = C[batch_idx]
            t_b  = rng.uniform(0, 1, len(batch_idx))
            x0_b = rng.standard_normal(len(batch_idx))

            loss, grads = _compute_loss_and_grad(
                vf, X1_b, C_b, t_b, x0_b, sigma)

            # Adam update
            step += 1
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads ** 2
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            flat = vf.net.get_flat_params()
            flat -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            vf.net.set_flat_params(flat)

            epoch_loss += loss
            n_batches  += 1

        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / max(n_batches, 1)
            print(f"    epoch {epoch+1}/{config.CFM_N_EPOCHS}  loss={avg:.6f}")

    return vf


# ── ODE integration (Euler) ───────────────────────────────────────────────────

def _euler_integrate(
    vf:    VelocityField,
    x0:    float,
    c:     np.ndarray,
    steps: int,
) -> float:
    """Integrate dx/dt = v_θ(x, t, c) from t=0 to t=1 using Euler method."""
    x = x0
    dt = 1.0 / steps
    for i in range(steps):
        t  = i * dt
        vt = vf(x, t, c)
        x  = x + dt * vt
    return x


# ── Feature construction ──────────────────────────────────────────────────────

def _build_condition(
    log_ret:   np.ndarray,   # full log-return history up to t
    macro_row: np.ndarray,   # macro signal values at t (standardised)
    feat_win:  int,
) -> np.ndarray:
    """
    Build the conditioning context vector for time t:
      c = [lagged_returns (feat_win bars, standardised), macro_changes]
    """
    if len(log_ret) < feat_win:
        return None
    ret_lags = log_ret[-feat_win:]
    # Standardise lags
    mu  = ret_lags.mean()
    std = ret_lags.std() + 1e-8
    ret_lags = (ret_lags - mu) / std

    c = np.concatenate([ret_lags, macro_row])
    return c.astype(np.float64)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_cfm_scores(
    prices:   pd.DataFrame,
    macro_df: pd.DataFrame,
    tickers:  List[str],
    window:   int,
) -> pd.Series:
    """
    Train a CFM model per ETF and return cross-sectional z-scores.

    For each ETF:
      1. Build (condition, forward_return) pairs over the rolling window
      2. Train OT-CFM velocity field on these pairs
      3. Sample from learned distribution conditioned on today's context
      4. Compute composite score from samples

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite CFM z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.PRED_HORIZON + config.FEATURE_WINDOW + 10
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    # Align macro to price index
    common   = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a = prices.loc[common]
    macro_a  = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    # Standardise macro
    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    m_mu  = np.nanmean(macro_vals, axis=0, keepdims=True) if macro_vals.shape[1] > 0 else macro_vals
    m_std = np.nanstd(macro_vals, axis=0, keepdims=True) + 1e-8
    macro_norm = (macro_vals - m_mu) / m_std if macro_vals.shape[1] > 0 else macro_vals

    cond_dim = config.FEATURE_WINDOW + macro_vals.shape[1]
    rng      = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values
        n_total = len(log_ret)

        # ── Build training pairs over the rolling window ──────────────────────
        # Training range: [window_start .. n_total - PRED_HORIZON - 1]
        train_start = max(config.FEATURE_WINDOW, n_total - window - config.PRED_HORIZON)
        train_end   = n_total - config.PRED_HORIZON

        conditions  = []
        fwd_returns = []

        for t in range(train_start, train_end):
            macro_row = macro_norm[t] if macro_norm.shape[1] > 0 else np.array([])
            c = _build_condition(log_ret[:t], macro_row, config.FEATURE_WINDOW)
            if c is None:
                continue
            fwd = log_ret[t:t + config.PRED_HORIZON].mean()
            if np.isnan(fwd):
                continue
            conditions.append(c)
            fwd_returns.append(fwd)

        if len(conditions) < config.CFM_BATCH_SIZE:
            continue

        X1 = np.array(fwd_returns)
        C  = np.array(conditions)

        # Standardise target
        x1_mu  = X1.mean()
        x1_std = X1.std() + 1e-8
        X1_norm = (X1 - x1_mu) / x1_std

        # ── Train CFM ─────────────────────────────────────────────────────────
        print(f"    Training CFM for {ticker}  (N={len(X1)}, cond_dim={cond_dim})")
        try:
            vf = _train_cfm(X1_norm, C, rng)
        except Exception as e:
            print(f"    Training failed for {ticker}: {e}")
            continue

        # ── Build today's condition ───────────────────────────────────────────
        macro_today = macro_norm[-1] if macro_norm.shape[1] > 0 else np.array([])
        c_today = _build_condition(log_ret, macro_today, config.FEATURE_WINDOW)
        if c_today is None:
            continue

        # ── Sample from learned distribution ──────────────────────────────────
        samples_norm = np.array([
            _euler_integrate(vf, rng.standard_normal(), c_today, config.CFM_ODE_STEPS)
            for _ in range(config.CFM_N_SAMPLES)
        ])
        # Denormalise
        samples = samples_norm * x1_std + x1_mu

        # ── Score components ──────────────────────────────────────────────────
        s_mean   = float(np.mean(samples))
        s_sharpe = float(np.mean(samples) / (np.std(samples) + 1e-8))
        s_tail   = float(np.mean(samples > 0) - 0.5)

        composite = (
            config.WEIGHT_MEAN   * s_mean
            + config.WEIGHT_SHARPE * s_sharpe
            + config.WEIGHT_TAIL   * s_tail
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu  = scores.mean()
    std = scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std

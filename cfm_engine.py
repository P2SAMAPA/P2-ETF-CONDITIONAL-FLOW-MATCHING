"""
cfm_engine.py — Conditional Flow Matching Engine (fast numpy backprop)
=======================================================================

Theory: see original docstring — OT-CFM, Lipman et al. 2022.

This version replaces finite-difference gradients with analytical backprop
through the MLP, making training O(P) per step instead of O(P²).

Network: tanh-MLP with sinusoidal time embedding.
Optimiser: Adam with analytical gradients.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

import config


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalTimeEmbedding:
    def __init__(self, dim: int):
        self.dim  = dim
        half      = dim // 2
        self.freqs = np.array([10000 ** (-2 * i / dim) for i in range(half)])

    def __call__(self, t: float) -> np.ndarray:
        args = t * self.freqs
        emb  = np.concatenate([np.sin(args), np.cos(args)])
        return emb[:self.dim]

    def batch(self, T: np.ndarray) -> np.ndarray:
        """T shape (B,) → (B, dim)"""
        args = T[:, None] * self.freqs[None, :]        # (B, half)
        return np.concatenate([np.sin(args), np.cos(args)], axis=1)[:, :self.dim]


# ── MLP with analytical forward + backward ────────────────────────────────────

class MLP:
    """
    Tanh MLP.  All layers use tanh except the last (linear output).
    Stores weights as lists of W, b arrays for easy gradient computation.
    """
    def __init__(self, layer_sizes: List[int], rng: np.random.Generator):
        self.Ws, self.bs = [], []
        for i in range(len(layer_sizes) - 1):
            fin, fout = layer_sizes[i], layer_sizes[i+1]
            self.Ws.append(rng.normal(0, np.sqrt(2.0/fin), (fin, fout)))
            self.bs.append(np.zeros(fout))
        self.n_layers = len(self.Ws)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        X: (B, in_dim)
        Returns (output (B, out_dim), cache list for backward)
        """
        cache = []
        A = X
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            Z = A @ W + b                          # (B, out)
            if i < self.n_layers - 1:
                A_new = np.tanh(Z)
                cache.append((A, Z, A_new))        # pre-act, linear, post-act
                A = A_new
            else:
                cache.append((A, Z, Z))            # output layer: linear
                A = Z
        return A, cache

    def backward(self, dL_dout: np.ndarray, cache: list) -> Tuple[list, list, np.ndarray]:
        """
        dL_dout: (B, out_dim)
        Returns (dWs, dbs, dX)
        """
        dWs = [None] * self.n_layers
        dbs = [None] * self.n_layers
        dA  = dL_dout

        for i in reversed(range(self.n_layers)):
            A_in, Z, A_out = cache[i]
            # Apply activation derivative (tanh' = 1 - tanh²)
            if i < self.n_layers - 1:
                dZ = dA * (1.0 - A_out ** 2)
            else:
                dZ = dA                            # linear output

            dWs[i] = A_in.T @ dZ / len(dA)
            dbs[i] = dZ.mean(axis=0)
            dA     = dZ @ self.Ws[i].T

        return dWs, dbs, dA

    def apply_adam(self, dWs, dbs, ms, vs, step, lr, b1=0.9, b2=0.999, eps=1e-8):
        """In-place Adam update.
        ms = [(mW0,mb0), (mW1,mb1), ...]
        vs = [(vW0,vb0), (vW1,vb1), ...]
        """
        for i in range(self.n_layers):
            mW, mb = ms[i]
            vW, vb = vs[i]
            for param, grad, mp, vp in [
                (self.Ws[i], dWs[i], mW, vW),
                (self.bs[i], dbs[i], mb, vb),
            ]:
                mp[:] = b1 * mp + (1 - b1) * grad
                vp[:] = b2 * vp + (1 - b2) * grad ** 2
                m_hat = mp / (1 - b1 ** step)
                v_hat = vp / (1 - b2 ** step)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def init_adam_state(self):
        ms = [(np.zeros_like(W), np.zeros_like(b)) for W, b in zip(self.Ws, self.bs)]
        vs = [(np.zeros_like(W), np.zeros_like(b)) for W, b in zip(self.Ws, self.bs)]
        return ms, vs


# ── Velocity field ────────────────────────────────────────────────────────────

class VelocityField:
    def __init__(self, cond_dim: int, rng: np.random.Generator):
        self.t_emb   = SinusoidalTimeEmbedding(config.CFM_TIME_DIM)
        in_dim       = 1 + config.CFM_TIME_DIM + cond_dim
        sizes        = [in_dim] + [config.CFM_HIDDEN_DIM] * config.CFM_N_LAYERS + [1]
        self.net     = MLP(sizes, rng)
        self.cond_dim = cond_dim

    def forward_batch(self, X: np.ndarray, T: np.ndarray,
                      C: np.ndarray) -> Tuple[np.ndarray, list, np.ndarray]:
        """
        X: (B,)  scalar x values
        T: (B,)  times in [0,1]
        C: (B, cond_dim)
        Returns: v (B,), cache, inp (B, in_dim)
        """
        t_emb = self.t_emb.batch(T)                     # (B, time_dim)
        inp   = np.concatenate([X[:, None], t_emb, C], axis=1)   # (B, in_dim)
        v, cache = self.net.forward(inp)
        return v.ravel(), cache, inp

    def backward_batch(self, dL_dv: np.ndarray, cache: list):
        dWs, dbs, _ = self.net.backward(dL_dv[:, None], cache)
        return dWs, dbs


# ── OT-CFM helpers ────────────────────────────────────────────────────────────

def _interpolant(x0, x1, t, s=config.CFM_SIGMA_MIN):
    return (1 - (1 - s) * t) * x0 + t * x1

def _target_vel(x0, x1, s=config.CFM_SIGMA_MIN):
    return x1 - (1 - s) * x0


# ── Training ──────────────────────────────────────────────────────────────────

def _train_cfm(X1: np.ndarray, C: np.ndarray,
               rng: np.random.Generator) -> VelocityField:
    """Train OT-CFM with analytical Adam backprop. No finite differences."""
    cond_dim = C.shape[1]
    vf       = VelocityField(cond_dim, rng)
    ms, vs   = vf.net.init_adam_state()
    N        = len(X1)
    B        = min(config.CFM_BATCH_SIZE, N)
    step     = 0

    for epoch in range(config.CFM_N_EPOCHS):
        idx        = rng.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, B):
            bi = idx[start:start + B]
            if len(bi) < 2:
                continue

            x1_b = X1[bi]
            c_b  = C[bi]
            t_b  = rng.uniform(0, 1, len(bi))
            x0_b = rng.standard_normal(len(bi))

            # Interpolate
            xt_b = _interpolant(x0_b, x1_b, t_b)  # (B,)
            ut_b = _target_vel(x0_b, x1_b)         # (B,)

            # Forward
            v_b, cache, _ = vf.forward_batch(xt_b, t_b, c_b)

            # MSE loss: L = mean((v - u)²)
            residual = v_b - ut_b                  # (B,)
            loss     = float(np.mean(residual ** 2))

            # Backward: dL/dv = 2*(v-u)/B
            dL_dv = 2.0 * residual / len(bi)
            dWs, dbs = vf.backward_batch(dL_dv, cache)

            step += 1
            vf.net.apply_adam(dWs, dbs, ms, vs, step, config.CFM_LR)

            epoch_loss += loss
            n_batches  += 1

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1}/{config.CFM_N_EPOCHS}  "
                  f"loss={epoch_loss/max(n_batches,1):.6f}")

    return vf


# ── ODE integration (Euler) ───────────────────────────────────────────────────

def _euler_integrate(vf: VelocityField, x0_batch: np.ndarray,
                     c: np.ndarray, steps: int) -> np.ndarray:
    """
    Integrate for a whole batch of x0 values at once.
    x0_batch: (N_samples,)
    c:        (cond_dim,)
    Returns:  (N_samples,) final positions
    """
    N  = len(x0_batch)
    C  = np.tile(c, (N, 1))
    x  = x0_batch.copy()
    dt = 1.0 / steps

    for i in range(steps):
        t      = np.full(N, i * dt)
        v, _, _ = vf.forward_batch(x, t, C)
        x      = x + dt * v

    return x


# ── Feature / condition builder ───────────────────────────────────────────────

def _build_condition(log_ret: np.ndarray, macro_row: np.ndarray,
                     feat_win: int):
    if len(log_ret) < feat_win:
        return None
    lags = log_ret[-feat_win:]
    mu, std = lags.mean(), lags.std() + 1e-8
    return np.concatenate([(lags - mu) / std, macro_row]).astype(np.float64)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_cfm_scores(prices: pd.DataFrame, macro_df: pd.DataFrame,
                       tickers: List[str], window: int) -> pd.Series:
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.PRED_HORIZON + config.FEATURE_WINDOW + 10
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    common   = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a = prices.loc[common]
    macro_a  = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu  = np.nanmean(macro_vals, axis=0, keepdims=True)
        m_std = np.nanstd(macro_vals,  axis=0, keepdims=True) + 1e-8
        macro_norm = (macro_vals - m_mu) / m_std
    else:
        macro_norm = macro_vals

    cond_dim   = config.FEATURE_WINDOW + macro_vals.shape[1]
    rng        = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values
        n_total = len(log_ret)

        train_start = max(config.FEATURE_WINDOW, n_total - window - config.PRED_HORIZON)
        train_end   = n_total - config.PRED_HORIZON

        conditions, fwd_returns = [], []
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

        x1_mu, x1_std = X1.mean(), X1.std() + 1e-8
        X1_norm = (X1 - x1_mu) / x1_std

        print(f"    Training CFM for {ticker}  (N={len(X1)}, cond_dim={cond_dim})")
        try:
            vf = _train_cfm(X1_norm, C, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            continue

        macro_today = macro_norm[-1] if macro_norm.shape[1] > 0 else np.array([])
        c_today = _build_condition(log_ret, macro_today, config.FEATURE_WINDOW)
        if c_today is None:
            continue

        x0_samples = rng.standard_normal(config.CFM_N_SAMPLES)
        samples_norm = _euler_integrate(vf, x0_samples, c_today, config.CFM_ODE_STEPS)
        samples = samples_norm * x1_std + x1_mu

        # Clamp to avoid exploding samples
        samples = np.clip(samples, -3 * x1_std, 3 * x1_std)

        s_mean   = float(np.mean(samples))
        s_sharpe = float(np.mean(samples) / (np.std(samples) + 1e-8))
        s_tail   = float(np.mean(samples > 0) - 0.5)

        composite = (config.WEIGHT_MEAN   * s_mean
                   + config.WEIGHT_SHARPE * s_sharpe
                   + config.WEIGHT_TAIL   * s_tail)
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std

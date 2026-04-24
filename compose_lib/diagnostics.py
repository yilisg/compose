"""Portfolio diagnostics — risk contributions and bootstrap CIs on weights."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from compose_lib.optimize import Constraints, Solution
from compose_lib.covariance import ledoit_wolf_cov

MONTHS = 12


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Decomposition of portfolio volatility into per-asset contributions.

    Columns:
    - weight
    - mrc  (marginal risk contribution = ∂σ/∂w_i = (Σw)_i / σ)
    - rc   (risk contribution = w_i · mrc_i; sums to σ)
    - pct  (rc / σ; sums to 1)
    """
    codes = list(cov.columns)
    w = weights.reindex(codes).values
    S = cov.values
    port_var = float(w @ S @ w)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else 0.0
    if port_vol == 0:
        return pd.DataFrame({
            "weight": w, "mrc": np.zeros(len(codes)),
            "rc": np.zeros(len(codes)), "pct": np.zeros(len(codes)),
        }, index=codes)
    mrc = S @ w / port_vol
    rc = w * mrc
    pct = rc / port_vol
    # Annualize vol-scale quantities for reporting
    mrc_ann = mrc * np.sqrt(MONTHS)
    rc_ann = rc * np.sqrt(MONTHS)
    return pd.DataFrame({
        "weight": w, "mrc": mrc_ann, "rc": rc_ann, "pct": pct,
    }, index=codes)


def bootstrap_weights(
    rets: pd.DataFrame,
    optimizer: Callable[..., Solution],
    cons: Constraints,
    rf_annual: float = 0.0,
    n_boot: int = 200,
    block: int = 6,
    seed: int = 0,
    cov_fn: Callable | None = None,
    **opt_kwargs,
) -> pd.DataFrame:
    """Stationary-block bootstrap of monthly returns → refit (μ, Σ) → solve →
    collect weights. Returns a DataFrame with rows = bootstrap draws,
    columns = asset codes.

    `block` is the expected block length (geometric). Default 6 months = a
    typical business-cycle quarter-ish.
    """
    rng = np.random.default_rng(seed)
    T, N = rets.shape
    codes = list(rets.columns)
    cov_fn = cov_fn or (lambda r: ledoit_wolf_cov(r).cov)

    out = np.full((n_boot, N), np.nan)
    for i in range(n_boot):
        idx = _stationary_bootstrap_index(T, block, rng)
        sample = rets.iloc[idx].reset_index(drop=True)
        mu_hat = sample.mean()
        cov_hat = cov_fn(sample)
        try:
            sol = optimizer(cov=cov_hat, mu=mu_hat, cons=cons,
                            rf_annual=rf_annual, **opt_kwargs)
            out[i] = sol.weights.reindex(codes).values
        except Exception:
            continue
    df = pd.DataFrame(out, columns=codes).dropna(how="all")
    return df


def _stationary_bootstrap_index(T: int, block: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: starts at a random index, then
    at each step either advances by 1 (prob = 1 - 1/block) or jumps to a
    new random index (prob = 1/block)."""
    p = 1.0 / max(1, block)
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(0, T)
    jumps = rng.random(T - 1) < p
    for t in range(1, T):
        idx[t] = rng.integers(0, T) if jumps[t - 1] else (idx[t - 1] + 1) % T
    return idx


def weight_ci(boot_df: pd.DataFrame, ci: float = 0.90) -> pd.DataFrame:
    """Summarize a bootstrap weight DataFrame: mean, median, CI bounds."""
    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    q = boot_df.quantile([lo, 0.5, hi]).T
    q.columns = [f"q{int(lo*100)}", "median", f"q{int(hi*100)}"]
    q["mean"] = boot_df.mean()
    q["std"] = boot_df.std()
    return q[["mean", "median", f"q{int(lo*100)}", f"q{int(hi*100)}", "std"]]
